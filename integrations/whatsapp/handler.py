import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import or_, select, update

from agent_runtime import (
    AGENT_AVAILABLE, graph_config, in_executor,
    invoke_sync, pending_interrupts, state_values,
)
from .client import (
    MAX_INTERACTIVE_BODY, mark_as_read, send_interactive, send_text,
)
from .db.connection import async_session
from .db.models import MessageRecord
from .db.repository import mark_processed, register_message
from .normalizer import WhatsAppMessage, _normalize_phone

logger = logging.getLogger("leyia-whatsapp")

MSG_HITL = ("⏸️ Tu solicitud está pendiente de revisión por un abogado del estudio. "
            "Te escribiremos apenas sea resuelta.")
MSG_ONLY_TEXT = "Por ahora solo puedo procesar mensajes de texto 🙏"
MSG_ERROR = "Tuvimos un problema técnico. Inténtalo en unos minutos."

PROCESSABLE_TYPES = ("text", "interactive")

_locks: dict[str, asyncio.Lock] = {}

# --- Configuración del Reaper ---
MAX_REAPER_RETRIES = 3
REAPER_BACKOFF_SECONDS = 120
NON_RETRYABLE_STATUS = "dead_letter"
RETRYABLE_EXHAUSTED_STATUS = "failed"


def _operator_number() -> str:
    """Número de la abogada/ejecutiva para alertas de HITL.
    Prioriza LAWYER_WHATSAPP_TO (igual que notify.py) y cae en
    WHATSAPP_OPERATOR_NUMBER para compatibilidad histórica.
    Lazy: se evalúa en cada envío para no depender del orden de carga de
    variables de entorno en el arranque."""
    raw = (os.getenv("LAWYER_WHATSAPP_TO")
           or os.getenv("WHATSAPP_OPERATOR_NUMBER") or "")
    return raw.replace("+", "").replace(" ", "").strip()


def _is_retryable(exc: Exception) -> bool:
    """
    True para errores transitorios de red/rate-limit.
    False para bugs de código (NameError, TypeError, AttributeError, etc.).
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


async def handle_messages(messages: list[WhatsAppMessage]) -> None:
    for msg in messages:
        # Normalizar una sola vez al recibir del webhook
        msg.customer_phone = _normalize_phone(msg.customer_phone)

        status = await register_message(msg)
        if status == "processed":
            continue

        lock = _locks.setdefault(msg.customer_phone, asyncio.Lock())
        async with lock:
            try:
                await process_message(msg)
                await mark_processed(msg.message_id)
                logger.info("Mensaje procesado %s", msg.message_id)
            except Exception:
                logger.exception("Fallo procesando %s", msg.message_id)
                # Registrar el intento fallido para que el Reaper respete backoff
                await _bump_failure_attempt(msg.message_id)


async def _bump_failure_attempt(message_id: str) -> None:
    """Incrementa retries/last_attempted_at sin cambiar el status."""
    try:
        async with async_session() as session:
            row = (
                await session.execute(
                    select(MessageRecord).where(
                        MessageRecord.whatsapp_message_id == message_id
                    )
                )
            ).scalar_one_or_none()

            if row:
                row.retries = (row.retries or 0) + 1
                row.last_attempted_at = datetime.now(timezone.utc)
                await session.commit()
    except Exception:
        logger.exception("No se pudo registrar intento fallido para %s", message_id)


async def process_message(msg: WhatsAppMessage) -> None:
    """
    Responde al cliente y, solo si todo sale bien, marca el mensaje como leído.
    Esto evita POSTs a Facebook en cada reintento fallido.
    """
    wa_id = _normalize_phone(msg.customer_phone)

    if msg.message_type not in PROCESSABLE_TYPES or not msg.text:
        await send_text(wa_id, MSG_ONLY_TEXT)
    else:
        reply = await run_agent(wa_id, msg.text, msg.customer_name)
        await _deliver(wa_id, reply)

    await mark_as_read(msg.message_id)


async def run_agent(wa_id: str, text: str, name: str | None) -> dict:
    wa_id = _normalize_phone(wa_id)

    if not AGENT_AVAILABLE:
        return {"text": MSG_ERROR, "interactive": None, "cta": None}

    thread_id = f"wa-{wa_id}"
    config = graph_config(thread_id, user_id=wa_id, channel="whatsapp")

    if await in_executor(pending_interrupts, config):
        operator = _operator_number()
        if operator:
            await send_text(
                operator,
                f"⚖️ HITL de {name or wa_id} ({thread_id}). Revísalo en el dashboard.",
            )
        return {"text": MSG_HITL, "interactive": None, "cta": None}

    payload = {"messages": [("human", text)], "thread_id": thread_id}
    await in_executor(invoke_sync, payload, config)

    pendings = await in_executor(pending_interrupts, config)
    if pendings:
        operator = _operator_number()
        if operator:
            await send_text(
                operator,
                f"⚖️ HITL de {name or wa_id} ({thread_id}). Revísalo en el dashboard.",
            )
        return {"text": MSG_HITL, "interactive": None, "cta": None}

    values = await in_executor(state_values, config)
    return {
        "text": values.get("response")
                or "No pude generar una respuesta, ¿puedes repetir tu consulta?",
        "interactive": values.get("response_interactive"),
        "cta": values.get("response_cta"),
    }


async def _deliver(to: str, reply: dict) -> None:
    to = _normalize_phone(to)
    text = (reply.get("text") or "").strip()

    interactives = reply.get("interactive")
    if isinstance(interactives, dict):
        interactives = [interactives]
    interactives = [i for i in (interactives or []) if isinstance(i, dict)]

    cta = reply.get("cta") if isinstance(reply.get("cta"), dict) else {}
    cta_text = (cta.get("text") or "").strip()
    cta_inter = cta.get("interactive")
    if not isinstance(cta_inter, dict):
        cta_inter = None

    if not interactives and not cta_inter:
        if text:
            await send_text(to, text)
        return

    text_in_first = bool(text) and len(text) <= MAX_INTERACTIVE_BODY

    if text and not (interactives and text_in_first):
        await send_text(to, text)

    for idx, inter in enumerate(interactives):
        if idx == 0 and text_in_first:
            body = text
        else:
            body = inter.get("body") or (
                "Pincha aquí 👇" if inter.get("kind") == "cta"
                else "Elige una opción 👇"
            )
        try:
            await send_interactive(to, body, inter)
        except ValueError:
            logger.warning("interactive inválido ignorado: %r", inter, exc_info=True)

    if cta_inter:
        try:
            await send_interactive(to, cta_text or "Pincha aquí 👇", cta_inter)
        except ValueError:
            logger.warning("cta inválido ignorado: %r", cta_inter, exc_info=True)


async def reaper_loop() -> None:
    while True:
        try:
            await _recover_pending()
        except Exception:
            logger.exception("Reaper falló")
        await asyncio.sleep(60)


async def _recover_pending() -> None:
    now = datetime.now(timezone.utc)

    async with async_session() as session:
        await session.execute(
            update(MessageRecord)
            .where(
                MessageRecord.status == "pending",
                MessageRecord.created_at < now - timedelta(hours=1),
            )
            .values(status=RETRYABLE_EXHAUSTED_STATUS)
        )

        rows = (
            await session.execute(
                select(MessageRecord)
                .where(
                    MessageRecord.status == "pending",
                    MessageRecord.created_at < now - timedelta(seconds=90),
                    MessageRecord.retries < MAX_REAPER_RETRIES,
                    or_(
                        MessageRecord.last_attempted_at.is_(None),
                        MessageRecord.last_attempted_at
                        < now - timedelta(seconds=REAPER_BACKOFF_SECONDS),
                    ),
                )
                .order_by(MessageRecord.created_at)
                .limit(20)
            )
        ).scalars().all()

        for row in rows:
            row.retries = (row.retries or 0) + 1
            row.last_attempted_at = now
        await session.commit()

    for row in rows:
        msg = WhatsAppMessage(
            message_id=row.whatsapp_message_id,
            phone_number_id=row.phone_number_id,
            customer_phone=_normalize_phone(row.customer_phone),
            message_type=row.message_type,
            text=row.content,
            timestamp="",
        )

        lock = _locks.setdefault(msg.customer_phone, asyncio.Lock())
        async with lock:
            try:
                await process_message(msg)
                await mark_processed(msg.message_id)
                logger.info("Reaper recuperó %s", row.whatsapp_message_id)
            except Exception as exc:
                logger.exception("Reaper: sigue fallando %s", row.whatsapp_message_id)

                if not _is_retryable(exc):
                    await _mark_dead_letter(
                        row.id,
                        f"{type(exc).__name__}: {str(exc)[:200]}",
                    )


async def _mark_dead_letter(record_id: uuid.UUID, reason: str) -> None:
    """Marca un mensaje como dead_letter para que no se reintente más."""
    try:
        async with async_session() as session:
            await session.execute(
                update(MessageRecord)
                .where(MessageRecord.id == record_id)
                .values(
                    status=NON_RETRYABLE_STATUS,
                    failed_reason=reason,
                )
            )
            await session.commit()
            logger.warning("Mensaje %s movido a dead_letter: %s", record_id, reason)
    except Exception:
        logger.exception("No se pudo marcar dead_letter para %s", record_id)
