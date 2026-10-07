import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from agent_runtime import (
    AGENT_AVAILABLE, graph_config, in_executor,
    invoke_sync, pending_interrupts, state_values,
)
from .client import mark_as_read, send_text
from .db.connection import async_session
from .db.models import MessageRecord
from .db.repository import mark_processed, register_message
from .normalizer import WhatsAppMessage

logger = logging.getLogger("leyia-whatsapp")

OPERATOR_NUMBER = os.getenv("WHATSAPP_OPERATOR_NUMBER")

MSG_HITL = ("⏸️ Tu solicitud está pendiente de revisión por un abogado del estudio. "
            "Te escribiremos apenas sea resuelta.")
MSG_ONLY_TEXT = "Por ahora solo puedo procesar mensajes de texto 🙏"
MSG_ERROR = "Tuvimos un problema técnico. Inténtalo en unos minutos."

_locks: dict[str, asyncio.Lock] = {}


async def handle_messages(messages: list[WhatsAppMessage]) -> None:
    for msg in messages:
        status = await register_message(msg)
        if status == "processed":
            continue

        lock = _locks.setdefault(msg.customer_phone, asyncio.Lock())
        async with lock:
            try:
                await process_message(msg)
                await mark_processed(msg.message_id)
            except Exception:
                logger.exception("Fallo procesando %s", msg.message_id)


async def process_message(msg: WhatsAppMessage) -> None:
    await mark_as_read(msg.message_id)
    if msg.message_type != "text" or not msg.text:
        await send_text(msg.customer_phone, MSG_ONLY_TEXT)
        return
    reply = await run_agent(msg.customer_phone, msg.text, msg.customer_name)
    await send_text(msg.customer_phone, reply)


async def run_agent(wa_id: str, text: str, name: str | None) -> str:
    if not AGENT_AVAILABLE:
        return MSG_ERROR

    thread_id = f"wa-{wa_id}"
    config = graph_config(thread_id, user_id=wa_id, channel="whatsapp")

    if await in_executor(pending_interrupts, config):
        return MSG_HITL

    payload = {"messages": [("human", text)], "thread_id": thread_id}
    await in_executor(invoke_sync, payload, config)

    pendings = await in_executor(pending_interrupts, config)
    if pendings:
        if OPERATOR_NUMBER:
            await send_text(OPERATOR_NUMBER,
                            f"⚖️ HITL de {name or wa_id} ({thread_id}). Revísalo en el dashboard.")
        return MSG_HITL

    values = await in_executor(state_values, config)
    return values.get("response") or "No pude generar una respuesta, ¿puedes repetir tu consulta?"


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
            .where(MessageRecord.status == "pending",
                   MessageRecord.created_at < now - timedelta(hours=1))
            .values(status="failed")
        )
        rows = (await session.execute(
            select(MessageRecord)
            .where(MessageRecord.status == "pending",
                   MessageRecord.created_at < now - timedelta(seconds=90))
            .limit(20)
        )).scalars().all()
        await session.commit()

    for row in rows:
        msg = WhatsAppMessage(
            message_id=row.whatsapp_message_id,
            phone_number_id=row.phone_number_id,
            customer_phone=row.customer_phone,
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
            except Exception:
                logger.exception("Reaper: sigue fallando %s", row.whatsapp_message_id)
