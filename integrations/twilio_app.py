"""Webhook de WhatsApp (Twilio) + endpoint HITL para operadores."""
import logging
from fastapi import (
    FastAPI,
    BackgroundTasks,
    Form,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from twilio.request_validator import RequestValidator
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from langgraph.types import Command

from core.config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_FROM,
    HITL_API_KEY,
    TWILIO_VALIDATE_SIGNATURE,
)
from core.contracts import make_config
from graph.builder import agent_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None
validator = RequestValidator(TWILIO_AUTH_TOKEN) if TWILIO_AUTH_TOKEN else None

MAX_WHATSAPP_CHARS = 1500
PROCESSED_MESSAGE_SIDS: set[str] = set()
security = HTTPBearer(auto_error=False)


def _phone_to_whatsapp(phone: str) -> str:
    return phone if phone.startswith("whatsapp:") else f"whatsapp:{phone}"


def _split_message(text: str, max_len: int = MAX_WHATSAPP_CHARS) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    lines = text.split("\n")
    current = ""
    for line in lines:
        if len(line) > max_len:
            if current:
                chunks.append(current.strip())
                current = ""
            words = line.split(" ")
            part = ""
            for word in words:
                if len(part) + len(word) + 1 > max_len:
                    chunks.append(part.strip())
                    part = word
                else:
                    part = f"{part} {word}".strip()
            if part:
                current = part
            continue

        if len(current) + len(line) + 1 > max_len:
            chunks.append(current.strip())
            current = line
        else:
            current = f"{current}\n{line}".strip()

    if current:
        chunks.append(current.strip())

    return chunks


async def _send_whatsapp_message(to: str, body: str) -> None:
    if not twilio_client or not TWILIO_WHATSAPP_FROM:
        logger.error("Twilio no está configurado. No se puede enviar mensaje a %s", to)
        return

    chunks = _split_message(body)
    for idx, chunk in enumerate(chunks, start=1):
        try:
            twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                to=_phone_to_whatsapp(to),
                body=chunk,
            )
            logger.info("📤 Mensaje enviado a %s (parte %d/%d)", to, idx, len(chunks))
        except Exception as exc:
            logger.exception("Error enviando WhatsApp a %s (parte %d): %s", to, idx, exc)


async def _is_thread_interrupted(config: dict) -> bool:
    try:
        state = await agent_graph.aget_state(config)
        return any(getattr(t, "status", None) == "interrupt" for t in getattr(state, "tasks", []))
    except Exception as exc:
        logger.warning("No se pudo consultar estado del hilo: %s", exc)
        return False


async def _process_incoming_message(message_sid: str, from_number: str, body: str) -> None:
    if message_sid in PROCESSED_MESSAGE_SIDS:
        logger.info("MessageSid %s ya procesado. Ignorando reintento.", message_sid)
        return
    PROCESSED_MESSAGE_SIDS.add(message_sid)

    thread_id = from_number.replace("whatsapp:", "")
    config = make_config(thread_id)

    try:
        if await _is_thread_interrupted(config):
            await _send_whatsapp_message(
                thread_id,
                "Tu consulta anterior sigue en revisión ⏳. Te respondo en cuanto un asesor la revise.",
            )
            return

        result = await agent_graph.ainvoke(
            {"messages": [("human", body)], "query": body, "thread_id": thread_id},
            config=config,
        )

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            logger.info("⏸️ HITL pendiente [%s]: %s", thread_id, payload)
            await _notify_operator(thread_id, payload)
            response_text = (
                "Recibí tu mensaje 👍. Un asesor lo está revisando y te respondo enseguida."
            )
        else:
            response_text = result.get("response") or (
                "Lo siento, tuve un problema procesando tu mensaje. "
                "Por favor intenta de nuevo en un momento."
            )

        await _send_whatsapp_message(thread_id, response_text)

    except Exception as exc:
        logger.exception("Error procesando mensaje de %s: %s", thread_id, exc)
        await _send_whatsapp_message(
            thread_id,
            "Ups, algo salió mal 🛠️. Estamos revisando el problema. Intenta de nuevo pronto.",
        )


async def _notify_operator(thread_id: str, payload) -> None:
    logger.info("🔔 Notificación de HITL: thread_id=%s payload=%s", thread_id, payload)


def _verify_twilio_request(request: Request, form_data: dict) -> bool:
    if not TWILIO_VALIDATE_SIGNATURE:
        logger.warning("Validación de firma de Twilio desactivada.")
        return True
    if not validator:
        logger.warning("Validador de Twilio no configurado; aceptando request sin firma.")
        return True

    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    return validator.validate(url, form_data, signature)


app = FastAPI(title="Agente Soporte — Twilio + LangGraph")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    From: str = Form(...),
    Body: str = Form(...),
    MessageSid: str | None = Form(None),
):
    form_data = dict(await request.form())

    if not _verify_twilio_request(request, form_data):
        logger.warning("Firma de Twilio inválida")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Firma inválida")

    thread_id = From.replace("whatsapp:", "")
    logger.info("📥 Mensaje de %s: %r", thread_id, Body)

    background_tasks.add_task(
        _process_incoming_message,
        MessageSid or f"unknown-{thread_id}-{hash(Body)}",
        From,
        Body,
    )

    return Response(content=str(MessagingResponse()), media_type="application/xml")


class HitlDecision(BaseModel):
    thread_id: str
    aprobado: bool = True
    nota: str = ""
    mensaje: str | None = None


def _require_hitl_key(credentials: HTTPAuthorizationCredentials | None) -> None:
    if not HITL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="HITL_API_KEY no configurada en el servidor",
        )
    if not credentials or credentials.credentials != HITL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.post("/hitl/decision")
async def hitl_decision(
    d: HitlDecision,
    credentials: HTTPAuthorizationCredentials | None = None,
):
    _require_hitl_key(credentials)

    config = make_config(d.thread_id)

    try:
        result = await agent_graph.ainvoke(
            Command(resume=d.model_dump(exclude={"thread_id"})),
            config=config,
        )
    except Exception as exc:
        logger.exception("Error reanudando grafo para %s: %s", d.thread_id, exc)
        raise HTTPException(status_code=500, detail="Error reanudando el agente") from exc

    response = result.get("response")
    if not response:
        if "__interrupt__" in result:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="El grafo sigue interrumpido. Revisa el payload.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El agente no generó respuesta",
        )

    await _send_whatsapp_message(d.thread_id, response)
    return {"status": "ok", "thread_id": d.thread_id, "response": response}
