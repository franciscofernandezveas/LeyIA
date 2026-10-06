"""Webhook de WhatsApp (Twilio) + endpoint HITL para operadores."""
import logging
import os
from contextlib import asynccontextmanager

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

# Operadores que pueden responder por WhatsApp directamente (opcional).
# Ejemplo: whatsapp:+56988887777,whatsapp:+56999998888
OPERATOR_NUMBERS = {
    n.strip()
    for n in os.environ.get("OPERATOR_WHATSAPP_NUMBERS", "").split(",")
    if n.strip()
}

# Memoria volátil: qué operador atiende qué conversación.
operador_activo: dict[str, str] = {}   # nº operador -> thread_id
hitl_pendientes: dict[str, dict] = {}  # thread_id -> payload


# ---------------------------------------------------------------------
# Utilidades de envío
# ---------------------------------------------------------------------
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


async def _send_whatsapp(to: str, body: str) -> None:
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


# ---------------------------------------------------------------------
# Lógica del agente
# ---------------------------------------------------------------------
async def _is_thread_interrupted(config: dict) -> bool:
    try:
        state = await agent_graph.aget_state(config)
        return any(
            getattr(t, "status", None) == "interrupt"
            for t in getattr(state, "tasks", [])
        )
    except Exception as exc:
        logger.warning("No se pudo consultar estado del hilo: %s", exc)
        return False


def _prompt_operador(payload: dict, user_number: str) -> str:
    texto = (
        "⏸️ *INTERVENCIÓN HUMANA REQUERIDA*\n"
        f"• Tipo    : {payload.get('tipo')}\n"
        f"• Cliente : {user_number}\n"
        f"• Consulta: {payload.get('query')}\n"
        f"• Detalle : {payload.get('detalle')}\n\n"
    )
    if "agendamiento" in str(payload.get("tipo", "")):
        texto += "Responde:\n• *A* → aprobar\n• *R <nota>* → rechazar"
    else:
        texto += "Responde:\n• *D* → enviar mensaje por defecto\n• *M <mensaje>* → mensaje personalizado"
    return texto


def _parsear_decision(texto: str, payload: dict) -> dict:
    primera, _, resto = texto.strip().partition(" ")
    op = primera.lower()

    if "agendamiento" in str(payload.get("tipo", "")):
        if op in ("a", "aprobar"):
            return {"aprobado": True}
        return {"aprobado": False, "nota": resto.strip()}

    if op in ("m", "mensaje"):
        return {"aprobado": True, "mensaje": resto.strip() or None}
    return {"aprobado": True, "mensaje": None}


async def _notify_operators(thread_id: str, payload: dict) -> None:
    user_number = _phone_to_whatsapp(thread_id)
    prompt = _prompt_operador(payload, user_number)
    for op_number in OPERATOR_NUMBERS:
        operador_activo[op_number] = thread_id
        await _send_whatsapp(op_number, prompt)
    logger.info("🔔 Notificación de HITL enviada a %s operadores", len(OPERATOR_NUMBERS))


async def _process_incoming_message(message_sid: str, from_number: str, body: str) -> None:
    # Idempotencia ante reintentos de Twilio
    if message_sid in PROCESSED_MESSAGE_SIDS:
        logger.info("MessageSid %s ya procesado. Ignorando.", message_sid)
        return
    PROCESSED_MESSAGE_SIDS.add(message_sid)

    # Si es un operador respondiendo por WhatsApp, lo procesamos como HITL
    if from_number in OPERATOR_NUMBERS:
        await _procesar_respuesta_operador(from_number, body)
        return

    thread_id = from_number.replace("whatsapp:", "")
    config = make_config(thread_id)

    try:
        # Si ya hay HITL pendiente, no invocamos con input nuevo
        if await _is_thread_interrupted(config):
            await _send_whatsapp(
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
            hitl_pendientes[thread_id] = payload
            await _notify_operators(thread_id, payload)
            response_text = (
                "Recibí tu mensaje 👍. Un asesor lo está revisando y te respondo enseguida."
            )
        else:
            response_text = result.get("response") or (
                "Lo siento, tuve un problema procesando tu mensaje. "
                "Por favor intenta de nuevo en un momento."
            )

        await _send_whatsapp(thread_id, response_text)

    except Exception as exc:
        logger.exception("Error procesando mensaje de %s: %s", thread_id, exc)
        await _send_whatsapp(
            thread_id,
            "Ups, algo salió mal 🛠️. Estamos revisando el problema. Intenta de nuevo pronto.",
        )


async def _procesar_respuesta_operador(op_number: str, body: str) -> None:
    thread_id = operador_activo.get(op_number)
    payload = hitl_pendientes.get(thread_id or "")
    if not thread_id or payload is None:
        await _send_whatsapp(op_number, "No tienes interrupciones pendientes.")
        return

    decision = _parsear_decision(body, payload)
    config = make_config(thread_id)

    try:
        result = await agent_graph.ainvoke(
            Command(resume=decision), config=config
        )
    except Exception as exc:
        logger.exception("Error reanudando grafo para %s: %s", thread_id, exc)
        await _send_whatsapp(op_number, "Error al aplicar la decisión. Intenta de nuevo.")
        return

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        hitl_pendientes[thread_id] = payload
        await _send_whatsapp(op_number, _prompt_operador(payload, _phone_to_whatsapp(thread_id)))
        return

    hitl_pendientes.pop(thread_id, None)
    operador_activo.pop(op_number, None)
    await _send_whatsapp(op_number, "✅ Decisión aplicada y cliente notificado.")
    if respuesta := result.get("response"):
        await _send_whatsapp(thread_id, respuesta)


# ---------------------------------------------------------------------
# Validación de firma Twilio
# ---------------------------------------------------------------------
def _verify_twilio_request(request: Request, form_data: dict) -> bool:
    if not TWILIO_VALIDATE_SIGNATURE:
        logger.warning("Validación de firma de Twilio desactivada.")
        return True
    if not validator:
        logger.warning("Validador de Twilio no configurado.")
        return True

    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    return validator.validate(url, form_data, signature)


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Twilio app iniciada")
    yield
    logger.info("🛑 Twilio app detenida")


app = FastAPI(
    title="Agente Soporte — Twilio + LangGraph",
    lifespan=lifespan,
)


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


# ---------------------------------------------------------------------
# Endpoint HITL (para panel/admin)
# ---------------------------------------------------------------------
class HitlDecision(BaseModel):
    thread_id: str
    aprobado: bool = True
    nota: str = ""
    mensaje: str | None = None


def _require_hitl_key(credentials: HTTPAuthorizationCredentials | None) -> None:
    if not HITL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="HITL_API_KEY no configurada",
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
                detail="El grafo sigue interrumpido.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El agente no generó respuesta",
        )

    await _send_whatsapp(d.thread_id, response)
    return {"status": "ok", "thread_id": d.thread_id, "response": response}
