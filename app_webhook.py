"""Webhook de WhatsApp (Twilio) para el agente LangGraph."""
import asyncio
import os

from fastapi import BackgroundTasks, FastAPI, Request, Response
from langgraph.types import Command
from twilio.rest import Client

from core.contracts import make_config
from graph.builder import agent_graph

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_NUMBER = os.environ["TWILIO_WHATSAPP_NUMBER"]  # ej. "whatsapp:+14155238886"

OPERATOR_NUMBERS = {
    n.strip()
    for n in os.environ.get("OPERATOR_WHATSAPP_NUMBERS", "").split(",")
    if n.strip()
}

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app = FastAPI()

# ---------------------------------------------------------------------------
# Estado en memoria (MVP) — en producción: Redis / DB
# ---------------------------------------------------------------------------
operador_activo: dict[str, str] = {}   # nº operador -> thread_id que atiende
hitl_pendientes: dict[str, dict] = {}  # thread_id -> payload de la interrupción
locks: dict[str, asyncio.Lock] = {}    # 1 mensaje a la vez por conversación


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def send_whatsapp(to: str, body: str) -> None:
    """WhatsApp vía Twilio permite ~1600 caracteres por mensaje."""
    if not body:
        return
    for i in range(0, len(body), 1500):
        client.messages.create(from_=TWILIO_NUMBER, body=body[i:i + 1500], to=to)


def thread_id_from(number: str) -> str:
    # "whatsapp:+52155..." -> "wa-52155..."
    return "wa-" + number.removeprefix("whatsapp:+")


def number_from(thread_id: str) -> str:
    return "whatsapp:+" + thread_id.removeprefix("wa-")


def prompt_operador(payload: dict, user_number: str) -> str:
    texto = (
        "⏸️ *INTERVENCIÓN HUMANA REQUERIDA*\n"
        f"• Tipo    : {payload.get('tipo')}\n"
        f"• Cliente : {user_number}\n"
        f"• Consulta: {payload.get('query')}\n"
        f"• Detalle : {payload.get('detalle')}\n\n"
    )
    if "agendamiento" in str(payload.get("tipo", "")):
        texto += ("Responde:\n"
                  "• *A* → aprobar y enviar link de Calendly\n"
                  "• *R <nota>* → rechazar (nota opcional)")
    else:
        texto += ("Responde:\n"
                  "• *D* → enviar mensaje por defecto\n"
                  "• *M <mensaje>* → mensaje personalizado")
    return texto


def parsear_decision(texto: str, payload: dict) -> dict:
    primera, _, resto = texto.strip().partition(" ")
    op = primera.lower()

    if "agendamiento" in str(payload.get("tipo", "")):
        if op in ("a", "aprobar"):
            return {"aprobado": True}
        return {"aprobado": False, "nota": resto.strip()}

    if op in ("m", "mensaje"):
        return {"aprobado": True, "mensaje": resto.strip() or None}
    return {"aprobado": True, "mensaje": None}


# ---------------------------------------------------------------------------
# Flujo del cliente — reemplaza tu bucle "tú>" del CLI
# ---------------------------------------------------------------------------
async def procesar_mensaje_usuario(from_number: str, body: str) -> None:
    thread_id = thread_id_from(from_number)
    config = make_config(thread_id)
    lock = locks.setdefault(thread_id, asyncio.Lock())

    async with lock:
        # invoke es síncrono y bloqueante → se corre en un hilo.
        # Mejor aún si usas: await agent_graph.ainvoke(...)
        result = await asyncio.to_thread(
            agent_graph.invoke,
            {"messages": [("human", body)], "query": body, "thread_id": thread_id},
            config,
        )

        # HITL: el grafo quedó pausado EN EL CHECKPOINTER.
        # No esperamos al operador aquí: avisamos y terminamos la petición.
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            hitl_pendientes[thread_id] = payload
            for op_number in OPERATOR_NUMBERS:
                operador_activo[op_number] = thread_id
                send_whatsapp(op_number, prompt_operador(payload, from_number))
            return

        if respuesta := result.get("response"):
            send_whatsapp(from_number, respuesta)


# ---------------------------------------------------------------------------
# Flujo del operador — reemplaza tu resolver_hitl() del CLI
# ---------------------------------------------------------------------------
async def procesar_respuesta_operador(op_number: str, body: str) -> None:
    thread_id = operador_activo.get(op_number)
    payload = hitl_pendientes.get(thread_id or "")
    if not thread_id or payload is None:
        send_whatsapp(op_number, "No tienes interrupciones pendientes.")
        return

    decision = parsear_decision(body, payload)
    lock = locks.setdefault(thread_id, asyncio.Lock())

    async with lock:
        # Gracias al checkpointer, resume funciona aunque hayan pasado
        # minutos u horas desde la interrupción.
        result = await asyncio.to_thread(
            agent_graph.invoke, Command(resume=decision), make_config(thread_id)
        )

        # Interrupciones encadenadas (igual que tu while en el CLI)
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            hitl_pendientes[thread_id] = payload
            send_whatsapp(op_number, prompt_operador(payload, number_from(thread_id)))
            return

    hitl_pendientes.pop(thread_id, None)
    operador_activo.pop(op_number, None)
    send_whatsapp(op_number, "✅ Decisión aplicada y cliente notificado.")
    if respuesta := result.get("response"):
        send_whatsapp(number_from(thread_id), respuesta)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, background: BackgroundTasks):
    form = await request.form()
    body = (form.get("Body") or "").strip()
    from_number = form.get("From") or ""

    if not body or not from_number:
        return Response(status_code=200)

    # El mismo webhook atiende usuarios y operadores, diferenciados por número
    if from_number in OPERATOR_NUMBERS:
        background.add_task(procesar_respuesta_operador, from_number, body)
    else:
        background.add_task(procesar_mensaje_usuario, from_number, body)

    # 200 inmediato sin TwiML: la respuesta llegará vía REST porque el agente
    # puede tardar más que el timeout del webhook (~15 s).
    return Response(status_code=200)
