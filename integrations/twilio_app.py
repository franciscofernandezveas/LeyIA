"""Webhook de WhatsApp (Twilio) + endpoint HITL para operadores."""
import logging
from fastapi import FastAPI, Form, HTTPException, Response
from pydantic import BaseModel
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from langgraph.types import Command

from core.config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
from core.contracts import make_config
from graph.builder import agent_graph

logger = logging.getLogger(__name__)
app = FastAPI(title="Agente Soporte — Twilio + LangGraph")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(From: str = Form(...), Body: str = Form(...)):
    # El teléfono ES el thread_id → memoria persistente por cliente
    thread_id = From.replace("whatsapp:", "")          # p.ej. +573001112233

    result = agent_graph.invoke(
        {"messages": [("human", Body)], "query": Body, "thread_id": thread_id},
        config=make_config(thread_id),
    )

    twiml = MessagingResponse()
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        logger.info("⏸️ HITL pendiente [%s]: %s", thread_id, payload)
        # Aquí puedes notificar al operador (Slack, correo, panel interno…)
        twiml.message("Recibí tu mensaje 👍. Un asesor lo está revisando y te respondo enseguida.")
    else:
        twiml.message(result.get("response", "..."))

    return Response(content=str(twiml), media_type="application/xml")


class HitlDecision(BaseModel):
    thread_id: str
    aprobado: bool = True
    nota: str = ""
    mensaje: str | None = None


@app.post("/hitl/decision")
async def hitl_decision(d: HitlDecision):
    """El operador humano reanuda el grafo pausado con su decisión."""
    result = agent_graph.invoke(
        Command(resume=d.model_dump(exclude={"thread_id"})),
        config=make_config(d.thread_id),
    )
    response = result.get("response")
    if not response:
        raise HTTPException(409, "El grafo sigue interrumpido.")

    # Push proactivo al cliente por WhatsApp
    if twilio_client:
        twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_FROM,
            to=f"whatsapp:{d.thread_id}",
            body=response,
        )
    return {"status": "ok", "response": response}
