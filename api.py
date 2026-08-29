# api.py
"""API FastAPI para atender clientes web desde LeyIA."""

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

from core.contracts import make_config
from graph.builder import agent_graph

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LeyIA Chat API")

# CORS restringido a tu dominio de producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qantyxlab.cl",
        "https://www.qantyxlab.cl",
        # Añade aquí tu localhost para pruebas si lo necesitas:
        # "http://localhost",
        # "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    meta: dict = {}
    escalated: bool = False


def extract_meta(result: dict) -> dict:
    keys = (
        "sentiment", "urgency", "intent", "category", "route",
        "escalated", "closed", "recolectando_datos_agenda",
        "esperando_confirmacion_booking", "esperando_slot",
    )
    return {k: result.get(k) for k in keys if result.get(k) not in (None, False)}


def resolve_interrupt(result: dict, thread_id: str) -> dict:
    """Resuelve automáticamente interrupciones HITL para clientes web."""
    interrupt = result.get("__interrupt__", [None])[0]
    if not interrupt:
        return result

    payload = interrupt.value
    tipo = str(payload.get("tipo", ""))

    if "agendamiento" in tipo:
        decision = {"aprobado": True}
    else:
        decision = {"aprobado": True, "mensaje": None}

    try:
        return agent_graph.invoke(Command(resume=decision), config=make_config(thread_id))
    except Exception as e:
        logger.exception("Error resolviendo HITL")
        return {
            "response": "Un representante se pondrá en contacto contigo a la brevedad.",
            "escalated": True,
        }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    thread_id = req.thread_id or f"web-{uuid.uuid4().hex[:8]}"
    config = make_config(thread_id)

    try:
        result = agent_graph.invoke(
            {
                "messages": [("human", req.message)],
                "query": req.message,
                "thread_id": thread_id,
            },
            config=config,
        )

        while "__interrupt__" in result:
            result = resolve_interrupt(result, thread_id)

        meta = extract_meta(result)

        return ChatResponse(
            response=result.get("response", "No tengo una respuesta en este momento."),
            thread_id=thread_id,
            meta=meta,
            escalated=bool(meta.get("escalated")),
        )

    except Exception as e:
        logger.exception("Error en chat")
        return ChatResponse(
            response=f"Lo siento, ocurrió un error: {type(e).__name__}",
            thread_id=thread_id,
            meta={},
            escalated=True,
        )


@app.get("/health")
def health():
    return {"status": "ok"}
