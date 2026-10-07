"""
agent_runtime.py - Helpers compartidos para invocar el grafo de LeyIA.

Este módulo extrae la lógica común que usan:
  - api.py (chat web con SSE + HITL)
  - integrations/whatsapp/handler.py (procesamiento de mensajes WhatsApp)

Evita duplicar pending_interrupts, state_values, invoke_sync, etc.
"""

import asyncio
import logging

from core.contracts import make_config
from graph.builder import agent_graph

logger = logging.getLogger("leyia-agent-runtime")

AGENT_AVAILABLE = agent_graph is not None


async def in_executor(func, *args):
    """Ejecuta una función síncrona en el thread pool por defecto."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


def graph_config(thread_id: str, user_id: str = "anon", channel: str = "api") -> dict:
    """
    Construye la config que recibe agent_graph.invoke/astream.

    Args:
        thread_id: id del hilo del checkpointer (determinístico por canal).
        user_id: identificador del usuario/cliente.
        channel: "api" para el chat web, "whatsapp" para WhatsApp.
    """
    config = dict(make_config(thread_id) or {})
    config["tags"] = ["leyia", channel, f"thread:{thread_id}"]
    config["metadata"] = {
        "thread_id": thread_id,
        "user_id": user_id,
        "channel": channel,
        "service": "leyia-api",
    }
    return config


def pending_interrupts(config: dict) -> list:
    """Interrupts pendientes del checkpoint (igual que en api.py)."""
    try:
        snap = agent_graph.get_state(config)
    except Exception as e:
        logger.warning(f"get_state falló: {e}")
        return []
    if not snap or not snap.tasks:
        return []
    return [i for task in snap.tasks for i in (task.interrupts or [])]


def state_values(config: dict) -> dict:
    """Valores del estado del grafo en el checkpoint actual."""
    try:
        snap = agent_graph.get_state(config)
        return (snap.values or {}) if snap else {}
    except Exception as e:
        logger.warning(f"get_state values falló: {e}")
        return {}


def invoke_sync(payload: dict, config: dict):
    """Invocación síncrona del grafo (fallback cuando no hay astream)."""
    return agent_graph.invoke(payload, config=config)
