"""graph/intake/supervisor.py — Routing interno del sub-agente INTAKE (v17).

v17: nueva acción "derivar_urgente" (detención vigente / orden de arresto →
handoff inmediato). Hacia el padre se traduce en intake_exit="handoff".
"""
import logging

from core.contracts import AgentState

logger = logging.getLogger(__name__)

_ACCIONES = {
    "iniciar_ficha": "iniciar_ficha",
    "procesar_respuesta": "procesar_respuesta",
    "reanudar": "reanudar",
    "pausar_para_faq": "pausar_para_faq",
    "pausar_ficha": "pausar_ficha",
    "derivar_parcial": "derivar_parcial",
    "derivar_urgente": "derivar_urgente",
    "abandonar_ficha": "abandonar_ficha",
    "sin_consentimiento": "sin_consentimiento",
}


def route_intake(state: AgentState) -> str:
    d = state.get("intake_decision") or {}
    accion = d.get("accion", "procesar_respuesta")
    if accion in _ACCIONES:
        return _ACCIONES[accion]
    logger.warning("[intake] acción desconocida '%s' → reanudar", accion)
    return "reanudar"
