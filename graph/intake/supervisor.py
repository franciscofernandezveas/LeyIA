"""graph/intake/supervisor.py — Routing interno del sub-agente INTAKE (v8)."""
from core.contracts import AgentState

_ACCIONES = {
    "iniciar_ficha": "iniciar_ficha",
    "procesar_respuesta": "procesar_respuesta",
    "reanudar": "reanudar",
    "pausar_para_faq": "pausar_para_faq",
    "derivar_parcial": "derivar_parcial",
    "abandonar_ficha": "abandonar_ficha",
    "sin_consentimiento": "sin_consentimiento",
}


def route_intake(state: AgentState) -> str:
    d = state.get("intake_decision") or {}
    return _ACCIONES.get(d.get("accion", "procesar_respuesta"),
                         "procesar_respuesta")
