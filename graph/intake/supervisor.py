"""graph/intake/supervisor.py — Routing interno del sub-agente INTAKE.

v7 — Corte definitivo: solo 3 acciones reales. La decisión es determinista
y viene del planner (iniciar_ficha / procesar_respuesta / sin_consentimiento).
"""
from core.contracts import AgentState


def route_intake(state: AgentState) -> str:
    """Mapea la decisión del planner al nodo de acción correspondiente.

    Defensivo: si intake_decision está corrupto o ausente, cae en
    procesar_respuesta (el nodo más robusto: puede re-inicializar o
    continuar según el ledger).
    """
    d = state.get("intake_decision") or {}
    accion = d.get("accion", "procesar_respuesta")

    return {
        "iniciar_ficha": "iniciar_ficha",
        "procesar_respuesta": "procesar_respuesta",
        "sin_consentimiento": "sin_consentimiento",
    }.get(accion, "procesar_respuesta")
