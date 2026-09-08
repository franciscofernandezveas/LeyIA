"""graph/faq/supervisor.py — Routing interno del sub-agente FAQ."""
from core.contracts import AgentState

from .contracts import FAQStageLabel


def route_faq(state: AgentState) -> str:
    d = state.get("faq_decision") or {}
    accion = d.get("accion", "responder_pregunta")

    if accion == "responder_pregunta":
        return "responder_pregunta"
    if accion == "pedir_clarificacion":
        return "pedir_clarificacion"
    if accion == "sugerir_agendar":
        return "sugerir_agendar"
    if accion == "hablar_humano":
        return "derivar_a_ejecutiva"
    if accion == "fuera_de_dominio":
        return "derivar_a_fuera_dominio"

    return "responder_pregunta"
