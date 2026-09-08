"""graph/faq/builder.py — Subgrafo especializado FAQ."""
from langgraph.graph import END, START, StateGraph

from core.contracts import AgentState

from .nodes import (
    derivar_a_ejecutiva,
    derivar_a_fuera_dominio,
    pedir_clarificacion,
    responder_pregunta,
    sugerir_agendar,
)
from .planner import faq_planner
from .supervisor import route_faq


def build_faq_graph():
    b = StateGraph(AgentState)

    b.add_node("faq_planner", faq_planner)
    b.add_node("responder_pregunta", responder_pregunta)
    b.add_node("pedir_clarificacion", pedir_clarificacion)
    b.add_node("sugerir_agendar", sugerir_agendar)
    b.add_node("derivar_a_ejecutiva", derivar_a_ejecutiva)
    b.add_node("derivar_a_fuera_dominio", derivar_a_fuera_dominio)

    b.add_edge(START, "faq_planner")
    b.add_conditional_edges("faq_planner", route_faq, {
        "responder_pregunta": "responder_pregunta",
        "pedir_clarificacion": "pedir_clarificacion",
        "sugerir_agendar": "sugerir_agendar",
        "derivar_a_ejecutiva": "derivar_a_ejecutiva",
        "derivar_a_fuera_dominio": "derivar_a_fuera_dominio",
    })

    for terminal in ("responder_pregunta", "pedir_clarificacion",
                     "sugerir_agendar", "derivar_a_ejecutiva",
                     "derivar_a_fuera_dominio"):
        b.add_edge(terminal, END)

    return b.compile()
