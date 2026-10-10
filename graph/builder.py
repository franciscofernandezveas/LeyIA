"""graph/builder.py — Grafo padre con 3 sub-agentes especializados.

v5.2 — Compatibilidad con intake v13:
  - route_post_intake puede devolver ROUTE_AGENDAR cuando el cierre del
    intake ofrece el CTA de agendar.
  - Mantiene FAQ → intake → handoff.
"""
from langgraph.graph import END, START, StateGraph

from core.contracts import (
    AgentState,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF, ROUTE_INTAKE,
)
from core.persistence import checkpointer
from graph.booking.builder import build_booking_graph
from graph.faq.builder import build_faq_graph
from graph.intake.builder import build_intake_graph
from graph.nodes import (
    analyze_sentiment, handoff_humano, receive_message,
    respuesta_fuera_dominio, route_post_faq, route_post_intake, route_query,
)


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("receive_message", receive_message)
    workflow.add_node("analyze_sentiment", analyze_sentiment)

    workflow.add_node(ROUTE_FAQ, build_faq_graph())
    workflow.add_node(ROUTE_AGENDAR, build_booking_graph())
    workflow.add_node(ROUTE_INTAKE, build_intake_graph())

    workflow.add_node(ROUTE_HANDOFF, handoff_humano)
    workflow.add_node(ROUTE_FUERA_DOMINIO, respuesta_fuera_dominio)

    workflow.add_edge(START, "receive_message")
    workflow.add_edge("receive_message", "analyze_sentiment")

    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_query,
        {
            ROUTE_FAQ: ROUTE_FAQ,
            ROUTE_AGENDAR: ROUTE_AGENDAR,
            ROUTE_INTAKE: ROUTE_INTAKE,
            ROUTE_HANDOFF: ROUTE_HANDOFF,
            ROUTE_FUERA_DOMINIO: ROUTE_FUERA_DOMINIO,
        },
    )

    # Terminales puros: su sub-flujo termina aquí y cierra el turno.
    for terminal in (ROUTE_AGENDAR, ROUTE_HANDOFF, ROUTE_FUERA_DOMINIO):
        workflow.add_edge(terminal, END)

    # Intake puede: pausar a FAQ (duda lateral), derivar a handoff
    # (ficha completa o parcial), pasar a agendar, o terminar el turno.
    workflow.add_conditional_edges(
        ROUTE_INTAKE,
        route_post_intake,
        {
            ROUTE_FAQ: ROUTE_FAQ,
            ROUTE_AGENDAR: ROUTE_AGENDAR,
            ROUTE_HANDOFF: ROUTE_HANDOFF,
            END: END,
        },
    )

    # FAQ: normalmente responde y cierra; si el intake estaba pausado, lo
    # reanuda. Además, si el planner FAQ detectó un pedido de humano,
    # señala intake (o handoff si la ficha ya estaba completa).
    workflow.add_conditional_edges(
        ROUTE_FAQ,
        route_post_faq,
        {
            ROUTE_INTAKE: ROUTE_INTAKE,
            ROUTE_HANDOFF: ROUTE_HANDOFF,
            END: END,
        },
    )

    return workflow.compile(checkpointer=checkpointer)


agent_graph = build_graph()
