"""graph/builder.py — Grafo padre con 3 sub-agentes especializados.

v5.5 — PLAN A: eliminación de la ruta de crisis:
  - Se quita ROUTE_CRISIS del import (ya no existe en core.contracts v6.8)
    y del mapeo de analyze_sentiment.
  - Nadie la emitía ya: graph/nodes.py v22.4 eliminó los backstops y las
    guardas de crisis; el clasificador etiqueta riesgo=true como
    hablar_humano + urgency alta y deriva por el flujo estándar.
  - Red de seguridad intacta: route_query (nodes.py) cae a ROUTE_FAQ con
    warning si algún estado viejo trajera route="crisis" (ya no está en
    VALID_ROUTES).
  - Si aplicaste el parche temporal `ROUTE_CRISIS` deprecated en
    contracts.py para desbloquear el arranque, elimínalo ahora.

v5.4 — FIX crisis: ROUTE_CRISIS ruteada a handoff_humano (eliminada en v5.5).
v5.3 — Handoff choice: bifurcación ficha breve vs contacto directo.
v5.2 — Compatibilidad con intake v13.
"""
from langgraph.graph import END, START, StateGraph

from core.contracts import (
    AgentState,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF,
    ROUTE_INTAKE, ROUTE_HANDOFF_CHOICE,
)
from core.persistence import checkpointer
from graph.booking.builder import build_booking_graph
from graph.faq.builder import build_faq_graph
from graph.intake.builder import build_intake_graph
from graph.nodes import (
    analyze_sentiment, handoff_humano, preguntar_preferencia_handoff,
    receive_message, respuesta_fuera_dominio, route_post_faq,
    route_post_intake, route_query,
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
    workflow.add_node(ROUTE_HANDOFF_CHOICE, preguntar_preferencia_handoff)

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
            ROUTE_HANDOFF_CHOICE: ROUTE_HANDOFF_CHOICE,
        },
    )

    # Terminales puros: su sub-flujo termina aquí y cierra el turno.
    # ROUTE_HANDOFF_CHOICE también es terminal: pregunta y espera respuesta
    # en el siguiente turno.
    for terminal in (ROUTE_AGENDAR, ROUTE_HANDOFF, ROUTE_FUERA_DOMINIO,
                     ROUTE_HANDOFF_CHOICE):
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
    # reanuda. Si el planner FAQ detectó pedido de humano (fix #2, v21.1),
    # señala ROUTE_HANDOFF_CHOICE → bifurcación ficha vs contacto directo
    # (misma política que analyze_sentiment para intent=hablar_humano).
    workflow.add_conditional_edges(
        ROUTE_FAQ,
        route_post_faq,
        {
            ROUTE_INTAKE: ROUTE_INTAKE,
            ROUTE_HANDOFF: ROUTE_HANDOFF,
            ROUTE_HANDOFF_CHOICE: ROUTE_HANDOFF_CHOICE,
            END: END,
        },
    )

    return workflow.compile(checkpointer=checkpointer)


agent_graph = build_graph()
