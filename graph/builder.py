"""graph/builder.py — Grafo padre con 3 sub-agentes especializados:
booking, faq, intake. Solo hace routing de alto nivel.

v4 — Tres subgrafos especializados:
  - ROUTE_FAQ      → subgrafo graph/faq/   (respuestas desde RAG)
  - ROUTE_AGENDAR  → subgrafo graph/booking/ (reservas)
  - ROUTE_INTAKE   → subgrafo graph/intake/  (ficha proactiva)
  - ROUTE_HANDOFF  → nodo transversal handoff_humano (resumen + escala)
  - ROUTE_FUERA_DOMINIO → nodo transversal respuesta_fuera_dominio

  El grafo padre NUNCA contiene la lógica de negocio de los sub-flujos:
  solo enruta hacia el sub-agente correcto y, en el caso de intake,
  decide post-ejecución si la ficha completó (handoff) o debe esperar
  otro turno (END).

  Invariante estructural: todo valor de VALID_ROUTES existe como nodo
  registrado (test_toda_ruta_tiene_nodo sigue pasando).
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
    respuesta_fuera_dominio, route_query,
)


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Nodos transversales del padre
    workflow.add_node("receive_message", receive_message)
    workflow.add_node("analyze_sentiment", analyze_sentiment)

    # Sub-agentes especializados (subgrafos compilados, sin checkpointer propio:
    # heredan el del padre y persisten su ledger entre turnos).
    workflow.add_node(ROUTE_FAQ, build_faq_graph())
    workflow.add_node(ROUTE_AGENDAR, build_booking_graph())
    workflow.add_node(ROUTE_INTAKE, build_intake_graph())

    # Nodos transversales de cierre
    workflow.add_node(ROUTE_HANDOFF, handoff_humano)
    workflow.add_node(ROUTE_FUERA_DOMINIO, respuesta_fuera_dominio)

    # Entrada
    workflow.add_edge(START, "receive_message")
    workflow.add_edge("receive_message", "analyze_sentiment")

    # Router principal: decide qué sub-agente ejecuta este turno
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

    # Sub-agentes terminales: cierran el turno con su respuesta al cliente
    for terminal in (ROUTE_FAQ, ROUTE_AGENDAR, ROUTE_HANDOFF, ROUTE_FUERA_DOMINIO):
        workflow.add_edge(terminal, END)

    # Intake es el único sub-agente que puede necesitar una continuación
    # inmediata: si la ficha se completó dentro del subgrafo, el padre
    # deriva a handoff_humano en el mismo turno; si no, espera el próximo
    # mensaje del cliente (END).
    workflow.add_conditional_edges(
        ROUTE_INTAKE,
        lambda state: ROUTE_HANDOFF if state.get("intake_completado") else END,
        {ROUTE_HANDOFF: ROUTE_HANDOFF, END: END},
    )

    return workflow.compile(checkpointer=checkpointer)


agent_graph = build_graph()
