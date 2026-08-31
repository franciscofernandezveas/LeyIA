"""graph/builder.py — ensambla y compila el StateGraph.

v2 — Integración del nodo de INTAKE:
  - Nodo intake_lead registrado y mapeado (ROUTE_INTAKE).
  - Edge condicional post-intake: ficha completa → handoff_humano en la
    misma invocación; incompleta → END (espera el próximo turno).

v1→v2 conserva:
  FIX BUG-1: respuesta_fuera_dominio registrada y mapeada (antes crash).

Invariante estructural (exigida por test_toda_ruta_tiene_nodo):
  todo valor de VALID_ROUTES debe existir como nodo registrado.

Flujo resultante:

  START → receive_message → analyze_sentiment ─┬─ respuestas_faq ────────→ END
                                               ├─ respuesta_fuera_dominio → END
                                               ├─ agendar_asesoria ──────→ END
                                               └─ intake_lead ─┬─────────→ END
                                                               └─→ handoff_humano → END
"""
from langgraph.graph import END, START, StateGraph

from core.contracts import (
    AgentState,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF, ROUTE_INTAKE,
)
from core.persistence import checkpointer
from graph.intake import despues_de_intake, intake_lead
from graph.nodes import (
    agendar_asesoria, analyze_sentiment, handoff_humano, receive_message,
    respuesta_fuera_dominio, respuestas_faq, route_query,
)


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Nodos
    workflow.add_node("receive_message", receive_message)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node(ROUTE_FAQ, respuestas_faq)
    workflow.add_node(ROUTE_AGENDAR, agendar_asesoria)
    workflow.add_node(ROUTE_INTAKE, intake_lead)
    workflow.add_node(ROUTE_HANDOFF, handoff_humano)
    workflow.add_node(ROUTE_FUERA_DOMINIO, respuesta_fuera_dominio)

    # Entrada
    workflow.add_edge(START, "receive_message")
    workflow.add_edge("receive_message", "analyze_sentiment")

    # Routing sobre la clasificación (las 5 rutas oficiales tienen destino)
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

    # Terminales simples
    workflow.add_edge(ROUTE_FAQ, END)
    workflow.add_edge(ROUTE_AGENDAR, END)
    workflow.add_edge(ROUTE_HANDOFF, END)
    workflow.add_edge(ROUTE_FUERA_DOMINIO, END)

    # Intake: ficha completa → derivar de inmediato; incompleta → esperar turno
    workflow.add_conditional_edges(
        ROUTE_INTAKE,
        despues_de_intake,
        {ROUTE_HANDOFF: ROUTE_HANDOFF, END: END},
    )

    # El checkpointer habilita persistencia + interrupt/resume (HITL)
    # y es lo que mantiene intake_idx/intake_respuestas entre turnos.
    return workflow.compile(checkpointer=checkpointer)


agent_graph = build_graph()
