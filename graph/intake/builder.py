"""graph/intake/builder.py — Subgrafo especializado INTAKE.

v7 — Corte definitivo de nodos legacy:
  - Se ELIMINAN completar_ficha y repetir_pregunta. La completitud de la
    ficha y el hold-on-error se resuelven inline dentro de
    procesar_respuesta; no necesitan nodos separados.
  - El subgrafo ahora tiene 4 nodos reales: planner + 3 acciones.

Flujo interno:
  START → intake_planner ─┬─ iniciar_ficha ─────── END
                          ├─ procesar_respuesta ─ END
                          └─ sin_consentimiento ─ END

El grafo padre (graph/builder.py) se encarga de la continuación post-intake:
  - intake_completado=True  → handoff_humano (mismo turno)
  - intake_completado=False → END (espera siguiente mensaje)
"""
from langgraph.graph import END, START, StateGraph

from core.contracts import AgentState

from .nodes import (
    iniciar_ficha,
    procesar_respuesta,
    sin_consentimiento,
)
from .planner import intake_planner
from .supervisor import route_intake


def build_intake_graph():
    b = StateGraph(AgentState)

    # Planner
    b.add_node("intake_planner", intake_planner)

    # Acciones terminales (cada una cierra el turno con respuesta al cliente)
    b.add_node("iniciar_ficha", iniciar_ficha)
    b.add_node("procesar_respuesta", procesar_respuesta)
    b.add_node("sin_consentimiento", sin_consentimiento)

    # Entrada
    b.add_edge(START, "intake_planner")

    # Routing desde el planner
    b.add_conditional_edges(
        "intake_planner",
        route_intake,
        {
            "iniciar_ficha": "iniciar_ficha",
            "procesar_respuesta": "procesar_respuesta",
            "sin_consentimiento": "sin_consentimiento",
        },
    )

    # Todos los caminos terminan en END; el grafo padre decide qué sigue.
    for terminal in ("iniciar_ficha", "procesar_respuesta", "sin_consentimiento"):
        b.add_edge(terminal, END)

    return b.compile()
