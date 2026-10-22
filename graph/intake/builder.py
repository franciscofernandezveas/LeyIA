"""graph/intake/builder.py — Subgrafo especializado INTAKE (v17).

Flujo interno:
  START → intake_planner ─┬─ iniciar_ficha ─────── END
                          ├─ procesar_respuesta ─ END      (padre: handoff/agendar)
                          ├─ pausar_para_faq ──── END      (padre: FAQ → reanudar)
                          ├─ pausar_ficha ─────── END      (padre: espera próximo msg)
                          ├─ reanudar ─────────── END
                          ├─ derivar_parcial ──── END      (padre: handoff)
                          ├─ derivar_urgente ──── END      (padre: handoff, urgente)
                          ├─ abandonar_ficha ──── END
                          └─ sin_consentimiento ─ END

v17: se agrega derivar_urgente (detención vigente / orden de arresto).
Emite intake_exit="handoff" igual que derivar_parcial — el contrato del
padre (IntakeExitLabel + route_post_intake) no requiere cambios.
"""
from langgraph.graph import END, START, StateGraph

from core.contracts import AgentState

from .nodes import (
    abandonar_ficha, derivar_parcial, derivar_urgente, iniciar_ficha,
    pausar_ficha, pausar_para_faq, procesar_respuesta, reanudar,
    sin_consentimiento,
)
from .planner import intake_planner
from .supervisor import route_intake

_NODOS = {
    "iniciar_ficha": iniciar_ficha,
    "procesar_respuesta": procesar_respuesta,
    "reanudar": reanudar,
    "pausar_para_faq": pausar_para_faq,
    "pausar_ficha": pausar_ficha,
    "derivar_parcial": derivar_parcial,
    "derivar_urgente": derivar_urgente,
    "abandonar_ficha": abandonar_ficha,
    "sin_consentimiento": sin_consentimiento,
}


def build_intake_graph():
    b = StateGraph(AgentState)

    b.add_node("intake_planner", intake_planner)
    for nombre, fn in _NODOS.items():
        b.add_node(nombre, fn)

    b.add_edge(START, "intake_planner")
    b.add_conditional_edges(
        "intake_planner",
        route_intake,
        {nombre: nombre for nombre in _NODOS},
    )
    for terminal in _NODOS:
        b.add_edge(terminal, END)

    return b.compile()
