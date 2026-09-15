"""graph/booking/builder.py — subgrafo compilado."""
from langgraph.graph import END, START, StateGraph

from core.contracts import AgentState

from .nodes import (
    aclarar_eleccion, cerrar_booking, confirmar_y_crear, consultar_disponibilidad,
    ofrecer_ejecutiva, pedir_email, pedir_modalidad, pedir_nombre,
    procesar_captura, proponer_slots, reintentar_eleccion,
    responder_lateral,
)
from .planner import booking_planner
from .supervisor import despues_de_confirmar, despues_de_consultar, route_booking


def build_booking_graph():
    b = StateGraph(AgentState)

    b.add_node("booking_planner", booking_planner)
    b.add_node("consultar_disponibilidad", consultar_disponibilidad)
    b.add_node("proponer_slots", proponer_slots)

    # Wizard de captura
    b.add_node("pedir_nombre", pedir_nombre)
    b.add_node("pedir_email", pedir_email)
    b.add_node("pedir_modalidad", pedir_modalidad)
    b.add_node("procesar_captura", procesar_captura)

    b.add_node("reintentar_eleccion", reintentar_eleccion)
    b.add_node("aclarar_eleccion", aclarar_eleccion)
    b.add_node("confirmar_y_crear", confirmar_y_crear)
    b.add_node("responder_lateral", responder_lateral)
    b.add_node("ofrecer_ejecutiva", ofrecer_ejecutiva)
    b.add_node("cerrar_booking", cerrar_booking)

    b.add_edge(START, "booking_planner")
    b.add_conditional_edges("booking_planner", route_booking, {
        "pedir_nombre": "pedir_nombre",
        "pedir_email": "pedir_email",
        "pedir_modalidad": "pedir_modalidad",
        "procesar_captura": "procesar_captura",
        "responder_lateral": "responder_lateral",
        "consultar_disponibilidad": "consultar_disponibilidad",
        "confirmar_y_crear": "confirmar_y_crear",
        "aclarar_eleccion": "aclarar_eleccion",
        "reintentar_eleccion": "reintentar_eleccion",
        "ofrecer_ejecutiva": "ofrecer_ejecutiva",
        "cerrar_booking": "cerrar_booking",
    })
    b.add_conditional_edges("consultar_disponibilidad", despues_de_consultar, {
        "proponer_slots": "proponer_slots",
        "ofrecer_ejecutiva": "ofrecer_ejecutiva",
        "responder_lateral": "responder_lateral",
    })
    b.add_conditional_edges("confirmar_y_crear", despues_de_confirmar, {
        "consultar_disponibilidad": "consultar_disponibilidad",
        END: END,
    })

    for terminal in ("pedir_nombre", "pedir_email", "pedir_modalidad",
                     "procesar_captura", "proponer_slots",
                     "reintentar_eleccion", "aclarar_eleccion",
                     "responder_lateral", "ofrecer_ejecutiva",
                     "cerrar_booking"):
        b.add_edge(terminal, END)

    return b.compile()
