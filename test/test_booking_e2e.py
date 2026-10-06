import copy
import pytest
from graph.builder import agent_graph
from langchain_core.messages import HumanMessage


def test_booking_golden_path():
    thread_id = "+56900000000"
    config = {"configurable": {"thread_id": thread_id}}

    # Limpia cualquier estado previo del mismo thread_id
    agent_graph.update_state(config, {
        "query": "",
        "messages": [],
        "lead_nombre": None,
        "lead_email": None,
        "lead_modalidad": None,
        "booking_stage": None,
        "slots_propuestos": [],
        "booking_match": None,
        "booking": None,
    }, as_node="__start__")

    def step(query):
        current = copy.deepcopy(agent_graph.get_state(config).values)
        current["messages"].append(HumanMessage(content=query))
        current["query"] = query
        out = agent_graph.invoke(current, config=config)
        return out

    # Turno 1: intención de agendar → inicia wizard, pide nombre
    r1 = step("Quiero agendar una hora")
    state = agent_graph.get_state(config).values
    assert state.get("booking_stage") == "captura_nombre"

    # Turno 2: nombre → pide email
    step("Juan Pérez")
    state = agent_graph.get_state(config).values
    assert state.get("lead_nombre")
    assert state.get("booking_stage") == "captura_email"

    # Turno 3: email → pide modalidad
    step("juan@test.com")
    state = agent_graph.get_state(config).values
    assert state.get("lead_email")
    assert state.get("booking_stage") == "captura_modalidad"

    # Turno 4: modalidad → propone slots
    r4 = step("Online")
    state = agent_graph.get_state(config).values
    assert state.get("lead_modalidad")
    assert state.get("slots_propuestos")
    assert state.get("booking_stage") == "propuesta"
    # Verificar que se generó payload interactivo
    assert r4.get("response_interactive") is not None

    # Turno 5: elegir opción 1
    r5 = step("1")
    # FIX: no leer booking_match del checkpoint final, porque confirmar_y_crear
    # lo limpia. Verificar el retorno del invoke o la existencia de booking.
    assert (r5.get("booking_match") is not None) or (r5.get("booking") is not None)
