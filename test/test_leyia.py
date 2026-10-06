"""test/test_agente_e2e.py — Tests end-to-end de los bugs críticos y P1.

Requiere:
  - Grafo compilado en graph.builder.agent_graph
  - Checkpointer sqlite configurado
  - Variables de entorno básicas (OPENAI_API_KEY)
"""

import copy

import pytest
from langchain_core.messages import HumanMessage

from core.contracts import ROUTE_INTAKE
from graph.builder import agent_graph


def _reset_thread(thread_id: str) -> dict:
    """Limpia el estado de un thread para tests aislados."""
    config = {"configurable": {"thread_id": thread_id}}
    agent_graph.update_state(config, {
        "query": "",
        "messages": [],
        "sentiment": None,
        "urgency": None,
        "intent": None,
        "category": None,
        "case_category": None,
        "clf_reason": "",
        "route": None,
        "lead_nombre": None,
        "lead_email": None,
        "lead_modalidad": None,
        "booking_stage": None,
        "slots_propuestos": [],
        "booking_match": None,
        "booking": None,
        "intake_activo": False,
        "intake_completado": False,
        "intake_exit": None,
        "intake_resume": False,
        "intake_respuestas": {},
        "intake_decision": None,
        "intake_stage": None,
        "intake_started_en": None,
        "intake_oferta_qid": None,
        "closed": False,
        "response_interactive": None,
    }, as_node="__start__")
    return config


def _step(config: dict, query: str) -> dict:
    """Envía un mensaje al grafo y retorna el output del turno."""
    current = copy.deepcopy(agent_graph.get_state(config).values)
    current["messages"].append(HumanMessage(content=query))
    current["query"] = query
    return agent_graph.invoke(current, config=config)


# ============================================================================
# P0 #1 — Agendar funciona E2E
# ============================================================================

def test_booking_golden_path_flujo_completo():
    """Wizard de agendamiento captura datos, propone horarios y elige slot."""
    tid = "+56900000001"
    config = _reset_thread(tid)

    _step(config, "Quiero agendar una hora")
    state = agent_graph.get_state(config).values
    assert state.get("booking_stage") == "captura_nombre"

    _step(config, "Juan Pérez")
    state = agent_graph.get_state(config).values
    assert state.get("lead_nombre")
    assert state.get("booking_stage") == "captura_email"

    _step(config, "juan@test.com")
    state = agent_graph.get_state(config).values
    assert state.get("lead_email")
    assert state.get("booking_stage") == "captura_modalidad"

    out = _step(config, "Online")
    state = agent_graph.get_state(config).values
    assert state.get("lead_modalidad")
    assert state.get("slots_propuestos")
    assert state.get("booking_stage") == "propuesta"
    # Payload interactivo de horarios presente
    assert out.get("response_interactive") is not None

    out = _step(config, "1")
    # booking_match o booking confirman que el slot fue elegido
    assert (out.get("booking_match") is not None) or (out.get("booking") is not None)


def test_booking_lenguaje_natural():
    """El usuario puede elegir horario con lenguaje natural."""
    tid = "+56900000002"
    config = _reset_thread(tid)

    _step(config, "Quiero agendar una hora")
    _step(config, "María González")
    _step(config, "maria@test.com")
    _step(config, "Presencial")
    out = _step(config, "el lunes a las 10:30")
    assert (out.get("booking_match") is not None) or (out.get("booking") is not None)


# ============================================================================
# P1 #4 — Categoría del caso es estable
# ============================================================================

def test_case_category_no_se_corrompe_por_mensajes_laterales():
    """Una vez que el caso es sustantivo, las preguntas genéricas no lo cambian."""
    tid = "+56900000003"
    config = _reset_thread(tid)

    out = _step(config, "Me demandaron por pensión de alimentos")
    state = agent_graph.get_state(config).values
    assert state.get("category") == "pension_alimentos"
    assert state.get("case_category") == "pension_alimentos"

    out = _step(config, "¿cuánto cuesta la consulta?")
    state = agent_graph.get_state(config).values
    # La categoría por TURNO puede volverse genérica
    assert state.get("category") in ("consulta_general", "otro", "pension_alimentos")
    # Pero la del CASO debe mantenerse
    assert state.get("case_category") == "pension_alimentos"

    # Verificar que el link wa.me hereda la categoría estable
    from graph.nodes import _wa_texto_intake
    texto = _wa_texto_intake(state)
    assert "pension_alimentos" in texto or "Área:" not in texto


# ============================================================================
# P1 #5 — Pausa real
# ============================================================================

def test_pausa_reanuda_intake():
    """Decir 'pausa' detiene la ficha; el siguiente mensaje reanuda."""
    tid = "+56900000004"
    config = _reset_thread(tid)

    # Iniciar intake (pide consentimiento)
    _step(config, "Quiero hablar con una ejecutiva")
    state = agent_graph.get_state(config).values
    assert state.get("booking_stage") is None
    assert state.get("intake_activo") is True

    # Autorizar para avanzar
    _step(config, "Sí, autorizo")
    state = agent_graph.get_state(config).values
    assert state.get("intake_activo") is True

    # Pedir pausa
    out = _step(config, "pausa")
    state = agent_graph.get_state(config).values
    assert state.get("intake_exit") == "pausa"
    assert state.get("intake_stage") == "pausado"

    # Reanudar
    out = _step(config, "ahora sigamos")
    state = agent_graph.get_state(config).values
    assert state.get("intake_activo") is True
    assert state.get("intake_exit") != "pausa"


# ============================================================================
# P1 #6 — Datos del mensaje inicial precapturados
# ============================================================================

def test_intake_precaptura_datos_iniciales():
    """El mensaje que dispara el intake ya puede traer nombre/email."""
    tid = "+56900000005"
    config = _reset_thread(tid)

    _step(config, "Hola soy Juan Pérez, mi correo es jp@gmail.com")
    state = agent_graph.get_state(config).values
    respuestas = state.get("intake_respuestas") or {}
    assert respuestas.get("nombre") == "Juan Pérez"
    assert respuestas.get("email") == "jp@gmail.com"


# ============================================================================
# P0 #3 — PII a Sheets sin consentimiento (inspección de payload)
# ============================================================================

def test_sync_lead_modo_minimo_excluye_pii():
    """Un lead sin consentimiento no debe llevar PII a SheetDB."""
    from integrations.sheetdb import _build_payload

    state = {
        "thread_id": "+56900000006",
        "intake_respuestas": {
            "nombre": "Juan Pérez",
            "email": "jp@gmail.com",
            "situacion_actual": "Me demandaron por pensión",
            "consentimiento_datos": False,
        },
        "category": "pension_alimentos",
        "case_category": "pension_alimentos",
        "sentiment": "negativo",
        "urgency": "alta",
        "intent": "hablar_humano",
        "route": "intake",
        "clf_reason": "pide humano",
    }

    payload = _build_payload(state, canal="whatsapp", modo="minimo")

    assert payload["Nombre completo"] == ""
    assert payload["Correo electrónico"] == ""
    assert payload["Situación / Caso"] == ""
    assert payload["Resumen ejecutiva"] == "Pendiente consentimiento datos — no contactar hasta autorización"
    assert payload["Estado lead"] == "pendiente_consentimiento"
    assert payload["Consentimiento datos"] == "NO"


def test_sync_lead_modo_completo_incluye_pii_con_consentimiento():
    """Un lead con consentimiento SÍ lleva la ficha completa."""
    from integrations.sheetdb import _build_payload

    state = {
        "thread_id": "+56900000007",
        "intake_respuestas": {
            "nombre": "Juan Pérez",
            "email": "jp@gmail.com",
            "situacion_actual": "Me demandaron por pensión",
            "consentimiento_datos": True,
        },
        "category": "pension_alimentos",
        "case_category": "pension_alimentos",
        "sentiment": "negativo",
        "urgency": "alta",
        "intent": "hablar_humano",
        "route": "intake",
        "clf_reason": "pide humano",
    }

    payload = _build_payload(state, canal="whatsapp", modo="completo")

    assert payload["Nombre completo"] == "Juan Pérez"
    assert payload["Correo electrónico"] == "jp@gmail.com"
    assert payload["Situación / Caso"] == "Me demandaron por pensión"
    assert payload["Consentimiento datos"] == "SÍ"
    assert payload["Estado lead"] == "nuevo"
