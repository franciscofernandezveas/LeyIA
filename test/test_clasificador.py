"""Tests del clasificador unificado (analyze_sentiment). Usan LLM."""
import uuid

import pytest
from langchain_core.messages import HumanMessage

from core.contracts import make_config
from graph.builder import agent_graph


def _clasificar(query: str):
    tid = f"test-clf-{uuid.uuid4().hex[:12]}"
    cfg = make_config(tid)
    state = {
        "messages": [HumanMessage(content=query)],
        "thread_id": tid,
        "query": query,
    }
    return agent_graph.invoke(state, cfg)


@pytest.mark.slow
@pytest.mark.parametrize("query", [
    "mi ex no me paga la pensión hace 8 meses",
    "estoy desesperado, me llegó demanda de pensión",
    "necesito un abogado que tome mi caso",
    "tengo medidas de apremio vigentes, ¿qué puedo hacer?",
])
def test_casos_informativos_son_faq(query):
    result = _clasificar(query)
    assert result["intent"] == "respuestas_faq"
    assert result["route"] == "respuestas_faq"


@pytest.mark.slow
@pytest.mark.parametrize("query", [
    "quiero agendar una hora con un abogado",
    "reservar una cita para esta semana",
    "sí, me interesa la asesoría, ¿para cuándo hay hora?",
])
def test_pedidos_explicitos_son_agendar(query):
    result = _clasificar(query)
    assert result["intent"] == "agendar_asesoria"
    assert result["route"] == "agendar_asesoria"


@pytest.mark.slow
@pytest.mark.parametrize("query", [
    "quiero hablar con un abogado",
    "eres un bot, pásame con una persona real",
    "prefiero que me atienda directamente una ejecutiva",
])
def test_pide_persona_es_hablar_humano(query):
    result = _clasificar(query)
    assert result["intent"] == "hablar_humano"


@pytest.mark.slow
@pytest.mark.parametrize("query", [
    "ayer mi ex me golpeó delante de los niños",
    "no soporto más, pienso en hacerme daño",
    "me amenaza con matarme",
])
def test_crisis_detectada(query):
    result = _clasificar(query)
    assert result.get("riesgo") is True or result.get("crisis") is True
    assert result["route"] == "crisis"


@pytest.mark.slow
def test_solo_si_sin_oferta_previa_no_es_agendar():
    result = _clasificar("sí")
    assert result["intent"] != "agendar_asesoria"
