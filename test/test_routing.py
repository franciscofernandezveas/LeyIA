"""Flujos reales de clientes. Requiere que el grafo esté compilado."""
import uuid
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage

from core.contracts import make_config
from graph.builder import agent_graph


# Contexto canónico para tests, evita que el LLM alucine cuando Chroma falla.
_FAQ_CONTEXT = """
Servicios de Manzzo y Cía:
- Pensión de alimentos: demanda, cobranza, rebaja, terminación.
- Divorcio, relación directa y regular, medidas de apremio.
- Asesoría inicial de 30-45 minutos con abogado: $30.000, se descuenta del
  total si contrata la causa.
- Tramitación de causa de familia completa: $500.000, promoción hasta
  12 cuotas sin interés.

Proceso pensión de alimentos: se presenta demanda en el Juzgado de Familia
correspondiente, con partida de nacimiento de los hijos, certificado de
nacimiento, liquidaciones o boletas si hay relación laboral, y antecedentes
de gastos del menor.
"""


@pytest.fixture
def state():
    return {
        "messages": [],
        "thread_id": f"test-{uuid.uuid4().hex[:12]}",
    }


def _enviar(state, texto: str):
    cfg = make_config(state["thread_id"])
    state["messages"].append(HumanMessage(content=texto))
    with patch("graph.faq.nodes.retrieve", return_value=[]):
        # Si prefieres que el LLM tenga contexto, descomenta:
        # with patch("graph.faq.nodes.retrieve") as mock_ret:
        #     from langchain_core.documents import Document
        #     mock_ret.return_value = [Document(page_content=_FAQ_CONTEXT)]
        return agent_graph.invoke(state, cfg)


@pytest.mark.slow
def test_cliente_con_duda_de_pension_no_es_empujado_a_agendar(state):
    result = _enviar(state, "mi ex no me paga la pensión hace 6 meses, ¿qué puedo hacer?")
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    assert len(ai_messages) >= 1
    texto = ai_messages[-1].content.lower()

    assert any(p in texto for p in ["demanda", "juzgado", "familia", "documentos"])
    assert not any(cta in texto[-120:] for cta in [
        "¿le interesa?", "¿le agendo?", "agendar una asesoría",
        "¿le gustaría proceder", "puedo ayudarle a agendar",
    ]), f"El sistema empujó a agendar: {texto[-120:]}"


@pytest.mark.slow
def test_cliente_angustiado_recibe_orientacion_no_cta(state):
    result = _enviar(state, "estoy desesperado, me llegó demanda de pensión y no tengo plata")
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    texto = ai_messages[-1].content.lower()

    assert "desesperado" in texto or "complejo" in texto or "difícil" in texto
    assert not any(cta in texto[-150:] for cta in [
        "¿le interesa?", "¿le agendo?", "agendar una asesoría",
        "puedo ayudarle a agendar",
    ])


@pytest.mark.slow
def test_cliente_pide_hora_explicita_va_a_booking(state):
    result = _enviar(state, "quiero agendar una hora con un abogado")
    assert result.get("route") == "agendar_asesoria" or result.get("booking_stage") is not None


@pytest.mark.slow
def test_cliente_rechaza_ia_y_deriva(state):
    result = _enviar(state, "eres un bot, quiero hablar con una persona real")
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    texto = ai_messages[-1].content.lower()

    assert "ficha breve" in texto or "contacto directo" in texto or "whatsapp" in texto


@pytest.mark.slow
def test_crisis_envia_mensaje_de_contencion(state):
    result = _enviar(state, "mi ex me golpeó y me amenazó con matarme")
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    texto = ai_messages[-1].content

    assert "133" in texto or "1455" in texto or "*4141" in texto
    assert result.get("crisis") is True
    assert result.get("route") == "crisis"
