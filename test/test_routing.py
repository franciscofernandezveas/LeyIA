"""Flujos reales de clientes. Requiere que el grafo esté compilado."""
import pytest

from core.contracts import make_config
from graph.builder import agent_graph


@pytest.fixture
def state():
    return {
        "messages": [],
        "thread_id": "test-whatsapp-56912345678",
    }


def _enviar(state, texto: str):
    from langchain_core.messages import HumanMessage
    cfg = make_config(state["thread_id"])
    state["messages"].append(HumanMessage(content=texto))
    return agent_graph.invoke(state, cfg)


@pytest.mark.slow
def test_cliente_con_duda_de_pension_no_es_empujado_a_agendar(state):
    result = _enviar(state, "mi ex no me paga la pensión hace 6 meses, ¿qué puedo hacer?")
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    assert len(ai_messages) >= 1
    texto = ai_messages[-1].content.lower()

    # Debe orientar, mencionar proceso/documentos
    assert any(p in texto for p in ["demanda", "juzgado", "familia", "documentos"])

    # NO debe terminar con CTA de agenda como cierre forzado
    assert not any(cta in texto[-120:] for cta in [
        "¿le interesa?", "¿le agendo?", "agendar una asesoría",
    ]), f"El sistema empujó a agendar: {texto[-120:]}"


@pytest.mark.slow
def test_cliente_angustiado_recibe_orientacion_no_cta(state):
    result = _enviar(state, "estoy desesperado, me llegó demanda de pensión y no tengo plata")
    ai_messages = [m for m in result["messages"] if m.type == "ai"]
    texto = ai_messages[-1].content.lower()

    assert "desesperado" in texto or "complejo" in texto or "difícil" in texto
    # No debe ofrecer agendar inmediatamente
    assert "¿le interesa?" not in texto[-150:]


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
