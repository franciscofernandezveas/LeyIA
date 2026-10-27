import pytest
from langchain_core.messages import HumanMessage

import graph.nodes as nodes
from graph.nodes import analyze_sentiment, _hidratar_contacto
from core.contracts import (
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_HANDOFF_CHOICE, AnalisisResult,
)

def _s(**kw):
    base = {"query": "", "messages": [], "closed": False, "thread_id": "t"}
    base.update(kw)
    return base


@pytest.fixture
def stub_classifier(monkeypatch):
    """Evita LLM y BD en analyze_sentiment cuando el flujo llega al clasificador."""
    class _StubPrompt:
        def __or__(self, other):
            return other

    class _StubChain:
        def __init__(self, result):
            self._r = result
        def with_retry(self, **kw):
            return self
        def invoke(self, _):
            return self._r

    def _factory(result: AnalisisResult):
        monkeypatch.setattr(nodes.ChatPromptTemplate, "from_messages",
                            lambda *_a, **_k: _StubPrompt())
        monkeypatch.setattr(nodes, "with_structured_output",
                            lambda *_a, **_k: _StubChain(result))
        monkeypatch.setattr(nodes, "upsert_conversation",
                            lambda **_kw: True)
    return _factory


# 1) Reapertura por humano explícito en hilo cerrado (0-LLM, no usa fixture)
def test_reabre_por_humano_en_hilo_cerrado():
    out = analyze_sentiment(_s(query="pásame con una ejecutiva", closed=True))
    assert out["route"] == ROUTE_HANDOFF_CHOICE
    assert out["closed"] is False
    assert out["crisis"] is False


# 2) Reapertura por CTA agendar aunque NO haya intake_completado
def test_reabre_por_agendar_sin_intake_completado():
    out = analyze_sentiment(_s(query="quiero agendar una hora", closed=True,
                               intake_completado=False))
    assert out["route"] == ROUTE_AGENDAR
    assert out["closed"] is False


# 3) Hidratación desde el historial (el caso de "amnesia" reportado)
def test_hidrata_nombre_y_email_desde_historial():
    msgs = [
        HumanMessage(content="soy Pedro Pérez"),
        HumanMessage(content="mi correo es pedro@mail.com"),
    ]
    pre = _hidratar_contacto(_s(messages=msgs))
    assert pre["intake_respuestas"] == {"nombre": "Pedro Pérez", "email": "pedro@mail.com"}
    assert pre["lead_nombre"] == "Pedro Pérez"
    assert pre["lead_email"] == "pedro@mail.com"


# 4) Idempotencia: cuando TODAS las fuentes ya están sincronizadas → None
def test_hidratacion_idempotente():
    s = _s(
        intake_respuestas={"nombre": "Ana", "email": "ana@x.cl"},
        lead_nombre="Ana",
        lead_email="ana@x.cl",
    )
    assert _hidratar_contacto(s) is None


# 5) Frase negada NO dispara el CTA agendar → sigue al clasificador, y si este
#    enruta a FAQ, el hilo cerrado NO se reabre
def test_cta_negado_no_reabre(stub_classifier):
    stub_classifier(AnalisisResult.model_construct(
        sentiment="negativo", urgency="baja", intent="respuestas_faq",
        category="otro", reason="test",
    ))
    out = analyze_sentiment(_s(query="no quiero agendar", closed=True))
    assert out.get("route") == ROUTE_FAQ
    assert out.get("closed", True) is True
