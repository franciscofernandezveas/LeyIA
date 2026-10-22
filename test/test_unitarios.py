"""Tests unitarios de funciones deterministas. No usan LLM."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from core.contracts import (
    AnalisisResult, ROUTE_AGENDAR, ROUTE_FAQ,
    ROUTE_HANDOFF_CHOICE, ROUTE_FUERA_DOMINIO,
)
from graph.faq.planner import _fast_path, _negada
from graph.nodes import (
    _es_crisis_keyword,
    _aceptacion_ambigua_agendar,
    _ultima_ai_ofrecio_agendar,
    compute_route,
    _pide_humano_explicito,
)
from graph.utils import RESET_SESION_NUEVA


class TestFastPathFAQ:
    # Casos donde el fast-path NO debe decidir nada (van al LLM).
    # La invariante correcta es: si _fast_path devuelve algo, NO sea agenda.
    @pytest.mark.parametrize("query", [
        "necesito un abogado que tome mi caso de pensión",
        "quiero que una abogada me ayude con la pensión alimenticia",
        "mi ex no me paga hace 8 meses, ¿qué puedo hacer?",
        "tengo la demanda de pensión encima y no sé qué hacer",
        "me despidieron sin finiquito, ¿ustedes toman causas laborales?",
    ])
    def test_no_fuerza_agenda_si_decide(self, query):
        d = _fast_path(query)
        if d is not None:
            assert d.accion != "sugerir_agendar", f"fast_path forzó agenda: {query}"

    # Casos donde el fast-path SÍ decide y debe ser NO agenda.
    @pytest.mark.parametrize("query", [
        "no quiero agendar todavía, solo información",
        "aún no quiero asesoría, quiero saber primero si puedo rebajar",
        "no me interesa agendar",
    ])
    def test_negaciones_no_son_agenda(self, query):
        d = _fast_path(query)
        assert d is not None, f"fast_path debería haber decidido: {query}"
        assert d.accion != "sugerir_agendar"

    @pytest.mark.parametrize("query", [
        "quiero agendar una hora",
        "reservar una cita para esta semana",
        "quiero coordinar una asesoría",
        "agendar asesoría con abogado",
    ])
    def test_si_acepta_agenda_explicita(self, query):
        d = _fast_path(query)
        assert d is not None, f"fast_path no detectó agenda explícita: {query}"
        assert d.accion == "sugerir_agendar"

    @pytest.mark.parametrize("query", [
        "eres un bot, pásame con alguien",
        "no quiero hablar con una máquina",
        "quiero hablar con un abogado",
        "hablar con una ejecutiva real",
        "pásame con una ejecutiva",
    ])
    def test_detecta_humano(self, query):
        d = _fast_path(query)
        assert d is not None, f"fast_path no detectó humano: {query}"
        assert d.accion == "hablar_humano"


class TestAntiNegacion:
    @pytest.mark.parametrize("query,keyword,esperado", [
        ("no quiero agendar", "agendar", True),
        ("aún no quiero asesoría", "asesoría", True),
        ("todavía no quiero agendar", "agendar", True),
        ("no quiero agendar todavía", "agendar", True),
        ("quiero agendar", "agendar", False),
        ("quiero agendar una hora", "agendar", False),
        ("no quiero dejar de pagar", "agendar", False),
    ])
    def test_negada(self, query, keyword, esperado):
        assert _negada(query, keyword) is esperado


class TestCrisisKeywords:
    @pytest.mark.parametrize("query", [
        "ayer mi ex me golpeó delante de los niños",
        "me pega y me va a matar",
        "estoy pensando en suicidarme",
        "no quiero vivir más",
        "mi pareja no me deja salir y me controla el celular",
        "me amenaza con un cuchillo",
        "me amenaza con matarme",
    ])
    def test_activa(self, query):
        assert _es_crisis_keyword(query) is True

    @pytest.mark.parametrize("query", [
        "tengo una demanda de pensión",
        "quiero saber cómo demandar",
        "me llegó un embargo al sueldo",
        "estoy estresado con el divorcio",
    ])
    def test_no_activa(self, query):
        assert _es_crisis_keyword(query) is False


class TestAceptacionesAmbiguas:
    def test_detecta(self):
        assert _aceptacion_ambigua_agendar("sí") is True
        assert _aceptacion_ambigua_agendar("dale") is True
        assert _aceptacion_ambigua_agendar("ok") is True

    def test_no_detecta(self):
        assert _aceptacion_ambigua_agendar("quiero agendar el lunes") is False
        assert _aceptacion_ambigua_agendar("no quiero") is False


class TestUltimaAIOfrecioAgendar:
    def test_si(self):
        state = {
            "messages": [
                AIMessage(content="Puedo agendarle una asesoría. ¿Le interesa?"),
                HumanMessage(content="sí"),
            ]
        }
        assert _ultima_ai_ofrecio_agendar(state) is True

    def test_no(self):
        state = {
            "messages": [
                AIMessage(content="La pensión de alimentos es un derecho de los hijos."),
                HumanMessage(content="sí"),
            ]
        }
        assert _ultima_ai_ofrecio_agendar(state) is False


class TestComputeRoute:
    def test_agendar(self):
        r = AnalisisResult(
            sentiment="neutro", urgency="media",
            intent="agendar_asesoria", category="pension_alimentos",
            reason="pide hora", riesgo=False,
        )
        assert compute_route(r) == ROUTE_AGENDAR

    def test_hablar_humano(self):
        r = AnalisisResult(
            sentiment="neutro", urgency="media",
            intent="hablar_humano", category="consulta_general",
            reason="pide persona", riesgo=False,
        )
        assert compute_route(r) == ROUTE_HANDOFF_CHOICE

    def test_faq(self):
        r = AnalisisResult(
            sentiment="neutro", urgency="media",
            intent="respuestas_faq", category="pension_alimentos",
            reason="duda", riesgo=False,
        )
        assert compute_route(r) == ROUTE_FAQ

    def test_fuera_dominio(self):
        r = AnalisisResult(
            sentiment="neutro", urgency="baja",
            intent="fuera_de_dominio", category="otro",
            reason="contador", riesgo=False,
        )
        assert compute_route(r) == ROUTE_FUERA_DOMINIO


class TestHumanoExplicito:
    @pytest.mark.parametrize("query", [
        "eres un bot",
        "no quiero hablar con una máquina",
        "quiero hablar con un abogado real",
        "pásame con una ejecutiva",
    ])
    def test_si(self, query):
        assert _pide_humano_explicito(query) is True

    @pytest.mark.parametrize("query", [
        "quiero agendar una hora",
        "mi ex no me paga la pensión",
        "no quiero agendar",
    ])
    def test_no(self, query):
        assert _pide_humano_explicito(query) is False


class TestResets:
    def test_sesion_nueva_incluye_crisis(self):
        assert RESET_SESION_NUEVA.get("crisis") is None
        assert RESET_SESION_NUEVA.get("closed") is False
        assert RESET_SESION_NUEVA.get("handoff_choice_pending") is False
