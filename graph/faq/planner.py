"""graph/faq/planner.py — Decide qué hacer con la pregunta del usuario.

Fast-path determinista para intenciones claras; LLM acotado para casos
de borde. El RAG y la generación de la respuesta los hacen los nodos de
acción, no este planner.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState
from core.llm import with_structured_output
from graph.nodes import _parece_abort

from .contracts import FAQDecision

logger = logging.getLogger(__name__)

# Fast-paths: palabras que gatillan acciones sin LLM
_AGENDAR = ("agendar", "reservar", "hora", "cita", "asesoría", "reunión",
            "quiero hablar con un abogado", "tomar mi caso", "coordinar una cita")
_HUMANO = ("hablar con", "ejecutiva", "persona", "humano", "abogado real",
           "me comunico con", "llámame", "llamame", "contacto humano")
_FUERA = ("pizza", "delivery", "comida", "restaurante", "clima", "deporte",
          "política", "noticias", "chiste", "cancha", "partido", "gimnasio")


def _fast_path(query: str) -> FAQDecision | None:
    q = query.lower()
    if _parece_abort(query):
        return None
    if any(p in q for p in _FUERA):
        return FAQDecision(accion="fuera_de_dominio",
                           razon="keywords fuera de dominio detectadas")
    if any(p in q for p in _HUMANO):
        return FAQDecision(accion="hablar_humano",
                           razon="usuario pide contacto humano")
    if any(p in q for p in _AGENDAR):
        return FAQDecision(accion="sugerir_agendar",
                           razon="usuario menciona agendar/hora/asesoría")
    return None


def _decidir_llm(state: AgentState) -> FAQDecision:
    """Planner LLM acotado: decide la acción, no genera la respuesta."""
    prompt_text = f"""Eres el planner de un agente FAQ de un estudio jurídico chileno (Manzzo y Cía).
Decide la mejor acción para ESTE mensaje del usuario.

Mensaje del usuario: {state['query']}

Historial reciente de la conversación:
{chr(10).join(f'- {m.type}: {m.content[:200]}' for m in state.get('messages', [])[-4:]) or '(sin historial)'}

Acciones posibles (devuelve UNA):
- responder_pregunta: pregunta sobre derecho familiar, pensiones, divorcio, visitas, etc.
- sugerir_agendar: el usuario quiere reservar una cita o que un abogado lo atienda.
- hablar_humano: pide hablar con una persona/ejecutiva real.
- fuera_de_dominio: no se relaciona con servicios legales.
- pedir_clarificacion: la pregunta es ambigua o falta contexto.

Reglas:
- "quiero hora", "agendar", "cita", "asesoría" → sugerir_agendar.
- "hablar con ejecutiva", "persona real", "llámame" → hablar_humano.
- Preguntas sobre ubicación, modalidades, especialidades → responder_pregunta.
- Devuelve SOLO la acción y una justificación breve."""

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
        ])
        extractor = prompt | with_structured_output(FAQDecision)
        return extractor.invoke({})
    except Exception as e:
        logger.warning("[faq] planner LLM falló: %s", e)
        return FAQDecision(accion="responder_pregunta",
                           razon="fallback por error del planner")


def faq_planner(state: AgentState) -> AgentState:
    raw = state.get("query", "")
    decision = _fast_path(raw) or _decidir_llm(state)
    logger.info("[faq] acción=%s confianza=%.2f razon=%s",
                decision.accion, decision.confianza, decision.razon)
    return {
        "faq_decision": decision.model_dump(),
        "faq_stage": "respondiendo",
    }
