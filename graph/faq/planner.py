"""graph/faq/planner.py — Decide qué hacer con la pregunta del usuario.

v2.1 — Fast-paths ampliados: saludos, presentación y consultas generales del
estudio van directo a responder_pregunta, donde el system prompt del FAQ
ya se presenta y orienta. Se mantiene la precaución en humano/agendar/fuera.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState
from core.llm import with_structured_output

from .contracts import FAQDecision

logger = logging.getLogger(__name__)

# Fast-paths EXPLÍCITOS. Si no calza, el LLM decide.
_SALUDO_PRESENTACION = (
    "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "como me pueden ayudar", "como pueden ayudarme", "quienes son",
    "qué hacen", "que hacen", "a que se dedican", "a qué se dedican",
    "en que me pueden ayudar", "en qué me pueden ayudar", "servicios",
    "que servicios ofrecen", "qué servicios ofrecen", "donde quedan",
    "dónde quedan", "quienes son ustedes", "quiénes son ustedes",
    "presentacion", "presentación",
)
_AGENDAR_EXPLICITO = (
    "agendar", "reservar", "quiero una cita", "quiero asesoría",
    "quiero asesoria", "tomar mi caso", "coordinar una cita",
    "necesito un abogado", "quiero hablar con un abogado",
)
_HUMANO_EXPLICITO = (
    "hablar con una ejecutiva", "hablar con una persona",
    "hablar con un humano", "persona real", "ejecutiva real",
    "llámame", "llamame", "contacto humano", "abogado real",
    "hablar directamente con alguien", "quiero una ejecutiva",
    "puedo hablar con alguien", "hablar con alguien de verdad",
)
_FUERA = (
    "pizza", "delivery", "comida", "restaurante", "clima", "deporte",
    "política", "noticias", "chiste", "cancha", "partido", "gimnasio",
    "cine", "película", "viaje", "hotel", "comprar", "vender",
)


def _fast_path(query: str) -> FAQDecision | None:
    q = query.lower().strip().rstrip("?").rstrip("!")

    # Saludos y preguntas de presentación/capacidades → responder_pregunta
    if any(p in q for p in _SALUDO_PRESENTACION):
        return FAQDecision(accion="responder_pregunta",
                           razon="saludo, presentación o consulta general del estudio")

    if any(p in q for p in _FUERA):
        return FAQDecision(accion="fuera_de_dominio",
                           razon="keywords fuera de dominio detectadas")

    if any(p in q for p in _HUMANO_EXPLICITO):
        return FAQDecision(accion="hablar_humano",
                           razon="solicitud explícita de contacto humano")

    if any(p in q for p in _AGENDAR_EXPLICITO):
        return FAQDecision(accion="sugerir_agendar",
                           razon="solicitud explícita de agendamiento")

    return None


def _decidir_llm(state: AgentState) -> FAQDecision:
    history = "\n".join(
        f"- {m.type}: {m.content[:200]}"
        for m in state.get("messages", [])[-5:]
    ) or "(sin historial)"

    prompt_text = f"""Eres el planner de un agente FAQ de un estudio jurídico chileno (Manzzo y Cía).
Decide la mejor acción para ESTE mensaje del usuario.

Mensaje del usuario: {state['query']}

Historial reciente:
{history}

Acciones posibles (devuelve UNA):
- responder_pregunta: saludos, presentación, preguntas sobre derecho familiar, pensiones, divorcio, visitas, costos, proceso, ubicación, servicios del estudio, etc.
- sugerir_agendar: el usuario quiere reservar una cita o que un abogado tome su caso.
- hablar_humano: pide explícitamente hablar con una persona/ejecutiva real.
- fuera_de_dominio: no se relaciona con servicios legales.
- pedir_clarificacion: SOLO si la pregunta es realmente ambigua o incomprensible.

Reglas importantes:
- "Hola", saludos o preguntas como "¿quiénes son?", "¿cómo me ayudan?", "¿qué servicios ofrecen?" → responder_pregunta.
- "hablar con usted" o frases corteses NO son hablar_humano; son responder_pregunta.
- "¿a qué hora atienden?" o "¿cuándo abren?" NO son agendar; son responder_pregunta.
- "quiero agendar", "quiero una asesoría" SÍ son sugerir_agendar.
- "hablar con una ejecutiva", "quiero una persona real" SÍ son hablar_humano.
- Devuelve SOLO la acción y una justificación breve. No inventes hechos."""  # noqa: E501

    try:
        prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])
        extractor = prompt | with_structured_output(FAQDecision)
        return extractor.invoke({})
    except Exception as e:
        logger.warning("[faq] planner LLM falló: %s", e)
        return FAQDecision(accion="responder_pregunta",
                           razon="fallback por error del planner")


def faq_planner(state: AgentState) -> AgentState:
    raw = state.get("query", "")
    decision = _fast_path(raw) or _decidir_llm(state)
    logger.info("[faq] acción=%s razon=%s", decision.accion, decision.razon)
    return {
        "faq_decision": decision.model_dump(),
        "faq_stage": "respondiendo",
    }
