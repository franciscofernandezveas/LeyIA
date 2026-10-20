"""graph/faq/planner.py — Decide qué hacer con la pregunta del usuario.

v2.2 — FIX #7 (auditoría): fast-paths por substring.
  - Reordena: HUMANO → AGENDAR → SALUDO (con gate de largo). Antes el
    saludo corría primero y enterraba intenciones en mensajes como
    "hola, ¿puedo hablar con alguien?" o "hola, llámame" — iban a
    responder_pregunta cuando el cliente pedía humano explícitamente.
  - Elimina _FUERA como fast-path. Su blacklist por substring colisionaba
    con vocabulario real de derecho de familia ("la comida de los niños",
    "se fue de viaje con mi hijo", "vender la casa en común") — falsos
    positivos que expulsaban a clientes angustiados con casos legítimos.
    fuera_de_dominio queda al LLM, que ve historial completo.
  - SALUDO limitado a mensajes cortos (≤ 8 palabras) como autodefensa:
    "hola" es la palabra más débil del idioma; un mensaje largo que la
    contiene casi nunca ES un saludo.
  - HUMANO y AGENDAR mantienen fast-path sin gate: sus keywords son
    específicas ("hablar con una persona", "reservar hora") y en un
    mensaje largo con esta frase la intención sigue siendo la misma.

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

# _FUERA — DEPRECATED (v2.2, fix #7). Eliminado como fast-path: el
# blacklist por substring colisionaba con vocabulario real de familia.
# Ejemplos que caían erróneamente en fuera_de_dominio:
#   "no me deja plata para la comida de los niños"  → pensión
#   "mi ex se fue de viaje con mi hijo sin permiso" → tupla/salida
#   "quiero vender la casa que tenemos en común"    → divorcio/bienes
# Rechazar a un cliente angustiado con un caso legítimo es el falso
# positivo más caro de este producto. fuera_de_dominio ahora lo decide
# SOLO el LLM (_decidir_llm), que ve historial completo.


def _fast_path(query: str) -> FAQDecision | None:
    q = query.lower().strip().rstrip("?").rstrip("!")
    n_palabras = len(q.split())

    # HUMANO primero (fix #7): "hola, ¿puedo hablar con alguien?" debe
    # resolverse como hablar_humano, no como saludo. Keyword específica
    # ("hablar con una persona", "quiero una ejecutiva") → no requiere
    # gate de largo: en un mensaje largo con esta frase, la intención
    # sigue siendo la misma (contacto humano explícito).
    if any(p in q for p in _HUMANO_EXPLICITO):
        return FAQDecision(accion="hablar_humano",
                           razon="solicitud explícita de contacto humano")

    # AGENDAR segundo (fix #7): misma lógica que HUMANO. "quiero
    # agendar" / "reservar hora" son específicas, no se entierran por
    # saludo previo.
    if any(p in q for p in _AGENDAR_EXPLICITO):
        return FAQDecision(accion="sugerir_agendar",
                           razon="solicitud explícita de agendamiento")

    # SALUDO al final y con gate de largo (fix #7): "hola" es el token
    # más débil del idioma; un mensaje largo que la contiene casi nunca
    # ES un saludo (es contenido con fórmula de cortesía). Casos:
    #   "hola, ¿quiénes son?"              ≤ 8 → responder_pregunta ✓
    #   "buenos días, ¿atienden sábados?"  ≤ 8 → responder_pregunta ✓
    #   "hola, me llamo X y tengo pensión" >  8 → LLM decide con contexto
    if n_palabras <= 8 and any(p in q for p in _SALUDO_PRESENTACION):
        return FAQDecision(accion="responder_pregunta",
                           razon="saludo, presentación o consulta general del estudio")

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
    # v2.3 — FIX #11 (auditoría): faq_stage eliminado. Era dead code y su
    # semántica era incorrecta (siempre "respondiendo", incluso cuando la
    # acción es pedir_clarificacion).
    return {"faq_decision": decision.model_dump()}

