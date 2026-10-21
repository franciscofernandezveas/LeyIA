"""graph/faq/planner.py — Decide qué hacer con la pregunta del usuario.

v2.3 — B4 (post-incidente producción 20/10): el CTA standalone de agenda
  deja de ser elegible por el LLM.

  Causal del bug: el LLM elegía sugerir_agendar ante descripciones de caso
  ("mi ex no me paga hace 8 meses", "¿cómo se demanda?") porque la
  definición de la acción ("que un abogado tome su caso") calzaba con el
  ~80% del tráfico real. El cliente recibía el CTA pelado — dos turnos
  seguidos — en vez de orientación.

  Ahora:
  - El catálogo del LLM tiene 4 acciones (responder_pregunta es default);
    sugerir_agendar NO existe para el LLM. La oferta de agenda vive
    embebida en la respuesta RAG (mission del faq_system_prompt v4.0.1).
  - sugerir_agendar subsiste en FAQDecision pero SOLO la emite el
    fast-path determinista (_AGENDAR_EXPLICITO: keywords reales).
  - La anti-repetición (CTA ×2) la resuelve el supervisor con el ledger
    cta_agenda_ofrecida (AgentState), no el planner — política de flujo
    vive en el router, idiom idéntico a intake_oferta_qid.
  - "quiero hablar con un abogado" pasa de _AGENDAR_EXPLICITO a
    _HUMANO_EXPLICITO: es pedir persona, no pedir hora (coherente con el
    clasificador padre).

v2.2 — FIX #7 (auditoría): fast-paths por substring.
  - Reordena: HUMANO → AGENDAR → SALUDO (con gate de largo). ...
  - Elimina _FUERA como fast-path. fuera_de_dominio queda al LLM.
  - SALUDO limitado a mensajes cortos (≤ 8 palabras).

v2.1 — Fast-paths ampliados: saludos, presentación y consultas generales.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState
from core.llm import with_structured_output

from .contracts import FAQDecision

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Fast-paths EXPLÍCITOS (0-LLM). Si no calza, el LLM decide.
# --------------------------------------------------------------------------
_SALUDO_PRESENTACION = (
    "hola", "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "como me pueden ayudar", "como pueden ayudarme", "quienes son",
    "qué hacen", "que hacen", "a que se dedican", "a qué se dedican",
    "en que me pueden ayudar", "en qué me pueden ayudar", "servicios",
    "que servicios ofrecen", "qué servicios ofrecen", "donde quedan",
    "dónde quedan", "quienes son ustedes", "quiénes son ustedes",
    "presentacion", "presentación",
)
# Única vía al CTA standalone (v2.3): keywords explícitas de agendamiento.
_AGENDAR_EXPLICITO = (
    "agendar", "reservar", "quiero una cita", "quiero asesoría",
    "quiero asesoria", "tomar mi caso", "coordinar una cita",
    "necesito un abogado",
)
_HUMANO_EXPLICITO = (
    "hablar con una ejecutiva", "hablar con una persona",
    "hablar con un humano", "persona real", "ejecutiva real",
    "llámame", "llamame", "contacto humano", "abogado real",
    "hablar directamente con alguien", "quiero una ejecutiva",
    "puedo hablar con alguien", "hablar con alguien de verdad",
    "quiero hablar con un abogado",   # v2.3: desde _AGENDAR_EXPLICITO
)

# _FUERA — DEPRECATED (v2.2, fix #7). Ver historial.


def _fast_path(query: str) -> FAQDecision | None:
    q = query.lower().strip().rstrip("?").rstrip("!")
    n_palabras = len(q.split())

    # HUMANO primero (v2.2): las keywords son específicas; sin gate de largo.
    if any(p in q for p in _HUMANO_EXPLICITO):
        return FAQDecision(accion="hablar_humano",
                           razon="solicitud explícita de contacto humano")

    # AGENDAR segundo (v2.2/2.3): única forma del CTA standalone.
    if any(p in q for p in _AGENDAR_EXPLICITO):
        return FAQDecision(accion="sugerir_agendar",
                           razon="solicitud explícita de agendamiento")

    # SALUDO al final y con gate de largo (v2.2).
    if n_palabras <= 8 and any(p in q for p in _SALUDO_PRESENTACION):
        return FAQDecision(accion="responder_pregunta",
                           razon="saludo, presentación o consulta general del estudio")

    return None


def _decidir_llm(state: AgentState) -> FAQDecision:
    history = "\n".join(
        f"- {m.type}: {m.content[:200]}"
        for m in state.get("messages", [])[-5:]
    ) or "(sin historial)"

    prompt_text = f"""Eres el planner del agente FAQ de Manzzo y Cía (Derecho de Familia, Chile).
Decide la mejor acción para ESTE mensaje del usuario.

Mensaje del usuario: {state['query']}

Historial reciente:
{history}

Acciones posibles (devuelve UNA):
- responder_pregunta: orientar con contenido (caso propio, proceso, costos,
  documentos, plazos, consecuencias). ACCIÓN DEFAULT y la más frecuente.
  Incluye: describir el problema legal, preguntar "¿cómo se demanda?",
  "¿cómo es el proceso?", "¿cómo pido rebaja?", costos, "necesito ayuda"
  sin pedir hora. El agente orienta PRIMERO.
- hablar_humano: pide EXPLÍCITAMENTE hablar con persona/ejecutiva real, o
  acepta una oferta de contacto humano previa.
- fuera_de_dominio: nada relacionado con servicios legales.
- pedir_clarificacion: SOLO si el mensaje es realmente incomprensible.

NOTA DE CONTRATO (v2.3 — inquebrantable):
- No existe acción de agendamiento standalone para ti. La oferta de
  agendar asesoría va SIEMPRE embebida dentro de la respuesta orientativa
  (cierra con el siguiente paso concreto), nunca como mensaje aparte.
- Si el último mensaje del asistente ya ofrecía agendar O contacto humano
  (texto breve que termina en "¿Le interesa?" / "¿Cómo prefiere?"), NO
  vuelvas a esa oferta como respuesta: este turno responde la nueva
  consulta. La aceptación implícita ("sí", "dale") la gestiona el
  clasificador padre, no tú.

Ejemplos normativos:
- "mi ex no me paga la pensión hace 8 meses" → responder_pregunta
- "¿cómo se hace para demandar? no sé de trámites" → responder_pregunta
- "¿cuánto cobran por demandar pensión?" → responder_pregunta
- "me llegó demanda, ¿qué hago?" → responder_pregunta
- "quiero que me atienda una persona" → hablar_humano
- "¿me recomiendan un contador?" → fuera_de_dominio

Devuelve SOLO la acción y una justificación breve. No inventes hechos."""

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
    logger.info("[faq] acción=%s razon=%s", decision.accion,
                decision.razon)
    return {"faq_decision": decision.model_dump()}
