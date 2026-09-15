"""graph/intake/planner.py — Planner semántico del sub-flujo de intake.

v8 — 1 llamada LLM por turno activo: decide el tipo de turno Y extrae los
campos explícitos (patrón BookingDecision). Los guardias estructurales
(primer turno, reanudación, consentimiento rechazado, ficha completa) siguen
siendo deterministas y de costo cero. Si el LLM falla, se degrada al
comportamiento clásico: tratar el mensaje como respuesta al campo activo.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState
from core.llm import with_structured_output
from graph.nodes import _recent_messages

from .contracts import IntakeDecision

logger = logging.getLogger(__name__)

_MAX_TOKENS_HISTORIA = 200

_PROMPT_TURNO = ChatPromptTemplate.from_messages([
    ("system",
     "Eres el planificador del registro de clientes de Manzzo y Cía (estudio "
     "jurídico chileno). El cliente está completando una ficha conversada por "
     "WhatsApp. Debes decidir QUÉ ES el último mensaje y EXTRAER los datos "
     "explícitos que traiga.\n\n"
     "Pregunta activa de la ficha: '{pregunta_activa}'\n"
     "Campos ya registrados: {registrados}\n\n"
     "Reglas:\n"
     "- Si el mensaje responde o aporta datos (aunque sea breve: un número de "
     "opción, un sí, una descripción) → respuesta_formulario y extrae TODO "
     "dato explícito (puede traer varios campos a la vez).\n"
     "- Si hace una pregunta o comenta algo ajeno a la ficha SIN cancelarla "
     "(precios, proceso, '¿y mi hija?', etc.) → duda_o_consulta.\n"
     "- Si pide hablar con una persona/ejecutiva, o es la segunda vez que lo "
     "pide, o muestra hartazgo con las preguntas → insiste_humano.\n"
     "- Si no quiere seguir con el registro → abandonar.\n"
     "- NUNCA inventes ni infieras datos; null en lo ausente.\n"
     "- consentimiento_datos NO existe en el esquema: esa respuesta se valida "
     "como sí/no con el mensaje crudo, nunca por inferencia.\n"
     "- nombre: solo nombres propios reales; una frase tipo 'quiero hablar "
     "con un humano' NO es un nombre (→ null).\n"
     "- email: normaliza dictados por voz ('arroba'→@, 'punto'→., "
     "'guión bajo'→_).\n"
     "- Un número solo (1-4) respondiendo a la pregunta de etapa → mapea a la "
     "opción correspondiente."),
    ("human",
     "Historial reciente:\n{historial}\n\nÚltimo mensaje del cliente: {query}"),
])


def _decision_fallback(motivo: str) -> dict:
    d = IntakeDecision(tipo="respuesta_formulario", confianza=0.0,
                       razon=f"fallback: {motivo}")
    return {"accion": "procesar_respuesta", **d.model_dump()}


def intake_planner(state: AgentState) -> AgentState:
    activo = bool(state.get("intake_activo"))
    completado = bool(state.get("intake_completado"))
    consent_rechazado = ((state.get("intake_respuestas") or {})
                         .get("consentimiento_datos") is False)

    # --- guardias deterministas (0 LLM) ---
    if state.get("intake_resume"):
        return {"intake_decision": {"accion": "reanudar"},
                "intake_stage": "preguntando"}

    if completado:
        # Inalcanzable por diseño (analyze no rutea aquí con completado);
        # reanudar con ledger vacío cierra en silencio sin romper el turno.
        logger.warning("[intake] planner invocado con ficha completada → noop")
        return {"intake_decision": {"accion": "reanudar"},
                "intake_stage": "completado"}

    if consent_rechazado:
        return {"intake_decision": {"accion": "sin_consentimiento"},
                "intake_stage": "sin_consentimiento"}

    if not activo:
        return {"intake_decision": {"accion": "iniciar_ficha"},
                "intake_stage": "apertura"}

    # --- turno activo: 1 LLM (decisión + extracción) ---
    from .nodes import QUESTIONS, _pendientes  # import local (evita ciclo)
    pend = _pendientes(state, state.get("intake_respuestas") or {})
    if not pend:
        return {"intake_decision": {"accion": "procesar_respuesta"},
                "intake_stage": "preguntando"}

    pregunta = pend[0]["prompt"][:300]
    registrados = {k: v for k, v in (state.get("intake_respuestas") or {}).items()
                   if v not in (None, "")}
    historial = "\n".join(
        f"- {m.type}: {m.content[:_MAX_TOKENS_HISTORIA]}"
        for m in _recent_messages(state, k=6)
    ) or "(sin historial)"

    chain = _PROMPT_TURNO | with_structured_output(IntakeDecision)
    try:
        dec: IntakeDecision = chain.with_retry(stop_after_attempt=2).invoke({
            "pregunta_activa": pregunta,
            "registrados": registrados,
            "historial": historial,
            "query": state.get("query") or "",
        })
    except Exception as e:
        logger.warning("planner de intake falló (fallback a respuesta): %s", e)
        return {"intake_decision": _decision_fallback(str(e)[:120]),
                "intake_stage": "preguntando"}

    accion = {
        "respuesta_formulario": "procesar_respuesta",
        "duda_o_consulta": "pausar_para_faq",
        "insiste_humano": "derivar_parcial",
        "abandonar": "abandonar_ficha",
    }.get(dec.tipo, "procesar_respuesta")

    # Baja confianza en escape/duda → conservador: tratar como respuesta.
    if accion != "procesar_respuesta" and dec.confianza < 0.55:
        accion = "procesar_respuesta"

    return {
        "intake_decision": {"accion": accion, **dec.model_dump()},
        "intake_stage": "preguntando",
    }
