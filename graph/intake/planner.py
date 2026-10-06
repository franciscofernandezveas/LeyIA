"""graph/intake/planner.py — Planner semántico del sub-flujo de intake.

v10 — Pausa real:
  - Nuevo tipo "pausar": el cliente pide pausa, un momento, más tarde,
    ahora no, continúo después → pausa la ficha y espera al próximo
    mensaje para reanudar.
  - Mantiene guardia determinista para botones de oferta y umbral por
    reversibilidad.

v9 — UX fluida:
  - Guardia determinista (costo 0) para los botones de la oferta de salida:
    "Hablar con humano" / "Lo intento de nuevo" se resuelven SIN LLM, así
    una misclasificación jamás rompe la oferta activa.
  - Umbral de confianza por REVERSIBILIDAD de la acción: derivar/abandonar
    son irreversibles para el turno → exigen confianza ≥ 0,55; pausar a FAQ
    es reversible (FAQ responde y reanudar retoma) → se permite con duda
    razonable en vez de castigar el mensaje contra el validador del campo.
  - Prompt: reconoce botones/listas ("Sí, autorizo", "No autorizo",
    "Lo intento de nuevo", títulos de la lista de etapa), la petición de
    pausa ("pausa", "un momento") y las correcciones de datos ya entregados.

v8 — 1 llamada LLM por turno activo: decide el tipo de turno Y extrae los
campos explícitos. Guardias estructurales deterministas; fallback clásico.
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
     "- Botones y listas: mensajes como 'Sí, autorizo', 'No autorizo', "
     "'Lo intento de nuevo' o el título de una opción de la lista de etapa "
     "→ respuesta_formulario (extrae lo que aplique; 'Lo intento de nuevo' "
     "no es un dato).\n"
     "- Si hace una pregunta o comenta algo ajeno a la ficha SIN cancelarla "
     "(precios, proceso, '¿y mi hija?', etc.) → duda_o_consulta.\n"
     "- Si pide pausa o un momento ('pausa', 'un segundo', 'espera', "
     "'más tarde', 'ahora no', 'continúo después') → pausar.\n"
     "- Si corrige un dato ya entregado ('no, mi correo es otro…', 'me "
     "equivoqué en el nombre') → respuesta_formulario y extrae el valor "
     "corregido en el campo correspondiente.\n"
     "- Si pide hablar con una persona/ejecutiva ('hablar con humano'), o es "
     "la segunda vez que lo pide, o muestra hartazgo → insiste_humano.\n"
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
        logger.warning("[intake] planner invocado con ficha completada → noop")
        return {"intake_decision": {"accion": "reanudar"},
                "intake_stage": "completado"}

    if consent_rechazado:
        return {"intake_decision": {"accion": "sin_consentimiento"},
                "intake_stage": "sin_consentimiento"}

    if not activo:
        return {"intake_decision": {"accion": "iniciar_ficha"},
                "intake_stage": "apertura"}

    # --- guardia: botones de la oferta de salida activa (0 LLM) ---
    if state.get("intake_oferta_qid"):
        from .nodes import RESPUESTAS_HUMANO, RESPUESTAS_REINTENTAR, _norm
        rn = _norm(state.get("query") or "")
        if rn in RESPUESTAS_HUMANO:
            return {"intake_decision": {"accion": "derivar_parcial"},
                    "intake_stage": "preguntando"}
        if rn in RESPUESTAS_REINTENTAR:
            return {"intake_decision": {"accion": "procesar_respuesta"},
                    "intake_stage": "preguntando"}

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
        "pausar": "pausar_ficha",
        "insiste_humano": "derivar_parcial",
        "abandonar": "abandonar_ficha",
    }.get(dec.tipo, "procesar_respuesta")

    # Umbral por reversibilidad: derivar/abandonar son irreversibles para
    # el turno → exigen confianza; la pausa a FAQ es reversible y barata.
    if accion in ("derivar_parcial", "abandonar_ficha") and dec.confianza < 0.55:
        accion = "procesar_respuesta"

    return {
        "intake_decision": {"accion": accion, **dec.model_dump()},
        "intake_stage": "preguntando",
    }
