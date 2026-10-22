"""graph/intake/planner.py — Planner semántico del sub-flujo de intake.

v17 — Ramas y urgencias:
  - Prompt actualizado con las reglas de extracción de los nuevos campos
    (area_caso se extrae proactivamente del relato: evita la pregunta
    clasificadora cuando el área es inequívoca).
  - Override de urgencia: detenido=True (detención VIGENTE declarada)
    fuerza la acción derivar_urgente incluso si el turno era una duda —
    un control de detención corre en horas y no puede esperar el FAQ.
  - Guardias deterministas intactas (0 LLM cuando corresponde).
"""
import logging

from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState
from core.llm import with_structured_output
from graph.utils import _recent_messages

from .contracts import IntakeDecision
from .nodes import (
    QUESTIONS, RESPUESTAS_HUMANO, RESPUESTAS_REINTENTAR, _norm, _pendientes,
)

logger = logging.getLogger(__name__)

_MAX_TOKENS_HISTORIA = 200

_PROMPT_TURNO = ChatPromptTemplate.from_messages([
    ("system",
     "Eres el planificador del registro de clientes de Manzzo y Cía (estudio "
     "jurídico chileno, especialista en derecho de familia; también laboral, "
     "civil y penal). El cliente está completando una ficha conversada por "
     "WhatsApp. Debes decidir QUÉ ES el último mensaje y EXTRAER los datos "
     "explícitos que traiga.\n\n"
     "Pregunta activa de la ficha: '{pregunta_activa}'\n"
     "Campos ya registrados: {registrados}\n\n"
     "Reglas de clasificación del turno:\n"
     "- Si el mensaje responde o aporta datos (aunque sea breve: un número de "
     "opción, un sí, una descripción) → respuesta_formulario y extrae TODO "
     "dato explícito (puede traer varios campos a la vez).\n"
     "- Botones y listas: mensajes como 'Sí, autorizo', 'No autorizo', "
     "'Lo intento de nuevo' o el título de una opción de una lista → "
     "respuesta_formulario (extrae lo que aplique; 'Lo intento de nuevo' "
     "no es un dato).\n"
     "- Si hace una pregunta o comenta algo ajeno a la ficha SIN cancelarla "
     "(precios, proceso, '¿y mi hija?', hipótesis como '¿me pueden detener "
     "si no pago?') → duda_o_consulta.\n"
     "- Si pide pausa o un momento ('pausa', 'un segundo', 'espera', "
     "'más tarde', 'ahora no', 'continúo después') → pausar.\n"
     "- Si corrige un dato ya entregado ('no, mi correo es otro…', 'me "
     "equivoqué en el nombre') → respuesta_formulario y extrae el valor "
     "corregido en el campo correspondiente.\n"
     "- Si pide hablar con una persona/ejecutiva ('hablar con humano'), o es "
     "la segunda vez que lo pide, o muestra hartazgo → insiste_humano.\n"
     "- Si no quiere seguir con el registro → abandonar.\n\n"
     "Reglas de extracción:\n"
     "- NUNCA inventes ni infieras datos; null en lo ausente.\n"
     "- consentimiento_datos NO existe en el esquema: esa respuesta se valida "
     "como sí/no con el mensaje crudo, nunca por inferencia.\n"
     "- nombre: solo nombres propios reales; una frase tipo 'quiero hablar "
     "con un humano' NO es un nombre (→ null).\n"
     "- email: normaliza dictados por voz ('arroba'→@, 'punto'→., "
     "'guión bajo'→_).\n"
     "- Un número solo respondiendo a una pregunta con opciones numeradas → "
     "mapea a la opción correspondiente de esa pregunta.\n"
     "- area_caso: extráela SIEMPRE que el mensaje revele el área, aunque la "
     "pregunta activa sea otra. Pensión/cuota/demanda de alimentos → "
     "pension_alimentos; no me deja ver a mis hijos / visitas → "
     "regimen_visitas; divorcio, compensación económica → divorcio; embargo, "
     "licencia retenida, registro de deudores, prohibición de salir del país "
     "→ medidas_apremio; detención, fiscalía, carabineros, PDI, querella, "
     "formalización, delito → penal; despido, empleador, contrato de trabajo, "
     "finiquito → laboral_civil.\n"
     "- detenido: True SOLO ante una detención VIGENTE declarada como hecho "
     "('me detuvieron anoche', 'mi hijo está detenido', 'el control de "
     "detención es mañana'). False si lo niega expresamente. Una pregunta "
     "hipotética sobre detención NO es True (→ null y tipo duda_o_consulta).\n"
     "- rol_penal: solo si el caso es penal. 'me acusan/me querellaron' → "
     "imputado; 'a mí me pasó / me robaron / me agredieron' → victima; "
     "'detuvieron a mi hermano' → familiar.\n"
     "- hijos_menores: cantidad y edades tal como las declara ('2, de 5 y 8 "
     "años'); si dice que no hay hijos menores, escribe 'no'.\n"
     "- medidas_apremio: las medidas vigentes mencionadas, o 'ninguna'.\n"
     "- audiencia_proxima: fecha o descripción, sin reinterpretar ('el "
     "martes', 'aún no me citan')."),
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
        rn = _norm(state.get("query") or "")
        if rn in RESPUESTAS_HUMANO:
            return {"intake_decision": {"accion": "derivar_parcial"},
                    "intake_stage": "preguntando"}
        if rn in RESPUESTAS_REINTENTAR:
            return {"intake_decision": {"accion": "procesar_respuesta"},
                    "intake_stage": "preguntando"}

    # --- turno activo: 1 LLM (decisión + extracción) ---
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

    if accion in ("derivar_parcial", "abandonar_ficha") and dec.confianza < 0.55:
        accion = "procesar_respuesta"

    # v17 — Override de urgencia: una detención VIGENTE declarada interrumpe
    # cualquier otro plan de este turno (incluida una duda lateral: el control
    # de detención corre en horas y no espera el FAQ). El nodo derivar_urgente
    # vuelca la extracción completa antes de salir, así que nada se pierde.
    # Si el LLM falló arriba, procesar_respuesta aplica _URGENTE_RE como
    # refuerzo determinista sobre el mensaje crudo.
    if dec.detenido is True and accion in ("procesar_respuesta",
                                           "pausar_para_faq",
                                           "pausar_ficha"):
        logger.info("[intake] detención vigente detectada → derivar_urgente")
        accion = "derivar_urgente"

    return {
        "intake_decision": {"accion": accion, **dec.model_dump()},
        "intake_stage": "preguntando",
    }
