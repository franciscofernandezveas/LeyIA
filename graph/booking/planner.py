"""graph/booking/planner.py — Comprensión del turno de agendamiento.

Ídem planner_node del núcleo BI, adaptado: el "catálogo" es freebusy, los
fast-paths deterministas (≡ _is_demand_forecast_question) corren ANTES del
LLM, y la señal de fallo se consume y marca (≡ replan_errors).

v2 — Fast-paths reordenados + navegación por día específico:
  - Franja ANTES de match_slot: "por la mañana" es FILTRO, no la fecha
    "mañana" (fix del hijack que quemaba attempts y botaba a ejecutiva).
  - extraer_dias_offset: "este miércoles", "el jueves", "18/09" →
    pedir_otra_fecha(dias_offset=N) determinista, sin LLM. Va DESPUÉS de
    match_slot (no secuestra elecciones vigentes) y ANTES de _OTRO_DIA.
  - Prompt del LLM con ancla de fecha ("hoy es ...") como respaldo para
    casos que el parser determinista no cubre.
"""
import logging
import re
import unicodedata
from datetime import datetime

from core.contracts import EMAIL_RE, AgentState
from core.llm import LLM
from graph.nodes import _ahora_iso, _parse_dt, _parece_abort  # helpers vigentes
from graph.slot_match import extraer_dias_offset, match_slot
from tools.google_calendar import TZ as GCAL_TZ

from .contracts import BookingDecision

logger = logging.getLogger(__name__)

BOOKING_LLM = LLM.with_structured_output(BookingDecision, method="function_calling")

_DIASEM = ("lunes", "martes", "miércoles", "jueves",
           "viernes", "sábado", "domingo")

_OTRO_DIA = ("otro dia", "otra fecha", "ninguna", "no me acomoda",
             "no me sirve", "mas adelante", "la proxima semana", "la otra semana")

# Franja EXIGE marcador explícito: "mañana" a solas = día siguiente
# (lo resuelve match_slot como fecha), NO franja manana.
_RE_MANANA = re.compile(r"(por|en|de)\s+la\s+manana|temprano|a primera hora"
                        r"|antes de las?\s+1[23]")
_RE_TARDE = re.compile(r"(por|en|de)\s+la\s+tarde|tipo tarde|al salir del trabajo"
                       r"|despues de las?\s+(1[4-9]|2[01])")
_RE_ONLINE = re.compile(r"online|videollamada|virtual|meet|zoom")
_RE_PRESENCIAL = re.compile(r"presencial|oficina|en persona")


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _slots(state: AgentState) -> list:
    return [s for s in (_parse_dt(x) for x in state.get("slots_propuestos") or [])
            if s]


def _slot_display(s) -> str:
    return f"{_DIASEM[s.weekday()]} {s.strftime('%d/%m a las %H:%M')}"


# ----------------------------------------------------------------------
# Fast-paths deterministas (0 LLM). Orden importa.
# ----------------------------------------------------------------------
def _fast_path(raw: str, stage: str | None, slots: list) -> BookingDecision | None:
    t = _norm(raw)
    if _parece_abort(raw):
        return BookingDecision(accion="abortar")

    # 1) Franja ANTES de match_slot: "por la mañana" es FILTRO, no fecha.
    #    "mañana" a solas no calza estos regex (exigen por/en/de la) y sigue
    #    a match_slot/extraer_dias_offset como día. Fix del hijack.
    if _RE_MANANA.search(t):
        return BookingDecision(accion="filtrar_franja", franja="manana")
    if _RE_TARDE.search(t):
        return BookingDecision(accion="filtrar_franja", franja="tarde")

    # 2) Elección directa sobre la propuesta vigente (match robusto cableado)
    ref = datetime.now(GCAL_TZ).date()
    if stage == "propuesta" and slots:
        m = match_slot(raw, slots, ref=ref)
        if m is not None:
            return BookingDecision(accion="elegir_horario")

    # 3) Día ESPECÍFICO sin match vigente: "este miércoles", "el jueves",
    #    "para el 18/09". Determinista, 0 LLM. Después de match_slot para no
    #    secuestrar una elección válida; antes de _OTRO_DIA porque es más
    #    preciso que el avance genérico de a 3 días.
    off = extraer_dias_offset(raw, ref)
    if off is not None:
        return BookingDecision(accion="pedir_otra_fecha", dias_offset=off)

    # 4) Navegación genérica
    if any(p in t for p in _OTRO_DIA):
        return BookingDecision(accion="pedir_otra_fecha")

    # 5) Datos sueltos detectables sin LLM (captura o corrección)
    email = EMAIL_RE.search(t)
    modalidad = ("online" if _RE_ONLINE.search(t)
                 else "presencial" if _RE_PRESENCIAL.search(t) else None)
    if stage in (None, "captura") and (email or modalidad):
        return BookingDecision(accion="entregar_datos",
                               email=email.group(0) if email else None,
                               modalidad=modalidad)
    return None


# ----------------------------------------------------------------------
# Decisión LLM acotada (el horario NUNCA lo decide: eso es match_slot)
# ----------------------------------------------------------------------
def _decidir_llm(state: AgentState, slots: list) -> BookingDecision | None:
    stage = state.get("booking_stage") or "entrada (aún no se piden datos)"
    datos = (f"nombre={state.get('lead_nombre')} | email={state.get('lead_email')} "
             f"| modalidad={state.get('lead_modalidad')} | "
             f"franja={state.get('booking_franja')}")
    opciones = ("\n".join(f"  {i}) {_slot_display(s)}"
                          for i, s in enumerate(slots, 1))
                or "(sin opciones vigentes)")
    señal = state.get("booking_signal") or "ninguna"
    hoy = datetime.now(GCAL_TZ).date()

    prompt = f"""Decides la ACCIÓN del turno en el agendamiento de una asesoría legal.
Hoy es {_DIASEM[hoy.weekday()]} {hoy.strftime('%d/%m/%Y')}.
Estado del sub-flujo: {stage}
Datos capturados: {datos}
Opciones vigentes (ÚNICAS horas válidas; está prohibido inventar otras):
{opciones}
Señal del intento anterior: {señal}

Reglas:
- Elige UNA acción. Si el mensaje aporta datos de contacto → entregar_datos.
- "por la mañana"/"en la tarde" → filtrar_franja; "mañana" A SOLAS es el día
  siguiente → pedir_otra_fecha con dias_offset=1.
- Si pide un día de semana o fecha específica que NO está en las opciones →
  pedir_otra_fecha con dias_offset calculado DESDE HOY (0=hoy, 1=mañana).
- elegir_horario SOLO si se refiere a una opción vigente (número u hora/día).
- consultar = duda lateral (duración, dirección, modalidad, precio) SIN
  avanzar el flujo. Preguntar por disponibilidad de OTRO día NO es duda
  lateral: es pedir_otra_fecha.
- Devuelve en los campos SOLO lo explícito en el mensaje; null lo demás.

Mensaje del cliente: {state['query']}"""
    try:
        d = BOOKING_LLM.with_retry(stop_after_attempt=2).invoke(prompt)
        return BookingDecision(**d) if isinstance(d, dict) else d
    except Exception as e:
        logger.warning("[booking] decisión LLM falló: %s", e)
        return None


def _fallback_decision(raw: str, stage: str | None) -> BookingDecision:
    """≡ _fallback_extract_forecast_params: conservador, nunca avanza el flujo."""
    accion = "elegir_horario" if stage == "propuesta" else "entregar_datos"
    return BookingDecision(accion=accion, confianza=0.3)


# ----------------------------------------------------------------------
# NODO
# ----------------------------------------------------------------------
def booking_planner(state: AgentState) -> AgentState:
    raw = (state.get("query") or "").strip()
    stage = state.get("booking_stage")
    slots = _slots(state)

    # Señal anterior: consumir (≡ _planner_result limpiando replan_errors)
    señal = state.get("booking_signal")
    if señal:
        logger.info("[booking] señal consumida: %s", señal)

    # Inicialización del ledger al entrar al sub-flujo
    init = {}
    if not stage:
        init = {"agenda_started_en": _ahora_iso(), "agenda_ventana_desde": 0,
                "booking_attempts": 0, "booking_franja": None,
                "booking_match": None, "booking_match_candidatos": []}

    # Guard: propuesta vencida (hilo reanudado días después) → regenerar
    if stage == "propuesta" and slots and min(slots) < datetime.now(GCAL_TZ):
        logger.info("[booking] slots vencidos → regenerar propuesta")
        return {**init, "booking_signal": None,
                "booking_decision": BookingDecision(
                    accion="pedir_otra_fecha", dias_offset=0).model_dump()}

    decision = _fast_path(raw, stage, slots) \
        or _decidir_llm(state, slots) \
        or _fallback_decision(raw, stage)

    # Match de slot: verdad determinista, con respaldo acotado del LLM
    match, candidatos = None, []
    if decision.accion == "elegir_horario" and slots:
        m = match_slot(raw, slots, ref=datetime.now(GCAL_TZ).date())
        if isinstance(m, int):
            match = m
        elif isinstance(m, list):
            candidatos = m
        elif decision.eleccion and 1 <= decision.eleccion <= len(slots):
            match = decision.eleccion - 1

    # Cortafuegos de datos (≡ validación léxica post-LLM del planner BI)
    updates = {}
    if decision.nombre:
        updates["lead_nombre"] = decision.nombre.strip()
    if decision.email:
        e = decision.email.strip().lower()
        if EMAIL_RE.fullmatch(e):
            updates["lead_email"] = e
        else:
            updates["lead_email"] = None        # formato inválido → re-pregunta
    if decision.modalidad:
        updates["lead_modalidad"] = decision.modalidad
    if decision.franja:
        updates["booking_franja"] = decision.franja

    logger.info("[booking] stage=%s acción=%s match=%s conf=%.2f",
                stage, decision.accion, match, decision.confianza)

    return {**init, **updates,
            "booking_decision": decision.model_dump(),
            "booking_match": match,
            "booking_match_candidatos": candidatos,
            "booking_signal": None}
