"""graph/booking/planner.py — Comprensión del turno de agendamiento.

v4.3 — Match exacto por título de botón:
  - Si el cliente pincha "Lun 06/10 09:00", el planner lo mapea
    directamente al slot correspondiente.
  - Lenguaje natural sigue siendo el camino principal; esto es un
    redoble defensivo para canales interactivos.

v4.2 — Lenguaje natural primero:
  - "a las 16 hrs este viernes", "el lunes a las 17:30", "la primera opción",
    "la última" → match_slot los interpreta.
  - "1", "2", "opción 3" → fallback numérico.

v4 — FIX primer ingreso:
  - stage=None siempre devuelve acción "iniciar_wizard".
"""

import logging
import re
import unicodedata
from datetime import datetime

from core.contracts import AgentState
from core.llm import LLM
from graph.nodes import _ahora_iso, _parse_dt, _parece_abort
from graph.slot_match import extraer_dias_offset, match_slot
from tools.google_calendar import TZ as GCAL_TZ

from .contracts import BookingDecision

logger = logging.getLogger(__name__)

BOOKING_LLM = LLM.with_structured_output(BookingDecision, method="function_calling")

_DIASEM = ("lunes", "martes", "miércoles", "jueves",
           "viernes", "sábado", "domingo")

_OTRO_DIA = ("otro dia", "otra fecha", "ninguna", "no me acomoda",
             "no me sirve", "mas adelante", "la proxima semana", "la otra semana")

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


# ---------------------------------------------------------------------------
# Helper duplicado de nodes.py: evita import circular
# ---------------------------------------------------------------------------
def _slot_button_label(s) -> str:
    """Etiqueta corta para botones/lista de WhatsApp (máx. ~16 chars)."""
    dias_abr = {"lunes": "Lun", "martes": "Mar", "miercoles": "Mie",
                "jueves": "Jue", "viernes": "Vie",
                "sabado": "Sab", "domingo": "Dom"}
    dia = dias_abr[_DIASEM[s.weekday()]]
    return f"{dia} {s.strftime('%d/%m')} {s.strftime('%H:%M')}"


_ORDINALES = {
    "primera": 1, "primero": 1, "primer": 1, "1ra": 1, "1ro": 1,
    "segunda": 2, "segundo": 2, "2do": 2, "2da": 2,
    "tercera": 3, "tercero": 3, "3ro": 3, "3ra": 3,
    "cuarta": 4, "cuarto": 4, "4to": 4, "4ta": 4,
    "quinta": 5, "quinto": 5, "5to": 5, "5ta": 5,
    "sexta": 6, "sexto": 6, "6to": 6, "6ta": 6,
    "septima": 7, "septimo": 7, "7mo": 7, "7ma": 7,
    "octava": 8, "octavo": 8, "8vo": 8, "8va": 8,
    "novena": 9, "noveno": 9, "9no": 9, "9na": 9,
    "decima": 10, "decimo": 10, "10mo": 10, "10ma": 10,
}


def _extraer_ordinal(texto: str) -> int | None:
    """Interpreta 'la primera', 'opción 3', 'elijo la 2', etc."""
    t = _norm(texto)
    m = re.search(r"(?:opcion|numero|nro|eleccion|elijo|escoja|seleccione|escoger|elegir)\s*(\d+)", t)
    if m:
        return int(m.group(1))
    for palabra, num in _ORDINALES.items():
        if palabra in t:
            return num
    return None


# ---------------------------------------------------------------------------
# Fast-paths deterministas
# ---------------------------------------------------------------------------
def _fast_path(raw: str, stage: str | None, slots: list) -> BookingDecision | None:
    t = _norm(raw)
    if _parece_abort(raw):
        return BookingDecision(accion="abortar")

    # --- captura de modalidad (sin cambios) ---
    if stage == "captura_modalidad":
        if re.search(r"\b1\b", t) or _RE_ONLINE.search(t):
            return BookingDecision(accion="entregar_modalidad", modalidad="online")
        if re.search(r"\b2\b", t) or _RE_PRESENCIAL.search(t):
            return BookingDecision(accion="entregar_modalidad", modalidad="presencial")

    if stage != "propuesta" or not slots:
        return None

    # ------------------------------------------------------------------
    # 1) Lenguaje natural PRIMERO: día + hora / hora única / ordinal
    # ------------------------------------------------------------------
    m = match_slot(raw, slots, ref=datetime.now(GCAL_TZ).date())
    if m is not None:
        if isinstance(m, int):
            logger.info("[booking] fast-path match_slot único: slot idx %d", m)
            return BookingDecision(accion="elegir_horario", eleccion=m + 1)
        if isinstance(m, list) and len(m) == 1:
            logger.info("[booking] fast-path match_slot único (lista): slot idx %d", m[0])
            return BookingDecision(accion="elegir_horario", eleccion=m[0] + 1)
        if isinstance(m, list) and len(m) > 1:
            logger.info("[booking] fast-path match_slot ambiguo: %s", m)
            return BookingDecision(accion="elegir_horario")

    # ------------------------------------------------------------------
    # 2) Día/fecha explícito que NO calzó exactamente → replanificar,
    #    NUNCA tratar como duda lateral ("el lunes", "mañana", "04/09"...).
    # ------------------------------------------------------------------
    off = extraer_dias_offset(raw, datetime.now(GCAL_TZ).date())
    if off is not None:
        logger.info("[booking] fast-path otro día: offset %d", off)
        return BookingDecision(accion="pedir_otra_fecha", dias_offset=off)

    # ------------------------------------------------------------------
    # 3) Franja horaria (cuando no pidió día específico)
    # ------------------------------------------------------------------
    if _RE_MANANA.search(t):
        return BookingDecision(accion="filtrar_franja", franja="manana")
    if _RE_TARDE.search(t):
        return BookingDecision(accion="filtrar_franja", franja="tarde")

    # ------------------------------------------------------------------
    # 4) Ordinales y números simples
    # ------------------------------------------------------------------
    ordinal = _extraer_ordinal(raw)
    if ordinal is not None and 1 <= ordinal <= len(slots):
        logger.info("[booking] fast-path ordinal: opción %d", ordinal)
        return BookingDecision(accion="elegir_horario", eleccion=ordinal)

    raw_stripped = raw.strip()
    if raw_stripped.isdigit():
        opcion = int(raw_stripped)
        if 1 <= opcion <= len(slots):
            logger.info("[booking] fallback numérico: opción %d", opcion)
            return BookingDecision(accion="elegir_horario", eleccion=opcion)
        logger.info("[booking] número fuera de rango: %d (slots=%d)",
                    opcion, len(slots))

    # ------------------------------------------------------------------
    # 5) Match exacto contra título de botón enviado por el adaptador
    # ------------------------------------------------------------------
    titulo_cliente = _norm(raw)
    for i, s in enumerate(slots, 1):
        if _norm(_slot_button_label(s)) == titulo_cliente:
            logger.info("[booking] match exacto por título de botón: %d", i)
            return BookingDecision(accion="elegir_horario", eleccion=i)

    return None


# ---------------------------------------------------------------------------
# Decisión LLM acotada
# ---------------------------------------------------------------------------
def _decidir_llm(state: AgentState, slots: list) -> BookingDecision | None:
    stage = state.get("booking_stage") or "inicio"
    hoy = datetime.now(GCAL_TZ).date()

    if stage in ("captura_nombre", "captura_email", "captura_modalidad"):
        prompt = f"""Eres el asistente de agendamiento de Manzzo y Cía.
El usuario está respondiendo la pregunta activa: {stage}.
Datos ya capturados: nombre={state.get('lead_nombre')} | email={state.get('lead_email')} | modalidad={state.get('lead_modalidad')}

Mensaje del cliente: {state['query']}

Extrae SOLO el dato correspondiente a la pregunta activa:
- captura_nombre → campo 'nombre'
- captura_email → campo 'email'
- captura_modalidad → campo 'modalidad' (online/presencial)

Devuelve la acción apropiada (entregar_nombre, entregar_email o entregar_modalidad) y el valor extraído. No inventes nada."""  # noqa: E501
    else:
        datos = (f"nombre={state.get('lead_nombre')} | email={state.get('lead_email')} "
                 f"| modalidad={state.get('lead_modalidad')} | "
                 f"franja={state.get('booking_franja')}")
        opciones = ("\n".join(f"  {i}) {_slot_display(s)}"
                              for i, s in enumerate(slots, 1))
                    or "(sin opciones vigentes)")
        señal = state.get("booking_signal") or "ninguna"

        prompt = f"""Decides la ACCIÓN del turno en el agendamiento de una asesoría legal.
Hoy es {_DIASEM[hoy.weekday()]} {hoy.strftime('%d/%m/%Y')}.
Estado: {stage}
Datos capturados: {datos}
Opciones vigentes:
{opciones}
Señal anterior: {señal}

Reglas:
- "la primera", "opción 3" → elegir_horario con elección 1-based.
- "por la mañana"/"en la tarde" → filtrar_franja; "mañana" A SOLAS es el día siguiente.
- Día de semana o fecha específica no en opciones → pedir_otra_fecha.
- elegir_horario SOLO si se refiere a una opción vigente.
- consultar = duda lateral SIN avanzar el flujo.
- Devuelve SOLO lo explícito en el mensaje.

Mensaje del cliente: {state['query']}"""

    try:
        d = BOOKING_LLM.with_retry(stop_after_attempt=2).invoke(prompt)
        return BookingDecision(**d) if isinstance(d, dict) else d
    except Exception as e:
        logger.warning("[booking] decisión LLM falló: %s", e)
        return None


def _fallback_decision(stage: str | None) -> BookingDecision:
    if stage == "propuesta":
        return BookingDecision(accion="elegir_horario", confianza=0.3)
    if stage == "captura_nombre":
        return BookingDecision(accion="entregar_nombre", confianza=0.3)
    if stage == "captura_email":
        return BookingDecision(accion="entregar_email", confianza=0.3)
    if stage == "captura_modalidad":
        return BookingDecision(accion="entregar_modalidad", confianza=0.3)
    return BookingDecision(accion="entregar_nombre", confianza=0.3)


# ---------------------------------------------------------------------------
# NODO
# ---------------------------------------------------------------------------
def booking_planner(state: AgentState) -> AgentState:
    raw = (state.get("query") or "").strip()
    stage = state.get("booking_stage")
    slots = _slots(state)

    init = {}
    if not stage:
        init = {"agenda_started_en": _ahora_iso(), "agenda_ventana_desde": 0,
                "booking_attempts": 0, "booking_franja": None,
                "booking_match": None, "booking_match_candidatos": []}

    # FIX v4: primer ingreso al booking → siempre iniciar wizard de captura
    if stage is None:
        logger.info("[booking] primer ingreso → iniciar wizard de captura")
        return {**init,
                "booking_decision": BookingDecision(accion="iniciar_wizard").model_dump(),
                "booking_stage": "captura_nombre",
                "booking_match": None,
                "booking_match_candidatos": [],
                "booking_signal": None}

    # Guard: slots vencidos
    if stage == "propuesta" and slots and min(slots) < datetime.now(GCAL_TZ):
        logger.info("[booking] slots vencidos → regenerar propuesta")
        return {**init, "booking_signal": None,
                "booking_decision": BookingDecision(
                    accion="pedir_otra_fecha", dias_offset=0).model_dump()}

    decision = _fast_path(raw, stage, slots) \
        or _decidir_llm(state, slots) \
        or _fallback_decision(stage)

    # Match de slot: usa elección explícita primero, luego match_slot
    match, candidatos = None, []
    if decision.accion == "elegir_horario" and slots:
        if decision.eleccion and 1 <= decision.eleccion <= len(slots):
            match = decision.eleccion - 1
            logger.info("[booking] match por elección explícita: %d", match)
        else:
            m = match_slot(raw, slots, ref=datetime.now(GCAL_TZ).date())
            logger.info("[booking] match_slot('%s') → %s", raw, m)
            if isinstance(m, int):
                match = m
            elif isinstance(m, list):
                candidatos = m

    updates = {}
    if decision.nombre:
        updates["lead_nombre"] = decision.nombre.strip()
    if decision.email:
        updates["lead_email"] = decision.email.strip().lower()
    if decision.modalidad:
        updates["lead_modalidad"] = decision.modalidad
    if decision.franja:
        updates["booking_franja"] = decision.franja

    logger.info("[booking] stage=%s acción=%s match=%s", stage, decision.accion, match)

    return {**init, **updates,
            "booking_decision": decision.model_dump(),
            "booking_match": match,
            "booking_match_candidatos": candidatos,
            "booking_signal": None}
