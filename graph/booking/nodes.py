"""graph/booking/nodes.py — Acciones del sub-agente BOOKING.

v4 — Wizard de captura secuencial:
  - captura_nombre → captura_email → captura_modalidad → propuesta.
  - Un campo por turno, validado deterministamente; si falla, repite.
  - Al completar los 3 datos, consulta disponibilidad y propone slots en el
    MISMO turno (menor fricción).
  - Mantiene anti-loop de tanda repetida, slot_stale y ofrecer ejecutiva.
"""
import logging
import re
import unicodedata
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState, EMAIL_RE, HitlPayload, TipoHITL
from core.db_client import insert_booking, upsert_conversation
from core.llm import LLM
from graph.nodes import (
    _cfg, _guardar_ai, _parse_dt, _primer_nombre, _telefono_cliente,
)
from tools.google_calendar import (
    DIRECCION_OFICINA, LeadCalendar, TZ as GCAL_TZ,
    crear_evento_asesoria, horarios_disponibles,
)

from .contracts import (
    AGENDA_DIAS_POR_TANDA, AGENDA_HORAS_POR_DIA, AGENDA_SLOTS_VENTANA_DIAS,
    AGENDA_VENTANA_MAX_DIAS, FRANJA_HORAS, REQUIERE_APROBACION_AGENDAMIENTO,
)

logger = logging.getLogger(__name__)

_DIASEM = ("lunes", "martes", "miércoles", "jueves",
           "viernes", "sábado", "domingo")


# ---------------------------------------------------------------------------
# Validadores deterministas de captura
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = (s or "").strip()
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def _v_nombre(s: str) -> tuple[str | None, str | None]:
    s = (s or "").strip()
    palabras = [p for p in s.split() if p]
    if len(palabras) < 2 or any(len(p) < 2 for p in palabras):
        return None, "indique su nombre y apellido (al menos dos palabras)"
    if len(palabras) > 6 or any(len(p) > 25 for p in palabras):
        return None, "el nombre parece ser una frase; indique solo nombre y apellido"
    return " ".join(p.capitalize() for p in palabras), None


def _v_email(s: str) -> tuple[str | None, str | None]:
    e = (s or "").strip().lower().replace(" ", "")
    if not EMAIL_RE.fullmatch(e):
        return None, "el formato del correo no es válido (ejemplo: nombre@correo.cl)"
    return e, None


def _v_modalidad(s: str) -> tuple[str | None, str | None]:
    t = _norm(s).rstrip(".").rstrip(")")
    if t in ("1", "online", "videollamada", "virtual", "meet", "google meet"):
        return "online", None
    if t in ("2", "presencial", "oficina", "en persona", "en la oficina"):
        return "presencial", None
    return None, "indique 1) Online o 2) Presencial"


# ---------------------------------------------------------------------------
# Display de slots (sin cambios funcionales)
# ---------------------------------------------------------------------------
def _slots(state: AgentState) -> list:
    return [s for s in (_parse_dt(x) for x in state.get("slots_propuestos") or [])
            if s]


def _slot_display(s) -> str:
    return f"{_DIASEM[s.weekday()]} {s.strftime('%d/%m a las %H:%M')}"


def _agrupar_por_dia(slots: list) -> list[tuple]:
    grupos: dict = {}
    for s in slots:
        grupos.setdefault(s.date(), []).append(s)
    return sorted(grupos.items())


def _opciones(slots: list) -> str:
    lineas, i = [], 0
    for _, dia_slots in _agrupar_por_dia(slots):
        primero = dia_slots[0]
        lineas.append(f"*{_DIASEM[primero.weekday()].capitalize()} "
                      f"{primero.strftime('%d/%m')}*")
        for s in dia_slots:
            i += 1
            lineas.append(f"  {i}) {s.strftime('%H:%M')}")
    return "\n".join(lineas)


def _opciones_indexadas(slots: list, idxs: list[int]) -> str:
    return "\n".join(f"  {i + 1}) {_slot_display(slots[i])}" for i in idxs)


# ---------------------------------------------------------------------------
# Disponibilidad (sin cambios funcionales)
# ---------------------------------------------------------------------------
def _slots_dia(dia, franja: str | None) -> list:
    libres = horarios_disponibles(dia)
    if franja:
        a, b = FRANJA_HORAS[franja]
        libres = [s for s in libres if a <= s.hour < b]
    return libres


def _representativos(libres: list, k: int = AGENDA_HORAS_POR_DIA) -> list:
    if len(libres) <= k:
        return libres
    idxs = sorted({round(i * (len(libres) - 1) / (k - 1)) for i in range(k)})
    return [libres[i] for i in idxs]


def consultar_disponibilidad(state: AgentState) -> AgentState:
    d = state.get("booking_decision") or {}
    accion, off = d.get("accion"), d.get("dias_offset")

    desde = state.get("agenda_ventana_desde") or 0
    if accion == "pedir_otra_fecha":
        desde = off if off is not None else desde + AGENDA_SLOTS_VENTANA_DIAS
    franja = d.get("franja") or state.get("booking_franja")

    if desde > AGENDA_VENTANA_MAX_DIAS:
        return {"booking_signal": "ventana_agotada", "agenda_dia_sin_cupos": None}

    hoy = datetime.now(GCAL_TZ).date()
    dia_pedido = (hoy + timedelta(days=off)
                  if accion == "pedir_otra_fecha" and off is not None else None)

    elegidos, dias_ok, libres_pedido = [], 0, None
    for delta in range(desde, AGENDA_VENTANA_MAX_DIAS + 1):
        if dias_ok >= AGENDA_DIAS_POR_TANDA:
            break
        f = hoy + timedelta(days=delta)
        libres = _slots_dia(f, franja)
        if dia_pedido is not None and f == dia_pedido:
            libres_pedido = libres
        if libres:
            elegidos.extend(_representativos(libres))
            dias_ok += 1

    if not elegidos:
        return {"booking_signal": "ventana_agotada", "booking_franja": franja,
                "agenda_dia_sin_cupos": None}

    nuevos = [s.isoformat() for s in elegidos]
    if nuevos == (state.get("slots_propuestos") or []):
        logger.info("[booking] tanda idéntica re-propuesta → duda lateral")
        return {"booking_signal": "tanda_repetida", "agenda_dia_sin_cupos": None}

    sin_cupos = None
    if dia_pedido is not None and libres_pedido == []:
        sin_cupos = f"{_DIASEM[dia_pedido.weekday()]} {dia_pedido.strftime('%d/%m')}"

    return {"slots_propuestos": nuevos,
            "agenda_ventana_desde": desde,
            "booking_franja": franja,
            "agenda_dia_sin_cupos": sin_cupos}


def proponer_slots(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    slots = _slots(state)
    señal = state.get("booking_signal")

    if señal == "slot_stale":
        response = atn["agenda_slot_ocupado"].format(opciones=_opciones(slots))
    else:
        response = atn["agenda_proponer_slots"].format(
            nombre=_primer_nombre(state.get("lead_nombre")),
            opciones=_opciones(slots))
        sin_cupos = state.get("agenda_dia_sin_cupos")
        if sin_cupos:
            response = (f"Para *{sin_cupos}* no me quedan horas, "
                        f"pero tengo estas alternativas:\n\n{response}")
    _guardar_ai(state, response)
    return {"response": response,
            "booking_stage": "propuesta",
            "booking_signal": None,
            "agenda_dia_sin_cupos": None,
            "booking_match": None,
            "booking_match_candidatos": [],
            "booking_attempts": 0,
            "messages": [AIMessage(content=response)]}


# ---------------------------------------------------------------------------
# Wizard de captura: nodos de pregunta
# ---------------------------------------------------------------------------
def pedir_nombre(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_pedir_nombre"]
    _guardar_ai(state, response)
    return {"response": response,
            "booking_stage": "captura_nombre",
            "messages": [AIMessage(content=response)]}


def pedir_email(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_pedir_email"]
    _guardar_ai(state, response)
    return {"response": response,
            "booking_stage": "captura_email",
            "messages": [AIMessage(content=response)]}


def pedir_modalidad(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_pedir_modalidad"]
    _guardar_ai(state, response)
    return {"response": response,
            "booking_stage": "captura_modalidad",
            "messages": [AIMessage(content=response)]}


def procesar_captura(state: AgentState) -> AgentState:
    """Nodo único de avance del wizard. Valida el campo de la etapa actual,
    avanza de etapa, y si ya completó datos consulta disponibilidad + propone
    slots en el mismo turno."""
    atn = _cfg()["atencion"]
    stage = state.get("booking_stage")
    decision = state.get("booking_decision") or {}
    raw = (state.get("query") or "").strip()

    # Inicialización del ledger al primer ingreso
    init = {}
    if not state.get("agenda_started_en"):
        init = {"agenda_started_en": datetime.now(GCAL_TZ).isoformat(),
                "agenda_ventana_desde": 0,
                "booking_attempts": 0,
                "booking_franja": None,
                "booking_match": None,
                "booking_match_candidatos": [],
                "slots_propuestos": []}

    updates = {**init}

    if stage == "captura_nombre":
        nombre = decision.get("nombre") or raw
        val, err = _v_nombre(nombre)
        if err:
            response = f"No fue posible registrar su nombre: {err}. {atn['agenda_pedir_nombre']}"
            _guardar_ai(state, response)
            return {**updates, "response": response,
                    "messages": [AIMessage(content=response)]}
        updates["lead_nombre"] = val
        response = atn["agenda_pedir_email"]
        _guardar_ai(state, response)
        return {**updates,
                "lead_nombre": val,
                "booking_stage": "captura_email",
                "response": response,
                "messages": [AIMessage(content=response)]}

    if stage == "captura_email":
        email = decision.get("email") or raw
        val, err = _v_email(email)
        if err:
            response = f"No fue posible registrar su correo: {err}. {atn['agenda_pedir_email']}"
            _guardar_ai(state, response)
            return {**updates, "response": response,
                    "messages": [AIMessage(content=response)]}
        updates["lead_email"] = val
        response = atn["agenda_pedir_modalidad"]
        _guardar_ai(state, response)
        return {**updates,
                "lead_email": val,
                "booking_stage": "captura_modalidad",
                "response": response,
                "messages": [AIMessage(content=response)]}

    if stage == "captura_modalidad":
        modalidad = decision.get("modalidad") or raw
        val, err = _v_modalidad(modalidad)
        if err:
            response = f"No fue posible registrar la modalidad: {err}. {atn['agenda_pedir_modalidad']}"
            _guardar_ai(state, response)
            return {**updates, "response": response,
                    "messages": [AIMessage(content=response)]}
        updates["lead_modalidad"] = val

        # Datos completos: consultar + proponer en el mismo turno
        estado_intermedio = {**state, **updates, "booking_stage": "propuesta",
                             "booking_decision": {"accion": "entregar_datos"}}
        disponibilidad = consultar_disponibilidad(estado_intermedio)
        estado_propuesta = {**estado_intermedio, **disponibilidad}
        return proponer_slots(estado_propuesta)

    # Fallback defensivo: volver a pedir nombre
    return pedir_nombre({**state, **updates})


def reintentar_eleccion(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_reintento_slots"].format(
        opciones=_opciones(_slots(state)))
    _guardar_ai(state, response)
    return {"response": response,
            "booking_attempts": (state.get("booking_attempts") or 0) + 1,
            "messages": [AIMessage(content=response)]}


def aclarar_eleccion(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    slots = _slots(state)
    idxs = state.get("booking_match_candidatos") or []
    response = atn["agenda_eleccion_ambigua"].format(
        opciones=_opciones_indexadas(slots, idxs))
    _guardar_ai(state, response)
    return {"response": response,
            "booking_attempts": (state.get("booking_attempts") or 0) + 1,
            "messages": [AIMessage(content=response)]}


def responder_lateral(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", atn["agenda_lateral_system"]),
            ("human", "{query}"),
        ])
        respuesta = (prompt | LLM).invoke({
            "query": state["query"],
            "opciones": _opciones(_slots(state)) or "(aún sin horarios)",
        }).content
    except Exception as e:
        logger.exception("[booking] respuesta lateral falló: %s", e)
        respuesta = ("La asesoría dura 30 minutos y puede ser online por Meet "
                     "o presencial en nuestra oficina.")
    _guardar_ai(state, respuesta)
    return {"response": respuesta, "messages": [AIMessage(content=response)]}


# ---------------------------------------------------------------------------
# Confirmación y creación (sin cambios funcionales)
# ---------------------------------------------------------------------------
def confirmar_y_crear(state: AgentState) -> AgentState:
    """Revalida el slot, crea el evento y persiste. Si faltan datos de
    contacto (estado residual), redirige al wizard de captura sin romper."""
    atn = _cfg()["atencion"]

    # GUARD: no confirmar sin datos completos
    if not state.get("lead_nombre"):
        return pedir_nombre(state)
    if not state.get("lead_email"):
        return pedir_email(state)
    if not state.get("lead_modalidad"):
        return pedir_modalidad(state)

    slots = _slots(state)
    if not slots or state.get("booking_match") is None:
        # Defensivo: no hay slot válido → volver a proponer
        response = atn["agenda_reintento_slots"].format(opciones="(sin horarios)")
        _guardar_ai(state, response)
        return {"response": response, "booking_stage": "propuesta",
                "messages": [AIMessage(content=response)]}

    elegido = slots[state["booking_match"]]
    hoy = datetime.now(GCAL_TZ).date()
    tid = state.get("thread_id", "")

    # Cortafuegos de carrera: el slot existía cuando se PROPUSO; ¿sigue libre?
    if elegido not in horarios_disponibles(elegido.date()):
        logger.warning("[booking] slot %s se ocupó post-propuesta → replan", elegido)
        return {"booking_signal": "slot_stale",
                "booking_match": None,
                "booking_franja": "manana" if elegido.hour < 13 else "tarde",
                "agenda_ventana_desde": max(0, (elegido.date() - hoy).days),
                "agenda_dia_sin_cupos": None}

    if REQUIERE_APROBACION_AGENDAMIENTO:
        from langgraph.types import interrupt
        decision = interrupt(HitlPayload(
            tipo=TipoHITL.APROBACION_AGENDAMIENTO.value,
            thread_id=tid,
            query=state["query"],
            detalle=f"El cliente eligió {_slot_display(elegido)}. ¿Aprobar creación?",
            lead={"nombre": state["lead_nombre"], "email": state["lead_email"],
                  "modalidad": state.get("lead_modalidad")},
            categoria=state.get("category", "otro"),
            urgencia=state.get("urgency", "media"),
        ))
        if not decision.get("aprobado"):
            response = ("Gracias por su interés. Una ejecutiva se contactará para "
                        "coordinar la cita. " + (decision.get("nota", ""))).strip()
            _guardar_ai(state, response)
            return {"response": response, "booking_stage": None,
                    "slots_propuestos": [], "messages": [AIMessage(content=response)]}

    evento = crear_evento_asesoria(
        LeadCalendar(
            nombre=state["lead_nombre"], email=state["lead_email"],
            telefono=_telefono_cliente(state),
            modalidad=state.get("lead_modalidad", "online"),
            categoria=state.get("category"), motivo=state.get("clf_reason"),
        ),
        inicio=elegido,
    )
    if evento is None:
        response = ("Tuve un problema técnico confirmando la hora. Si lo prefiere, "
                    "una ejecutiva la agenda manualmente contigo. ¿Le parece?")
        _guardar_ai(state, response)
        return {"response": response, "booking_stage": None,
                "booking_signal": "creacion_fallida",
                "slots_propuestos": [], "messages": [AIMessage(content=response)]}

    modalidad_linea = (
        f"Videollamada por Google Meet: {evento.meet_link}"
        if evento.meet_link
        else f"Presencial en nuestra oficina: {DIRECCION_OFICINA}")
    response = atn["agenda_confirmada"].format(
        nombre=_primer_nombre(state["lead_nombre"]),
        fecha=evento.fecha, hora_inicio=evento.hora_inicio,
        hora_fin=evento.hora_fin, modalidad_linea=modalidad_linea,
        html_link=evento.html_link)

    insert_booking(thread_id=tid, booking=evento.model_dump())
    upsert_conversation(thread_id=tid, status="agendado")

    try:
        from integrations.sheetdb import actualizar_booking
        actualizar_booking({**state, "booking": evento.model_dump()})
    except Exception as e:
        logger.exception("[booking] SheetDB (fallback) falló: %s", e)

    _guardar_ai(state, response)
    return {"response": response,
            "booking": evento.model_dump(),
            "booking_stage": None,
            "slots_propuestos": [], "booking_match": None,
            "booking_attempts": 0, "booking_franja": None,
            "agenda_started_en": None, "agenda_ventana_desde": 0,
            "agenda_dia_sin_cupos": None,
            "messages": [AIMessage(content=response)]}



def ofrecer_ejecutiva(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_sin_horarios"]
    _guardar_ai(state, response)
    return {"response": response, "booking_stage": None,
            "slots_propuestos": [], "booking_match": None,
            "booking_attempts": 0, "booking_signal": None,
            "agenda_dia_sin_cupos": None,
            "messages": [AIMessage(content=response)]}


def cerrar_booking(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_abortado"]
    _guardar_ai(state, response)
    return {"response": response, "booking_stage": None,
            "slots_propuestos": [], "booking_match": None,
            "booking_attempts": 0, "booking_franja": None,
            "booking_signal": None,
            "agenda_dia_sin_cupos": None,
            "messages": [AIMessage(content=response)]}
