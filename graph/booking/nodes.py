"""graph/booking/nodes.py — Acciones del sub-agente BOOKING.

v3 — Anti-loop de tanda repetida + honestidad de día pedido:
  - consultar_disponibilidad: si la tanda nueva es IDÉNTICA a la vigente,
    el mensaje no era navegación sino una duda mal clasificada por el LLM
    → booking_signal="tanda_repetida" → responder_lateral (antes: el mismo
    mensaje se re-enviaba en loop sin que BOOKING_MAX_ATTEMPTS se enterara,
    porque proponer_slots resetea booking_attempts).
  - Si pedir_otra_fecha trae dias_offset explícito y ESE día no tiene
    cupos, agenda_dia_sin_cupos queda seteado y proponer_slots lo dice
    ("para el miércoles 09/09 no me quedan horas, pero sí estas:") en vez
    de saltar de día en silencio — el cliente lo leía como que lo ignoraron.

v2 — Hooks de persistencia PostgreSQL (messages, bookings, SheetDB fallback).
"""
import logging
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState, HitlPayload, TipoHITL
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
# Display: propuesta AGRUPADA POR DÍA con varias horas por día
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
    """Numeración GLOBAL sobre la lista plana → match_slot por número sigue
    siendo la fuente de verdad."""
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
    """Subset con numeración ORIGINAL (renumerar rompería match_slot)."""
    return "\n".join(f"  {i + 1}) {_slot_display(slots[i])}" for i in idxs)


# ---------------------------------------------------------------------------
# Disponibilidad: horas representativas por día + franja opcional
# ---------------------------------------------------------------------------
def _slots_dia(dia, franja: str | None) -> list:
    libres = horarios_disponibles(dia)
    if franja:
        a, b = FRANJA_HORAS[franja]
        libres = [s for s in libres if a <= s.hour < b]
    return libres


def _representativos(libres: list, k: int = AGENDA_HORAS_POR_DIA) -> list:
    """Espaciados uniformes sobre el día/franja."""
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
    # Día explícitamente pedido ("este miércoles", "el 18/11"): es el delta de
    # partida, así que el loop lo consulta SÍ o SÍ → sin llamada freebusy extra.
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

    if not elegidos:          # freebusy caído o agenda/franja llena
        return {"booking_signal": "ventana_agotada", "booking_franja": franja,
                "agenda_dia_sin_cupos": None}

    nuevos = [s.isoformat() for s in elegidos]
    if nuevos == (state.get("slots_propuestos") or []):
        # Tanda IDÉNTICA a la vigente: el LLM clasificó una duda como
        # navegación. Re-proponer lo mismo es un bucle invisible (proponer_slots
        # resetea attempts) → tratar como pregunta lateral con las opciones.
        logger.info("[booking] tanda idéntica re-propuesta → duda lateral")
        return {"booking_signal": "tanda_repetida", "agenda_dia_sin_cupos": None}

    sin_cupos = None
    if dia_pedido is not None and libres_pedido == []:
        sin_cupos = f"{_DIASEM[dia_pedido.weekday()]} {dia_pedido.strftime('%d/%m')}"
        logger.info("[booking] día pedido sin cupos: %s", sin_cupos)

    return {"slots_propuestos": nuevos,
            "agenda_ventana_desde": desde,
            "booking_franja": franja,
            "agenda_dia_sin_cupos": sin_cupos}


def proponer_slots(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    slots = _slots(state)
    señal = state.get("booking_signal")          # slot_stale = replan en caliente

    if señal == "slot_stale":
        response = atn["agenda_slot_ocupado"].format(opciones=_opciones(slots))
    else:
        response = atn["agenda_proponer_slots"].format(
            nombre=_primer_nombre(state.get("lead_nombre")),
            opciones=_opciones(slots))
        sin_cupos = state.get("agenda_dia_sin_cupos")
        if sin_cupos:
            response = (f"Para *{sin_cupos}* no me quedan horas 😕, "
                        f"pero tengo estas alternativas:\n\n{response}")
    _guardar_ai(state, response)
    return {"response": response,
            "booking_stage": "propuesta",
            "booking_signal": None,               # consumida
            "agenda_dia_sin_cupos": None,         # consumido
            "booking_match": None,
            "booking_match_candidatos": [],
            "booking_attempts": 0,
            "messages": [AIMessage(content=response)]}


# ----------------------------------------------------------------------
# Terminales de captura / elección
# ----------------------------------------------------------------------
def pedir_datos(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    faltantes = []
    if not state.get("lead_nombre"):
        faltantes.append("tu nombre completo")
    if not state.get("lead_email"):
        faltantes.append("tu correo electrónico (formato nombre@correo.cl)")
    if not state.get("lead_modalidad"):
        faltantes.append("si la prefieres online o presencial")

    response = (atn["agenda_pedir_datos"] if len(faltantes) == 3
                else f"¡Casi listo! Solo me falta: {' y '.join(faltantes)} 🙌")
    _guardar_ai(state, response)
    return {"response": response, "booking_stage": "captura",
            "booking_match": None, "booking_match_candidatos": [],
            "messages": [AIMessage(content=response)]}


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
    """Responde la duda y re-muestra la propuesta vigente. También aterriza
    aquí booking_signal="tanda_repetida" (el LLM leyó duda como navegación):
    en ese caso `query` es la duda original y el lateral la contesta con las
    opciones vigentes — en vez de re-proponer idéntico por tercera vez."""
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
    return {"response": respuesta, "messages": [AIMessage(content=respuesta)]}


# ----------------------------------------------------------------------
# Confirmación: revalidación anti-choque → crear → persistir
# ----------------------------------------------------------------------
def confirmar_y_crear(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    slots = _slots(state)
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
            response = ("Gracias por tu interés. Una ejecutiva te contactará para "
                        "coordinar la cita. " + decision.get("nota", "")).strip()
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
        response = ("Tuve un problema técnico confirmando la hora 😕. Si quieres, "
                    "una ejecutiva la agenda manualmente contigo. ¿Te parece?")
        _guardar_ai(state, response)
        return {"response": response, "booking_stage": None,
                "booking_signal": "creacion_fallida",
                "slots_propuestos": [], "messages": [AIMessage(content=response)]}

    modalidad_linea = (
        f"💻 Videollamada por Google Meet: {evento.meet_link}"
        if evento.meet_link
        else f"📍 Presencial en nuestra oficina: {DIRECCION_OFICINA}")
    response = atn["agenda_confirmada"].format(
        nombre=_primer_nombre(state["lead_nombre"]),
        fecha=evento.fecha, hora_inicio=evento.hora_inicio,
        hora_fin=evento.hora_fin, modalidad_linea=modalidad_linea,
        html_link=evento.html_link)

    # Persistencia PRIMARIA en Postgres (fuente de verdad operativa)
    insert_booking(thread_id=tid, booking=evento.model_dump())
    upsert_conversation(thread_id=tid, status="agendado")

    # Fallback secundario: Google Sheets vía SheetDB (best-effort)
    try:
        from integrations.sheetdb import actualizar_booking
        actualizar_booking({**state, "booking": evento.model_dump()})
    except Exception as e:
        logger.exception("[booking] SheetDB (fallback) falló: %s", e)

    _guardar_ai(state, response)
    return {"response": response,
            "booking": evento.model_dump(),
            "booking_stage": None,           # sub-flujo cerrado
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
