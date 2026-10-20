"""graph/booking/nodes.py — Acciones del sub-agente BOOKING.

v8.5 — UX horarios: cuando hay interactive nativo (buttons/lista), el mensaje
       de texto solo muestra la introducción; la lista numerada ya no se
       duplica en el cuerpo. Los horarios se leen exclusivamente desde los
       botones/lista nativos de WhatsApp.
v8.4 — Cierres exhaustivos: todos los retornos finales (éxito, HITL
       rechazado, fallo técnico, sin horarios, abortado) usan RESET_BOOKING.
v8.3 — Fix aviso de privacidad repetido.
v8.2 — FIX import constantes de booking.
v8.1 — Limpieza: se elimina `_slot_desde_raw`.
v8 — Interactivos reales en WhatsApp.
"""

import logging
import unicodedata
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from core.contracts import AgentState, EMAIL_RE, HitlPayload, TipoHITL
from core.db_client import insert_booking, upsert_conversation, upsert_lead

from core.llm import LLM
from graph.utils import (
    RESET_BOOKING,
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
_DIAS_ABR = ("Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom")

# Botones nativos de modalidad.
_BOTONES_MODALIDAD = {"kind": "buttons",
                      "buttons": [{"id": "online", "title": "Online"},
                                  {"id": "presencial", "title": "Presencial"}]}

# Etiquetas humanas para el ack de precarga.
_PRECARGA_LABELS = {"lead_nombre": "su nombre", "lead_email": "su correo"}


# ---------------------------------------------------------------------------
# Payloads interactivos de horarios
# ---------------------------------------------------------------------------
def _slot_button_label(s) -> str:
    """Etiqueta corta para botones/lista de WhatsApp."""
    return f"{_DIAS_ABR[s.weekday()]} {s.strftime('%d/%m')} {s.strftime('%H:%M')}"


def _interactive_slots(slots: list, idxs: list[int] | None = None) -> dict | None:
    """Botones nativos (≤3 opciones) o lista nativa agrupada por día (≤10).

    La numeración de título e id es ABSOLUTA respecto de slots_propuestos:
    el tap en WhatsApp entrega "N) <slot>" y calza con el texto numerado
    (match_slot lo resuelve por fecha+hora exacta).
    """
    if not slots:
        return None
    if idxs is None:
        idxs = list(range(len(slots)))
    idxs = [i for i in idxs if 0 <= i < len(slots)][:10]
    if not idxs:
        return None

    options = [
        {"id": f"slot_{i + 1}",
         "title": f"{i + 1}) {_slot_button_label(slots[i])}",
         "description": f"Asesoría a las {slots[i].strftime('%H:%M')} hrs"}
        for i in idxs
    ]

    if len(options) <= 3:
        return {"kind": "buttons",
                "buttons": [{"id": o["id"], "title": o["title"]} for o in options]}

    # Lista nativa: una sección por día (Meta soporta secciones múltiples)
    por_dia: dict = {}
    for i, o in zip(idxs, options):
        por_dia.setdefault(slots[i].date(), []).append(o)

    sections = [
        {"title": f"{_DIASEM[d.weekday()].capitalize()} {d.strftime('%d/%m')}",
         "rows": rows}
        for d, rows in sorted(por_dia.items())
    ]
    return {"kind": "list", "cta": "Ver horarios",
            "title": "Próximos horarios", "sections": sections}


# ---------------------------------------------------------------------------
# Aviso de privacidad para agendamiento directo
# ---------------------------------------------------------------------------
def _aviso_privacidad(state: AgentState) -> str | None:
    """Devuelve el texto del aviso SOLO si corresponde emitirlo ahora:
    - el cliente NO dio consentimiento explícito en la ficha de intake
      (presencia de intake_respuestas ≠ autorización — Ley 21.719), y
    - el flag persistido booking_privacidad_dicha aún no está levantado.

    v8.7 — FIX #8 (auditoría): antes bastaba con que intake_respuestas
    existiera (`bool(state.get("intake_respuestas"))`) para omitir el aviso.
    Pero iniciar_ficha pre-carga 'telefono' y sin_consentimiento conserva
    intake_respuestas aunque el consent haya sido explícitamente False —
    la presencia de la ficha no es prueba de autorización. Ahora se chequea
    el campo `consentimiento_datos is True` en lugar de la mera presencia.

    El texto vive en prompts.yaml (atencion.agenda_inicio_aviso); el hardcode
    previo aquí se elimina."""
    if (state.get("intake_respuestas") or {}).get("consentimiento_datos") is True:
        return None
    if state.get("booking_privacidad_dicha"):
        return None
    return _cfg()["atencion"]["agenda_inicio_aviso"]



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
# Precarga desde la ficha de intake
# ---------------------------------------------------------------------------
def _precargar_desde_intake(state: AgentState) -> dict:
    r = state.get("intake_respuestas") or {}
    out: dict = {}
    if not state.get("lead_nombre") and r.get("nombre"):
        val, _err = _v_nombre(str(r["nombre"]))
        if val:
            out["lead_nombre"] = val
    if not state.get("lead_email") and r.get("email"):
        val, _err = _v_email(str(r["email"]))
        if val:
            out["lead_email"] = val
    return out


# ---------------------------------------------------------------------------
# Display de slots
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
# Disponibilidad
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


def _intro_proponer_slots(state: AgentState, señal: str | None,
                          sin_cupos: str | None) -> str:
    """Texto introductorio cuando los horarios se muestran por interactive.
    No incluye la lista numerada — esa vive en los botones/lista nativos."""
    atn = _cfg()["atencion"]
    nombre = _primer_nombre(state.get("lead_nombre"))

    if señal == "slot_stale":
        base = atn["agenda_slot_ocupado"].format(nombre=nombre, opciones="")
    else:
        base = atn["agenda_proponer_slots"].format(nombre=nombre, opciones="")
        if sin_cupos:
            base = (f"Para *{sin_cupos}* no me quedan horas, "
                    f"pero tengo estas alternativas:\n\n{base}")

    return base.rstrip()


def proponer_slots(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    slots = _slots(state)
    señal = state.get("booking_signal")
    interactive = _interactive_slots(slots)

    if interactive is None:
        # Fallback textual (p.ej. CLI/web sin botones nativos)
        response = atn["agenda_proponer_slots"].format(
            nombre=_primer_nombre(state.get("lead_nombre")),
            opciones=_opciones(slots))
    else:
        response = _intro_proponer_slots(state, señal, state.get("agenda_dia_sin_cupos"))

    _guardar_ai(state, response)
    return {"response": response,
            "response_bubbles": [response],
            "response_interactive": interactive,
            "booking_stage": "propuesta",
            "booking_signal": None,
            "agenda_dia_sin_cupos": None,
            "booking_match": None,
            "booking_match_candidatos": [],
            "booking_attempts": 0,
            "messages": [AIMessage(content=response)]}


# ---------------------------------------------------------------------------
# Wizard de captura: cadena con skip por precarga
# ---------------------------------------------------------------------------
def pedir_nombre(state: AgentState, _pre: dict | None = None) -> AgentState:
    """Paso 1 del wizard y ÚNICO punto de emisión del aviso de privacidad.

    El flag booking_privacidad_dicha es un canal de AgentState (v6.4)
    persistido por el checkpointer → el aviso sale a lo más una vez por hilo.
    Regla de escritura: se levanta SOLO cuando se emite; aquí jamás se limpia,
    y los resets del padre (abort/TTL) tampoco lo incluyen (política lead_*).
    """
    pre = {**(_pre or {}), **_precargar_desde_intake(state)}

    if pre.get("lead_nombre") or state.get("lead_nombre"):
        return {**pre, **pedir_email({**state, **pre}, _pre=pre)}

    atn = _cfg()["atencion"]
    aviso = _aviso_privacidad(state)
    burbujas = [aviso] if aviso else []
    burbujas.append(atn["agenda_pedir_nombre"])
    response = "\n\n".join(burbujas)
    _guardar_ai(state, response)

    out = {**pre,
           "response": response,
           "response_bubbles": burbujas,
           "response_interactive": None,
           "booking_stage": "captura_nombre",
           "messages": [AIMessage(content=response)]}
    if aviso:
        # One-shot: se escribe únicamente al emitir. Si aviso es None porque
        # el flag ya estaba levantado, NO se devuelve la clave → se conserva.
        out["booking_privacidad_dicha"] = True
    return out


def pedir_email(state: AgentState, _pre: dict | None = None) -> AgentState:
    """Paso 2 del wizard. Emite SOLO agenda_pedir_email (v8.3).

    El aviso de privacidad es responsabilidad exclusiva de pedir_nombre:
    este nodo ya no lo evalúa, no lo concatena y NO devuelve la clave
    booking_privacidad_dicha (antes la pisaba con bool(aviso), lo que
    re-armaba el aviso en el paso siguiente).
    """
    pre = {**(_pre or {}), **_precargar_desde_intake(state)}

    if pre.get("lead_email") or state.get("lead_email"):
        return {**pre, **pedir_modalidad({**state, **pre}, _pre=pre)}

    atn = _cfg()["atencion"]
    response = atn["agenda_pedir_email"]
    _guardar_ai(state, response)
    return {**pre,
            "response": response,
            "response_bubbles": [response],
            "response_interactive": None,
            "booking_stage": "captura_email",
            "messages": [AIMessage(content=response)]}


def pedir_modalidad(state: AgentState, _pre: dict | None = None) -> AgentState:
    """Paso 3 del wizard. Emite SOLO agenda_pedir_modalidad, más el ack de
    precarga de intake si corresponde, y los botones nativos.

    NUNCA reinyecta el aviso de privacidad ni toca booking_privacidad_dicha
    (v8.3): esa emisión es one-shot y pertenece a pedir_nombre.
    """
    atn = _cfg()["atencion"]
    pre = _pre or {}

    burbujas: list[str] = []
    if pre:
        burbujas.append("De su registro ya tengo "
                        + " y ".join(_PRECARGA_LABELS[k] for k in pre
                                     if k in _PRECARGA_LABELS)
                        + "; solo me falta la modalidad:")
    burbujas.append(atn["agenda_pedir_modalidad"])
    response = "\n\n".join(burbujas)
    _guardar_ai(state, response)

    return {**pre,
            "response": response,
            "response_bubbles": burbujas,
            "response_interactive": _BOTONES_MODALIDAD,
            "booking_stage": "captura_modalidad",
            "messages": [AIMessage(content=response)]}


def procesar_captura(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    stage = state.get("booking_stage")
    decision = state.get("booking_decision") or {}
    raw = (state.get("query") or "").strip()

    init = {}
    if not state.get("agenda_started_en"):
        # v8.7 — FIX #8 (auditoría): el flag booking_privacidad_dicha se
        # levanta SOLO si el cliente dio consentimiento explícito en la
        # ficha. Antes bastaba con que intake_respuestas existiera —
        # vulneraba Ley 21.719 cuando el cliente había negado la
        # autorización pero iniciar_ficha ya había pre-cargado 'telefono'.
        #
        # Variable LOCAL (no canal del estado): consent_dado NO existe en
        # AgentState, escribirlo como clave del dict lanzaría
        # InvalidUpdateError (mismo patrón que el fix #1 de session_id).
        consent_dado = ((state.get("intake_respuestas") or {})
                        .get("consentimiento_datos") is True)

        init = {"agenda_started_en": datetime.now(GCAL_TZ).isoformat(),
                "agenda_ventana_desde": 0,
                "booking_attempts": 0,
                "booking_franja": None,
                "booking_match": None,
                "booking_match_candidatos": [],
                "slots_propuestos": [],
                # v8.3 (conservado): preservar un True ya persistido.
                # v8.7: el origen del True es ahora el consent EXPLÍCITO.
                "booking_privacidad_dicha": bool(state.get("booking_privacidad_dicha"))
                                            or consent_dado}


    updates = {**init}

    if stage == "captura_nombre":
        nombre = decision.get("nombre") or raw
        val, err = _v_nombre(nombre)
        if err:
            response = f"No fue posible registrar su nombre: {err}. {atn['agenda_pedir_nombre']}"
            _guardar_ai(state, response)
            return {**updates, "response": response,
                    "response_bubbles": [response],
                    "response_interactive": None,
                    "messages": [AIMessage(content=response)]}
        res = pedir_email({**state, **updates, "lead_nombre": val})
        return {**updates, "lead_nombre": val, **res}

    if stage == "captura_email":
        email = decision.get("email") or raw
        val, err = _v_email(email)
        if err:
            response = f"No fue posible registrar su correo: {err}. {atn['agenda_pedir_email']}"
            _guardar_ai(state, response)
            return {**updates, "response": response,
                    "response_bubbles": [response],
                    "response_interactive": None,
                    "messages": [AIMessage(content=response)]}
        res = pedir_modalidad({**state, **updates, "lead_email": val})
        return {**updates, "lead_email": val, **res}

    if stage == "captura_modalidad":
        modalidad = decision.get("modalidad") or raw
        val, err = _v_modalidad(modalidad)
        if err:
            response = f"No fue posible registrar la modalidad: {err}. {atn['agenda_pedir_modalidad']}"
            _guardar_ai(state, response)
            return {**updates, "response": response,
                    "response_bubbles": [response],
                    "response_interactive": _BOTONES_MODALIDAD,
                    "messages": [AIMessage(content=response)]}
        updates["lead_modalidad"] = val

        estado_intermedio = {**state, **updates, "booking_stage": "propuesta",
                             "booking_decision": {"accion": "entregar_datos"}}
        disponibilidad = consultar_disponibilidad(estado_intermedio)

        # v8.6 — Fix #5 (auditoría): replicar el manejo de "ventana_agotada"
        # que ya hace despues_de_consultar en el camino normal del planner.
        # Sin este branch, proponer_slots recibía slots=[] y emitía el
        # template de propuesta SIN opciones (texto colgado) tras capturar
        # modalidad. ofrecer_ejecutiva ofrece el contacto humano y aplica
        # RESET_BOOKING (limpieza de stage/signal) — el próximo turno sale
        # del sub-flujo sin arrastre.
        if disponibilidad.get("booking_signal") == "ventana_agotada":
            oferta = ofrecer_ejecutiva({**estado_intermedio, **disponibilidad})
            return {**updates, **disponibilidad, **oferta}

        pack = proponer_slots({**estado_intermedio, **disponibilidad})
        return {**updates, **disponibilidad, **pack}

    # Fallback defensivo
    return pedir_nombre({**state, **updates})



def reintentar_eleccion(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    slots = _slots(state)
    interactive = _interactive_slots(slots)

    if interactive is None:
        response = atn["agenda_reintento_slots"].format(opciones=_opciones(slots))
    else:
        # v8.5: los horarios ya están en los botones/lista; no duplicar.
        response = atn["agenda_reintento_slots"].format(opciones="").rstrip()

    _guardar_ai(state, response)
    return {"response": response,
            "response_bubbles": [response],
            "response_interactive": interactive,
            "booking_attempts": (state.get("booking_attempts") or 0) + 1,
            "messages": [AIMessage(content=response)]}


def aclarar_eleccion(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    slots = _slots(state)
    idxs = [i for i in (state.get("booking_match_candidatos") or [])
            if 0 <= i < len(slots)]
    interactive = _interactive_slots(slots, idxs)

    if interactive is None:
        response = atn["agenda_eleccion_ambigua"].format(
            opciones=_opciones_indexadas(slots, idxs))
    else:
        # v8.5: los candidatos ya están en el interactive; no duplicar.
        response = atn["agenda_eleccion_ambigua"].format(opciones="").rstrip()

    _guardar_ai(state, response)
    return {"response": response,
            "response_bubbles": [response],
            "response_interactive": interactive,
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
        respuesta = ("La asesoría dura 30 a 45 minutos y puede ser online por Meet "
                     "o presencial en nuestra oficina.")
    _guardar_ai(state, respuesta)
    return {"response": respuesta,
            "response_bubbles": [respuesta],
            "response_interactive": None,
            "messages": [AIMessage(content=respuesta)]}


# ---------------------------------------------------------------------------
# Confirmación y creación
# ---------------------------------------------------------------------------
def confirmar_y_crear(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]

    if not state.get("lead_nombre"):
        return pedir_nombre(state)
    if not state.get("lead_email"):
        return pedir_email(state)
    if not state.get("lead_modalidad"):
        return pedir_modalidad(state)

    slots = _slots(state)
    if not slots or state.get("booking_match") is None:
        response = atn["agenda_reintento_slots"].format(opciones="(sin horarios)")
        _guardar_ai(state, response)
        return {"response": response,
                "response_bubbles": [response],
                "response_interactive": None,
                "booking_stage": "propuesta",
                "messages": [AIMessage(content=response)]}

    elegido = slots[state["booking_match"]]
    hoy = datetime.now(GCAL_TZ).date()
    tid = state.get("thread_id", "")

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
            return {
                "response": response,
                "response_bubbles": [response],
                "response_interactive": None,
                "messages": [AIMessage(content=response)],
                **RESET_BOOKING,
            }

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
        return {
            "response": response,
            "response_bubbles": [response],
            "response_interactive": None,
            "messages": [AIMessage(content=response)],
            **RESET_BOOKING,
        }

    if state.get("lead_modalidad") == "online" and evento.meet_link:
        modalidad_linea = f"Videollamada por Google Meet: {evento.meet_link}"
    elif state.get("lead_modalidad") == "presencial":
        modalidad_linea = f"Presencial en nuestra oficina: {DIRECCION_OFICINA}"
    else:
        modalidad_linea = (f"Modalidad: {state.get('lead_modalidad')}. "
                           f"Presencial en nuestra oficina: {DIRECCION_OFICINA}")

    response = atn["agenda_confirmada"].format(
        nombre=_primer_nombre(state["lead_nombre"]),
        fecha=evento.fecha, hora_inicio=evento.hora_inicio,
        hora_fin=evento.hora_fin, modalidad_linea=modalidad_linea,
        html_link=evento.html_link)

    payload = {
        **evento.model_dump(),
        "cliente_nombre": state["lead_nombre"],
        "cliente_email": state["lead_email"],
        "inicio_cita": elegido.isoformat(),
    }

    upsert_lead(thread_id=tid, intake_respuestas={
        **(state.get("intake_respuestas") or {}),
        "nombre": state["lead_nombre"],
        "email": state["lead_email"],
        "telefono": _telefono_cliente(state),
    })

    insert_booking(thread_id=tid, booking=payload)
    upsert_conversation(thread_id=tid, status="agendado")

    try:
        from integrations.sheetdb import actualizar_booking
        actualizar_booking({**state, "booking": payload})
    except Exception as e:
        logger.exception("[booking] SheetDB (fallback) falló: %s", e)

    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": None,
        "booking": payload,
        "closed": False,
        "messages": [AIMessage(content=response)],
        **RESET_BOOKING,
    }


def ofrecer_ejecutiva(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_sin_horarios"]
    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": None,
        "messages": [AIMessage(content=response)],
        **RESET_BOOKING,
    }


def cerrar_booking(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["agenda_abortado"]
    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": None,
        "messages": [AIMessage(content=response)],
        **RESET_BOOKING,
    }
