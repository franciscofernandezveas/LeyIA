"""graph/intake/nodes.py — Acciones del sub-agente INTAKE.

v10 — Links wa.me limpios (oculta la URL enorme):
  - Todos los links hacia la ejecutiva se renderizan como markdown
    [Hablar con un humano](wa.me?text=...). El cliente ve texto limpio;
    al pinchar, WhatsApp abre con el diagnóstico prellenado.
  - _wa_link_display() envuelve _wa_link() en markdown; se usa en
    _completar_ficha (vía _wa_link_diagnostico), _derivar_parcial,
    abandonar_ficha y sin_consentimiento.
  - El diagnóstico completo (ficha validada + thread_id) sigue viajando
    oculto en el query param ?text= para la ejecutiva.

v9 — Link de completado con diagnóstico:
  - _completar_ficha arma el wa.me con _wa_link_diagnostico (ficha validada
    + thread_id para cruce CRM) en vez del saludo genérico.
  - GOTCHA: fusiona las respuestas del turno en una copia del state antes
    de llamarlo — state["intake_respuestas"] aún no tiene la última
    respuesta e intake_completado aún viene False (el intro del diagnóstico
    depende de él); se fuerza True en la copia.

v8 — Semántico + leads parciales:
  - Consentimiento = pregunta 1 → upsert incremental (completed=False) tras
    cada campo válido; completed=True al cerrar la ficha.
  - Extracción multi-campo desde intake_decision (el planner ya la hizo);
    los validadores deterministas siguen siendo la fuente de verdad.
  - Nombre NUNCA se acepta del mensaje crudo (solo extracción validada) —
    adiós al bug "nombre = quiero hablar con humano".
  - 2 fallos en campo requerido → derivar_parcial automático con ficha parcial.
  - Nuevos nodos: reanudar, pausar_para_faq, derivar_parcial, abandonar_ficha.
  - Completado ahora también setea `response`.
"""
import logging
import unicodedata

from langchain_core.messages import AIMessage

from core.contracts import EMAIL_RE, AgentState, ROUTE_FAQ
from core.db_client import upsert_lead
from graph.nodes import (
    _ahora_iso, _cfg, _guardar_ai, _primer_nombre, _telefono_cliente,
    _wa_link, _wa_link_diagnostico, _wa_link_display,
)

logger = logging.getLogger(__name__)

MAX_INTENTOS_CAMPO = 2

SALTAR = ("saltar", "salta", "skip", "omitir", "paso",
          "prefiero no", "prefiero no decir", "prefiero no responder",
          "no quiero responder", "no aplica", "n/a", "-")

_CAMPOS_SOLO_EXTRACCION = {"nombre"}        # jamás aceptan el mensaje crudo


# ---------------------------------------------------------------------------
# NORMALIZACIÓN + VALIDADORES (fuente de verdad de formato)
# ---------------------------------------------------------------------------
def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _v_text(minlen: int):
    def check(s):
        s = (s or "").strip()
        if len(s) < minlen:
            return None, f"se requieren al menos {minlen} caracteres para este dato"
        return s, None
    return check


def _v_email(s):
    s = (s or "").strip().lower().replace(" ", "")
    if not EMAIL_RE.fullmatch(s):
        return None, ("el formato del correo no es válido "
                      "(ejemplo: nombre@correo.cl)")
    return s, None


def _v_si_no(s):
    t = _sin_tildes((s or "").strip().lower().rstrip("."))
    if t in ("si", "sipo", "sip", "yes", "claro", "por supuesto",
             "afirmativo", "de acuerdo", "autorizo", "lo autorizo", "acepto"):
        return True, None
    if t in ("no", "nop", "nope", "negativo", "para nada",
             "no autorizo", "no acepto"):
        return False, None
    return None, "por favor, responda únicamente sí o no"


def _v_opciones(*opciones: str):
    def check(s):
        t = _sin_tildes((s or "").strip().lower().rstrip("."))
        if t.isdigit() and 1 <= int(t) <= len(opciones):
            return opciones[int(t) - 1], None
        exact = [o for o in opciones if _sin_tildes(o.lower()) == t]
        if exact:
            return exact[0], None
        subs = [o for o in opciones if t and t in _sin_tildes(o.lower())]
        if len(subs) == 1:
            return subs[0], None
        if subs:
            return None, ("su respuesta admite más de una opción — "
                          "¿a cuál se refiere: " + " · ".join(subs) + "?")
        return None, ("indique una de las opciones de la lista "
                      "(puede responder con el número)")
    return check


def _v_nombre(s):
    """Sanity mínima del nombre extraído por el LLM (el crudo NUNCA entra)."""
    s = (s or "").strip()
    palabras = [p for p in s.split() if p]
    if len(s) < 3 or not palabras:
        return None, "no logré identificar su nombre; indíquelo completo"
    if len(palabras) > 6 or any(len(p) > 20 for p in palabras):
        return None, "el nombre parece ser una frase; indique solo su nombre y apellido"
    return " ".join(p.capitalize() for p in palabras), None


# ---------------------------------------------------------------------------
# REGISTRY — consentimiento primero: desbloquea persistencia incremental
# ---------------------------------------------------------------------------
QUESTIONS: list[dict] = [
    {"id": "consentimiento_datos", "requerida": True, "extractable": False,
     "prompt": ("Conforme a la Ley N° 21.719 de protección de datos "
                "personales: ¿autoriza usted a Manzzo y Cía a tratar los "
                "datos que nos entregue exclusivamente para gestionar su "
                "consulta y contactarlo(a)? (responda sí o no)"),
     "validate": _v_si_no},

    {"id": "nombre", "requerida": True,
     "prompt": "Indique su nombre completo, tal como aparece en su cédula de identidad:",
     "validate": _v_nombre},

    {"id": "email", "requerida": True,
     "prompt": ("Indique su correo electrónico (se utilizará exclusivamente "
                "para el seguimiento de su caso):"),
     "validate": _v_email},

    {"id": "situacion_actual", "requerida": True,
     "prompt": ("Describa los hechos de su caso en 2 o 3 líneas: "
                "qué ocurrió, quiénes intervienen y desde cuándo."),
     "validate": _v_text(20)},

    {"id": "etapa_proceso", "requerida": False,
     "prompt": ("¿En qué estado se encuentra su caso actualmente?\n"
                "  1) Aún no he iniciado acciones legales\n"
                "  2) Fui demandado(a) o notificado(a) por un tribunal\n"
                "  3) Hay una causa judicial en curso\n"
                "  4) Ya existe sentencia o acuerdo previo"),
     "validate": _v_opciones("aún no inicio nada", "me demandaron o me notificaron",
                             "causa en curso", "sentencia o acuerdo previo"),
     "label_map": {"sin_inicio": "aún no inicio nada",
                   "demandado": "me demandaron o me notificaron",
                   "causa_en_curso": "causa en curso",
                   "sentencia_previa": "sentencia o acuerdo previo"}},
]

_BY_ID = {q["id"]: q for q in QUESTIONS}


# ---------------------------------------------------------------------------
# HELPERS DE PROGRESO
# ---------------------------------------------------------------------------
def _aplicables(state: AgentState, respuestas: dict | None = None) -> list[dict]:
    resp = respuestas if respuestas is not None else (state.get("intake_respuestas") or {})
    return [q for q in QUESTIONS
            if not q.get("skip_if", lambda s, r=None: False)(state, resp)]


def _pendientes(state: AgentState, respuestas: dict) -> list[dict]:
    hechas = {q["id"] for q in _aplicables(state, respuestas)
              if q["id"] in respuestas}
    return [q for q in _aplicables(state, respuestas) if q["id"] not in hechas]


def _form_pregunta(state: AgentState, q: dict, respuestas: dict) -> str:
    sec = _cfg()["intake"]
    aplicables = _aplicables(state, respuestas)
    respondidas = len([x for x in aplicables if x["id"] in respuestas])
    prefix = f"_{respondidas + 1} de {len(aplicables)}_ · "
    suffix = "" if q["requerida"] else f"\n{sec['aviso_saltar']}"
    return prefix + q["prompt"] + suffix


def _idx_of(qid: str) -> int:
    return next(i for i, q in enumerate(QUESTIONS) if q["id"] == qid)


# ---------------------------------------------------------------------------
# EXTRACCIÓN MULTI-CAMPO
# ---------------------------------------------------------------------------
def _aplicar_extraccion(respuestas: dict, decision: dict) -> None:
    """Persiste SOLO campos explícitos que pasan su validador determinista."""
    for qid, qdef in _BY_ID.items():
        if not qdef.get("extractable", True):
            continue
        raw = decision.get(qid)
        if raw is None:
            continue
        if isinstance(raw, bool):
            respuestas[qid] = raw
            continue
        if "label_map" in qdef:
            raw = qdef["label_map"].get(str(raw))
            if raw is None:
                continue
        val, err = qdef["validate"](raw)
        if err is None and val is not None:
            respuestas[qid] = val
        else:
            logger.info("[intake] extracción rechazada por validador: %s=%r (%s)",
                        qid, raw, err)


def _persistir_parcial(state: AgentState, respuestas: dict) -> None:
    """Upsert incremental del lead — solo con consentimiento expreso."""
    if respuestas.get("consentimiento_datos") is not True:
        return
    try:
        upsert_lead(
            thread_id=state.get("thread_id", ""),
            intake_respuestas=respuestas,
            category=state.get("category"),
            completed=False,
        )
    except Exception as e:
        logger.warning("upsert_lead parcial falló (best-effort): %s", e)


# ---------------------------------------------------------------------------
# NODOS DE ACCIÓN
# ---------------------------------------------------------------------------
def iniciar_ficha(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas: dict = {}
    tel = _telefono_cliente(state)
    if tel:
        respuestas["telefono"] = tel

    pend = _pendientes(state, respuestas)
    msg = sec["apertura"].rstrip() + "\n\n" + _form_pregunta(state, pend[0], respuestas)

    _guardar_ai(state, msg)
    return {
        "intake_activo": True,
        "intake_idx": _idx_of(pend[0]["id"]),
        "intake_respuestas": respuestas,
        "intake_attempts": 0,
        "intake_exit": None,
        "intake_resume": False,
        "intake_started_en": _ahora_iso(),
        "intake_completado": False,
        "intake_decision": None,
        "intake_stage": "preguntando",
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }


def procesar_respuesta(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    decision = state.get("intake_decision") or {}
    attempts = int(state.get("intake_attempts") or 0)
    raw = (state.get("query") or "").strip()
    error = None

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas)
    q_actual = pend[0]

    if raw.lower() in SALTAR:
        if q_actual["requerida"]:
            error = ("este dato es obligatorio para poder derivar su caso; "
                     "no puede omitirse")
        else:
            respuestas[q_actual["id"]] = None
    else:
        _aplicar_extraccion(respuestas, decision)
        if q_actual["id"] not in respuestas:
            if q_actual["id"] in _CAMPOS_SOLO_EXTRACCION:
                error = ("no logré identificar su nombre en el mensaje; "
                         "indique solo su nombre y apellido")
            else:
                val, err = q_actual["validate"](raw)
                if err:
                    error = err
                else:
                    respuestas[q_actual["id"]] = val

    if respuestas.get("consentimiento_datos") is False:
        return sin_consentimiento(state, respuestas)

    pend = _pendientes(state, respuestas)

    if not pend:
        return _completar_ficha(state, respuestas)

    if error:
        attempts += 1
        if attempts >= MAX_INTENTOS_CAMPO and q_actual["requerida"]:
            logger.info("[intake] %d fallos en '%s' → derivación parcial",
                        attempts, q_actual["id"])
            return _derivar_parcial(state, respuestas)
        msg = "\n".join([f"No fue posible registrar su respuesta: {error}",
                         _form_pregunta(state, q_actual, respuestas)])
        _guardar_ai(state, msg)
        return {
            "intake_respuestas": respuestas,
            "intake_attempts": attempts,
            "intake_decision": None,
            "response": msg,
            "messages": [AIMessage(content=msg)],
        }

    _persistir_parcial(state, respuestas)

    siguiente = pend[0]
    msg = _form_pregunta(state, siguiente, respuestas)
    _guardar_ai(state, msg)
    return {
        "intake_idx": _idx_of(siguiente["id"]),
        "intake_respuestas": respuestas,
        "intake_attempts": 0,
        "intake_decision": None,
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }


def _completar_ficha(state: AgentState, respuestas: dict) -> AgentState:
    # state["intake_respuestas"] aún NO tiene la última respuesta del turno
    # e intake_completado aún viene False (el intro del diagnóstico depende
    # de él) → fusionar ambos en una copia antes de armar el link.
    state_con_ficha = {
        **state, "intake_respuestas": respuestas, "intake_completado": True,
    }
    ack = _cfg()["intake"]["ack_completado"].format(
        nombre=_primer_nombre(respuestas.get("nombre")) or "",
        wa_link=_wa_link_diagnostico(state_con_ficha),
    )
    try:
        upsert_lead(
            thread_id=state.get("thread_id", ""),
            intake_respuestas=respuestas,
            category=state.get("category"),
            completed=True,
        )
    except Exception as e:
        logger.warning("upsert_lead (completado) falló: %s", e)

    _guardar_ai(state, ack)
    return {
        "intake_activo": False,
        "intake_completado": True,
        "intake_respuestas": respuestas,
        "intake_exit": None,
        "intake_decision": None,
        "intake_stage": "completado",
        "response": ack,
        "messages": [AIMessage(content=ack)],
    }


def pausar_para_faq(state: AgentState) -> AgentState:
    logger.info("[intake] duda lateral → pausa a FAQ")
    return {
        "intake_exit": "faq",
        "intake_resume": True,
        "intake_stage": "pausado",
        "intake_decision": None,
    }


def reanudar(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    pend = _pendientes(state, respuestas)
    base = {
        "intake_exit": None,
        "intake_resume": False,
        "intake_decision": None,
        "intake_stage": "preguntando",
    }
    if not pend or not state.get("intake_activo"):
        return base

    msg = _cfg()["intake"]["reanudar"].rstrip() + "\n\n" \
        + _form_pregunta(state, pend[0], respuestas)
    _guardar_ai(state, msg)
    return {**base, "response": msg, "messages": [AIMessage(content=msg)]}


def derivar_parcial(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    return _derivar_parcial(state, respuestas)


def _derivar_parcial(state: AgentState, respuestas: dict) -> AgentState:
    msg = _cfg()["intake"]["derivacion_parcial"].format(
        wa_link=_wa_link_display(
            f"Hola, soy {respuestas.get('nombre', '') or 'un cliente del asistente'}, "
            "prefiero continuar directamente con la ejecutiva.",
            display="Hablar con un humano",
        )
    )
    _persistir_parcial(state, respuestas)
    _guardar_ai(state, msg)
    return {
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": respuestas,
        "intake_exit": "handoff",
        "intake_stage": "completado",
        "intake_decision": None,
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }


def abandonar_ficha(state: AgentState) -> AgentState:
    """El cliente cancela el registro: cierre amable. Si ya había consentido,
    sus datos parciales quedan en el CRM para seguimiento."""
    respuestas = dict(state.get("intake_respuestas") or {})
    _persistir_parcial(state, respuestas)
    msg = _cfg()["intake"]["cierre_abandono"].format(
        wa_link=_wa_link_display(display="Hablar con un humano")
    )
    _guardar_ai(state, msg)
    return {
        "intake_activo": False,
        "intake_completado": False,
        "intake_exit": None,
        "intake_stage": "abandonado",
        "intake_decision": None,
        "route": ROUTE_FAQ,
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }


def sin_consentimiento(state: AgentState, respuestas: dict | None = None) -> AgentState:
    msg = _cfg()["intake"]["sin_consentimiento"].format(
        wa_link=_wa_link_display(display="Hablar con un humano")
    )
    _guardar_ai(state, msg)
    return {
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": (respuestas if respuestas is not None
                              else state.get("intake_respuestas")),
        "intake_exit": None,
        "intake_decision": None,
        "intake_stage": "sin_consentimiento",
        "route": ROUTE_FAQ,
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }


