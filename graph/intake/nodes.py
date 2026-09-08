"""graph/intake/nodes.py — Acciones del sub-agente INTAKE.

v7.2 — Hooks de persistencia PostgreSQL con regla de consentimiento:
  - Los mensajes AI del intake se guardan en `messages` (log de conversación).
  - La tabla `leads` SOLO se escribe cuando consentimiento_datos=True
    (ficha completada). Persistir datos personales antes del consentimiento
    contradice la Ley 21.719 que el propio sistema cita — si el cliente
    abandona a mitad de la ficha o rechaza el consentimiento, no queda
    registro en el CRM. Es el mismo comportamiento que tenía SheetDB
    (solo sincronizaba tras handoff).

v7.1 — FIX: el extractor ya no alucina campos cruzados (campo_actual fijado).
v7   — encuesta de 5 preguntas; fix intake_activo; clamp de índice.
"""
import logging
import unicodedata

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from core.contracts import (
    EMAIL_RE, AgentState, IntakeExtract, ROUTE_FAQ,
)
from core.db_client import upsert_lead
from core.llm import with_structured_output
from graph.nodes import (
    _ahora_iso, _cfg, _guardar_ai, _primer_nombre, _telefono_cliente,
)

logger = logging.getLogger(__name__)

SALTAR = ("saltar", "salta", "skip", "omitir", "paso",
          "prefiero no", "prefiero no decir", "prefiero no responder",
          "no quiero responder", "no aplica", "n/a", "-")


# ---------------------------------------------------------------------------
# NORMALIZACIÓN
# ---------------------------------------------------------------------------
def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------------------
# VALIDADORES DETERMINISTAS
# ---------------------------------------------------------------------------
def _v_text(minlen: int):
    def check(s):
        s = (s or "").strip()
        if len(s) < minlen:
            return None, f"se requieren al menos {minlen} caracteres para este dato"
        return s, None
    return check


def _v_email(s):
    s = (s or "").strip().lower()
    if not EMAIL_RE.fullmatch(s):
        return None, "el formato del correo no es válido (ejemplo: nombre@correo.cl)"
    return s, None


def _v_si_no(s):
    t = _sin_tildes((s or "").strip().lower())
    if t in ("si", "sipo", "sip", "yes", "y", "claro", "ok", "dale",
             "por supuesto", "afirmativo", "de acuerdo", "autorizo"):
        return True, None
    if t in ("no", "nop", "n", "nope", "negativo", "para nada", "no autorizo"):
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


# ---------------------------------------------------------------------------
# REGISTRY DE PREGUNTAS — encuesta de 5
# ---------------------------------------------------------------------------
QUESTIONS: list[dict] = [
    {"id": "nombre", "requerida": True,
     "prompt": "Indique su nombre completo, tal como aparece en su cédula de identidad:",
     "validate": _v_text(3)},

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

    {"id": "consentimiento_datos", "requerida": True, "extractable": False,
     "prompt": ("Para finalizar, conforme a la Ley N° 21.719 de protección "
                "de datos personales: ¿autoriza usted a Manzzo y Cía a "
                "tratar los datos entregados exclusivamente para gestionar "
                "su consulta y contactarlo(a)? (responda sí o no)"),
     "validate": _v_si_no},
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


def _idx_actual(state: AgentState) -> int:
    idx = state.get("intake_idx", 0) or 0
    return min(max(idx, 0), len(QUESTIONS) - 1)


# ---------------------------------------------------------------------------
# COSTURA LLM (solo el campo actual)
# ---------------------------------------------------------------------------
def _extraer_intake(state: AgentState, campo_actual: str) -> IntakeExtract:
    extractor = ChatPromptTemplate.from_messages([
        ("system",
         "Extraes datos para la ficha de cliente de un estudio jurídico "
         "chileno. Devuelve SOLO el esquema, con null en todo dato que no "
         "aparezca EXPLÍCITAMENTE en el mensaje. Nunca infieras ni completes "
         "datos por contexto.\n"
         f"En este turno el usuario está respondiendo ÚNICAMENTE sobre: '{campo_actual}'.\n"
         "REGLA CRÍTICA: extrae SOLO ese campo. Los demás campos deben quedar "
         "en null, aunque parezcan deducibles del mensaje. Por ejemplo, NUNCA "
         "extraigas 'nombre' desde un email, ni 'etapa_proceso' desde una "
         "descripción de hechos. Solo responde el esquema JSON.\n"
         "Reglas por campo:\n"
         "· nombre: tal como lo escribe el cliente.\n"
         "· email: solo si aparece un correo con @ y dominio.\n"
         "· situacion_actual: resume los hechos del cliente sin agregar nada.\n"
         "· etapa_proceso: usa EXACTAMENTE las etiquetas del esquema.\n"
         "· consentimiento: NI SIQUIERA está en el esquema — debe responderse "
         "directamente, nunca inferirse."),
        ("human", "{query}"),
    ]) | with_structured_output(IntakeExtract)
    try:
        return extractor.with_retry(stop_after_attempt=2).invoke(
            {"query": state["query"]})
    except Exception as e:
        logger.warning("extractor de intake falló (se usa validador crudo): %s", e)
        return IntakeExtract()


def _aplicar_extraccion(respuestas: dict, extra: IntakeExtract,
                        campo_actual: str | None = None) -> None:
    """Aplica extracción SOLO al campo que el usuario responde este turno."""
    for qid, qdef in _BY_ID.items():
        if campo_actual is not None and qid != campo_actual:
            continue                       # v7.1: anti-alucinación cruzada
        if not qdef.get("extractable", True):
            continue
        raw = getattr(extra, qid, None)
        if raw is None:
            continue
        if isinstance(raw, bool):
            respuestas[qid] = raw
            continue
        if "label_map" in qdef:
            raw = qdef["label_map"].get(str(raw), None)
            if raw is None:
                continue
        val, _err = qdef["validate"](raw)
        if _err is None:
            respuestas[qid] = val


# ---------------------------------------------------------------------------
# NODOS DE ACCIÓN
# ---------------------------------------------------------------------------
def iniciar_ficha(state: AgentState) -> AgentState:
    """Apertura del intake: activa el ledger y envía la primera pregunta.
    NO escribe en `leads` (aún no hay consentimiento) — solo el mensaje AI
    queda registrado en `messages` como parte del log de conversación."""
    sec = _cfg()["intake"]
    respuestas: dict = {}
    tel = _telefono_cliente(state)
    if tel:
        respuestas["telefono"] = tel

    pend = _pendientes(state, respuestas)
    msg = sec["apertura"].rstrip() + "\n\n" + _form_pregunta(state, pend[0], respuestas)
    idx0 = next(i for i, q in enumerate(QUESTIONS) if q["id"] == pend[0]["id"])

    _guardar_ai(state, msg)
    return {
        "intake_activo": True,
        "intake_idx": idx0,
        "intake_respuestas": respuestas,
        "intake_started_en": _ahora_iso(),
        "intake_completado": False,
        "intake_decision": None,
        "intake_stage": "preguntando",
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }


def procesar_respuesta(state: AgentState) -> AgentState:
    """Turno normal: validar respuesta → avanzar, repetir (hold-on-error),
    completar (cierra el sub-flujo + persiste lead) o rechazar consentimiento."""
    respuestas = dict(state.get("intake_respuestas") or {})
    q_actual = QUESTIONS[_idx_actual(state)]
    raw = (state.get("query") or "").strip()
    error = None

    if raw.lower() in SALTAR:
        if q_actual["requerida"]:
            error = ("este dato es obligatorio para poder derivar su caso; "
                     "no puede omitirse")
        else:
            respuestas[q_actual["id"]] = None
    else:
        extra = _extraer_intake(state, q_actual["id"])
        _aplicar_extraccion(respuestas, extra, campo_actual=q_actual["id"])
        if q_actual["id"] not in respuestas:
            val, err = q_actual["validate"](raw)
            if err:
                error = err
            else:
                respuestas[q_actual["id"]] = val

    # Consentimiento rechazado → cerrar sin derivar y SIN persistir lead
    if respuestas.get("consentimiento_datos") is False:
        return sin_consentimiento(state, respuestas)

    pend = _pendientes(state, respuestas)

    # Ficha completa (consentimiento=True) → persistir lead + handoff inmediato
    if not pend:
        ack = _cfg()["intake"]["ack_completado"].format(
            nombre=_primer_nombre(respuestas.get("nombre")))
        # Hook Postgres: único punto donde intake escribe en `leads`.
        upsert_lead(
            thread_id=state.get("thread_id", ""),
            intake_respuestas=respuestas,
            category=state.get("category"),
            completed=True,
        )
        _guardar_ai(state, ack)
        return {
            "intake_activo": False,
            "intake_completado": True,
            "intake_respuestas": respuestas,
            "intake_decision": None,
            "intake_stage": "completado",
            "messages": [AIMessage(content=ack)],
        }

    # Hold-on-error
    if error:
        msg = "\n".join([f"No fue posible registrar su respuesta: {error}",
                         _form_pregunta(state, q_actual, respuestas)])
        _guardar_ai(state, msg)
        return {
            "intake_respuestas": respuestas,
            "intake_decision": None,
            "response": msg,
            "messages": [AIMessage(content=msg)],
        }

    # Avanzar a la siguiente pregunta
    siguiente = pend[0]
    idx_sig = next(i for i, q in enumerate(QUESTIONS) if q["id"] == siguiente["id"])
    msg = _form_pregunta(state, siguiente, respuestas)
    _guardar_ai(state, msg)
    return {
        "intake_idx": idx_sig,
        "intake_respuestas": respuestas,
        "intake_decision": None,
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }


def sin_consentimiento(state: AgentState, respuestas: dict | None = None) -> AgentState:
    """Rechazó el tratamiento de datos: cerrar sin derivar y SIN escribir
    en `leads`. Solo se registra el mensaje de cierre en el log de conversación."""
    sec = _cfg()["intake"]
    msg = sec["sin_consentimiento"]
    _guardar_ai(state, msg)
    return {
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": (respuestas if respuestas is not None
                              else state.get("intake_respuestas")),
        "intake_decision": None,
        "intake_stage": "sin_consentimiento",
        "route": ROUTE_FAQ,
        "response": msg,
        "messages": [AIMessage(content=msg)],
    }
