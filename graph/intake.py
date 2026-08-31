"""graph/intake.py — Intake proactivo de lead + caso (patrón Interview Intake).

Adaptación del patrón tv_merge (deepwiki 5.1) a este agente:
- QUESTIONS: registry de preguntas con prompt, validador puro, flag
  `requerida` y `skip_if` condicional (≡ su lista GRAPH de 12 nodos).
- Validadores deterministas: devuelven (valor, None) o (None, error tipado).
  Si falla, el estado se MANTIENE en la pregunta actual — jamás se fabrica
  un dato (su política cero-alucinación, aquí sobre datos de persona).
- Costura LLM ("LLM layer in FRONT of submit"): _extraer_intake mapea texto
  libre → varios campos a la vez; los validadores siguen siendo la verdad.
- Al completar → edge `despues_de_intake` envía a handoff_humano en la MISMA
  invocación: la ejecutiva recibe resumen + ficha (JSON + WhatsApp).

Cumplimiento: `consentimiento_datos` es extractable=False — debe responderse
directamente (sí/no), nunca inferirse con el LLM.
"""
import logging
import re

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from core.contracts import (
    EMAIL_RE, AgentState, IntakeExtract,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_HANDOFF, ROUTE_INTAKE,
    ComoConocioLabel, EtapaProcesoLabel, HorarioContactoLabel,
)
from core.llm import with_structured_output
from graph.nodes import _ahora_iso, _cfg, _primer_nombre, _telefono_cliente

logger = logging.getLogger(__name__)

SALTAR = ("saltar", "skip", "omitir", "prefiero no", "prefiero no decir",
          "no aplica", "n/a", "-")


# ---------------------------------------------------------------------------
# VALIDADORES DETERMINISTAS  (≡ _v_text/_v_number/_v_yesno de tv_merge)
# ---------------------------------------------------------------------------
def _v_text(minlen: int):
    def check(s):
        s = (s or "").strip()
        if len(s) < minlen:
            return None, f"necesito al menos {minlen} caracteres"
        return s, None
    return check


def _v_email(s):
    s = (s or "").strip().lower()
    if not EMAIL_RE.fullmatch(s):
        return None, "ese correo no parece válido (ej: nombre@correo.com)"
    return s, None


def _v_telefono(s):
    """Normaliza móvil chileno a E.164 (+569XXXXXXXX)."""
    t = re.sub(r"[\s\-.()+]", "", (s or ""))
    if t.startswith("56"):
        t = t[2:]
    if t.isdigit() and len(t) == 9 and t.startswith("9"):
        return "+56" + t, None
    return None, "formato esperado: +56 9 1234 5678"


def _v_si_no(s):
    t = (s or "").strip().lower()
    if t in ("si", "sí", "sipo", "sip", "yes", "y", "claro", "ok", "dale",
             "por supuesto", "afirmativo"):
        return True, None
    if t in ("no", "nop", "n", "nope", "negativo", "para nada"):
        return False, None
    return None, "respóndeme con un sí o un no"


def _v_opciones(*opciones: str):
    """≡ _match_choice de tv_merge: exacto > substring único > número."""
    def check(s):
        t = (s or "").strip().lower().rstrip(".")
        if t.isdigit() and 1 <= int(t) <= len(opciones):
            return opciones[int(t) - 1], None
        exact = [o for o in opciones if o.lower() == t]
        if exact:
            return exact[0], None
        subs = [o for o in opciones if t and t in o.lower()]
        if len(subs) == 1:
            return subs[0], None
        if subs:
            return None, "ambiguo — ¿te refieres a: " + " · ".join(subs) + "?"
        return None, "elige una opción de la lista (puedes responder con el número)"
    return check


# ---------------------------------------------------------------------------
# REGISTRY DE PREGUNTAS  (≡ GRAPH de tv_merge)
# ---------------------------------------------------------------------------
_FAMILIA = {"pension_alimentos", "rebaja_pension", "regimen_visitas",
            "terminacion_pension", "divorcio", "compensacion_economica",
            "medidas_apremio"}

QUESTIONS: list[dict] = [
    {"id": "nombre", "requerida": True,
     "prompt": "¿Cuál es tu nombre completo?",
     "validate": _v_text(3)},

    {"id": "email", "requerida": True,
     "prompt": "¿Cuál es tu correo electrónico?",
     "validate": _v_email},

    {"id": "telefono", "requerida": False,
     "prompt": "¿Tienes un teléfono de contacto? (ej: +56 9 1234 5678)",
     "validate": _v_telefono,
     "skip_if": lambda st: _telefono_cliente(st) is not None},   # en prod ya lo tenemos

    {"id": "situacion_actual", "requerida": True,
     "prompt": "Cuéntame tu situación en 2-3 líneas: ¿qué pasó y desde cuándo?",
     "validate": _v_text(20)},

    {"id": "etapa_proceso", "requerida": False,
     "prompt": ("¿En qué etapa está tu caso?\n"
                "  1) Aún no inicio nada\n"
                "  2) Me demandaron / me notificaron\n"
                "  3) Hay una causa en curso\n"
                "  4) Ya existe sentencia o acuerdo previo"),
     "validate": _v_opciones("aún no inicio nada", "me demandaron o notificaron",
                             "causa en curso", "sentencia o acuerdo previo"),
     "label_map": {"sin_inicio": "aún no inicio nada",
                   "demandado": "me demandaron o notificaron",
                   "causa_en_curso": "causa en curso",
                   "sentencia_previa": "sentencia o acuerdo previo"}},

    {"id": "hijos_menores", "requerida": False,
     "prompt": "¿Hay hijos menores de edad de por medio? (sí/no)",
     "validate": _v_si_no,
     "skip_if": lambda st: st.get("category") not in _FAMILIA},

    {"id": "comuna", "requerida": False,
     "prompt": "¿En qué comuna vives? (por si prefieres atención presencial)",
     "validate": _v_text(2)},

    {"id": "horario_contacto", "requerida": False,
     "prompt": "¿En qué horario te puede llamar la ejecutiva?\n"
               "  1) Mañana (9-13)\n  2) Tarde (14-18)\n  3) Indiferente",
     "validate": _v_opciones("mañana (9-13)", "tarde (14-18)", "indiferente"),
     "label_map": {"manana": "mañana (9-13)", "tarde": "tarde (14-18)",
                   "indiferente": "indiferente"}},

    {"id": "como_nos_conocio", "requerida": False,
     "prompt": "Última: ¿cómo nos conociste?\n"
               "  1) Google\n  2) Instagram\n  3) TikTok\n"
               "  4) Recomendación\n  5) Otro",
     "validate": _v_opciones("google", "instagram", "tiktok",
                             "recomendación", "otro"),
     "label_map": {"google": "google", "instagram": "instagram",
                   "tiktok": "tiktok", "recomendacion": "recomendación",
                   "otro": "otro"}},

    {"id": "consentimiento_datos", "requerida": True, "extractable": False,
     "prompt": ("Para terminar: ¿autorizas que Manzzo y Cía use estos datos "
                "solo para gestionar tu consulta y contactarte? (sí/no)"),
     "validate": _v_si_no},
]

_BY_ID = {q["id"]: q for q in QUESTIONS}


# ---------------------------------------------------------------------------
# HELPERS DE PROGRESO (≡ IntakeSession.current/progress)
# ---------------------------------------------------------------------------
def _aplicables(state: AgentState) -> list[dict]:
    return [q for q in QUESTIONS
            if not q.get("skip_if", lambda s: False)(state)]


def _pendientes(state: AgentState, respuestas: dict) -> list[dict]:
    hechas = {q["id"] for q in _aplicables(state) if q["id"] in respuestas}
    return [q for q in _aplicables(state) if q["id"] not in hechas]


def _form_pregunta(state: AgentState, q: dict, respuestas: dict) -> str:
    sec = _cfg()["intake"]
    aplicables = _aplicables(state)
    respondidas = len([x for x in aplicables if x["id"] in respuestas])
    prefix = f"_{respondidas + 1} de {len(aplicables)}_ · "
    suffix = "" if q["requerida"] else f"\n{sec['aviso_saltar']}"
    return prefix + q["prompt"] + suffix


# ---------------------------------------------------------------------------
# COSTURA LLM: texto libre → varios campos (los validadores mandan)
# ---------------------------------------------------------------------------
def _extraer_intake(state: AgentState, campo_actual: str) -> IntakeExtract:
    """1 llamada LLM. Nunca lanza: ante error devuelve esquema vacío y el
    flujo cae al validador determinista del campo actual."""
    extractor = ChatPromptTemplate.from_messages([
        ("system",
         "Extrae datos de una ficha de cliente legal desde su mensaje. "
         "Responde SOLO el esquema, con null en lo que no aparezca. "
         f"El cliente probablemente está respondiendo sobre: '{campo_actual}'. "
         "Reglas: teléfonos chilenos normaliza a +569...; 'etapa_proceso' usa "
         "las etiquetas del esquema; NO inferjas consentimiento"),
        ("human", "{query}"),
    ]) | with_structured_output(IntakeExtract)
    try:
        return extractor.with_retry(stop_after_attempt=2).invoke(
            {"query": state["query"]})
    except Exception as e:
        logger.warning("extractor de intake falló (se usa validador crudo): %s", e)
        return IntakeExtract()


def _aplicar_extraccion(respuestas: dict, extra: IntakeExtract) -> None:
    """Cada valor extraído pasa por el validador de su pregunta. Solo se
    guardan valores VÁLIDOS (y pueden corregir respuestas anteriores)."""
    for qid, qdef in _BY_ID.items():
        if not qdef.get("extractable", True):
            continue                       # consentimiento: solo respuesta directa
        raw = getattr(extra, qid, None)
        if raw is None:
            continue
        if "label_map" in qdef:            # menús: literal del esquema → display
            raw = qdef["label_map"].get(str(raw), None)
            if raw is None:
                continue
        val, _err = qdef["validate"](raw if isinstance(raw, str) else str(raw))
        if _err is None:
            respuestas[qid] = val


# ---------------------------------------------------------------------------
# EL NODO
# ---------------------------------------------------------------------------
def intake_lead(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]

    # ── Apertura: activar ficha + primera pregunta ───────────────────────
    if not state.get("intake_activo"):
        respuestas: dict = {}
        tel = _telefono_cliente(state)                 # prefill en prod (WhatsApp)
        if tel:
            respuestas["telefono"] = tel
        pend = _pendientes(state, respuestas)
        msg = sec["apertura"].rstrip() + "\n\n" + _form_pregunta(state, pend[0], respuestas)
        idx0 = next(i for i, q in enumerate(QUESTIONS) if q["id"] == pend[0]["id"])
        return {"intake_activo": True,
                "intake_idx": idx0,
                "intake_respuestas": respuestas,
                "intake_started_en": _ahora_iso(),
                "response": msg,
                "messages": [AIMessage(content=msg)]}

    # ── Turno normal: consumir la respuesta ──────────────────────────────
    respuestas = dict(state.get("intake_respuestas") or {})
    q_actual = QUESTIONS[state.get("intake_idx", 0)]
    raw = (state.get("query") or "").strip()
    error = None

    if raw.lower() in SALTAR and not q_actual["requerida"]:
        respuestas[q_actual["id"]] = None              # ≡ "skip" de tv_merge
    else:
        extra = _extraer_intake(state, q_actual["id"])  # costura LLM
        _aplicar_extraccion(respuestas, extra)
        if q_actual["id"] not in respuestas:
            # fallback determinista: el texto crudo ES la respuesta actual
            val, err = q_actual["validate"](raw)
            if err:
                error = err                            # hold-on-error
            else:
                respuestas[q_actual["id"]] = val

    # Consentimiento rechazado → cerrar sin derivar (cumplimiento)
    if respuestas.get("consentimiento_datos") is False:
        msg = sec["sin_consentimiento"]
        return {"intake_activo": False,
                "intake_respuestas": respuestas,
                "response": msg,
                "route": ROUTE_FAQ,
                "messages": [AIMessage(content=msg)]}

    pend = _pendientes(state, respuestas)

    # ── Ficha completa → handoff_humano en la misma invocación ───────────
    if not pend:
        ack = sec["ack_completado"].format(
            nombre=_primer_nombre(respuestas.get("nombre")))
        return {"intake_activo": False,
                "intake_completado": True,
                "intake_respuestas": respuestas,
                # sin "response": quien responde es handoff_humano
                "messages": [AIMessage(content=ack)]}

    # ── Siguiente pregunta (hold si hubo error de validación) ────────────
    idx = next(i for i, q in enumerate(QUESTIONS) if q["id"] == pend[0]["id"])
    partes = ([f"⚠️ {error}"] if error else []) + \
             [_form_pregunta(state, pend[0], respuestas)]
    msg = "\n".join(partes)
    return {"intake_idx": idx,
            "intake_respuestas": respuestas,
            "response": msg,
            "messages": [AIMessage(content=msg)]}


def despues_de_intake(state: AgentState) -> str:
    """Edge condicional post-intake: ficha completa → ejecutar el handoff
    al tiro (la ejecutiva recibe resumen + ficha este mismo turno)."""
    return ROUTE_HANDOFF if state.get("intake_completado") else END
