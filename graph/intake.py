"""graph/intake.py — Intake proactivo de lead + caso (patrón Interview Intake).

v6 — Registro formal + robustez de parseo. El agente HABLA como mesa de
     partes de un estudio jurídico (trato de usted, sin emojis), pero
     ENTIENDE el español real de WhatsApp ("sipo", tildes omitidas,
     números para elegir opciones).

Cambios v6:
- Todos los prompts y mensajes de error en registro formal.
- _sin_tildes(): las validaciones eran sensibles a tildes ("recomendacion"
  jamás matcheaba "recomendación"). Normalización NFD en _v_si_no/_v_opciones.
- FIX: _aplicar_extraccion descartaba valores bool del LLM — str(True) es
  "True" y _v_si_no no lo reconocía → hijos_menores extraído por la costura
  NUNCA se guardaba. Ahora los bools se aceptan directo.
- FIX: "saltar" en pregunta REQUERIDA caía al validador como texto y podía
  quedar guardado (ej: nombre="saltar", 6 caracteres). Ahora: hold + error.
- skip_if recibe (state, respuestas): las respuestas recién validadas del
  turno habilitan skips condicionales sin esperar al siguiente turno.
- Extractor LLM: prompt con reglas por campo; typo corregido ("infieras").

Arquitectura (sin cambios):
- QUESTIONS: registry de preguntas con prompt, validador puro, flag
  `requerida` y `skip_if` condicional (≡ lista GRAPH de tv_merge).
- Validadores deterministas: (valor, None) o (None, error tipado). Si fallan,
  el estado se MANTIENE en la pregunta actual — jamás se fabrica un dato de
  persona (política cero-alucinación sobre datos personales).
- Costura LLM ("LLM layer in FRONT of submit"): _extraer_intake mapea texto
  libre → varios campos a la vez; los validadores siguen siendo la verdad.
- Al completar → edge `despues_de_intake` envía a handoff_humano en la MISMA
  invocación: la ejecutiva recibe resumen + ficha (JSON + WhatsApp).

Cumplimiento: `consentimiento_datos` es extractable=False — debe responderse
directamente (sí/no), nunca inferirse con el LLM (Ley 21.719).
"""
import logging
import re
import unicodedata

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from core.contracts import (
    EMAIL_RE, AgentState, IntakeExtract, ROUTE_FAQ, ROUTE_HANDOFF,
)
from core.llm import with_structured_output
from graph.nodes import _ahora_iso, _cfg, _primer_nombre, _telefono_cliente

logger = logging.getLogger(__name__)

SALTAR = ("saltar", "salta", "skip", "omitir", "paso",
          "prefiero no", "prefiero no decir", "prefiero no responder",
          "no quiero responder", "no aplica", "n/a", "-")


# ---------------------------------------------------------------------------
# NORMALIZACIÓN
# ---------------------------------------------------------------------------
def _sin_tildes(s: str) -> str:
    """'recomendación' → 'recomendacion'. Los usuarios de WhatsApp omiten
    tildes constantemente; las comparaciones no pueden depender de ellas."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------------------
# VALIDADORES DETERMINISTAS  (≡ _v_text/_v_number/_v_yesno de tv_merge)
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


def _v_telefono(s):
    """Normaliza móvil chileno a E.164 (+569XXXXXXXX)."""
    t = re.sub(r"[\s\-.()+]", "", (s or ""))
    if t.startswith("56"):
        t = t[2:]
    if t.isdigit() and len(t) == 9 and t.startswith("9"):
        return "+56" + t, None
    return None, ("el número debe ser un móvil chileno de 9 dígitos "
                  "(ejemplo: +56 9 1234 5678)")


def _v_si_no(s):
    """El agente habla formal, pero debe entender el español real de WhatsApp."""
    t = _sin_tildes((s or "").strip().lower())
    if t in ("si", "sipo", "sip", "yes", "y", "claro", "ok", "dale",
             "por supuesto", "afirmativo", "de acuerdo", "autorizo"):
        return True, None
    if t in ("no", "nop", "n", "nope", "negativo", "para nada", "no autorizo"):
        return False, None
    return None, "por favor, responda únicamente sí o no"


def _v_opciones(*opciones: str):
    """≡ _match_choice de tv_merge: exacto > substring único > número."""
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
# REGISTRY DE PREGUNTAS  (≡ GRAPH de tv_merge)
#
# Nota: las strings de _v_opciones / label_map son las formas NATURALES en
# que responde la gente (y lo que queda guardado en la ficha); el formato
# formal vive en el prompt visible. Cambiarlas exige actualizar label_map.
# ---------------------------------------------------------------------------
_FAMILIA = {"pension_alimentos", "rebaja_pension", "regimen_visitas",
            "terminacion_pension", "divorcio", "compensacion_economica",
            "medidas_apremio"}

QUESTIONS: list[dict] = [
    {"id": "nombre", "requerida": True,
     "prompt": "Indique su nombre completo, tal como aparece en su cédula de identidad:",
     "validate": _v_text(3)},

    {"id": "email", "requerida": True,
     "prompt": ("Indique su correo electrónico (se utilizará exclusivamente "
                "para el seguimiento de su caso):"),
     "validate": _v_email},

    {"id": "telefono", "requerida": False,
     "prompt": ("¿Dispone de un teléfono de contacto directo? "
                "Formato esperado: +56 9 1234 5678."),
     "validate": _v_telefono,
     "skip_if": lambda st, r=None: _telefono_cliente(st) is not None},  # en prod ya lo tenemos

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

    {"id": "hijos_menores", "requerida": False,
     "prompt": ("¿Existen hijos menores de edad involucrados en este asunto? "
                "(responda sí o no)"),
     "validate": _v_si_no,
     "skip_if": lambda st, r=None: st.get("category") not in _FAMILIA},

    {"id": "comuna", "requerida": False,
     "prompt": "¿En qué comuna reside? Esto permite coordinar una eventual atención presencial.",
     "validate": _v_text(2)},

    {"id": "horario_contacto", "requerida": False,
     "prompt": "¿En qué horario prefiere que la ejecutiva lo contacte?\n"
               "  1) Mañana (9-13)\n  2) Tarde (14-18)\n  3) Indiferente",
     "validate": _v_opciones("mañana (9-13)", "tarde (14-18)", "indiferente"),
     "label_map": {"manana": "mañana (9-13)", "tarde": "tarde (14-18)",
                   "indiferente": "indiferente"}},

    {"id": "como_nos_conocio", "requerida": False,
     "prompt": ("¿A través de qué medio llegó a nosotros?\n"
                "  1) Google\n  2) Instagram\n  3) TikTok\n"
                "  4) Recomendación\n  5) Otro"),
     "validate": _v_opciones("google", "instagram", "tiktok",
                             "recomendación", "otro"),
     "label_map": {"google": "google", "instagram": "instagram",
                   "tiktok": "tiktok", "recomendacion": "recomendación",
                   "otro": "otro"}},

    {"id": "consentimiento_datos", "requerida": True, "extractable": False,
     # contracts.py ya cita la Ley 21.719; si prefieres citar la ley
     # actualmente vigente, cambia por "Ley N° 19.628 sobre protección
     # de la vida privada".
     "prompt": ("Para finalizar, conforme a la Ley N° 21.719 de protección "
                "de datos personales: ¿autoriza usted a Manzzo y Cía a "
                "tratar los datos entregados exclusivamente para gestionar "
                "su consulta y contactarlo(a)? (responda sí o no)"),
     "validate": _v_si_no},
]

_BY_ID = {q["id"]: q for q in QUESTIONS}


# ---------------------------------------------------------------------------
# HELPERS DE PROGRESO (≡ IntakeSession.current/progress)
# ---------------------------------------------------------------------------
def _aplicables(state: AgentState, respuestas: dict | None = None) -> list[dict]:
    """Preguntas que aplican ahora. skip_if recibe (state, respuestas):
    las respuestas VALIDADAS de este turno ya habilitan skips condicionales
    (antes se leía state['intake_respuestas'], desactualizado por un turno)."""
    resp = respuestas if respuestas is not None \
        else (state.get("intake_respuestas") or {})
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


# ---------------------------------------------------------------------------
# COSTURA LLM: texto libre → varios campos (los validadores mandan)
# ---------------------------------------------------------------------------
def _extraer_intake(state: AgentState, campo_actual: str) -> IntakeExtract:
    """1 llamada LLM. Nunca lanza: ante error devuelve esquema vacío y el
    flujo cae al validador determinista del campo actual."""
    extractor = ChatPromptTemplate.from_messages([
        ("system",
         "Extraes datos para la ficha de cliente de un estudio jurídico "
         "chileno. Devuelve SOLO el esquema, con null en todo dato que no "
         "aparezca EXPLÍCITAMENTE en el mensaje. Nunca infieras ni completes "
         "datos por contexto.\n"
         f"El cliente probablemente está respondiendo sobre: '{campo_actual}'.\n"
         "Reglas:\n"
         "· nombre: tal como lo escribe el cliente.\n"
         "· email: solo si aparece un correo con @ y dominio.\n"
         "· telefono: normaliza móviles chilenos a +569XXXXXXXX.\n"
         "· situacion_actual: resume los hechos del cliente sin agregar nada.\n"
         "· etapa_proceso, horario_contacto y como_nos_conocio: usa "
         "EXACTAMENTE las etiquetas del esquema.\n"
         "· hijos_menores: true solo si menciona explícitamente hijos menores "
         "de edad.\n"
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


def _aplicar_extraccion(respuestas: dict, extra: IntakeExtract) -> None:
    """Cada valor extraído pasa por el validador de su pregunta. Solo se
    guardan valores VÁLIDOS (y pueden corregir respuestas anteriores)."""
    for qid, qdef in _BY_ID.items():
        if not qdef.get("extractable", True):
            continue                       # consentimiento: solo respuesta directa
        raw = getattr(extra, qid, None)
        if raw is None:
            continue
        if isinstance(raw, bool):
            # FIX v6: str(True) → "True" jamás pasaba por _v_si_no, así que
            # hijos_menores extraído por la costura se descartaba siempre.
            respuestas[qid] = raw
            continue
        if "label_map" in qdef:            # menús: literal del esquema → display
            raw = qdef["label_map"].get(str(raw), None)
            if raw is None:
                continue
        val, _err = qdef["validate"](raw)
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

    if raw.lower() in SALTAR:
        if q_actual["requerida"]:
            # FIX v6: antes "saltar" caía al validador como texto y podía
            # quedar guardado (ej: nombre="saltar"). Ahora hold + error.
            error = ("este dato es obligatorio para poder derivar su caso; "
                     "no puede omitirse")
        else:
            respuestas[q_actual["id"]] = None          # ≡ "skip" de tv_merge
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
    partes = ([f"No fue posible registrar su respuesta: {error}"] if error else []) \
             + [_form_pregunta(state, pend[0], respuestas)]
    msg = "\n".join(partes)
    return {"intake_idx": idx,
            "intake_respuestas": respuestas,
            "response": msg,
            "messages": [AIMessage(content=msg)]}


def despues_de_intake(state: AgentState) -> str:
    """Edge condicional post-intake: ficha completa → ejecutar el handoff
    al tiro (la ejecutiva recibe resumen + ficha este mismo turno)."""
    return ROUTE_HANDOFF if state.get("intake_completado") else END
