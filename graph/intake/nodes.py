"""graph/intake/nodes.py — Acciones del sub-agente INTAKE.

v13 — Refactor crítico:
  - Rompe ciclo de importación (usa graph.utils en vez de graph.nodes).
  - Elimina _CAMPOS_SOLO_EXTRACCION; todo campo acepta extracción LLM + fallback
    validador directo (nombre ya no depende solo del LLM).
  - intake_exit: 'agendar' alinea CTA de cierre con routing del padre.
  - abandono/sin_consentimiento terminan el turno (intake_exit='end', closed=True).
  - Eventos de negocio estructurados.
  - Truncado defensivo de burbujas contra WA_TEXTO_MAX.
"""

import logging
import unicodedata

from langchain_core.messages import AIMessage

from core.contracts import EMAIL_RE, AgentState
from core.db_client import upsert_lead
from graph.intake.utils import _evento_intake, _pack
from graph.utils import (
    _ahora_iso, _cfg, _guardar_ai, _primer_nombre, _recent_messages,
    _telefono_cliente, _wa_link, _wa_link_diagnostico, _wa_link_display,
)

logger = logging.getLogger(__name__)

MAX_INTENTOS_CAMPO = 2

SALTAR = ("saltar", "salta", "skip", "omitir", "paso",
          "prefiero no", "prefiero no decir", "prefiero no responder",
          "no quiero responder", "no aplica", "n/a", "-")

RESPUESTAS_REINTENTAR = ("lo intento de nuevo", "intento de nuevo",
                         "intentar de nuevo", "intentarlo de nuevo",
                         "reintentar", "otra vez")
RESPUESTAS_HUMANO = ("hablar con humano", "hablar con una persona",
                     "hablar con un humano", "hablar con una ejecutiva",
                     "hablar con la ejecutiva")


DEFAULT_AP_IA = ("👋 ¡Hola! Soy el *asistente virtual* de Manzzo y Cía — "
                 "un sistema automatizado, no una persona, pero estoy aquí "
                 "para ayudarte.")
DEFAULT_AP_EXP = ("Te haré unas preguntas rápidas (unos 2 minutos) para "
                  "preparar tu caso.\nEn cualquier momento puedes hacer una "
                  "pregunta, pedir *hablar con una persona* o escribir "
                  "*pausa* para retomar después.")
DEFAULT_AVISO_SALTAR = "Si prefieres no responder, escribe *saltar*."
DEFAULT_ERROR_NO_SALTAR = ("Este dato sí lo necesito para poder derivar tu "
                           "caso 🙏 Si te complica, también puedes pedir "
                           "*hablar con una persona*.")
DEFAULT_OFERTA = ("{nombre}Este dato se nos está resistiendo 😅 ¿Qué "
                  "prefieres?\n• Te paso con una ejecutiva que lo toma "
                  "directamente contigo, o\n• lo intentamos una vez más.")
DEFAULT_REANUDA = "¡Seguimos donde quedamos 💪!"
DEFAULT_REANUDA_DUDA = ("Sobre lo que preguntaste {duda}: espero haberte "
                        "orientado 🙌 Seguimos donde quedamos:")
DEFAULT_CIERRE_LISTO = "🎉 ¡Listo{nombre}! Tu ficha quedó registrada."
DEFAULT_CIERRE_SLA = ("Una ejecutiva revisará tu caso y te contactará *hoy "
                      "dentro del horario hábil* (09:00–18:30). Quedó todo "
                      "anotado a tu nombre.")
DEFAULT_CIERRE_CTAS = ("¿Quieres asegurar la hora? Agenda ahora mismo 👇\n"
                       "Si prefieres que te contactemos nosotros, escríbele "
                       "directo a la ejecutiva: {wa_link}")
DEFAULT_DERIVA = ("Ningún problema{nombre} — te paso directamente con una "
                  "ejecutiva para que lo vean contigo 🤝\n{wa_link}")


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm(s) -> str:
    return _sin_tildes((s or "").strip().lower().rstrip("."))


def _v_text(minlen: int):
    def check(s):
        s = (s or "").strip()
        if len(s) < minlen:
            return None, f"menos de {minlen} caracteres"
        return s, None
    return check


def _v_email(s):
    s = (s or "").strip().lower().replace(" ", "")
    if not EMAIL_RE.fullmatch(s):
        return None, f"no calza EMAIL_RE: {s!r}"
    return s, None


def _v_si_no(s):
    t = _norm(s)
    if t in ("si", "sipo", "sip", "yes", "claro", "por supuesto",
             "afirmativo", "de acuerdo", "autorizo", "lo autorizo",
             "acepto", "si autorizo", "si, autorizo", "si por supuesto"):
        return True, None
    if t in ("no", "nop", "nope", "negativo", "para nada",
             "no autorizo", "no acepto", "no gracias", "no, gracias"):
        return False, None
    return None, f"no binario: {s!r}"


def _v_opciones(*opciones: str, aliases: dict | None = None):
    alias_map = {_norm(k): v for k, v in (aliases or {}).items()}

    def check(s):
        t = _norm(s)
        if t in alias_map:
            return alias_map[t], None
        if t.isdigit() and 1 <= int(t) <= len(opciones):
            return opciones[int(t) - 1], None
        exact = [o for o in opciones if _norm(o) == t]
        if exact:
            return exact[0], None
        subs = [o for o in opciones if t and t in _norm(o)]
        if len(subs) == 1:
            return subs[0], None
        return None, f"ambigua o fuera de lista: {s!r}"
    return check


def _v_nombre(s):
    s = (s or "").strip()
    palabras = [p for p in s.split() if p]
    if len(s) < 3 or not palabras:
        return None, "nombre no identificable"
    if len(palabras) > 6 or any(len(p) > 20 for p in palabras):
        return None, "parece frase, no nombre"
    return " ".join(p.capitalize() for p in palabras), None


def _botones(*id_title) -> dict:
    return {"kind": "buttons",
            "buttons": [{"id": i, "title": t} for i, t in id_title]}


_INTERACTIVE_CONSENTIMIENTO = _botones(("si", "Sí, autorizo"),
                                       ("no", "No autorizo"))
_BOTONES_OFERTA = _botones(("humano", "Hablar con humano"),
                           ("retry", "Lo intento de nuevo"))
_BOTON_AGENDAR = _botones(("agendar", "Agendar una hora"))

_ETAPA_TITULOS = ["Aún no inicio nada", "Me demandaron / avisaron",
                  "Causa en curso", "Hay sentencia previa"]
_INTERACTIVE_ETAPA = {
    "kind": "list", "cta": "Ver opciones", "title": "Estado del caso",
    "options": [{"id": f"etapa_{i + 1}", "title": t}
                for i, t in enumerate(_ETAPA_TITULOS)],
}

QUESTIONS: list[dict] = [
    {"id": "consentimiento_datos", "requerida": True, "extractable": False,
     "label": "tu autorización",
     "prompt": ("Para poder revisar tu caso y que un abogado te contacte, "
                "necesito tu *autorización para usar tus datos* (Ley N° "
                "21.719). Los usaremos solo para gestionar tu consulta. "
                "¿Me la das?"),
     "retry_prompt": ("Perdona, necesito un *sí* o un *no* claros: "
                      "¿autorizas el uso de tus datos solo para gestionar "
                      "tu consulta? Puedes usar los botones 👇"),
     "ack": "Gracias ✓",
     "interactive": _INTERACTIVE_CONSENTIMIENTO,
     "validate": _v_si_no},

    {"id": "nombre", "requerida": True,
     "label": "tu nombre",
     "prompt": ("Para partir, ¿me dices tu *nombre completo*, tal como "
                "aparece en tu cédula?"),
     "retry_prompt": ("Perdona, no logré distinguir tu nombre en el mensaje "
                      "🙈 Escríbemelo solo con nombre y apellido — por "
                      "ejemplo: *María Pérez Soto*"),
     "ack": None,
     "validate": _v_nombre},

    {"id": "email", "requerida": True,
     "label": "tu correo",
     "prompt": ("¿A qué *correo* te enviamos el seguimiento de tu caso? "
                "(lo usaremos solo para eso)"),
     "retry_prompt": ("Ese correo no me cuadra 🤔 A veces se pega un "
                      "espacio o falta la '@'. ¿Me lo repites? Algo como "
                      "*maria@gmail.com*"),
     "ack": "Listo, anoté tu correo ✓",
     "validate": _v_email},

    {"id": "situacion_actual", "requerida": True,
     "label": "tu relato",
     "prompt": ("Cuéntame brevemente qué está pasando con tu caso, "
                "con tus palabras:"),
     "retry_prompt": ("Me falta un poco más para derivarte bien: ¿*qué "
                      "ocurrió*, *quiénes* están involucrados y *desde "
                      "cuándo*?"),
     "ack": None,
     "validate": _v_text(20)},

    {"id": "etapa_proceso", "requerida": False,
     "label": "la etapa del caso",
     "prompt": ("¿En qué estado está tu caso hoy?\n"
                "  1) Aún no he iniciado acciones\n"
                "  2) Fui demandado(a) o notificado(a)\n"
                "  3) Hay una causa judicial en curso\n"
                "  4) Ya existe sentencia o acuerdo previo\n"
                "(elige de la lista o respóndeme con el número)"),
     "retry_prompt": ("No logré ubicarlo entre las opciones 😅 Elige de la "
                      "lista o respóndeme solo con el número (1 al 4):"),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_ETAPA,
     "validate": _v_opciones(
         "aún no inicio nada", "me demandaron o me notificaron",
         "causa en curso", "sentencia o acuerdo previo",
         aliases={"aún no inicio nada": "aún no inicio nada",
                  "me demandaron / avisaron": "me demandaron o me notificaron",
                  "causa en curso": "causa en curso",
                  "hay sentencia previa": "sentencia o acuerdo previo"}),
     "label_map": {"sin_inicio": "aún no inicio nada",
                   "demandado": "me demandaron o me notificaron",
                   "causa_en_curso": "causa en curso",
                   "sentencia_previa": "sentencia o acuerdo previa"}},
]

_BY_ID = {q["id"]: q for q in QUESTIONS}


def _aplicables(state: AgentState, respuestas: dict | None = None) -> list[dict]:
    resp = respuestas if respuestas is not None else (state.get("intake_respuestas") or {})
    return [q for q in QUESTIONS
            if not q.get("skip_if", lambda s, r=None: False)(state, resp)]


def _pendientes(state: AgentState, respuestas: dict) -> list[dict]:
    hechas = {q["id"] for q in _aplicables(state, respuestas)
              if q["id"] in respuestas}
    return [q for q in _aplicables(state, respuestas) if q["id"] not in hechas]


def _form_pregunta(state: AgentState, q: dict, respuestas: dict,
                   nombre: str | None = None) -> str:
    sec = _cfg()["intake"]
    total = len(_aplicables(state, respuestas))
    restantes = len(_pendientes(state, respuestas))
    if restantes <= 1:
        head = f"Ya casi{', ' + nombre if nombre else ''} 🙌 Solo me falta esto:\n"
    else:
        head = f"_{total - restantes + 1} de {total}_\n"
    body = q["prompt"]
    if not q["requerida"]:
        body += "\n" + sec.get("aviso_saltar", DEFAULT_AVISO_SALTAR)
    return head + body


def _idx_of(qid: str) -> int:
    return next(i for i, q in enumerate(QUESTIONS) if q["id"] == qid)


def _interactive_de(q: dict) -> dict | None:
    return q.get("interactive")


def _ack_campo(qid: str, respuestas: dict) -> str:
    nombre = _primer_nombre(respuestas.get("nombre"))
    if qid == "nombre":
        return f"Perfecto, {nombre} ✓" if nombre else "Perfecto ✓"
    if qid == "situacion_actual":
        resumen = str(respuestas.get(qid) or "").strip()
        if len(resumen) > 180:
            resumen = resumen[:177].rstrip() + "…"
        return (f"Gracias por contármelo{', ' + nombre if nombre else ''}; "
                "sé que no siempre es fácil 🙏 "
                f"Si te entendí bien: “{resumen}”\n"
                "Si se me escapó algo, dímelo; si no, seguimos 👇")
    tpl = _BY_ID[qid].get("ack") or "Anotado ✓"
    return tpl.format(nombre=nombre or "")


def _extras_turno(q_actual: dict, antes: dict, despues: dict) -> list[str]:
    return [qid for qid in _BY_ID
            if qid != q_actual["id"]
            and despues.get(qid) is not None
            and despues.get(qid) != antes.get(qid)]


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " y " + labels[-1]


def _acks_turno(q_actual: dict, antes: dict, respuestas: dict) -> list[str]:
    acks: list[str] = []
    qid = q_actual["id"]
    if respuestas.get(qid) is not None and respuestas.get(qid) != antes.get(qid):
        acks.append(_ack_campo(qid, respuestas))
    elif qid in respuestas and respuestas[qid] is None and qid not in antes:
        acks.append("Ningún problema, lo dejamos así ✓")
    extras = _extras_turno(q_actual, antes, respuestas)
    if extras:
        acks.append("De paso ya tengo "
                    + _join_labels([_BY_ID[q]["label"] for q in extras])
                    + " 👌")
    return acks


def _aplicar_extraccion(respuestas: dict, decision: dict) -> None:
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


def iniciar_ficha(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas: dict = {}
    tel = _telefono_cliente(state)
    if tel:
        respuestas["telefono"] = tel

    _aplicar_extraccion(respuestas, state.get("intake_decision") or {})

    pend = _pendientes(state, respuestas)
    q0 = pend[0]

    burbujas = [
        sec.get("apertura_ia") or (sec.get("apertura") or DEFAULT_AP_IA),
        sec.get("apertura_expectativa") or DEFAULT_AP_EXP,
        _form_pregunta(state, q0, respuestas),
    ]
    pack = _pack(burbujas, interactive=_interactive_de(q0))
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_iniciado", state, pregunta=q0["id"])
    return {
        **pack,
        "intake_activo": True,
        "intake_idx": _idx_of(q0["id"]),
        "intake_respuestas": respuestas,
        "intake_attempts": 0,
        "intake_exit": None,
        "intake_resume": False,
        "intake_started_en": _ahora_iso(),
        "intake_completado": False,
        "intake_decision": None,
        "intake_stage": "preguntando",
        "intake_oferta_qid": None,
    }


def procesar_respuesta(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas = dict(state.get("intake_respuestas") or {})
    antes = dict(respuestas)
    decision = state.get("intake_decision") or {}
    attempts = int(state.get("intake_attempts") or 0)
    raw = (state.get("query") or "").strip()

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas)
    q_actual = pend[0]
    nombre = _primer_nombre(respuestas.get("nombre"))
    ya_ofrecido = state.get("intake_oferta_qid") == q_actual["id"]

    if ya_ofrecido and _norm(raw) in RESPUESTAS_REINTENTAR:
        pack = _pack([f"Dale 💪 {q_actual['retry_prompt']}"],
                     interactive=_interactive_de(q_actual))
        _guardar_ai(state, pack["response"])
        return {
            **pack,
            "intake_respuestas": respuestas,
            "intake_attempts": attempts,
            "intake_decision": None,
        }

    error_key = None
    if _norm(raw) in SALTAR:
        if q_actual["requerida"]:
            error_key = "no_skip"
        else:
            respuestas[q_actual["id"]] = None
    else:
        _aplicar_extraccion(respuestas, decision)
        if q_actual["id"] not in respuestas:
            val, err = q_actual["validate"](raw)
            if err:
                logger.info("[intake] validador rechazó %s: %s",
                            q_actual["id"], err)
                error_key = "retry"
            else:
                respuestas[q_actual["id"]] = val

    if respuestas.get("consentimiento_datos") is False:
        return sin_consentimiento(state, respuestas)

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas)

    if error_key:
        extras = _extras_turno(q_actual, antes, respuestas)
        if not extras:
            attempts += 1

        if (not extras and q_actual["requerida"]
                and attempts >= MAX_INTENTOS_CAMPO):
            if ya_ofrecido:
                logger.info("[intake] reintento agotado en '%s' → "
                            "derivación parcial", q_actual["id"])
                return _derivar_parcial(state, respuestas)
            msg_oferta = sec.get("oferta_salida", DEFAULT_OFERTA).format(
                nombre=f"{nombre}, " if nombre else "")
            pack = _pack([msg_oferta], interactive=_BOTONES_OFERTA)
            _guardar_ai(state, pack["response"])
            return {
                **pack,
                "intake_respuestas": respuestas,
                "intake_attempts": attempts,
                "intake_decision": None,
                "intake_oferta_qid": q_actual["id"],
            }

        burbujas = []
        if extras:
            burbujas.append("De paso ya tengo " + _join_labels(
                [_BY_ID[q]["label"] for q in extras]) + " 👌")
        if error_key == "no_skip":
            burbujas.append(sec.get("error_no_saltar",
                                    DEFAULT_ERROR_NO_SALTAR))
        else:
            burbujas.append(q_actual.get("retry_prompt")
                            or q_actual["prompt"])
        pack = _pack(burbujas, interactive=_interactive_de(q_actual))
        _guardar_ai(state, pack["response"])
        return {
            **pack,
            "intake_respuestas": respuestas,
            "intake_attempts": attempts,
            "intake_decision": None,
        }

    _persistir_parcial(state, respuestas)

    siguiente = pend[0]
    pack = _pack(
        _acks_turno(q_actual, antes, respuestas)
        + [_form_pregunta(state, siguiente, respuestas, nombre)],
        interactive=_interactive_de(siguiente),
    )
    _guardar_ai(state, pack["response"])
    return {
        **pack,
        "intake_idx": _idx_of(siguiente["id"]),
        "intake_respuestas": respuestas,
        "intake_attempts": 0,
        "intake_oferta_qid": None,
        "intake_decision": None,
    }


def _completar_ficha(state: AgentState, respuestas: dict) -> AgentState:
    sec = _cfg()["intake"]
    state_con_ficha = {
        **state, "intake_respuestas": respuestas, "intake_completado": True,
    }
    nombre = _primer_nombre(respuestas.get("nombre"))
    link_humano = _wa_link_diagnostico(state_con_ficha)

    burbujas = [
        sec.get("cierre_listo", DEFAULT_CIERRE_LISTO).format(
            nombre=f", {nombre}" if nombre else ""),
        sec.get("cierre_sla", DEFAULT_CIERRE_SLA),
        sec.get("cierre_ctas", DEFAULT_CIERRE_CTAS).format(wa_link=link_humano),
    ]
    pack = _pack(burbujas, interactive=_BOTON_AGENDAR)

    try:
        upsert_lead(
            thread_id=state.get("thread_id", ""),
            intake_respuestas=respuestas,
            category=state.get("category"),
            completed=True,
        )
    except Exception as e:
        logger.warning("upsert_lead (completado) falló: %s", e)

    _guardar_ai(state, pack["response"])
    _evento_intake("intake_completado", state,
                   respuestas_keys=list(respuestas.keys()),
                   tiene_email=bool(respuestas.get("email")))
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": True,
        "intake_respuestas": respuestas,
        "intake_exit": "agendar",
        "intake_decision": None,
        "intake_stage": "completado",
        "intake_oferta_qid": None,
    }


def pausar_ficha(state: AgentState) -> AgentState:
    """El cliente pide pausa: detener la ficha y esperar al próximo mensaje."""
    pack = _pack(["Perfecto, quedo aquí esperando. Cuando quieras retomar, "
                  "solo escríbeme 🤙"])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_pausado", state)
    return {
        **pack,
        "intake_activo": True,
        "intake_resume": False,
        "intake_exit": "pausa",
        "intake_stage": "pausado",
        "intake_decision": None,
        "intake_oferta_qid": None,
    }


def pausar_para_faq(state: AgentState) -> AgentState:
    logger.info("[intake] duda lateral → pausa a FAQ")
    _evento_intake("intake_duda_lateral", state)
    return {
        "intake_exit": "faq",
        "intake_resume": True,
        "intake_stage": "pausado",
        "intake_decision": None,
        "response_interactive": None,
    }


def _ultima_duda(state: AgentState) -> str | None:
    actual = (state.get("query") or "").strip()
    for m in reversed(_recent_messages(state, k=8)):
        if getattr(m, "type", "") != "human":
            continue
        txt = (m.content or "").strip()
        if not txt or txt == actual:
            continue
        if "?" not in txt and "¿" not in txt:
            return None
        frag = txt if len(txt) <= 90 else txt[:87].rstrip() + "…"
        return f"«{frag}»"
    return None


def reanudar(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas = dict(state.get("intake_respuestas") or {})
    pend = _pendientes(state, respuestas)
    base = {
        "intake_exit": None,
        "intake_resume": False,
        "intake_decision": None,
        "intake_stage": "preguntando",
        "intake_oferta_qid": None,
        "response_interactive": None,
    }
    if not pend or not state.get("intake_activo"):
        return base

    q = pend[0]
    nombre = _primer_nombre(respuestas.get("nombre"))
    duda = _ultima_duda(state)
    puente = (sec.get("reanudar_duda", DEFAULT_REANUDA_DUDA).format(duda=duda)
              if duda else sec.get("reanudar", DEFAULT_REANUDA))

    pack = _pack([puente, _form_pregunta(state, q, respuestas, nombre)],
                 interactive=_interactive_de(q))
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_reanudado", state, pregunta=q["id"])
    return {**base, **pack}


def derivar_parcial(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    return _derivar_parcial(state, respuestas)


def _derivar_parcial(state: AgentState, respuestas: dict) -> AgentState:
    nombre = _primer_nombre(respuestas.get("nombre"))
    msg = _cfg()["intake"].get("derivacion_parcial", DEFAULT_DERIVA).format(
        nombre=f", {nombre}" if nombre else "",
        wa_link=_wa_link_display(
            f"Hola, soy {respuestas.get('nombre', '') or 'un cliente del asistente'}, "
            "prefiero continuar directamente con la ejecutiva.",
            display="Hablar con un humano",
        )
    )
    _persistir_parcial(state, respuestas)
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_derivacion_parcial", state,
                   respuestas_keys=list(respuestas.keys()))
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": respuestas,
        "intake_exit": "handoff",
        "intake_stage": "completado",
        "intake_decision": None,
        "intake_oferta_qid": None,
    }


def abandonar_ficha(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    _persistir_parcial(state, respuestas)
    msg = _cfg()["intake"]["cierre_abandono"].format(
        wa_link=_wa_link_display(display="Hablar con un humano")
    )
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_abandonado", state,
                   respuestas_keys=list(respuestas.keys()))
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": False,
        "intake_exit": "end",
        "intake_stage": "abandonado",
        "intake_decision": None,
        "intake_oferta_qid": None,
        "closed": True,
    }


def sin_consentimiento(state: AgentState, respuestas: dict | None = None) -> AgentState:
    msg = _cfg()["intake"]["sin_consentimiento"].format(
        wa_link=_wa_link_display(display="Hablar con un humano")
    )
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_sin_consentimiento", state)
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": (respuestas if respuestas is not None
                              else state.get("intake_respuestas")),
        "intake_exit": "end",
        "intake_decision": None,
        "intake_stage": "sin_consentimiento",
        "intake_oferta_qid": None,
        "closed": True,
    }
