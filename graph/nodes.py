"""graph/nodes.py — Nodos transversales del agente (Manzzo y Cía).

v16.1 — Fix #5 (pausa real en intake):
  - route_post_intake reconoce intake_exit == "pausa" y termina el turno.
  - analyze_sentiment detecta intake pausado y reanuda al próximo mensaje.
  - Mantiene FIX #2 (email) y FIX #3 (SheetDB con consentimiento).

v16 — FIX #2 + FIX #3 en handoff_humano:
  - Notificación por email a ejecutiva cuando se deriva un lead.
  - SheetDB sincroniza solo con consentimiento explícito.
  - Flag notificacion_pendiente combina WhatsApp + email.

v15 — Integración del intake v11 (UX fluida) en el grafo padre:
  - analyze_sentiment: excepción 0-LLM en hilo cerrado — CTA agendar.
  - handoff_humano silencioso cuando intake ya despidió el turno.
  - response_bubbles / response_interactive one-shot.
  - _RESET_INTAKE incluye intake_oferta_qid.
  - TEMPLATE_ARGS valida subconjunto de placeholders.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import yaml
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END


from core.contracts import (
    AgentState, AnalisisResult,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF, ROUTE_INTAKE,
    VALID_ROUTES, fallback_analisis,
)
from core.db_client import (
    insert_escalation, insert_message, upsert_conversation, upsert_lead,
)
from core.llm import LLM, with_structured_output
from core.notifications import notificar_email
from tools.notify_whatsapp import WHATSAPP_EJECUTIVA, notificar_escalamiento

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "core" / "prompts.yaml"
ESCALATIONS_DIR = Path("escalations")

AGENDA_CAPTURA_TTL_HORAS = 24
INTAKE_TTL_HORAS = 24

WA_TEXTO_MAX = 900

_AGENDAR_CTA = ("agendar una hora", "agendar", "agendar ahora",
                "agendar hora", "quiero agendar", "quiero agendar una hora")

REQUIRED_KEYS = {
    "atencion": {"disclosure", "cta_agendar", "faq_system_prompt", "tonos",
                 "summary_system_prompt", "fuera_dominio_message",
                 "handoff_message", "hilo_cerrado_message",
                 "agenda_pedir_nombre", "agenda_pedir_email",
                 "agenda_pedir_modalidad", "agenda_proponer_slots",
                 "agenda_reintento_slots", "agenda_sin_horarios",
                 "agenda_confirmada", "agenda_slot_ocupado",
                 "agenda_eleccion_ambigua", "agenda_abortado",
                 "agenda_lateral_system"},
    "classification": {"system_prompt", "few_shot_examples"},
    "intake": {"aviso_saltar", "sin_consentimiento", "reanudar",
               "derivacion_parcial", "cierre_abandono"},
}

TEMPLATE_ARGS = {
    "atencion.faq_system_prompt": {"disclosure", "tono", "context"},
    "atencion.fuera_dominio_message": {"disclosure"},
    "atencion.handoff_message": {"disclosure", "whatsapp_ejecutiva", "thread_id"},
    "atencion.hilo_cerrado_message": {"whatsapp_ejecutiva", "thread_id"},
    "atencion.cta_agendar": set(),
    "atencion.agenda_pedir_nombre": set(),
    "atencion.agenda_pedir_email": set(),
    "atencion.agenda_pedir_modalidad": set(),
    "atencion.agenda_proponer_slots": {"nombre", "opciones"},
    "atencion.agenda_reintento_slots": {"opciones"},
    "atencion.agenda_sin_horarios": set(),
    "atencion.agenda_confirmada": {"nombre", "fecha", "hora_inicio",
                                   "hora_fin", "modalidad_linea", "html_link"},
    "atencion.agenda_slot_ocupado": {"opciones"},
    "atencion.agenda_eleccion_ambigua": {"opciones"},
    "atencion.agenda_abortado": set(),
    "atencion.agenda_lateral_system": {"opciones"},
    "intake.apertura_ia": set(),
    "intake.apertura_expectativa": set(),
    "intake.aviso_saltar": set(),
    "intake.sin_consentimiento": {"wa_link"},
    "intake.reanudar": set(),
    "intake.reanudar_duda": {"duda"},
    "intake.oferta_salida": {"nombre"},
    "intake.error_no_saltar": set(),
    "intake.cierre_listo": {"nombre"},
    "intake.cierre_sla": set(),
    "intake.cierre_ctas": {"wa_link"},
    "intake.derivacion_parcial": {"wa_link", "nombre"},
    "intake.cierre_abandono": {"wa_link"},
}
_PLACEHOLDER_RE = re.compile(r"{(\w+)}")

_ABORT_BOOKING = ("no quiero", "mejor no", "olvídalo", "olvidalo",
                  "dejalo", "déjalo", "ya no me interesa")

_EMAIL_BUSCA = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NOMBRE_RE = re.compile(
    r"(?:soy|me llamo|mi nombre es)\s+"
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}"
    r"(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}){0,3})",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _cfg() -> dict:
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [f"{sec}.{key}" for sec, keys in REQUIRED_KEYS.items()
               for key in keys if key not in (cfg.get(sec) or {})]
    if missing:
        raise KeyError(f"prompts.yaml incompleto — faltan: {missing}")
    bad = []
    for dotted, esperados in TEMPLATE_ARGS.items():
        seccion, key = dotted.split(".", 1)
        tpl = (cfg.get(seccion) or {}).get(key) or ""
        encontrados = set(_PLACEHOLDER_RE.findall(tpl))
        if not encontrados.issubset(esperados):
            bad.append(f"{dotted}: plantilla usa "
                       f"{sorted(encontrados - esperados)} que el nodo no "
                       f"entrega (disponibles: {sorted(esperados)})")
    if bad:
        raise ValueError("prompts.yaml — placeholders desalineados:\n" + "\n".join(bad))
    return cfg


def _wa_link(texto: str = "") -> str:
    num = re.sub(r"\D", "", WHATSAPP_EJECUTIVA or "")
    base = f"https://wa.me/{num}"
    return f"{base}?text={quote(texto)}" if texto else base


def _wa_link_display(texto: str = "", display: str = "Hablar con un humano") -> str:
    return f"[{display}]({_wa_link(texto)})"


def _wa_texto_intake(state: AgentState) -> str:
    r = state.get("intake_respuestas") or {}
    nombre = r.get("nombre")

    lineas = [
        f"Hola, soy {nombre}." if nombre
        else "Hola, vengo del asistente virtual de Manzzo y Cía.",
        ("Acabo de completar mi registro. Resumen de mi caso:"
         if state.get("intake_completado") else
         "Prefiero hablar directamente con una ejecutiva. Datos que alcancé a registrar:"),
    ]
    area = state.get("case_category") or state.get("category")
    if area:
        lineas.append(f"• Área: {area}")
    if r.get("email"):
        lineas.append(f"• Correo: {r['email']}")
    if r.get("situacion_actual"):
        sit = str(r["situacion_actual"])
        if len(sit) > 280:
            sit = sit[:277].rstrip() + "…"
        lineas.append(f"• Mi situación: {sit}")
    if r.get("etapa_proceso"):
        lineas.append(f"• Etapa de mi caso: {r['etapa_proceso']}")
    if r.get("comuna"):
        lineas.append(f"• Comuna: {r['comuna']}")
    if r.get("hijos_menores") is True:
        lineas.append("• Tengo hijos menores de edad")
    if r.get("horario_contacto"):
        lineas.append(f"• Horario preferido: {r['horario_contacto']}")

    tid = state.get("thread_id")
    if tid:
        lineas.append(f"(ID de mi atención: {tid})")

    txt = "\n".join(lineas)
    if len(txt) > WA_TEXTO_MAX:
        txt = txt[:WA_TEXTO_MAX - 1].rstrip() + "…"
    return txt


def _wa_link_diagnostico(state: AgentState) -> str:
    return _wa_link_display(_wa_texto_intake(state), display="Hablar con un humano")


def _wa_link_cliente(state: AgentState) -> str:
    if state.get("intake_respuestas"):
        return _wa_link_diagnostico(state)
    return _wa_link_display(
        "Hola, vengo del asistente virtual de Manzzo y Cía.",
        display="Hablar con un humano",
    )


def _recent_messages(state: AgentState, k: int = 6) -> list:
    return state.get("messages", [])[-k:]


def _format_few_shots(examples: list[dict]) -> str:
    return "\n\n".join(
        f'ENTRADA: "{ex["input"]}"\n'
        f"sentiment={ex['sentiment']} | urgency={ex['urgency']} | "
        f"intent={ex['intent']} | category={ex['category']} | reason={ex['reason']}"
        for ex in examples
    )


def _ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(iso_ts: str | None) -> datetime | None:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _expirado(iso_ts: str | None, ttl_horas: int) -> bool:
    dt = _parse_dt(iso_ts)
    return bool(dt) and (datetime.now(timezone.utc) - dt > timedelta(hours=ttl_horas))


def _primer_nombre(nombre: str | None) -> str:
    partes = (nombre or "").split()
    return partes[0] if partes else ""


def _norm_simple(s: str) -> str:
    return (s or "").strip().lower().rstrip(".")


def _transcript(state: AgentState) -> list[dict]:
    return [{"rol": m.type, "contenido": m.content}
            for m in state.get("messages", [])]


def _transcript_resumen(state: AgentState, max_chars: int = 6000) -> str:
    txt = "\n".join(f"{t['rol']}: {t['contenido']}" for t in _transcript(state))
    if len(txt) > max_chars:
        txt = "…[inicio de la conversación omitido]\n" + txt[-max_chars:]
    return txt


def _ficha_intake(state: AgentState) -> str:
    r = state.get("intake_respuestas") or {}
    if not r:
        return ""
    filas = "\n".join(f"- {k}: {v}" for k, v in r.items() if v not in (None, ""))
    return f"FICHA DE INTAKE:\n{filas}\n\n"


def _parece_abort(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _ABORT_BOOKING)


def _telefono_cliente(state: AgentState) -> str | None:
    tid = state.get("thread_id") or ""
    return tid if re.fullmatch(r"\+?\d{8,15}", tid) else None


def _precaptura_contacto(state: AgentState) -> dict | None:
    """Llena nombre/email del mensaje actual sin depender del routing.
    No persiste: solo alimenta intake_respuestas en memoria."""
    resp = dict(state.get("intake_respuestas") or {})
    q = state.get("query") or ""

    if "nombre" not in resp and (m := _NOMBRE_RE.search(q)):
        palabras = m.group(1).split()
        if len(palabras) >= 2:
            resp["nombre"] = " ".join(p.capitalize() for p in palabras)

    if "email" not in resp and (m := _EMAIL_BUSCA.search(q)):
        e = m.group(0).lower().replace(" ", "")
        if EMAIL_RE.fullmatch(e):
            resp["email"] = e

    return resp if resp != (state.get("intake_respuestas") or {}) else None


def _canal_desconocido(thread_id: str | None) -> str:
    tid = thread_id or ""
    if tid.startswith("cli-"):
        return "cli"
    if tid.startswith("web-"):
        return "web"
    if re.fullmatch(r"\+?\d{8,15}", tid):
        return "whatsapp"
    return "desconocido"


def _guardar_ai(state: AgentState, content: str) -> None:
    insert_message(
        thread_id=state.get("thread_id", ""),
        role="ai", content=content,
        route=state.get("route"), sentiment=state.get("sentiment"),
        urgency=state.get("urgency"), intent=state.get("intent"),
        category=state.get("category"),
    )


def _persist_escalation(state: AgentState, summary: str) -> Path:
    payload = {
        "thread_id": state.get("thread_id"),
        "cerrado_en": _ahora_iso(),
        "clasificacion": {
            "sentiment": state.get("sentiment"), "urgency": state.get("urgency"),
            "intent": state.get("intent"), "category": state.get("category"),
        },
        "lead": {"nombre": state.get("lead_nombre"),
                 "email": state.get("lead_email"),
                 "modalidad": state.get("lead_modalidad")},
        "intake": state.get("intake_respuestas"),
        "booking": state.get("booking"),
        "resumen_ejecutiva": summary,
        "transcript": _transcript(state),
    }
    ESCALATIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(state.get("thread_id") or "sin-id"))
    path = ESCALATIONS_DIR / f"{safe_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


_RESET_INTAKE = {
    "intake_activo": False, "intake_idx": 0, "intake_respuestas": {},
    "intake_attempts": 0, "intake_exit": None, "intake_resume": False,
    "intake_decision": None, "intake_stage": None, "intake_started_en": None,
    "intake_completado": False, "intake_oferta_qid": None,
}


def receive_message(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if m.type == "human"), None
    )
    q = last_human.content if last_human else ""
    if not isinstance(q, str):
        q = str(q)

    tid = state.get("thread_id", "")
    upsert_conversation(thread_id=tid, channel=_canal_desconocido(tid),
                        status="abierto")

    insert_message(thread_id=tid, role="human", content=q)
    return {"query": q}


def analyze_sentiment(state: AgentState) -> AgentState:
    query = state["query"]

    if state.get("closed"):
        if state.get("intake_completado") \
                and _norm_simple(query) in _AGENDAR_CTA:
            return {"route": ROUTE_AGENDAR,
                    "clf_reason": "CTA agendar post-intake completado (0 LLM)"}
        return {"route": ROUTE_HANDOFF,
                "clf_reason": "seguimiento de caso cerrado"}

    en_booking = bool(state.get("booking_stage"))
    en_intake = bool(state.get("intake_activo")) and not state.get("intake_completado")
    en_intake_pausado = en_intake and state.get("intake_exit") == "pausa"
    reset: dict = {}
    forzar_intake = False

    # --- intake pausado: cualquier mensaje retoma la ficha ---
    if en_intake_pausado:
        return {
            "route": ROUTE_INTAKE,
            "clf_reason": "intake pausado → reanudar",
            "intake_exit": None,
            "intake_resume": True,
        }

    # --- booking: comportamiento histórico intacto ---
    if en_booking:
        if _parece_abort(query):
            logger.info("cliente aborta booking → reclasificar")
            reset = {"booking_stage": None, "slots_propuestos": [],
                     "booking_match": None, "booking_match_candidatos": [],
                     "booking_signal": None, "booking_attempts": 0}
        elif _expirado(state.get("agenda_started_en"), AGENDA_CAPTURA_TTL_HORAS):
            logger.info("booking expirado → reclasificar")
            reset = {"booking_stage": None, "slots_propuestos": [],
                     "booking_match": None, "booking_match_candidatos": [],
                     "booking_signal": None, "booking_attempts": 0}
        else:
            return {"route": ROUTE_AGENDAR,
                    "clf_reason": f"sub-flujo booking/{state.get('booking_stage')} (0 LLM)"}

    # --- intake: TTL expira ledger completo; si no, clasifica + fuerza ruta ---
    if en_intake:
        if _expirado(state.get("intake_started_en"), INTAKE_TTL_HORAS):
            logger.info("ficha de intake expirada (> %dh) → reset total",
                        INTAKE_TTL_HORAS)
            reset = {**reset, **_RESET_INTAKE}
        else:
            forzar_intake = True

    cfg = _cfg()["classification"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg["system_prompt"]),
        ("system",
         "Historial reciente (clasifica el ÚLTIMO mensaje del cliente usando "
         "este contexto; los follow-ups breves heredan el tema del hilo):\n{history}"),
        ("system",
         "Ejemplos de etiquetado estándar (referencia obligatoria):\n{few_shots}"),
        ("human", "{query}"),
    ])
    history = "\n".join(
        f"- {m.type}: {m.content[:200]}" for m in _recent_messages(state)
    ) or "(sin historial)"

    chain = prompt | with_structured_output(AnalisisResult)
    try:
        result: AnalisisResult = chain.with_retry(stop_after_attempt=2).invoke({
            "query": query,
            "few_shots": _format_few_shots(cfg["few_shot_examples"]),
            "history": history,
        })
    except Exception as e:
        logger.warning("analyze_sentiment cayó a fallback: %s", e)
        result = fallback_analisis(str(e)[:120])

    route = compute_route(result)
    reason = result.reason
    if forzar_intake:
        route = ROUTE_INTAKE
        reason = f"intake activo (clasificación solo metadata) | {result.reason}"

    # Determinar categoría estable del caso (primera sustantiva gana)
    current_case_category = state.get("case_category")
    new_category = result.category
    case_category = current_case_category
    if not current_case_category or (
        current_case_category in ("otro", "consulta_general")
        and new_category not in ("otro", "consulta_general")
    ):
        case_category = new_category

    logger.info("clf | %s/%s/%s/%s → %s | %s",
                result.sentiment, result.urgency, result.intent,
                result.category, route, reason)

    upsert_conversation(thread_id=state.get("thread_id", ""),
                        intent=result.intent, category=result.category)

    contacto_updates = {} if (pre := _precaptura_contacto(state)) is None else {"intake_respuestas": pre}

    return {
        "sentiment": result.sentiment, "urgency": result.urgency,
        "intent": result.intent, "category": result.category,
        "case_category": case_category,
        "clf_reason": reason, "route": route,
        **contacto_updates,
        **reset,
    }


def compute_route(result: AnalisisResult) -> str:
    if result.intent == "fuera_de_dominio":
        return ROUTE_FUERA_DOMINIO
    if result.intent == "hablar_humano":
        return ROUTE_INTAKE
    return ROUTE_AGENDAR if result.intent == "agendar_asesoria" else ROUTE_FAQ


def route_query(state: AgentState) -> str:
    route = state.get("route", ROUTE_FAQ)
    if route == ROUTE_INTAKE and state.get("intake_completado"):
        return ROUTE_HANDOFF
    if route not in VALID_ROUTES:
        logger.warning("route inesperado '%s' → %s", route, ROUTE_FAQ)
        return ROUTE_FAQ
    return route


def route_post_intake(state: AgentState) -> str:
    if state.get("intake_exit") == "faq":
        return ROUTE_FAQ
    if state.get("intake_exit") == "pausa":
        return END
    if state.get("intake_completado") or state.get("intake_exit") == "handoff":
        return ROUTE_HANDOFF
    return END


def route_post_faq(state: AgentState) -> str:
    if state.get("intake_resume") and state.get("intake_activo") \
            and not state.get("intake_completado"):
        return ROUTE_INTAKE
    return END


def respuesta_fuera_dominio(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(disclosure=atn["disclosure"])
    _guardar_ai(state, response)
    return {"response": response, "response_bubbles": [response],
            "response_interactive": None,
            "messages": [AIMessage(content=response)]}


def handoff_humano(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    tid = state.get("thread_id", "")

    if state.get("closed"):
        response = atn["hilo_cerrado_message"].format(
            whatsapp_ejecutiva=_wa_link_cliente(state), thread_id=tid,
        )
        _guardar_ai(state, response)
        return {"response": response, "response_bubbles": [response],
                "response_interactive": None,
                "messages": [AIMessage(content=response)]}

    try:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", atn["summary_system_prompt"]),
            ("human", "{transcript}"),
        ])
        summary = (summary_prompt | LLM).invoke(
            {"transcript": _ficha_intake(state) + "CONVERSACIÓN:\n"
                         + _transcript_resumen(state)}
        ).content
    except Exception as e:
        logger.exception("LLM de resumen falló: %s", e)
        summary = "(resumen automático no disponible — revisar transcript adjunto)"

    upsert_conversation(thread_id=tid, status="escalado",
                        summary=summary, closed=True)

    respuestas = state.get("intake_respuestas") or {}
    if respuestas.get("consentimiento_datos") is True:
        upsert_lead(
            thread_id=tid,
            intake_respuestas=respuestas,
            category=state.get("case_category") or state.get("category"),
            completed=bool(state.get("intake_completado")),
        )

    notificacion_ok = True
    try:
        notificar_escalamiento(
            thread_id=tid or "sin-id",
            resumen=summary,
            telefono_cliente=_telefono_cliente(state),
        )
    except Exception as e:
        logger.exception("notificación WhatsApp falló: %s", e)
        notificacion_ok = False

    insert_escalation(thread_id=tid, summary=summary,
                      notificado_whatsapp=notificacion_ok)

    _persist_escalation(state, summary)

    # FIX #2: Notificación por email a ejecutiva
    telefono = _telefono_cliente(state)
    wa_link = _wa_link_cliente(state)
    nombre = respuestas.get("nombre") or state.get("lead_nombre") or "Sin nombre"
    categoria_final = state.get("case_category") or state.get("category") or "otro"
    email_body = f"""
    <h3>Nuevo lead derivado — Manzzo y Cía</h3>
    <p><b>Thread ID:</b> {tid}</p>
    <p><b>Nombre:</b> {nombre}</p>
    <p><b>Teléfono:</b> {telefono or 'No disponible'}</p>
    <p><b>Categoría:</b> {categoria_final}</p>
    <p><b>Urgencia:</b> {state.get('urgency', 'media')}</p>
    <p><b>Resumen:</b> {summary}</p>
    <p><b>Link WhatsApp cliente:</b> <a href="{wa_link}">{wa_link}</a></p>
    <p><b>Ficha completa:</b> {respuestas}</p>
    """
    email_ok = notificar_email(
        subject=f"[Manzzo Bot] Nuevo lead: {nombre} | {categoria_final}",
        body_html=email_body,
    )

    # FIX #3: SheetDB solo con consentimiento explícito
    try:
        from integrations.sheetdb import sync_lead
        if respuestas.get("consentimiento_datos") is True:
            sync_lead(state, modo="completo")
        else:
            sync_lead(state, modo="minimo")
    except Exception as e:
        logger.exception("SheetDB (fallback) falló: %s", e)

    intake_ya_despidio = state.get("intake_exit") in ("booking", "handoff") \
        or bool(state.get("intake_completado"))
    if intake_ya_despidio:
        return {
            "summary": summary, "closed": True,
            "notificacion_pendiente": not (notificacion_ok or email_ok),
            "intake_exit": None,
        }

    response = atn["handoff_message"].format(
        disclosure=atn["disclosure"],
        whatsapp_ejecutiva=_wa_link_cliente(state),
        thread_id=tid,
    )
    _guardar_ai(state, response)
    return {
        "response": response, "response_bubbles": [response],
        "response_interactive": None,
        "summary": summary, "closed": True,
        "notificacion_pendiente": not (notificacion_ok or email_ok),
        "messages": [AIMessage(content=response)],
    }
