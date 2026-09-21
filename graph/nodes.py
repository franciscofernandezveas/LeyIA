"""graph/nodes.py — Nodos transversales del agente (Manzzo y Cía).

v14 — Links wa.me renderizados como markdown limpio:
  - _wa_link_display() envuelve el wa.me en [display](url): el cliente ve
    texto limpio, la URL enorme queda oculta.
  - _wa_link_diagnostico() y _wa_link_cliente() ahora devuelven markdown
    [Hablar con un humano](wa.me?text=...). Al pinchar, WhatsApp abre
    con el diagnóstico completo ya escrito para la ejecutiva.
  - El resumen LLM largo sigue yendo a la ejecutiva por
    notificar_escalamiento + DB; el link solo lleva el diagnóstico compacto.

v13 — Links wa.me con diagnóstico del intake:
  - _wa_texto_intake(): "tarjeta de presentación" determinista (0 LLM) con
    la ficha ya validada + thread_id (la ejecutiva cruza con el CRM al
    recibir el mensaje del cliente). Acotada a WA_TEXTO_MAX porque el
    ?text= se URL-encodea (~1.5-2x).
  - _wa_link_diagnostico(): wa.me con ese diagnóstico prellenado.
  - _wa_link_cliente() ahora usa el diagnóstico cuando hay intake → el
    cliente recibe el MISMO link en ack_completado y en handoff_message
    (consistencia); sin ficha, cae al saludo genérico. Con ficha parcial
    (derivación anticipada) el intro refleja que NO completó el registro.

v12 — Intake semántico + leads parciales + links wa.me:
  - analyze_sentiment clasifica SIEMPRE durante intake (metadata fresca
    para el lead: category/intent ya no quedan congelados); la precedencia
    de estado fuerza route=ROUTE_INTAKE. Booking conserva su short-circuit.
  - Abort por substrings (_parece_abort) queda SOLO para booking; el escape
    del intake lo decide su planner semántico.
  - TTL de intake expira el ledger COMPLETO (no más fichas zombie).
  - route_post_intake / route_post_faq: pausa lateral INTAKE→FAQ→INTAKE.
  - handoff_humano: lead completed según intake_completado real; {whatsapp_ejecutiva}
    ahora recibe link wa.me clickable con texto prellenado.
  - _wa_link(): helper único para links de la ejecutiva.

v11 — Hooks de persistencia PostgreSQL.
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
from tools.notify_whatsapp import WHATSAPP_EJECUTIVA, notificar_escalamiento

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "core" / "prompts.yaml"
ESCALATIONS_DIR = Path("escalations")

AGENDA_CAPTURA_TTL_HORAS = 24
INTAKE_TTL_HORAS = 24

# Largo máx. del texto crudo del wa.me (tras URL-encoding crece ~1.5-2x).
WA_TEXTO_MAX = 900

REQUIRED_KEYS = {
    "atencion": {"disclosure", "cta_agendar", "faq_system_prompt", "tonos",
                 "summary_system_prompt", "fuera_dominio_message",
                 "handoff_message", "hilo_cerrado_message",
                 "agenda_pedir_nombre", "agenda_pedir_email", "agenda_pedir_modalidad","agenda_proponer_slots",
                 "agenda_reintento_slots", "agenda_sin_horarios",
                 "agenda_confirmada", "agenda_slot_ocupado",
                 "agenda_eleccion_ambigua", "agenda_abortado",
                 "agenda_lateral_system"},
    "classification": {"system_prompt", "few_shot_examples"},
    "intake": {"apertura", "ack_completado", "aviso_saltar",
               "sin_consentimiento", "reanudar",
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
    "intake.apertura": set(),
    "intake.ack_completado": {"nombre", "wa_link"},
    "intake.aviso_saltar": set(),
    "intake.sin_consentimiento": {"wa_link"},
    "intake.reanudar": set(),
    "intake.derivacion_parcial": {"wa_link"},
    "intake.cierre_abandono": {"wa_link"},
}
_PLACEHOLDER_RE = re.compile(r"{(\w+)}")

# Abort por substrings: SOLO booking (el intake usa su planner semántico).
_ABORT_BOOKING = ("no quiero", "mejor no", "olvídalo", "olvidalo",
                  "dejalo", "déjalo", "ya no me interesa")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
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
        if encontrados != esperados:
            bad.append(f"{dotted}: plantilla usa {sorted(encontrados)} "
                       f"pero el nodo entrega {sorted(esperados)}")
    if bad:
        raise ValueError("prompts.yaml — placeholders desalineados:\n" + "\n".join(bad))
    return cfg


def _wa_link(texto: str = "") -> str:
    """Link clickable wa.me hacia la ejecutiva, con texto prellenado opcional."""
    num = re.sub(r"\D", "", WHATSAPP_EJECUTIVA or "")
    base = f"https://wa.me/{num}"
    return f"{base}?text={quote(texto)}" if texto else base


def _wa_link_display(texto: str = "", display: str = "Hablar con un humano") -> str:
    """Markdown link [display](wa.me?text=...): el cliente ve texto limpio,
    pero al pinchar abre el chat con el diagnóstico prellenado."""
    return f"[{display}]({_wa_link(texto)})"


def _wa_texto_intake(state: AgentState) -> str:
    """Diagnóstico compacto del intake para el ?text= del wa.me.

    Determinista (0 LLM): la ficha ya pasó los validadores, solo se formatea.
    El resumen largo le llega a la ejecutiva por notificación/CRM; este texto
    es la tarjeta de presentación que ella ve cuando el cliente le escribe.
    Ficha parcial (derivación anticipada) → intro que NO afirma completado.
    """
    r = state.get("intake_respuestas") or {}
    nombre = r.get("nombre")

    lineas = [
        f"Hola, soy {nombre}." if nombre
        else "Hola, vengo del asistente virtual de Manzzo y Cía.",
        ("Acabo de completar mi registro. Resumen de mi caso:"
         if state.get("intake_completado") else
         "Prefiero hablar directamente con una ejecutiva. Datos que alcancé a registrar:"),
    ]
    if state.get("category"):
        lineas.append(f"• Área: {state['category']}")
    if r.get("email"):
        lineas.append(f"• Correo: {r['email']}")
    if r.get("situacion_actual"):
        sit = str(r["situacion_actual"])
        if len(sit) > 280:                      # campo libre: acotar
            sit = sit[:277].rstrip() + "…"
        lineas.append(f"• Mi situación: {sit}")
    if r.get("etapa_proceso"):                   # ya viene como label legible
        lineas.append(f"• Etapa de mi caso: {r['etapa_proceso']}")
    # opcionales del planner, por si se agregan a QUESTIONS:
    if r.get("comuna"):
        lineas.append(f"• Comuna: {r['comuna']}")
    if r.get("hijos_menores") is True:
        lineas.append("• Tengo hijos menores de edad")
    if r.get("horario_contacto"):
        lineas.append(f"• Horario preferido: {r['horario_contacto']}")

    tid = state.get("thread_id")
    if tid:
        lineas.append(f"(ID de mi atención: {tid})")   # trazabilidad con el CRM

    txt = "\n".join(lineas)
    if len(txt) > WA_TEXTO_MAX:
        txt = txt[:WA_TEXTO_MAX - 1].rstrip() + "…"
    return txt


def _wa_link_diagnostico(state: AgentState) -> str:
    """Link limpio hacia la ejecutiva; el diagnóstico viaja oculto en ?text=."""
    return _wa_link_display(_wa_texto_intake(state), display="Hablar con un humano")


def _wa_link_cliente(state: AgentState) -> str:
    """wa.me prellenado con el diagnóstico del intake (si hay ficha)."""
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
    return tid if re.fullmatch(r"\+\d{8,15}", tid) else None


def _canal_desconocido(thread_id: str | None) -> str:
    tid = thread_id or ""
    if tid.startswith("cli-"):
        return "cli"
    if tid.startswith("web-"):
        return "web"
    if re.fullmatch(r"\+\d{8,15}", tid):
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


# Reset TOTAL del ledger de intake (TTL o inconsistencias) — mata fichas zombie.
_RESET_INTAKE = {
    "intake_activo": False, "intake_idx": 0, "intake_respuestas": {},
    "intake_attempts": 0, "intake_exit": None, "intake_resume": False,
    "intake_decision": None, "intake_stage": None, "intake_started_en": None,
    "intake_completado": False,
}


# ---------------------------------------------------------------------------
# ENTRADA + CLASIFICACIÓN
# ---------------------------------------------------------------------------
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
    if q != state.get("query"):
        insert_message(thread_id=tid, role="human", content=q)
        return {"query": q}
    return {}


def analyze_sentiment(state: AgentState) -> AgentState:
    """Clasificación unificada. Durante booking: short-circuit 0 LLM (como
    siempre). Durante intake: clasifica igual (metadata fresca) pero la
    precedencia de estado fuerza ROUTE_INTAKE; el planner del subgrafo
    entiende el mensaje en contexto (respuesta/duda/escape)."""

    if state.get("closed"):
        return {"route": ROUTE_HANDOFF, "clf_reason": "seguimiento de caso cerrado"}

    query = state["query"]
    en_booking = bool(state.get("booking_stage"))
    en_intake = bool(state.get("intake_activo")) and not state.get("intake_completado")
    reset: dict = {}
    forzar_intake = False

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

    # --- clasificación unificada (siempre que no haya short-circuit 0 LLM) ---
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

    logger.info("clf | %s/%s/%s/%s → %s | %s",
                result.sentiment, result.urgency, result.intent,
                result.category, route, reason)

    upsert_conversation(thread_id=state.get("thread_id", ""),
                        intent=result.intent, category=result.category)

    return {
        "sentiment": result.sentiment, "urgency": result.urgency,
        "intent": result.intent, "category": result.category,
        "clf_reason": reason, "route": route,
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


# ---------------------------------------------------------------------------
# CONTINUACIONES POST-SUBGRAFO (pausa lateral del intake + handoff)
# ---------------------------------------------------------------------------
def route_post_intake(state: AgentState) -> str:
    """Tras el subgrafo intake:
    - pausa por duda lateral → FAQ responde en el mismo turno;
    - ficha completa o derivación parcial → handoff;
    - resto → END (esperar próximo mensaje)."""
    if state.get("intake_exit") == "faq":
        return ROUTE_FAQ
    if state.get("intake_completado") or state.get("intake_exit") == "handoff":
        return ROUTE_HANDOFF
    return END


def route_post_faq(state: AgentState) -> str:
    """Si el FAQ respondió una duda lateral con el intake pausado, el subgrafo
    de intake se re-invocaba para re-anexar la pregunta pendiente."""
    if state.get("intake_resume") and state.get("intake_activo") \
            and not state.get("intake_completado"):
        return ROUTE_INTAKE
    return END


# ---------------------------------------------------------------------------
# NODOS TERMINALES TRANSVERSALES
# ---------------------------------------------------------------------------
def respuesta_fuera_dominio(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(disclosure=atn["disclosure"])
    _guardar_ai(state, response)
    return {"response": response, "messages": [AIMessage(content=response)]}


def handoff_humano(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    tid = state.get("thread_id", "")

    if state.get("closed"):
        response = atn["hilo_cerrado_message"].format(
            whatsapp_ejecutiva=_wa_link_cliente(state), thread_id=tid,
        )
        _guardar_ai(state, response)
        return {"response": response, "messages": [AIMessage(content=response)]}

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

    # Lead en CRM: solo con consentimiento expreso; completed refleja si la
    # ficha se terminó o es parcial (derivación anticipada por insistencia
    # del cliente o por reintentos agotados).
    respuestas = state.get("intake_respuestas") or {}
    if respuestas.get("consentimiento_datos") is True:
        upsert_lead(
            thread_id=tid,
            intake_respuestas=respuestas,
            category=state.get("category"),
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
    try:
        from integrations.sheetdb import sync_lead
        sync_lead(state)
    except Exception as e:
        logger.exception("SheetDB (fallback) falló: %s", e)

    response = atn["handoff_message"].format(
        disclosure=atn["disclosure"],
        whatsapp_ejecutiva=_wa_link_cliente(state),
        thread_id=tid,
    )
    _guardar_ai(state, response)
    return {
        "response": response, "summary": summary, "closed": True,
        "notificacion_pendiente": not notificacion_ok,
        "messages": [AIMessage(content=response)],
    }
