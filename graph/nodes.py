"""graph/nodes.py — Nodos transversales del agente (Manzzo y Cía).

v11 — Hooks de persistencia PostgreSQL (core/db_client.py):
  - receive_message: upsert de conversations + insert del mensaje humano.
  - analyze_sentiment: actualiza intent/category tras clasificar.
  - Cada nodo terminal guarda su respuesta AI vía _guardar_ai().
  - handoff_humano: conversations (cerrado/escalado) + leads (ficha) +
    escalations + SheetDB como fallback secundario.
  - PostgreSQL pasa a ser la fuente de verdad operativa. El JSON local
    (_persist_escalation) y SheetDB quedan como respaldos best-effort.

v10 — Sub-agentes especializados (faq/booking/intake en sus propios
subgrafos); este archivo conserva solo nodos transversales + helpers.
"""
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

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
ESCALATIONS_DIR = Path("escalations")     # respaldo local; la fuente es Postgres

# Políticas de sub-flujos (TTLs los aplica analyze_sentiment).
AGENDA_CAPTURA_TTL_HORAS = 24
INTAKE_TTL_HORAS = 24

# Fail-fast de prompts.yaml al primer uso.
REQUIRED_KEYS = {
    "atencion": {"disclosure", "cta_agendar", "faq_system_prompt", "tonos",
                 "summary_system_prompt", "fuera_dominio_message",
                 "handoff_message", "hilo_cerrado_message",
                 # booking
                 "agenda_pedir_datos", "agenda_proponer_slots",
                 "agenda_reintento_slots", "agenda_sin_horarios",
                 "agenda_confirmada", "agenda_slot_ocupado",
                 "agenda_eleccion_ambigua", "agenda_abortado",
                 "agenda_lateral_system"},
    "classification": {"system_prompt", "few_shot_examples"},
    "intake": {"apertura", "ack_completado", "aviso_saltar",
               "sin_consentimiento"},
}

TEMPLATE_ARGS = {
    "atencion.faq_system_prompt": {"disclosure", "tono", "context"},
    "atencion.fuera_dominio_message": {"disclosure"},
    "atencion.handoff_message": {"disclosure", "whatsapp_ejecutiva", "thread_id"},
    "atencion.hilo_cerrado_message": {"whatsapp_ejecutiva", "thread_id"},
    "atencion.cta_agendar": set(),
    "atencion.agenda_pedir_datos": set(),
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
    "intake.ack_completado": {"nombre"},
    "intake.aviso_saltar": set(),
    "intake.sin_consentimiento": set(),
}
_PLACEHOLDER_RE = re.compile(r"{(\w+)}")

# Abort explícito de cualquier sub-flujo (booking o intake)
_ABORT = ("no quiero", "mejor no", "olvídalo", "olvidalo", "cancela",
          "déjalo", "dejalo", "ya no me interesa", "no gracias")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cfg() -> dict:
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [
        f"{sec}.{key}" for sec, keys in REQUIRED_KEYS.items()
        for key in keys if key not in (cfg.get(sec) or {})
    ]
    if missing:
        raise KeyError(f"prompts.yaml incompleto — faltan: {missing}")

    bad = []
    for dotted, esperados in TEMPLATE_ARGS.items():
        seccion, key = dotted.split(".", 1)
        tpl = (cfg.get(seccion) or {}).get(key) or ""
        encontrados = set(_PLACEHOLDER_RE.findall(tpl))
        if encontrados != esperados:
            bad.append(
                f"{dotted}: plantilla usa {sorted(encontrados)} "
                f"pero el nodo entrega {sorted(esperados)}"
            )
    if bad:
        raise ValueError("prompts.yaml — placeholders desalineados:\n" + "\n".join(bad))

    return cfg


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
    return any(p in q for p in _ABORT)


def _telefono_cliente(state: AgentState) -> str | None:
    tid = state.get("thread_id") or ""
    return tid if re.fullmatch(r"\+\d{8,15}", tid) else None


def _canal_desconocido(thread_id: str | None) -> str:
    """Detecta el canal desde el thread_id ('cli-', 'web-' o teléfono E.164)."""
    tid = thread_id or ""
    if tid.startswith("cli-"):
        return "cli"
    if tid.startswith("web-"):
        return "web"
    if re.fullmatch(r"\+\d{8,15}", tid):
        return "whatsapp"
    return "desconocido"


def _guardar_ai(state: AgentState, content: str) -> None:
    """Persist best-effort de la respuesta del agente en Postgres.
    Lo usan los nodos de TODOS los subgrafos (importado desde aquí)."""
    insert_message(
        thread_id=state.get("thread_id", ""),
        role="ai",
        content=content,
        route=state.get("route"),
        sentiment=state.get("sentiment"),
        urgency=state.get("urgency"),
        intent=state.get("intent"),
        category=state.get("category"),
    )


def _persist_escalation(state: AgentState, summary: str) -> Path:
    """Respaldo local en JSON (secundario; Postgres es la fuente operativa)."""
    payload = {
        "thread_id": state.get("thread_id"),
        "cerrado_en": _ahora_iso(),
        "clasificacion": {
            "sentiment": state.get("sentiment"),
            "urgency": state.get("urgency"),
            "intent": state.get("intent"),
            "category": state.get("category"),
        },
        "lead": {
            "nombre": state.get("lead_nombre"),
            "email": state.get("lead_email"),
            "modalidad": state.get("lead_modalidad"),
        },
        "intake": state.get("intake_respuestas"),
        "booking": state.get("booking"),
        "resumen_ejecutiva": summary,
        "transcript": _transcript(state),
    }
    ESCALATIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(state.get("thread_id") or "sin-id"))
    path = ESCALATIONS_DIR / f"{safe_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("conversación guardada localmente (backup): %s", path)
    return path


# ---------------------------------------------------------------------------
# ENTRADA + CLASIFICACIÓN
# ---------------------------------------------------------------------------
def receive_message(state: AgentState) -> AgentState:
    """query se deriva SIEMPRE del último mensaje humano.
    Hook Postgres: asegura la fila en conversations y guarda el mensaje
    humano (solo si el query cambió — evita duplicar en resumes)."""
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
    """Clasificación unificada con short-circuits de sub-flujos (0 LLM).
    Hook Postgres: tras clasificar (LLM real), actualiza la conversación
    con intent/category."""

    # Caso cerrado/derivado → seguimiento por handoff
    if state.get("closed"):
        return {"route": ROUTE_HANDOFF, "clf_reason": "seguimiento de caso cerrado"}

    query = state["query"]
    en_booking = bool(state.get("booking_stage"))
    en_intake = bool(state.get("intake_activo"))
    reset: dict = {}

    if en_booking or en_intake:
        if _parece_abort(query):
            logger.info("cliente aborta sub-flujo → reclasificar")
            reset = {"intake_activo": False,
                     "booking_stage": None,
                     "slots_propuestos": [],
                     "booking_match": None,
                     "booking_match_candidatos": [],
                     "booking_signal": None,
                     "booking_attempts": 0}
        elif en_intake and _expirado(state.get("intake_started_en"), INTAKE_TTL_HORAS):
            logger.info("ficha de intake expirada (> %dh) → reclasificar",
                        INTAKE_TTL_HORAS)
            reset = {"intake_activo": False}
        elif en_booking and _expirado(
                state.get("agenda_started_en"), AGENDA_CAPTURA_TTL_HORAS):
            logger.info("sub-flujo booking expirado (> %dh) → reclasificar",
                        AGENDA_CAPTURA_TTL_HORAS)
            reset = {"booking_stage": None,
                     "slots_propuestos": [],
                     "booking_match": None,
                     "booking_match_candidatos": [],
                     "booking_signal": None,
                     "booking_attempts": 0}
        elif en_intake:
            return {"route": ROUTE_INTAKE,
                    "clf_reason": "sub-flujo: ficha de intake (0 LLM)"}
        elif en_booking:
            return {"route": ROUTE_AGENDAR,
                    "clf_reason": f"sub-flujo booking/{state.get('booking_stage')} (0 LLM)"}

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
    logger.info("clf | %s/%s/%s/%s → %s | %s",
                result.sentiment, result.urgency, result.intent,
                result.category, route, result.reason)

    # Hook Postgres: etiquetar la conversación con la clasificación
    upsert_conversation(
        thread_id=state.get("thread_id", ""),
        intent=result.intent,
        category=result.category,
    )

    return {
        "sentiment": result.sentiment, "urgency": result.urgency,
        "intent": result.intent, "category": result.category,
        "clf_reason": result.reason, "route": route,
        **reset,
    }


def compute_route(result: AnalisisResult) -> str:
    """Regla oficial de enrutamiento."""
    if result.intent == "fuera_de_dominio":
        return ROUTE_FUERA_DOMINIO
    if result.intent == "hablar_humano":
        return ROUTE_INTAKE
    return ROUTE_AGENDAR if result.intent == "agendar_asesoria" else ROUTE_FAQ


def route_query(state: AgentState) -> str:
    """Lee la ruta del estado con guardias defensivas."""
    route = state.get("route", ROUTE_FAQ)
    if route == ROUTE_INTAKE and state.get("intake_completado"):
        return ROUTE_HANDOFF
    if route not in VALID_ROUTES:
        logger.warning("route inesperado '%s' → %s", route, ROUTE_FAQ)
        return ROUTE_FAQ
    return route


# ---------------------------------------------------------------------------
# NODOS TERMINALES TRANSVERSALES
# ---------------------------------------------------------------------------
def respuesta_fuera_dominio(state: AgentState) -> AgentState:
    """Redirección amable + CTA."""
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(disclosure=atn["disclosure"])
    _guardar_ai(state, response)
    return {"response": response, "messages": [AIMessage(content=response)]}


# ---------------------------------------------------------------------------
# ESCALAMIENTO — solo si el cliente lo pide/acepta, TRAS completar la ficha
# ---------------------------------------------------------------------------
def handoff_humano(state: AgentState) -> AgentState:
    """Escalamiento/cierre humano.
    Persistencia: Postgres (primario: conversations+leads+escalations),
    JSON local (backup), SheetDB (fallback para ejecutivas)."""
    atn = _cfg()["atencion"]
    tid = state.get("thread_id", "")

    # Seguimiento post-cierre: no re-escalar ni re-guardar
    if state.get("closed"):
        response = atn["hilo_cerrado_message"].format(
            whatsapp_ejecutiva=WHATSAPP_EJECUTIVA,
            thread_id=tid,
        )
        _guardar_ai(state, response)
        return {"response": response, "messages": [AIMessage(content=response)]}

    # 1) Resumen del caso = FICHA DE INTAKE + transcript truncado (1 LLM)
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

    # 2) Persistencia PRIMARIA en Postgres (fuente de verdad operativa)
    upsert_conversation(
        thread_id=tid,
        status="escalado",
        summary=summary,
        closed=True,
    )
    upsert_lead(
        thread_id=tid,
        intake_respuestas=state.get("intake_respuestas") or {},
        category=state.get("category"),
        completed=True,
    )

    # 3) Notificar por WhatsApp ANTES de registrar el escalamiento
    #    (el flag notificado_whatsapp queda reflejando la realidad)
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

    insert_escalation(
        thread_id=tid,
        summary=summary,
        notificado_whatsapp=notificacion_ok,
    )

    # 4) Respaldos secundarios (best-effort, jamás rompen el flujo)
    _persist_escalation(state, summary)              # JSON local
    try:
        from integrations.sheetdb import sync_lead   # Google Sheets
        sync_lead(state)
    except Exception as e:
        logger.exception("SheetDB (fallback) falló: %s", e)

    # 5) Respuesta al cliente
    response = atn["handoff_message"].format(
        disclosure=atn["disclosure"],
        whatsapp_ejecutiva=WHATSAPP_EJECUTIVA,
        thread_id=tid,
    )
    _guardar_ai(state, response)
    return {
        "response": response,
        "summary": summary,
        "closed": True,
        "notificacion_pendiente": not notificacion_ok,
        "messages": [AIMessage(content=response)],
    }
