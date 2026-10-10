"""graph/nodes.py — Nodos transversales del agente (Manzzo y Cía).

v17 — Refactor crítico:
  - Helpers compartidos movidos a graph/utils.py; rompe ciclo de imports con
    graph/intake/nodes.py.
  - route_post_intake reconoce intake_exit == "agendar" y rutea a booking.
  - route_post_intake termina el turno (END) para abandono/sin_consentimiento.
  - _precaptura_contacto normaliza email sin depender de EMAIL_RE.
  - handoff_humano reconoce intake_exit="agendar" como "ya despidió".
"""

import logging
import re

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
from graph.utils import (
    _ahora_iso, _cfg, _expirado, _ficha_intake, _format_few_shots,
    _guardar_ai, _norm_simple, _persist_escalation, _primer_nombre,
    _recent_messages, _telefono_cliente, _transcript_resumen, _upsert_conversacion_abierta,
    _wa_link_cliente, _wa_link_display,
)
from tools.notify_whatsapp import WHATSAPP_EJECUTIVA, notificar_escalamiento

logger = logging.getLogger(__name__)

WA_TEXTO_MAX = 900

_AGENDAR_CTA = ("agendar una hora", "agendar", "agendar ahora",
                "agendar hora", "quiero agendar", "quiero agendar una hora")

_ABORT_BOOKING = ("no quiero", "mejor no", "olvídalo", "olvidalo",
                  "dejalo", "déjalo", "ya no me interesa")

_EMAIL_BUSCA = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NOMBRE_RE = re.compile(
    r"(?:soy|me llamo|mi nombre es)\s+"
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}"
    r"(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}){0,3})",
    re.IGNORECASE,
)


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
        resp["email"] = m.group(0).lower().replace(" ", "")

    return resp if resp != (state.get("intake_respuestas") or {}) else None


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
    _upsert_conversacion_abierta(tid)

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
    if state.get("intake_exit") == "agendar":
        return ROUTE_AGENDAR
    if state.get("intake_exit") in ("end",):
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

    intake_ya_despidio = state.get("intake_exit") in ("booking", "handoff", "agendar") \
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


def _parece_abort(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _ABORT_BOOKING)
