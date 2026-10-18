"""graph/nodes.py — Nodos transversales del agente (Manzzo y Cía).

v19 — Añade handoff_choice: bifurcación previa a handoff/intake que pregunta
       al usuario si prefiere rellenar una ficha breve o recibir contacto
       directo, reduciendo la fricción de forzar siempre el intake completo.

v18 — FIX import: verificado, sin constantes de booking en core.contracts.
       Este archivo solo importa AgentState, AnalisisResult, rutas y
       fallback_analisis del módulo central.

v17 — Refactor crítico:
  - Helpers compartidos movidos a graph/utils.py; rompe ciclo de imports con
    graph/intake/nodes.py.
  - route_post_intake reconoce intake_exit == "agendar" y rutea a booking.
  - route_post_intake termina el turno (END) para abandono/sin_consentimiento.
  - _precaptura_contacto normaliza email sin depender de EMAIL_RE.
  - handoff_humano reconoce intake_exit="agendar" como "ya despidió".
"""

import logging
import os
import re

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from core.contracts import (
    AgentState, AnalisisResult,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF, ROUTE_INTAKE,
    ROUTE_HANDOFF_CHOICE,
    VALID_ROUTES, fallback_analisis,
)
from core.db_client import (
    insert_escalation, insert_message, upsert_conversation, upsert_lead,
)
from core.llm import LLM, with_structured_output
from core.notifications import notificar_email
from graph.utils import (
    _ahora_iso, _cfg, _expirado, _ficha_intake, _format_few_shots,
    _guardar_ai, _norm_simple, _parece_abort, _persist_escalation,
    _primer_nombre, _recent_messages, _telefono_cliente,
    _transcript_resumen, _upsert_conversacion_abierta, _wa_link_cliente,
    _wa_link_display,
)

from tools.notify_whatsapp import WHATSAPP_EJECUTIVA, notificar_escalamiento

logger = logging.getLogger(__name__)

WA_TEXTO_MAX = 900

INTAKE_TTL_HORAS = int(os.getenv("INTAKE_TTL_HORAS", "24"))
AGENDA_CAPTURA_TTL_HORAS = int(os.getenv("AGENDA_CAPTURA_TTL_HORAS", "2"))

_AGENDAR_CTA = ("agendar una hora", "agendar", "agendar ahora",
                "agendar hora", "quiero agendar", "quiero agendar una hora")


_EMAIL_BUSCA = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NOMBRE_RE = re.compile(
    r"(?:soy|me llamo|mi nombre es)\s+"
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}"
    r"(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}){0,3})",
    re.IGNORECASE,
)


_RESET_INTAKE = {
    "intake_activo": False, "intake_idx": 0, "intake_respuestas": {},
    "intake_attempts": 0, "intake_exit": None, "intake_resume": False,
    "intake_decision": None, "intake_stage": None, "intake_started_en": None,
    "intake_completado": False, "intake_oferta_qid": None,
}

_RESET_HANDOFF_CHOICE = {
    "handoff_choice_pending": False,
    "handoff_choice_attempts": 0,
    "handoff_choice_en": None,
}

# Normalizadas con _norm_simple para el parser 0-LLM.
_ELECCION_FICHA_RAW = (
    "ficha", "rellenar", "relleno", "formulario", "breve", "agilizar",
    "opcion 1", "hc_ficha",
)
_ELECCION_DIRECTO_RAW = (
    "directo", "directa", "contacto", "numero", "número", "telefono",
    "teléfono", "llamar", "llamada", "whatsapp", "ejecutiva", "abogada",
    "correo", "opcion 2", "hc_directo",
)
_ELECCION_FICHA = tuple(_norm_simple(k) for k in _ELECCION_FICHA_RAW)
_ELECCION_DIRECTO = tuple(_norm_simple(k) for k in _ELECCION_DIRECTO_RAW)
_DIGITO_RE = re.compile(r"(?<!\d)([12])(?!\d)")


def _interpretar_eleccion_handoff(q: str) -> str | None:
    """Interpreta la respuesta a '¿ficha breve o contacto directo?'.
    Devuelve 'ficha', 'directo' o None si es ambigua."""
    t = _norm_simple(q)
    if not t:
        return None
    if m := _DIGITO_RE.search(t):
        return "ficha" if m.group(1) == "1" else "directo"
    # "directo" primero: frases como "no quiero ficha, dame el número"
    # contienen ambas keywords y deben ir por la vía de menor fricción.
    if any(k in t for k in _ELECCION_DIRECTO):
        return "directo"
    if any(k in t for k in _ELECCION_FICHA):
        return "ficha"
    return None


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
    choice_pendiente = False
    choice_attempts = int(state.get("handoff_choice_attempts") or 0)

    if en_intake_pausado:
        return {
            "route": ROUTE_INTAKE,
            "clf_reason": "intake pausado → reanudar",
            "intake_exit": None,
            "intake_resume": True,
        }

    # --- Bifurcación handoff: ofrecer ficha breve vs contacto directo ----
    if state.get("handoff_choice_pending"):
        if _expirado(state.get("handoff_choice_en"), INTAKE_TTL_HORAS):
            logger.info("elección handoff expirada (> %dh) → reclasificar",
                        INTAKE_TTL_HORAS)
            reset = {**reset, **_RESET_HANDOFF_CHOICE}
        else:
            eleccion = _interpretar_eleccion_handoff(query)
            if eleccion == "ficha":
                return {
                    "route": ROUTE_INTAKE,
                    **_RESET_HANDOFF_CHOICE,
                    "clf_reason": "handoff_choice → ficha breve (0 LLM)",
                }
            if eleccion == "directo":
                pre = _precaptura_contacto(state)
                return {
                    "route": ROUTE_HANDOFF,
                    **_RESET_HANDOFF_CHOICE,
                    **({} if pre is None else {"intake_respuestas": pre}),
                    "clf_reason": "handoff_choice → contacto directo (0 LLM)",
                }
            # Respuesta ambigua: dejamos que el LLM decida y re-preguntamos.
            choice_pendiente = True

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

    if choice_pendiente:
        if route == ROUTE_HANDOFF_CHOICE:
            if choice_attempts >= 1:
                # Segunda respuesta ambigua: evitar bucle, ir por menor fricción.
                route = ROUTE_HANDOFF
                reason = f"handoff_choice ambigua x2 → directo | {result.reason}"
                reset = {**reset, **_RESET_HANDOFF_CHOICE}
            else:
                reason = f"handoff_choice ambigua → re-preguntar | {result.reason}"
                reset = {**reset, "handoff_choice_attempts": choice_attempts + 1}
        else:
            # El usuario cambió de tema (faq/agendar): limpiar elección pendiente.
            reset = {**reset, **_RESET_HANDOFF_CHOICE}
    elif forzar_intake:
        route = ROUTE_INTAKE
        reason = f"intake activo (clasificación solo metadata) | {result.reason}"

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
        # Opción de bypass para urgencia alta: salta directo a handoff.
        # if result.urgency == "alta":
        #     return ROUTE_HANDOFF
        return ROUTE_HANDOFF_CHOICE
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


def preguntar_preferencia_handoff(state: AgentState) -> AgentState:
    """Nodo intermedio que reduce fricción: ficha breve vs contacto directo.
    Es defensivo: si faltan las plantillas en config, usa un fallback
    hardcodeado para evitar crash en producción."""
    atn = _cfg()["atencion"]
    pre = _precaptura_contacto(state)

    fallback_default = (
        "Para derivarte con nuestro equipo, ¿cómo prefieres continuar?\n\n"
        "1️⃣ Rellenar una ficha breve (2 min): la abogada revisa tu caso con\n"
        "   todo el contexto.\n"
        "2️⃣ Contacto directo: te entrego ahora el WhatsApp de la ejecutiva.\n\n"
        "Responde 1 o 2."
    )

    choice_attempts = state.get("handoff_choice_attempts") or 0
    key = ("handoff_choice_retry_message"
           if choice_attempts > 0
           else "handoff_choice_message")

    # Defensivo: resuelve en cascata para nunca crashear.
    template = atn.get(key) or atn.get("handoff_choice_message") or fallback_default

    if template == fallback_default:
        logger.warning(
            "handoff_choice: no se encontró '%s' en config; usando fallback por defecto",
            key,
        )

    disclosure = atn.get("disclosure", "Soy LeyIA, la asistente virtual de Manzzo y Cía.")
    response = template.format(disclosure=disclosure)

    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": {
            "type": "buttons",
            "body": response,
            "buttons": [
                {"id": "hc_ficha",   "title": "1 Rellenar ficha breve"},
                {"id": "hc_directo", "title": "2 Contacto directo"},
            ],
        },
        "handoff_choice_pending": True,
        "handoff_choice_en": _ahora_iso(),
        **({} if pre is None else {"intake_respuestas": pre}),
        "messages": [AIMessage(content=response)],
    }


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

    motivo = "hablar_abogada" if state.get("intake_completado") else "contacto_directo"
    notificacion_ok = insert_escalation(
        thread_id=tid,
        summary=summary,
        motivo=motivo,
        notificar=True,
    )

    _persist_escalation(state, summary)

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
