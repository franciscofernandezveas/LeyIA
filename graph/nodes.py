"""graph/nodes.py — Nodos del agente LeyIA (Manzzo y Cía).

Versión integrada final:
- Clasificación unificada v4 (1 LLM: sentiment/urgency/intent/category + reason),
  etiquetas cerradas Pydantic, few-shots desde prompts.yaml, historial en contexto.
- Routing determinista por INTENT (el sentimiento modula tono, no routea).
- Nodos de ATENCIÓN conversacional (ven el historial, CTA agendar siempre).
- Sub-flujo de agendamiento: captura nombre/email/modalidad → link Calendly
  prellenado (v2) → VERIFICACIÓN de booking por polling (plan Free) → resumen
  de la cita en el chat. Hook listo para Scheduling API v3 al pagar plan.
- Escalamiento SOLO si el cliente lo pide/acepta: resumen → guardado
  escalations/{thread_id}.json → WhatsApp a la ejecutiva → closed=True.
"""
import json
import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import interrupt

from core.contracts import (
    AgentState, AnalisisResult, LeadExtract,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF,
    VALID_ROUTES, fallback_analisis,
)
from core.llm import LLM, with_structured_output
from core.rag import retrieve
from tools.calendly import (
    DIRECCION_OFICINA, MODALIDAD_LABELS, LeadData,
    buscar_booking, crear_link_agendamiento,          # ← buscar_booking: NUEVO
)
from tools.notify_whatsapp import WHATSAPP_EJECUTIVA, notificar_escalamiento

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "core" / "prompts.yaml"
ESCALATIONS_DIR = Path("escalations")
ESCALATIONS_DIR.mkdir(parents=True, exist_ok=True)

# False = el link se envía directo (prioridad negocio: agendar).
# True  = reactiva el HITL de aprobación por operador (main.py/app.py ya lo soportan).
REQUIERE_APROBACION_AGENDAMIENTO = False

# Validación fail-fast de prompts.yaml al primer uso (KeyError con lista de faltantes)
REQUIRED_KEYS = {
    "atencion": {"disclosure", "cta_agendar", "faq_system_prompt", "tonos",
                 "summary_system_prompt", "fuera_dominio_message",
                 "handoff_message", "hilo_cerrado_message",
                 "agenda_pedir_datos", "agenda_link_enviado",
                 "agenda_confirmada", "agenda_verificacion_pendiente"},   # ← NUEVO
    "classification": {"system_prompt", "few_shot_examples"},
}

# Palabras que gatillan la verificación de booking (sin pagar LLM)
_CONFIRMACIONES = ("listo", "ya agend", "agendé", "agende", "hecho",
                   "confirme", "confirmé", "quedó", "quedo", "ya está", "ya esta")


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
        raise KeyError(
            f"prompts.yaml incompleto — faltan: {missing}. "
            "Revisa indentación (2 espacios por nivel)."
        )
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


def _transcript(state: AgentState) -> list[dict]:
    return [{"rol": m.type, "contenido": m.content}
            for m in state.get("messages", [])]


def _parece_confirmacion(query: str) -> bool:                        # ← NUEVO
    """'listo', 'ya agendé', 'quedó'... → activa verificación de booking."""
    q = query.lower()
    return any(p in q for p in _CONFIRMACIONES)


def _telefono_cliente(state: AgentState) -> str | None:              # ← NUEVO
    """En prod, thread_id = teléfono WhatsApp (E.164: +56...).
    En consola/Streamlit (cli-*, web-*) → None, para no ensuciar el formulario."""
    tid = state.get("thread_id") or ""
    return tid if re.fullmatch(r"\+\d{8,15}", tid) else None


def _extraer_lead(state: AgentState) -> LeadExtract:
    """Extrae nombre + email + modalidad del mensaje del cliente (1 llamada LLM)."""
    extractor = ChatPromptTemplate.from_messages([
        ("system",
         "Extrae del mensaje: nombre completo, email y modalidad "
         "('online' o 'presencial'). Normaliza sinónimos: 'videollamada', "
         "'meet', 'virtual' → online; 'oficina', 'en persona' → presencial. "
         "Si un dato falta, devuélvelo null. Responde solo el esquema."),
        ("human", "{query}"),
    ]) | with_structured_output(LeadExtract)
    return extractor.invoke({"query": state["query"]})


def _persist_escalation(state: AgentState, summary: str) -> Path:
    """Cierra y guarda la conversación completa bajo su thread_id."""
    payload = {
        "thread_id": state.get("thread_id"),
        "cerrado_en": datetime.now(timezone.utc).isoformat(),
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
        "booking": state.get("booking"),                 # ← cita confirmada (si hay)
        "resumen_ejecutiva": summary,
        "transcript": _transcript(state),
    }
    path = ESCALATIONS_DIR / f"{state.get('thread_id', 'sin-id')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("conversación guardada y cerrada: %s", path)
    return path


# ---------------------------------------------------------------------------
# ENTRADA + CLASIFICACIÓN (enrichment silencioso, con contexto del hilo)
# ---------------------------------------------------------------------------
def receive_message(state: AgentState) -> AgentState:
    """Normaliza la entrada: usa state['query'] o el último mensaje humano."""
    if state.get("query"):
        return {}
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    return {"query": last_human.content if last_human else ""}


def analyze_sentiment(state: AgentState) -> AgentState:
    """Clasificación unificada con contexto. Short-circuits de sub-flujos
    para no reclasificar y no gastar LLM cuando el hilo ya tiene un curso."""

    # Caso cerrado/escalado → seguimiento por handoff (0 LLM)
    if state.get("closed") or state.get("escalated"):
        logger.info("caso cerrado/escalado → handoff (modo seguimiento)")
        return {"route": ROUTE_HANDOFF,
                "clf_reason": "seguimiento de caso cerrado"}

    # Sub-flujo de captura de datos de agendamiento → seguir en agendar (0 LLM)
    if state.get("recolectando_datos_agenda"):
        return {"route": ROUTE_AGENDAR,
                "clf_reason": "sub-flujo: captura de datos de agendamiento"}

    # Sub-flujo de elección de horario (v3 Scheduling API; inactivo en plan Free)
    if state.get("esperando_slot"):
        return {"route": ROUTE_AGENDAR,
                "clf_reason": "sub-flujo: elección de horario"}

    # Link enviado y el cliente dice "listo" → verificar booking (0 LLM) ← NUEVO
    if (state.get("esperando_confirmacion_booking")
            and _parece_confirmacion(state["query"])):
        return {"route": ROUTE_AGENDAR,
                "clf_reason": "verificación de booking tras envío de link"}

    cfg = _cfg()["classification"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg["system_prompt"]),
        ("system",
         "Historial reciente (clasifica el ÚLTIMO mensaje del cliente usando "
         "este contexto; los follow-ups breves heredan el tema del hilo):\n"
         "{history}"),
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
            "query": state["query"],
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
    return {
        "sentiment": result.sentiment, "urgency": result.urgency,
        "intent": result.intent, "category": result.category,
        "clf_reason": result.reason, "route": route,
    }


def compute_route(result: AnalisisResult) -> str:
    """Regla oficial v4 — espejo de expected_route() en core/eval_classifier.py:

    - intent=fuera_de_dominio → rechazo amable + CTA
    - intent=hablar_humano    → escalamiento (el cliente lo pidió o lo aceptó)
    - intent=agendar_asesoria → sub-flujo de agendamiento
    - resto                   → atención/orientación (FAQ)
    El sentimiento NUNCA routea; solo modula el tono de respuestas_faq.
    """
    if result.intent == "fuera_de_dominio":
        return ROUTE_FUERA_DOMINIO
    if result.intent == "hablar_humano":
        return ROUTE_HANDOFF
    return ROUTE_AGENDAR if result.intent == "agendar_asesoria" else ROUTE_FAQ


def route_query(state: AgentState) -> str:
    """Lee la ruta del estado. Check defensivo ante valores inesperados."""
    route = state.get("route", ROUTE_FAQ)
    if route not in VALID_ROUTES:
        logger.warning("route inesperado '%s' → %s", route, ROUTE_FAQ)
        return ROUTE_FAQ
    return route


# ---------------------------------------------------------------------------
# NODOS DE ATENCIÓN (conversan con historial + siempre ofrecen contacto humano)
# ---------------------------------------------------------------------------
def respuestas_faq(state: AgentState) -> AgentState:
    """Nodo de ATENCIÓN: orienta con info completa (caso + documentos + proceso
    + precios), tono modulado por clasificación, cierre con doble oferta
    (agendar / hablar con ejecutiva)."""
    docs = retrieve(state["query"], k=3)
    context = "\n\n---\n\n".join(d.page_content for d in docs) or "Sin contexto."

    atn = _cfg()["atencion"]
    tono = atn["tonos"].get(state.get("sentiment", "neutro"), atn["tonos"]["neutro"])
    es_primer_contacto = not _recent_messages(state)

    prompt = ChatPromptTemplate.from_messages([
        ("system", atn["faq_system_prompt"]),
        MessagesPlaceholder("history"),          # ← chat fluido con contexto
        ("human", "{query}"),
    ])
    response = (prompt | LLM).invoke({
        "context": context,
        "query": state["query"],
        "history": _recent_messages(state),
        "disclosure": (atn["disclosure"] if es_primer_contacto
                       else "Eres el asistente virtual de Manzzo y Cía (ya te presentaste)."),
        "tono": tono,
    }).content

    return {
        "response": response,
        "context": [d.page_content for d in docs],
        "messages": [AIMessage(content=response)],
    }


def respuesta_fuera_dominio(state: AgentState) -> AgentState:
    """Redirección amable + CTA. Si el cliente acepta, su próximo mensaje
    se clasifica (con historial) como hablar_humano/agendar → flujo continuo."""
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(disclosure=atn["disclosure"])
    return {"response": response, "messages": [AIMessage(content=response)]}


# ---------------------------------------------------------------------------
# AGENDAMIENTO — captura datos → link prellenado → verificación de booking
# ---------------------------------------------------------------------------
def agendar_asesoria(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]

    # ── ETAPA C0: link ya enviado → VERIFICAR si el cliente agendó ─────  # ← NUEVO
    if state.get("esperando_confirmacion_booking") and state.get("lead_email"):
        desde = (datetime.fromisoformat(state["agenda_enviada_en"])
                 if state.get("agenda_enviada_en") else None)
        booking = buscar_booking(state["lead_email"], desde=desde)

        if booking:
            response = atn["agenda_confirmada"].format(
                nombre=(state.get("lead_nombre") or "").split()[0],
                **booking.model_dump(),
            )
            return {"response": response,
                    "booking": booking.model_dump(),
                    "esperando_confirmacion_booking": False,   # cierra el loop ✔
                    "messages": [AIMessage(content=response)]}

        # no encontrado (latencia Calendly o no completó) → recordatorio + link
        link = crear_link_agendamiento(LeadData(
            nombre=state.get("lead_nombre", "Cliente"),
            email=state["lead_email"],
            telefono=None,                       # no re-enviar teléfono aquí
            categoria=state.get("category"),
            modalidad=state.get("lead_modalidad"),
        ))
        response = atn["agenda_verificacion_pendiente"].format(link=link)
        return {"response": response, "messages": [AIMessage(content=response)]}

    # ── ETAPA A1: primera llegada al nodo → pedir los 3 datos ──────────
    if not state.get("recolectando_datos_agenda") and not state.get("lead_email"):
        response = atn["agenda_pedir_datos"]
        return {"response": response, "recolectando_datos_agenda": True,
                "messages": [AIMessage(content=response)]}

    # ── ETAPA A2: el cliente respondió con sus datos → extraer ─────────
    datos = _extraer_lead(state)
    nombre = datos.nombre or state.get("lead_nombre")
    email = datos.email or state.get("lead_email")
    modalidad = datos.modalidad or state.get("lead_modalidad")

    faltantes = []
    if not nombre:
        faltantes.append("tu nombre completo")
    if not email:
        faltantes.append("tu correo electrónico")
    if not modalidad:
        faltantes.append("si la prefieres online o presencial")
    if faltantes:
        response = f"¡Casi listo! Solo me falta: {' y '.join(faltantes)} 🙌"
        return {"response": response,
                "lead_nombre": nombre, "lead_email": email,
                "lead_modalidad": modalidad,
                "messages": [AIMessage(content=response)]}

    # ── HITL opcional: operador aprueba antes de enviar el link ────────
    if REQUIERE_APROBACION_AGENDAMIENTO:
        decision = interrupt({
            "tipo": "aprobacion_agendamiento",
            "thread_id": state.get("thread_id"),
            "query": state["query"],
            "lead": {"nombre": nombre, "email": email, "modalidad": modalidad},
            "categoria": state.get("category"),
            "urgencia": state.get("urgency"),
            "detalle": "El cliente completó sus datos. ¿Apruebas enviar el link?",
        })
        if not decision.get("aprobado"):
            response = ("Gracias por tu interés. Un asesor te contactará muy "
                        "pronto para coordinar la cita. "
                        + decision.get("nota", "")).strip()
            return {"response": response, "recolectando_datos_agenda": False,
                    "messages": [AIMessage(content=response)]}

    # ── ETAPA B: datos completos → lead + link prellenado (v2) ─────────
    lead = LeadData(
        nombre=nombre, email=email,
        telefono=_telefono_cliente(state),   # ← FIX: solo si thread_id es E.164
        categoria=state.get("category"),
        motivo=state.get("clf_reason"),
        modalidad=modalidad,
    )
    link = crear_link_agendamiento(lead)

    detalle_mod = (f"📍 {DIRECCION_OFICINA}"
                   if lead.modalidad == "presencial"
                   else "💻 Te llegará el link de Google Meet al confirmar")

    response = atn["agenda_link_enviado"].format(
        nombre=lead.nombre.split()[0],
        link=link,
        email=lead.email,
        modalidad_label=MODALIDAD_LABELS[lead.modalidad],
        detalle_mod=detalle_mod,
    )

    return {
        "response": response,
        "lead_nombre": lead.nombre,
        "lead_email": lead.email,
        "lead_modalidad": lead.modalidad,
        "recolectando_datos_agenda": False,     # cierra captura de datos
        "esperando_confirmacion_booking": True,                      # ← NUEVO
        "agenda_enviada_en": datetime.now(timezone.utc).isoformat(), # ← NUEVO
        "messages": [AIMessage(content=response)],
    }


# ---------------------------------------------------------------------------
# ESCALAMIENTO — solo si el cliente lo pide/acepta (intent: hablar_humano)
# ---------------------------------------------------------------------------
def handoff_humano(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]

    # Seguimiento post-cierre: no re-escalar ni re-guardar; anexar al caso.
    if state.get("closed"):
        response = atn["hilo_cerrado_message"].format(
            whatsapp_ejecutiva=WHATSAPP_EJECUTIVA,
            thread_id=state.get("thread_id"),
        )
        return {"response": response, "messages": [AIMessage(content=response)]}

    # 1) Resumen del caso para la ejecutiva (1 llamada LLM)
    transcript_txt = "\n".join(
        f"{t['rol']}: {t['contenido']}" for t in _transcript(state))
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", atn["summary_system_prompt"]),
        ("human", "{transcript}"),
    ])
    summary = (summary_prompt | LLM).invoke({"transcript": transcript_txt}).content

    # 2) Cerrar y guardar la conversación bajo su thread_id
    _persist_escalation(state, summary)

    # 3) Notificar al WhatsApp de la ejecutiva
    notificar_escalamiento(
        thread_id=state.get("thread_id", "sin-id"),
        resumen=summary,
        telefono_cliente=state.get("thread_id"),  # en prod: thread_id = teléfono
    )

    # 4) Respuesta al cliente: transparencia + derivación + código de expediente
    response = atn["handoff_message"].format(
        disclosure=atn["disclosure"],
        whatsapp_ejecutiva=WHATSAPP_EJECUTIVA,
        thread_id=state.get("thread_id"),
    )
    return {
        "response": response,
        "summary": summary,
        "escalated": True,
        "closed": True,
        "messages": [AIMessage(content=response)],
    }
