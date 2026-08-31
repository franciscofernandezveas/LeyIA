"""graph/nodes.py — Nodos del agente (Manzzo y Cía).

v6 — Integración del nodo INTAKE (ficha proactiva de lead + caso):
  - ROUTE_INTAKE: intent=hablar_humano ya no deriva directo; primero arma
    la ficha (graph/intake.py) y handoff_humano se ejecuta al completarla
    (edge condicional intake → handoff, misma invocación).
  - analyze_sentiment: short-circuit del intake (0 LLM) integrado al bloque
    de sub-flujos, con abort ("no quiero", "olvidalo"...) y TTL de 24 h.
  - route_query: si intake_completado y el cliente vuelve a pedir humano,
    salta directo a handoff (no repetir la ficha).
  - handoff_humano: el resumen incluye la FICHA DE INTAKE formateada y el
    JSON persistido lleva las respuestas de intake.
  - _cfg(): valida también la sección `intake:` de prompts.yaml.

v7 — Integración SheetDB:
  - handoff_humano → sync_lead() (crea o actualiza lead completo).
  - agendar_asesoria (booking confirmado) → actualizar_booking() (marca
    lead_status='agendado' y guarda fecha/link/modalidad).
  - Toda sincronización es best-effort: fallos se loguean, no crashean el agente.
"""
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import interrupt

from core.contracts import (
    EMAIL_RE,
    AgentState, AnalisisResult, HitlPayload, LeadExtract, TipoHITL,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF, ROUTE_INTAKE,
    VALID_ROUTES, fallback_analisis,
)
from core.llm import LLM, with_structured_output
from core.rag import retrieve
from tools.calendly import (
    DIRECCION_OFICINA, MODALIDAD_LABELS, LeadData,
    buscar_booking, crear_link_agendamiento,
)
from tools.notify_whatsapp import WHATSAPP_EJECUTIVA, notificar_escalamiento

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "core" / "prompts.yaml"
ESCALATIONS_DIR = Path("escalations")     # TODO prod: mover a DB/S3 cifrado

# False = el link se envía directo (prioridad negocio: agendar).
# True  = reactiva el HITL de aprobación por operador (main.py/app.py lo soportan).
REQUIERE_APROBACION_AGENDAMIENTO = False

# Políticas de sub-flujos
AGENDA_MAX_VERIFICACIONES = 3      # escape del loop de verificación de booking
AGENDA_CAPTURA_TTL_HORAS = 24      # captura de datos de agenda expira
AGENDA_LINK_TTL_HORAS = 48         # link enviado sin confirmar expira
INTAKE_TTL_HORAS = 24              # ficha de intake abandonada expira

# Fail-fast de prompts.yaml al primer uso
REQUIRED_KEYS = {
    "atencion": {"disclosure", "cta_agendar", "faq_system_prompt", "tonos",
                 "summary_system_prompt", "fuera_dominio_message",
                 "handoff_message", "hilo_cerrado_message",
                 "agenda_pedir_datos", "agenda_link_enviado",
                 "agenda_confirmada", "agenda_verificacion_pendiente"},
    "classification": {"system_prompt", "few_shot_examples"},
    "intake": {"apertura", "ack_completado", "aviso_saltar",
               "sin_consentimiento"},
}

# Contrato de placeholders YAML ↔ kwargs que entrega cada nodo.
TEMPLATE_ARGS = {
    "atencion.faq_system_prompt": {"disclosure", "tono", "context"},
    "atencion.fuera_dominio_message": {"disclosure"},
    "atencion.handoff_message": {"disclosure", "whatsapp_ejecutiva", "thread_id"},
    "atencion.hilo_cerrado_message": {"whatsapp_ejecutiva", "thread_id"},
    "atencion.agenda_pedir_datos": set(),
    "atencion.agenda_link_enviado": {"nombre", "link", "email",
                                     "modalidad_label", "detalle_mod"},
    "atencion.agenda_confirmada": {"nombre", "fecha", "hora_inicio", "hora_fin",
                                   "nombre_evento", "modalidad_linea",
                                   "reschedule_url", "cancel_url"},
    "atencion.agenda_verificacion_pendiente": {"link"},
    "intake.apertura": set(),
    "intake.ack_completado": {"nombre"},
    "intake.aviso_saltar": set(),
    "intake.sin_consentimiento": set(),
}
_PLACEHOLDER_RE = re.compile(r"{(\w+)}")

# Confirmación de booking: palabra completa y sin negación
_CONF_RE = re.compile(
    r"\b(listo|ya agend\w*|agend[ée]|hecho|confirm[ée]|qued[óo])\b", re.IGNORECASE
)
_NEGACIONES = ("no ", "aún no", "aun no", "todavía no", "todavia no", "no he")

# Abort explícito de cualquier sub-flujo (agenda o intake)
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

    # Contrato de placeholders: la plantilla pide EXACTAMENTE lo que el nodo da.
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
    """Truncado para el resumen: conversaciones largas no inflan el prompt."""
    txt = "\n".join(f"{t['rol']}: {t['contenido']}" for t in _transcript(state))
    if len(txt) > max_chars:
        txt = "…[inicio de la conversación omitido]\n" + txt[-max_chars:]
    return txt


def _ficha_intake(state: AgentState) -> str:
    """Bloque 'FICHA DE INTAKE' para el resumen de la ejecutiva."""
    r = state.get("intake_respuestas") or {}
    if not r:
        return ""
    filas = "\n".join(f"- {k}: {v}" for k, v in r.items() if v not in (None, ""))
    return f"FICHA DE INTAKE:\n{filas}\n\n"


def _parece_confirmacion(query: str) -> bool:
    """Palabra completa + descarta negaciones ('no estoy listo')."""
    q = query.lower()
    return bool(_CONF_RE.search(q)) and not any(n in q for n in _NEGACIONES)


def _parece_abort(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _ABORT)


def _telefono_cliente(state: AgentState) -> str | None:
    """En prod thread_id = teléfono WhatsApp (E.164). En CLI/web → None."""
    tid = state.get("thread_id") or ""
    return tid if re.fullmatch(r"\+\d{8,15}", tid) else None


def _extraer_lead(state: AgentState) -> LeadExtract:
    """Extrae nombre/email/modalidad para agendamiento (1 LLM). Nunca lanza;
    ante error devuelve LeadExtract vacío → el nodo re-pregunta."""
    extractor = ChatPromptTemplate.from_messages([
        ("system",
         "Extrae del mensaje: nombre completo, email y modalidad "
         "('online' o 'presencial'). Normaliza sinónimos: 'videollamada', "
         "'meet', 'virtual' → online; 'oficina', 'en persona' → presencial. "
         "Si un dato falta, devuélvelo null. Responde solo el esquema."),
        ("human", "{query}"),
    ]) | with_structured_output(LeadExtract)
    try:
        return extractor.with_retry(stop_after_attempt=2).invoke({"query": state["query"]})
    except Exception as e:
        logger.warning("extractor de lead falló (se re-preguntará): %s", e)
        return LeadExtract()


def _persist_escalation(state: AgentState, summary: str) -> Path:
    """Cierra y guarda la conversación (incluye ficha de intake).
    Filename sanitizado. TODO prod: DB/S3 cifrado en vez de filesystem."""
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
        "intake": state.get("intake_respuestas"),        # ficha completa
        "booking": state.get("booking"),
        "resumen_ejecutiva": summary,
        "transcript": _transcript(state),
    }
    ESCALATIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(state.get("thread_id") or "sin-id"))
    path = ESCALATIONS_DIR / f"{safe_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("conversación guardada y cerrada: %s", path)
    return path


# ---------------------------------------------------------------------------
# ENTRADA + CLASIFICACIÓN
# ---------------------------------------------------------------------------
def receive_message(state: AgentState) -> AgentState:
    """query se deriva SIEMPRE del último mensaje humano: `messages` es la
    única fuente de verdad (elimina stale query cuando el canal solo manda
    el mensaje nuevo)."""
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if m.type == "human"), None
    )
    q = last_human.content if last_human else ""
    if not isinstance(q, str):
        q = str(q)
    return {} if q == state.get("query") else {"query": q}


def analyze_sentiment(state: AgentState) -> AgentState:
    """Clasificación unificada con contexto + short-circuits de sub-flujo
    (0 LLM): agenda (captura / verificación de booking) e intake (ficha).
    Abort explícito y TTLs limpian flags y dejan pasar al clasificador."""

    # Caso cerrado/derivado → seguimiento por handoff (0 LLM)
    if state.get("closed"):
        return {"route": ROUTE_HANDOFF, "clf_reason": "seguimiento de caso cerrado"}

    query = state["query"]
    recolectando = bool(state.get("recolectando_datos_agenda"))
    esperando_booking = bool(state.get("esperando_confirmacion_booking"))
    en_intake = bool(state.get("intake_activo"))
    reset: dict = {}

    if recolectando or esperando_booking or en_intake:
        if _parece_abort(query):
            logger.info("cliente aborta sub-flujo → reclasificar")
            reset = {"recolectando_datos_agenda": False,
                     "esperando_confirmacion_booking": False,
                     "intake_activo": False}
        elif en_intake and _expirado(state.get("intake_started_en"), INTAKE_TTL_HORAS):
            logger.info("ficha de intake expirada (> %dh) → reclasificar",
                        INTAKE_TTL_HORAS)
            reset = {"intake_activo": False}
        elif recolectando and _expirado(state.get("agenda_started_en"),
                                        AGENDA_CAPTURA_TTL_HORAS):
            logger.info("captura de agenda expirada (> %dh) → reclasificar",
                        AGENDA_CAPTURA_TTL_HORAS)
            reset = {"recolectando_datos_agenda": False}
        elif esperando_booking and _expirado(state.get("agenda_enviada_en"),
                                             AGENDA_LINK_TTL_HORAS):
            logger.info("link sin confirmar expirado (> %dh) → reclasificar",
                        AGENDA_LINK_TTL_HORAS)
            reset = {"esperando_confirmacion_booking": False}
        elif en_intake:
            return {"route": ROUTE_INTAKE,
                    "clf_reason": "sub-flujo: ficha de intake (0 LLM)"}
        elif recolectando:
            return {"route": ROUTE_AGENDAR,
                    "clf_reason": "sub-flujo: captura de datos de agendamiento"}
        elif esperando_booking and _parece_confirmacion(query):
            return {"route": ROUTE_AGENDAR,
                    "clf_reason": "verificación de booking tras envío de link"}

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
    return {
        "sentiment": result.sentiment, "urgency": result.urgency,
        "intent": result.intent, "category": result.category,
        "clf_reason": result.reason, "route": route,
        **reset,                                  # limpieza por abort/TTL
    }


def compute_route(result: AnalisisResult) -> str:
    """Regla oficial — espejo de expected_route() en core/eval_classifier.py.

    - intent=fuera_de_dominio → respuesta_fuera_dominio
    - intent=hablar_humano    → INTAKE (ficha) → handoff al completarla
    - intent=agendar_asesoria → sub-flujo de agendamiento
    - resto                   → atención/orientación (FAQ)
    El sentimiento NUNCA routea; solo modula el tono de respuestas_faq.
    """
    if result.intent == "fuera_de_dominio":
        return ROUTE_FUERA_DOMINIO
    if result.intent == "hablar_humano":
        return ROUTE_INTAKE
    return ROUTE_AGENDAR if result.intent == "agendar_asesoria" else ROUTE_FAQ


def route_query(state: AgentState) -> str:
    """Lee la ruta del estado con dos guardias defensivas:
    - intake_completado + nuevo pedido de humano → handoff directo
      (no repetir la ficha).
    - ruta fuera de VALID_ROUTES → FAQ."""
    route = state.get("route", ROUTE_FAQ)
    if route == ROUTE_INTAKE and state.get("intake_completado"):
        return ROUTE_HANDOFF
    if route not in VALID_ROUTES:
        logger.warning("route inesperado '%s' → %s", route, ROUTE_FAQ)
        return ROUTE_FAQ
    return route


# ---------------------------------------------------------------------------
# NODOS DE ATENCIÓN
# ---------------------------------------------------------------------------
def respuestas_faq(state: AgentState) -> AgentState:
    """Atención con RAG + historial. Ni retrieve ni el LLM lanzan excepciones."""
    atn = _cfg()["atencion"]

    try:
        docs = retrieve(state["query"], k=3)
    except Exception as e:
        logger.warning("retrieve falló; continúo sin contexto: %s", e)
        docs = []
    context = "\n\n---\n\n".join(d.page_content for d in docs) or "Sin contexto."

    tono = atn["tonos"].get(state.get("sentiment", "neutro"), atn["tonos"]["neutro"])

    # Primer contacto = aún no hay mensajes del asistente en el hilo.
    es_primer_contacto = not any(m.type == "ai" for m in state.get("messages", []))

    # history sin el último humano (ese va como {query} al final).
    history = _recent_messages(state, k=7)[:-1]

    prompt = ChatPromptTemplate.from_messages([
        ("system", atn["faq_system_prompt"]),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    try:
        response = (prompt | LLM).invoke({
            "context": context,
            "query": state["query"],
            "history": history,
            "disclosure": (atn["disclosure"] if es_primer_contacto
                           else "Eres el asistente virtual de Manzzo y Cía (ya te presentaste)."),
            "tono": tono,
        }).content
    except Exception as e:
        logger.exception("LLM de FAQ falló: %s", e)
        response = ("Disculpa, tuve un problema técnico procesando tu mensaje 🙏. "
                    "¿Podrías repetírmelo en unos minutos? Si es urgente, dímelo "
                    "y te comunico con una ejecutiva.")

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

    # ── ETAPA C0: link enviado y vigente → VERIFICAR booking ────────────
    if (state.get("esperando_confirmacion_booking")
            and state.get("lead_email")
            and not _expirado(state.get("agenda_enviada_en"), AGENDA_LINK_TTL_HORAS)):

        booking = buscar_booking(
            state["lead_email"], desde=_parse_dt(state.get("agenda_enviada_en"))
        )
        if booking:
            try:
                response = atn["agenda_confirmada"].format(
                    nombre=_primer_nombre(state.get("lead_nombre")),
                    **booking.model_dump(),
                )
            except KeyError as e:
                logger.exception("agenda_confirmada no calza con BookingInfo: %s", e)
                response = (f"✅ ¡Agendado! Te enviamos la confirmación y los "
                            f"detalles de tu cita a {state['lead_email']}.")

            # v7: sincronizar con SheetDB el booking confirmado (mejor esfuerzo)
            try:
                from integrations.sheetdb import actualizar_booking
                actualizar_booking(state)
            except Exception as e:
                logger.exception("Error actualizando booking en SheetDB: %s", e)

            return {"response": response,
                    "booking": booking.model_dump(),
                    "esperando_confirmacion_booking": False,
                    "verificaciones_fallidas": 0,
                    "messages": [AIMessage(content=response)]}

        # No encontrado → contador con escape a ejecutiva
        fallidas = state.get("verificaciones_fallidas", 0) + 1
        if fallidas >= AGENDA_MAX_VERIFICACIONES:
            response = ("Calendly aún no registra tu agendamiento 😕. "
                        "Si prefieres, te comunico con una ejecutiva y ella "
                        "agenda la hora directamente contigo. ¿Te parece?")
            return {"response": response,
                    "esperando_confirmacion_booking": False,
                    "verificaciones_fallidas": 0,
                    "messages": [AIMessage(content=response)]}

        # Reusar el link ya enviado (idempotente)
        link = state.get("agenda_link") or crear_link_agendamiento(LeadData(
            nombre=state.get("lead_nombre", "Cliente"),
            email=state["lead_email"],
            telefono=None,
            categoria=state.get("category"),
            modalidad=state.get("lead_modalidad"),
        ))
        response = atn["agenda_verificacion_pendiente"].format(link=link)
        return {"response": response,
                "verificaciones_fallidas": fallidas,
                "agenda_link": link,
                "messages": [AIMessage(content=response)]}

    # ── ETAPA A1: primera llegada → pedir los 3 datos ───────────────────
    if not state.get("recolectando_datos_agenda") and not state.get("lead_email"):
        response = atn["agenda_pedir_datos"]
        return {"response": response,
                "recolectando_datos_agenda": True,
                "agenda_started_en": _ahora_iso(),
                "messages": [AIMessage(content=response)]}

    # ── ETAPA A2: el cliente respondió con sus datos → extraer ──────────
    datos = _extraer_lead(state)                     # nunca lanza (fallback vacío)
    nombre = datos.nombre or state.get("lead_nombre")
    modalidad = datos.modalidad or state.get("lead_modalidad")

    # El email solo se acepta si pasa EMAIL_RE (nunca se delega al LLM).
    email_invalido = bool(datos.email) and not EMAIL_RE.fullmatch(datos.email)
    email = (datos.email
             if datos.email and not email_invalido
             else state.get("lead_email"))

    faltantes = []
    if not nombre:
        faltantes.append("tu nombre completo")
    if not email:
        faltantes.append("un correo válido (ej: nombre@correo.com)"
                         if email_invalido else "tu correo electrónico")
    if not modalidad:
        faltantes.append("si la prefieres online o presencial")
    if faltantes:
        response = f"¡Casi listo! Solo me falta: {' y '.join(faltantes)} 🙌"
        return {"response": response,
                "lead_nombre": nombre, "lead_email": email,
                "lead_modalidad": modalidad,
                "messages": [AIMessage(content=response)]}

    # ── HITL opcional: operador aprueba antes de enviar el link ─────────
    if REQUIERE_APROBACION_AGENDAMIENTO:
        decision = interrupt(HitlPayload(
            tipo=TipoHITL.APROBACION_AGENDAMIENTO.value,
            thread_id=state.get("thread_id", ""),
            query=state["query"],
            detalle="El cliente completó sus datos. ¿Apruebas enviar el link?",
            lead={"nombre": nombre, "email": email, "modalidad": modalidad},
            categoria=state.get("category", "otro"),
            urgencia=state.get("urgency", "media"),
        ))
        if not decision.get("aprobado"):
            response = ("Gracias por tu interés. Un asesor te contactará muy "
                        "pronto para coordinar la cita. "
                        + decision.get("nota", "")).strip()
            return {"response": response,
                    "recolectando_datos_agenda": False,
                    "messages": [AIMessage(content=response)]}

    # ── ETAPA B: datos completos y válidos → lead + link prellenado ─────
    lead = LeadData(
        nombre=nombre, email=email,
        telefono=_telefono_cliente(state),       # solo si thread_id es E.164
        categoria=state.get("category"),
        motivo=state.get("clf_reason"),
        modalidad=modalidad,
    )
    link = crear_link_agendamiento(lead)
    detalle_mod = (f"📍 {DIRECCION_OFICINA}"
                   if lead.modalidad == "presencial"
                   else "💻 Te llegará el link de Google Meet al confirmar")
    response = atn["agenda_link_enviado"].format(
        nombre=_primer_nombre(lead.nombre),
        link=link, email=lead.email,
        modalidad_label=MODALIDAD_LABELS[lead.modalidad],
        detalle_mod=detalle_mod,
    )
    return {
        "response": response,
        "lead_nombre": lead.nombre,
        "lead_email": lead.email,
        "lead_modalidad": lead.modalidad,
        "recolectando_datos_agenda": False,
        "esperando_confirmacion_booking": True,   # abre loop de verificación
        "agenda_enviada_en": _ahora_iso(),
        "agenda_link": link,                      # reusar en verificaciones
        "verificaciones_fallidas": 0,
        "messages": [AIMessage(content=response)],
    }


# ---------------------------------------------------------------------------
# ESCALAMIENTO — solo si el cliente lo pide/acepta, TRAS completar la ficha
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

    # 2) Cerrar y guardar ANTES de notificar (fuente de verdad local)
    _persist_escalation(state, summary)

    # 3) Notificar por WhatsApp (best-effort)
    notificacion_ok = True
    try:
        notificar_escalamiento(
            thread_id=state.get("thread_id", "sin-id"),
            resumen=summary,
            telefono_cliente=_telefono_cliente(state),
        )
    except Exception as e:
        logger.exception("notificación WhatsApp falló (caso guardado igual): %s", e)
        notificacion_ok = False

    # v7: sincronizar con Google Sheets vía SheetDB (best-effort)
    try:
        from integrations.sheetdb import sync_lead
        sync_lead(state)
    except Exception as e:
        logger.exception("Error sincronizando lead con SheetDB: %s", e)

    # 4) Respuesta al cliente
    response = atn["handoff_message"].format(
        disclosure=atn["disclosure"],
        whatsapp_ejecutiva=WHATSAPP_EJECUTIVA,
        thread_id=state.get("thread_id"),
    )
    return {
        "response": response,
        "summary": summary,
        "closed": True,
        "notificacion_pendiente": not notificacion_ok,
        "messages": [AIMessage(content=response)],
    }
