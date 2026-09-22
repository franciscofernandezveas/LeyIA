"""core/db_client.py — Cliente PostgreSQL/Supabase para datos de negocio.

Persiste:
  - conversations: hilo completo
  - messages: cada mensaje humano/AI
  - leads: fichas de intake
  - escalations: handoffs a ejecutiva
  - bookings: citas creadas en Google Calendar

Todas las operaciones son best-effort: no lanzan excepción hacia el grafo.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        raw = DATABASE_URL or ""
        if not raw:
            raise RuntimeError("DATABASE_URL no está configurada")

        # Supabase a veces añade ?pgbouncer=true; psycopg2 no lo entiende.
        cleaned = raw.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
        if "?" in cleaned and cleaned.endswith("?"):
            cleaned = cleaned[:-1]

        _engine = create_engine(
            cleaned,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val)[:5000]


# ---------------------------------------------------------------------------
# CONVERSACIONES
# ---------------------------------------------------------------------------
def upsert_conversation(
    thread_id: str,
    channel: str | None = None,
    status: str | None = None,
    intent: str | None = None,
    category: str | None = None,
    summary: str | None = None,
    closed: bool | None = None,
    metadata: dict | None = None,
) -> None:
    try:
        meta = metadata or {}
        meta["updated_at"] = _now()
        if closed:
            meta["closed_at"] = _now()

        with _get_engine().connect() as conn:
            conn.execute(
                text("""
                insert into conversations (thread_id, channel, status, intent_first, category, summary, metadata, created_at, updated_at)
                values (:thread_id, :channel, :status, :intent, :category, :summary, CAST(:metadata_json AS jsonb), now(), now())
                on conflict (thread_id) do update set
                    status = coalesce(:status, conversations.status),
                    intent_first = coalesce(:intent, conversations.intent_first),
                    category = coalesce(:category, conversations.category),
                    summary = coalesce(:summary, conversations.summary),
                    metadata = coalesce(conversations.metadata, '{}'::jsonb) || CAST(:metadata_json AS jsonb),
                    updated_at = now(),
                    closed_at = case when :closed then now() else conversations.closed_at end
                """),
                {
                    "thread_id": thread_id,
                    "channel": _safe(channel),
                    "status": _safe(status),
                    "intent": _safe(intent),
                    "category": _safe(category),
                    "summary": _safe(summary),
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                    "closed": bool(closed),
                },
            )
            conn.commit()
    except Exception as e:
        logger.exception("[db] upsert_conversation falló: %s", e)


def insert_message(
    thread_id: str,
    role: str,
    content: str,
    route: str | None = None,
    sentiment: str | None = None,
    urgency: str | None = None,
    intent: str | None = None,
    category: str | None = None,
) -> None:
    try:
        with _get_engine().connect() as conn:
            conn.execute(
                text("""
                insert into messages (conversation_id, thread_id, role, content, route, sentiment, urgency, intent, category, created_at)
                select c.id, :thread_id, :role, :content, :route, :sentiment, :urgency, :intent, :category, now()
                from conversations c where c.thread_id = :thread_id
                """),
                {
                    "thread_id": thread_id,
                    "role": role,
                    "content": _safe(content),
                    "route": _safe(route),
                    "sentiment": _safe(sentiment),
                    "urgency": _safe(urgency),
                    "intent": _safe(intent),
                    "category": _safe(category),
                },
            )
            conn.commit()
    except Exception as e:
        logger.exception("[db] insert_message falló: %s", e)


# ---------------------------------------------------------------------------
# LEADS / INTAKE
# ---------------------------------------------------------------------------
def upsert_lead(
    thread_id: str,
    intake_respuestas: dict,
    category: str | None = None,
    completed: bool = False,
) -> None:
    try:
        r = intake_respuestas or {}
        meta = {}
        if completed:
            meta["completed_at"] = _now()

        consent = r.get("consentimiento_datos")
        consent_bool = consent if isinstance(consent, bool) else None

        with _get_engine().connect() as conn:
            conn.execute(
                text("""
                insert into leads (conversation_id, thread_id, nombre, email, telefono, situacion, etapa_proceso, categoria, consentimiento, completed_at, metadata)
                select c.id, :thread_id, :nombre, :email, :telefono, :situacion, :etapa, :categoria, :consentimiento,
                       case when :completed then now() else null end,
                       CAST(:metadata_json AS jsonb)
                from conversations c where c.thread_id = :thread_id
                on conflict (thread_id) do update set
                    nombre = coalesce(:nombre, leads.nombre),
                    email = coalesce(:email, leads.email),
                    telefono = coalesce(:telefono, leads.telefono),
                    situacion = coalesce(:situacion, leads.situacion),
                    etapa_proceso = coalesce(:etapa, leads.etapa_proceso),
                    categoria = coalesce(:categoria, leads.categoria),
                    consentimiento = coalesce(:consentimiento, leads.consentimiento),
                    completed_at = case when :completed and leads.completed_at is null then now() else leads.completed_at end,
                    metadata = coalesce(leads.metadata, '{}'::jsonb) || CAST(:metadata_json AS jsonb)
                """),
                {
                    "thread_id": thread_id,
                    "nombre": _safe(r.get("nombre")),
                    "email": _safe(r.get("email")),
                    "telefono": _safe(r.get("telefono")),
                    "situacion": _safe(r.get("situacion_actual")),
                    "etapa": _safe(r.get("etapa_proceso")),
                    "categoria": _safe(category),
                    "consentimiento": consent_bool,
                    "completed": completed,
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                },
            )
            conn.commit()
    except Exception as e:
        logger.exception("[db] upsert_lead falló: %s", e)


# ---------------------------------------------------------------------------
# ESCALAMIENTOS
# ---------------------------------------------------------------------------
def insert_escalation(
    thread_id: str,
    summary: str,
    notificado_whatsapp: bool = False,
) -> None:
    try:
        with _get_engine().connect() as conn:
            conn.execute(
                text("""
                insert into escalations (conversation_id, thread_id, lead_id, summary, notificado_whatsapp, created_at)
                select c.id, :thread_id, l.id, :summary, :notificado, now()
                from conversations c
                left join leads l on l.thread_id = :thread_id
                where c.thread_id = :thread_id
                on conflict (thread_id) do update set
                    summary = :summary,
                    notificado_whatsapp = :notificado
                """),
                {
                    "thread_id": thread_id,
                    "summary": _safe(summary),
                    "notificado": notificado_whatsapp,
                },
            )
            conn.commit()
    except Exception as e:
        logger.exception("[db] insert_escalation falló: %s", e)


# ---------------------------------------------------------------------------
# BOOKINGS / AGENDA
# ---------------------------------------------------------------------------
def insert_booking(
    thread_id: str,
    booking: dict,
) -> None:
    try:
        b = booking or {}
        with _get_engine().connect() as conn:
            conn.execute(
                text("""
                insert into bookings (conversation_id, thread_id, lead_id, evento_id, fecha, hora_inicio, hora_fin, modalidad, meet_link, html_link, cliente_nombre, cliente_email, inicio_cita, created_at)
                select c.id, :thread_id, l.id, :evento_id, :fecha, :hora_inicio, :hora_fin, :modalidad, :meet_link, :html_link, :cliente_nombre, :cliente_email, CAST(:inicio_cita AS timestamptz), now()
                from conversations c
                left join leads l on l.thread_id = :thread_id
                where c.thread_id = :thread_id
                on conflict (thread_id) do update set
                    evento_id = :evento_id,
                    fecha = :fecha,
                    hora_inicio = :hora_inicio,
                    hora_fin = :hora_fin,
                    modalidad = :modalidad,
                    meet_link = :meet_link,
                    html_link = :html_link,
                    cliente_nombre = coalesce(:cliente_nombre, bookings.cliente_nombre),
                    cliente_email = coalesce(:cliente_email, bookings.cliente_email),
                    inicio_cita = coalesce(CAST(:inicio_cita AS timestamptz), bookings.inicio_cita),
                    lead_id = coalesce(bookings.lead_id, excluded.lead_id)
                """),
                {
                    "thread_id": thread_id,
                    "evento_id": _safe(b.get("evento_id")),
                    "fecha": _safe(b.get("fecha")),
                    "hora_inicio": _safe(b.get("hora_inicio")),
                    "hora_fin": _safe(b.get("hora_fin")),
                    "modalidad": _safe(b.get("modalidad_label") or b.get("modalidad")),
                    "meet_link": _safe(b.get("meet_link")),
                    "html_link": _safe(b.get("html_link")),
                    "cliente_nombre": _safe(b.get("cliente_nombre")),
                    "cliente_email": _safe(b.get("cliente_email")),
                    "inicio_cita": b.get("inicio_cita"),
                },
            )
            conn.commit()
    except Exception as e:
        logger.exception("[db] insert_booking falló: %s", e)
