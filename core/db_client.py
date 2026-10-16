"""core/db_client.py — Cliente PostgreSQL/Supabase para datos de negocio.

Persiste:
  - conversations: hilo completo (una fila por SESIÓN de 24h)
  - messages: cada mensaje humano/AI
  - leads: fichas de intake (una por thread_id)
  - escalations: handoffs a ejecutiva (una por thread_id)
  - bookings: citas creadas en Google Calendar (una por thread_id)

Todas las operaciones son best-effort: no lanzan excepción hacia el grafo.

Sesiones: `conversations` ya no tiene UNIQUE(thread_id); cada 24h de
inactividad se cierra y se crea una nueva. Los INSERTs usan un COALESCE
que prioriza la sesión ACTIVA (closed_at IS NULL), con fallback a la
última para no perder linkage histórico.

Status normalizado a vocabulario cerrado:
  'open'       → sesión activa (closed_at IS NULL)
  'closed'     → cerrada por cierre normal o TTL
  'escalado'   → derivada a ejecutiva humana

Nota driver: psycopg3. Tras el pooler en modo transacción los prepared
 statements colisionan entre sesiones (DuplicatePreparedStatement), por eso
el engine se crea con prepare_threshold=None — mismo criterio que
core.persistence.build_postgres_checkpointer.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from core.threads import normalize_thread_id

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
_engine: Engine | None = None


# Subquery que devuelve el conversation_id "correspondiente":
#   - La sesión ACTIVA si existe (caso normal).
#   - Fallback: la última sesión conocida (por si algún flujo llama fuera de turno).
# Nunca devuelve una fila arbitraria.
_CONV_ID_SQL = """
    COALESCE(
        (SELECT c.id FROM conversations c
         WHERE c.thread_id = :thread_id AND c.closed_at IS NULL
         ORDER BY c.created_at DESC LIMIT 1),
        (SELECT c.id FROM conversations c
         WHERE c.thread_id = :thread_id
         ORDER BY c.created_at DESC LIMIT 1)
    )
"""


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        raw = DATABASE_URL or ""
        if not raw:
            raise RuntimeError("DATABASE_URL no está configurada")

        cleaned = raw.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
        if cleaned.endswith("?"):
            cleaned = cleaned[:-1]

        _engine = create_engine(
            cleaned,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"prepare_threshold": None},
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
# CONVERSATIONS
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
    """
    Actualiza la conversación ACTIVA (si existe) o crea una nueva.

    Reglas de status:
      - Si `closed=True` y no se pasa `status` explícito → 'closed'.
      - Si no, se respeta el `status` que venga (o el actual).
      - INSERT fallback: default 'open'.

    Ya no usa ON CONFLICT porque puede haber varias filas por thread_id.
    """
    try:
        thread_id = normalize_thread_id(thread_id)
        meta = metadata or {}
        meta["updated_at"] = _now()
        if closed:
            meta["closed_at"] = _now()

        with _get_engine().connect() as conn:
            # 1) Actualizar la conversación ACTIVA
            updated = conn.execute(
                text("""
                UPDATE conversations SET
                    channel = CASE
                        WHEN channel IS NULL
                          OR channel IN ('desconocido','sin_canal')
                        THEN COALESCE(:channel, channel)
                        ELSE channel
                    END,
                    status = CASE
                        WHEN :closed AND :status IS NULL THEN 'closed'
                        ELSE COALESCE(:status, status)
                    END,
                    intent_first = COALESCE(intent_first, :intent),
                    category = CASE
                        WHEN category IS NULL
                          OR category IN ('otro','consulta_general')
                          OR (:category IS NOT NULL
                              AND :category NOT IN ('otro','consulta_general'))
                        THEN COALESCE(:category, category)
                        ELSE category
                    END,
                    summary = COALESCE(:summary, summary),
                    metadata = COALESCE(metadata, '{}'::jsonb)
                               || CAST(:metadata_json AS jsonb),
                    updated_at = now(),
                    closed_at = CASE WHEN :closed THEN now() ELSE closed_at END
                WHERE id = (
                    SELECT c.id FROM conversations c
                    WHERE c.thread_id = :thread_id AND c.closed_at IS NULL
                    ORDER BY c.created_at DESC LIMIT 1
                )
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

            # 2) Si no había ninguna activa, crear una nueva
            if updated.rowcount == 0:
                conn.execute(
                    text("""
                    INSERT INTO conversations
                        (thread_id, channel, status, intent_first, category,
                         summary, metadata, created_at, updated_at)
                    VALUES (:thread_id, :channel,
                            COALESCE(:status, 'open'),
                            :intent, :category,
                            :summary, CAST(:metadata_json AS jsonb),
                            now(), now())
                    """),
                    {
                        "thread_id": thread_id,
                        "channel": _safe(channel),
                        "status": _safe(status),
                        "intent": _safe(intent),
                        "category": _safe(category),
                        "summary": _safe(summary),
                        "metadata_json": json.dumps(meta, ensure_ascii=False),
                    },
                )

            conn.commit()
    except Exception as e:
        logger.exception("[db] upsert_conversation falló: %s", e)


# ---------------------------------------------------------------------------
# MESSAGES
# ---------------------------------------------------------------------------
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
        thread_id = normalize_thread_id(thread_id)
        with _get_engine().connect() as conn:
            result = conn.execute(
                text(f"""
                INSERT INTO messages
                    (conversation_id, thread_id, role, content, route,
                     sentiment, urgency, intent, category, created_at)
                SELECT
                    ({_CONV_ID_SQL}),
                    :thread_id, :role, :content, :route, :sentiment,
                    :urgency, :intent, :category, now()
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
            if result.rowcount == 0:
                logger.warning(
                    "[db] insert_message: NO hay conversación para thread_id=%r. "
                    "¿Falta ensure_conversation() antes del primer mensaje?",
                    thread_id,
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
        thread_id = normalize_thread_id(thread_id)
        r = intake_respuestas or {}
        meta = {}
        if completed:
            meta["completed_at"] = _now()

        consent = r.get("consentimiento_datos")
        consent_bool = consent if isinstance(consent, bool) else None

        with _get_engine().connect() as conn:
            # ¿Ya estaba completo este lead? (evita notificación duplicada)
            prev = conn.execute(
                text("select completed_at from leads where thread_id = :thread_id"),
                {"thread_id": thread_id},
            ).fetchone()
            ya_notificado = prev is not None and prev.completed_at is not None

            conn.execute(
                text(f"""
                INSERT INTO leads
                    (conversation_id, thread_id, nombre, email, telefono,
                     situacion, etapa_proceso, categoria, consentimiento,
                     completed_at, metadata)
                SELECT
                    ({_CONV_ID_SQL}),
                    :thread_id, :nombre, :email, :telefono, :situacion,
                    :etapa, :categoria, :consentimiento,
                    CASE WHEN :completed THEN now() ELSE NULL END,
                    CAST(:metadata_json AS jsonb)
                ON CONFLICT (thread_id) DO UPDATE SET
                    nombre = COALESCE(:nombre, leads.nombre),
                    email = COALESCE(:email, leads.email),
                    telefono = COALESCE(:telefono, leads.telefono),
                    situacion = COALESCE(:situacion, leads.situacion),
                    etapa_proceso = COALESCE(:etapa, leads.etapa_proceso),
                    categoria = COALESCE(:categoria, leads.categoria),
                    consentimiento = COALESCE(:consentimiento, leads.consentimiento),
                    completed_at = CASE
                        WHEN :completed AND leads.completed_at IS NULL
                        THEN now() ELSE leads.completed_at END,
                    metadata = COALESCE(leads.metadata, '{{}}'::jsonb)
                               || CAST(:metadata_json AS jsonb)
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

        # Notificar a la abogada solo la primera vez que se completa la ficha
        if completed and not ya_notificado:
            from integrations.whatsapp.notify import notify_new_lead
            notify_new_lead({
                "nombre": r.get("nombre"),
                "telefono": r.get("telefono"),
                "email": r.get("email"),
                "categoria": category,
                "situacion": r.get("situacion_actual"),
                "etapa_proceso": r.get("etapa_proceso"),
            })
    except Exception as e:
        logger.exception("[db] upsert_lead falló: %s", e)


# ---------------------------------------------------------------------------
# ESCALACIONES
# ---------------------------------------------------------------------------
def insert_escalation(
    thread_id: str,
    summary: str,
    notificado_whatsapp: bool = False,
) -> None:
    try:
        thread_id = normalize_thread_id(thread_id)
        with _get_engine().connect() as conn:
            conn.execute(
                text(f"""
                INSERT INTO escalations
                    (conversation_id, thread_id, lead_id, summary,
                     notificado_whatsapp, created_at)
                SELECT
                    ({_CONV_ID_SQL}),
                    :thread_id,
                    (SELECT l.id FROM leads l
                     WHERE l.thread_id = :thread_id LIMIT 1),
                    :summary, :notificado, now()
                ON CONFLICT (thread_id) DO UPDATE SET
                    summary = :summary,
                    notificado_whatsapp = :notificado
                """),
                {
                    "thread_id": thread_id,
                    "summary": _safe(summary),
                    "notificado": notificado_whatsapp,
                },
            )

            prev = conn.execute(
                text("select notificado_whatsapp from escalations "
                     "where thread_id = :thread_id"),
                {"thread_id": thread_id},
            ).fetchone()
            ya_notificado = bool(prev and prev[0]) or notificado_whatsapp

            conn.commit()

        # Fallback: si el handoff no notificó por su canal, avisamos aquí
        if not ya_notificado:
            from integrations.whatsapp.notify import notify_escalation
            try:
                with _get_engine().connect() as conn2:
                    row = conn2.execute(
                        text("select telefono from leads where thread_id = :t"),
                        {"t": thread_id},
                    ).fetchone()
                telefono = row[0] if row else None
            except Exception:
                telefono = None

            ok = notify_escalation(thread_id, summary or "", telefono)
            if ok:
                with _get_engine().connect() as conn3:
                    conn3.execute(
                        text("update escalations set notificado_whatsapp = true "
                             "where thread_id = :t"),
                        {"t": thread_id},
                    )
                    conn3.commit()
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
        thread_id = normalize_thread_id(thread_id)
        b = booking or {}
        with _get_engine().connect() as conn:
            # ¿Ya existía booking para este hilo? (true = re-agenda)
            ya_existia = (
                conn.execute(
                    text("select 1 from bookings where thread_id = :thread_id"),
                    {"thread_id": thread_id},
                ).fetchone()
                is not None
            )

            conn.execute(
                text(f"""
                INSERT INTO bookings
                    (conversation_id, thread_id, lead_id, evento_id, fecha,
                     hora_inicio, hora_fin, modalidad, meet_link, html_link,
                     cliente_nombre, cliente_email, inicio_cita, created_at)
                SELECT
                    ({_CONV_ID_SQL}),
                    :thread_id,
                    (SELECT l.id FROM leads l
                     WHERE l.thread_id = :thread_id LIMIT 1),
                    :evento_id, :fecha, :hora_inicio, :hora_fin, :modalidad,
                    :meet_link, :html_link, :cliente_nombre, :cliente_email,
                    CAST(:inicio_cita AS timestamptz), now()
                ON CONFLICT (thread_id) DO UPDATE SET
                    evento_id = :evento_id,
                    fecha = :fecha,
                    hora_inicio = :hora_inicio,
                    hora_fin = :hora_fin,
                    modalidad = :modalidad,
                    meet_link = :meet_link,
                    html_link = :html_link,
                    cliente_nombre = COALESCE(:cliente_nombre, bookings.cliente_nombre),
                    cliente_email = COALESCE(:cliente_email, bookings.cliente_email),
                    inicio_cita = COALESCE(CAST(:inicio_cita AS timestamptz),
                                           bookings.inicio_cita),
                    lead_id = COALESCE(bookings.lead_id, EXCLUDED.lead_id)
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

        # Notificar a la abogada solo citas nuevas
        if not ya_existia:
            from integrations.whatsapp.notify import notify_booking
            notify_booking(b)
    except Exception as e:
        logger.exception("[db] insert_booking falló: %s", e)
