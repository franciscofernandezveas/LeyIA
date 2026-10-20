"""core/db_client.py — Cliente PostgreSQL/Supabase para datos de negocio.

Persiste:
  - conversations: hilo completo (una fila por SESIÓN de 24h)
  - messages: cada mensaje humano/AI
  - leads: ficha de intake ACTIVA por thread_id (ciclo nuevo → el caso
    anterior se archiva en metadata['casos_anteriores'])
  - escalations: handoffs a ejecutiva (una por CONVERSACIÓN + motivo)
  - bookings: citas creadas en Google Calendar (una por evento_id)

Regla de oro SQL (psycopg3): un parámetro NUNCA debe aparecer más de una
vez en la misma sentencia sin CAST, o PostgreSQL lanza AmbiguousParameter.
Por eso las banderas derivadas (effective_status / category_is_specific /
nuevo_ciclo) se computan en Python y se pasan como parámetros distintos.

Todas las operaciones son best-effort: no lanzan excepción hacia el grafo.

Nota driver: psycopg3. Tras el pooler en modo transacción los prepared
 statements colisionan entre sesiones (DuplicatePreparedStatement), por eso
el engine se crea con prepare_threshold=None — mismo criterio que
core.persistence.build_postgres_checkpointer.

REQUIERE migración (ver migración bookings/escalaciones):
  - bookings: DROP unique(thread_id); ADD unique(evento_id)
  - escalations: DROP unique(thread_id); ADD unique parcial
    (conversation_id, motivo) WHERE conversation_id IS NOT NULL
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
#   - Fallback: la última sesión conocida.
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
def get_active_conversation_id(thread_id: str) -> str | None:
    """ID de la sesión activa (closed_at IS NULL) para el thread.
    None si no hay sesión o si la BD falla (best-effort)."""
    try:
        thread_id = normalize_thread_id(thread_id)
        with _get_engine().connect() as conn:
            row = conn.execute(
                text("""
                    SELECT id FROM conversations
                    WHERE thread_id = :tid AND closed_at IS NULL
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"tid": thread_id},
            ).fetchone()
            return str(row[0]) if row else None
    except Exception as e:
        logger.warning("[db] get_active_conversation_id falló: %s", e)
        return None


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
    Actualiza la conversación ACTIVA (o crea una nueva).

    Reglas de status:
      - closed=True sin status explícito → 'closed'.
      - Caso contrario → se respeta el status que venga (o el actual).
      - INSERT fallback: default 'open'.
    """
    try:
        thread_id = normalize_thread_id(thread_id)
        meta = metadata or {}
        meta["updated_at"] = _now()
        if closed:
            meta["closed_at"] = _now()

        # --- Cómputo en Python (evita AmbiguousParameter de psycopg3) ---
        effective_status = status
        if closed and status is None:
            effective_status = "closed"

        # Bandera "la categoría nueva es específica" (≠ 'otro'/'consulta_general')
        category_is_specific = (
            category is not None
            and category not in ("otro", "consulta_general")
        )

        with _get_engine().connect() as conn:
            updated = conn.execute(
                text("""
                UPDATE conversations SET
                    channel = CASE
                        WHEN channel IS NULL
                          OR channel IN ('desconocido','sin_canal')
                        THEN COALESCE(:channel, channel)
                        ELSE channel
                    END,
                    status = COALESCE(:status, status),
                    intent_first = COALESCE(intent_first, :intent),
                    category = CASE
                        WHEN category IS NULL
                          OR category IN ('otro','consulta_general')
                          OR :category_is_specific
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
                    "status": _safe(effective_status),
                    "intent": _safe(intent),
                    "category": _safe(category),
                    "category_is_specific": category_is_specific,
                    "summary": _safe(summary),
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                    "closed": bool(closed),
                },
            )

            # Sin sesión activa → crear una nueva
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
                        "status": _safe(effective_status),
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
                    "[db] insert_message: NO hay conversación para thread_id=%r.",
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
    notificar_crm: bool = True,
) -> None:
    """
    Una ficha ACTIVA por thread_id. Si la ficha anterior ya se completó y
    el intake llega desde OTRA sesión de conversación, se inicia un ciclo
    nuevo: el caso anterior se archiva en metadata['casos_anteriores'] y
    completed_at se reinicia (→ la nueva ficha completa vuelve a notificar).

    v1.1 (fix #10, auditoría) — Parámetro notificar_crm:
      - True (default): comportamiento histórico. Al completarse la ficha
        (primera vez del ciclo), dispara notify_new_lead (WA) +
        insert_escalation('ficha_completada', silenciosa).
      - False: persiste el lead con completed_at=now() pero NO dispara
        ninguna notificación. Usado por _completar_ficha del intake, porque
        route_post_intake SIEMPRE envía intake_completado=True a
        ROUTE_HANDOFF en el mismo turno y handoff_humano es el emisor único
        (notify_escalation + email + sync_lead). Antes había doble WA por
        el mismo lead.
      - ⚠️ Si en el futuro se añade un flujo "ficha completa → END" (sin
        handoff inmediato), ese call site debe usar notificar_crm=True
        (explícito) para que el equipo reciba el aviso.
    """
    try:
        thread_id = normalize_thread_id(thread_id)
        r = intake_respuestas or {}
        meta = {}
        if completed:
            meta["completed_at"] = _now()

        consent = r.get("consentimiento_datos")
        consent_bool = consent if isinstance(consent, bool) else None

        with _get_engine().connect() as conn:
            prev = conn.execute(
                text("SELECT completed_at, conversation_id FROM leads "
                     "WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            ).fetchone()

            sesion = conn.execute(
                text("SELECT id FROM conversations "
                     "WHERE thread_id = :thread_id AND closed_at IS NULL "
                     "ORDER BY created_at DESC LIMIT 1"),
                {"thread_id": thread_id},
            ).fetchone()

            # Ciclo nuevo: ficha anterior completada Y otra sesión activa
            nuevo_ciclo = (
                prev is not None
                and prev.completed_at is not None
                and sesion is not None
                and prev.conversation_id != sesion[0]
            )
            ya_notificado = (
                prev is not None
                and prev.completed_at is not None
                and not nuevo_ciclo
            )

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
                        WHEN :completed
                             AND (leads.completed_at IS NULL OR :nuevo_ciclo)
                        THEN now()
                        WHEN :nuevo_ciclo THEN NULL
                        ELSE leads.completed_at
                    END,
                    metadata = CASE WHEN :nuevo_ciclo THEN
                            COALESCE(leads.metadata, '{{}}'::jsonb)
                            || jsonb_build_object('casos_anteriores',
                                 COALESCE(leads.metadata->'casos_anteriores',
                                          '[]'::jsonb)
                                 || jsonb_build_array(jsonb_build_object(
                                      'situacion', leads.situacion,
                                      'categoria', leads.categoria,
                                      'completed_at', leads.completed_at)))
                        ELSE COALESCE(leads.metadata, '{{}}'::jsonb)
                        END
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
                    "nuevo_ciclo": nuevo_ciclo,
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                },
            )
            conn.commit()

        # Primera vez que se completa este ciclo → notificar + escalación silenciosa
            # Primera vez que se completa este ciclo → notificar (si corresponde)
        # + fila silenciosa en escalations.
        #
        # v1.1 (fix #10): se añade el flag notificar_crm para suprimir la
        # notificación cuando un nodo posterior (handoff_humano) será el
        # emisor único. Antes, la combinación _completar_ficha + handoff_humano
        # en el mismo turno producía doble WA (notify_new_lead +
        # notify_escalation) por el mismo evento — la ejecutiva recibía
        # ≥2 WhatsApp + email redundantes por lead.
        if completed and not ya_notificado:
            if notificar_crm:
                from integrations.whatsapp.notify import notify_new_lead
                notify_new_lead({
                    "nombre": r.get("nombre"),
                    "telefono": r.get("telefono"),
                    "email": r.get("email"),
                    "categoria": category,
                    "situacion": r.get("situacion_actual"),
                    "etapa_proceso": r.get("etapa_proceso"),
                })

                # Registro silencioso: si el cliente desaparece sin decidir,
                # la abogada aún lo ve en su cola de trabajo. Solo tiene
                # sentido si notificar_crm=True — con False, el nodo
                # posterior (handoff_humano) crea su propia fila
                # (hablar_abogada o contacto_directo), por lo que el summary
                # "está considerando" (incorrecto en el escenario de handoff
                # inmediato) se omite aquí.
                insert_escalation(
                    thread_id=thread_id,
                    summary="Cliente completó la ficha y está considerando "
                            "agendar o hablar con la abogada.",
                    motivo="ficha_completada",
                    notificar=False,
                )
    
    except Exception as e:
        logger.exception("[db] upsert_lead falló: %s", e)


# ---------------------------------------------------------------------------
# ESCALACIONES
# ---------------------------------------------------------------------------
def insert_escalation(
    thread_id: str,
    summary: str | None = None,
    motivo: str = "hablar_abogada",
    notificar: bool = True,
) -> bool:
    """
    Una escalación por (conversación, motivo) — NO por thread_id.

    Devuelve True si la notif quedó enviada (o no hacía falta).

    motivo:    'ficha_completada' | 'hablar_abogada' | 'derivacion_parcial'
    notificar: True → envía WhatsApp si esta fila no fue notificada antes.
               False → solo upserta la fila (silencioso).

    Comportamiento: el mismo cliente que escala en la sesión A y luego en
    la sesión B genera 2 filas y 2 notificaciones. Dos escalaciones con el
    mismo motivo dentro de la misma sesión → 1 fila, 1 notificación.
    """
    try:
        thread_id = normalize_thread_id(thread_id)
        motivo = motivo or "hablar_abogada"

        with _get_engine().connect() as conn:
            row = conn.execute(
                text(f"""
                INSERT INTO escalations
                    (conversation_id, thread_id, lead_id, summary, motivo,
                     notificado_whatsapp, created_at)
                SELECT
                    ({_CONV_ID_SQL}),
                    :thread_id,
                    (SELECT l.id FROM leads l
                     WHERE l.thread_id = :thread_id
                     ORDER BY l.created_at DESC LIMIT 1),
                    :summary, :motivo, FALSE, now()
                ON CONFLICT (conversation_id, motivo)
                    WHERE conversation_id IS NOT NULL
                DO UPDATE SET
                    summary = COALESCE(:summary, escalations.summary)
                RETURNING id, notificado_whatsapp
                """),
                {
                    "thread_id": thread_id,
                    "summary": _safe(summary),
                    "motivo": motivo,
                },
            ).fetchone()
            conn.commit()

        esc_id = row[0] if row else None
        ya_notificado = bool(row[1]) if row else False

        if not notificar or ya_notificado:
            return True

        # Teléfono del cliente (best-effort)
        try:
            with _get_engine().connect() as conn:
                r = conn.execute(
                    text("SELECT telefono FROM leads WHERE thread_id = :tid"),
                    {"tid": thread_id},
                ).fetchone()
            telefono = r[0] if r else None
        except Exception:
            telefono = None

        from integrations.whatsapp.notify import notify_escalation
        ok = notify_escalation(
            thread_id=thread_id,
            summary=summary or "",
            telefono=telefono,
            motivo=motivo,
        )
        if ok and esc_id is not None:
            with _get_engine().connect() as conn:
                conn.execute(
                    # Por FILA, no por thread: cada escalación lleva su propio flag
                    text("UPDATE escalations SET notificado_whatsapp = true "
                         "WHERE id = :id"),
                    {"id": esc_id},
                )
                conn.commit()
        return ok
    except Exception as e:
        logger.exception("[db] insert_escalation falló: %s", e)
        return False


# ---------------------------------------------------------------------------
# BOOKINGS / AGENDA
# ---------------------------------------------------------------------------
def insert_booking(
    thread_id: str,
    booking: dict,
) -> None:
    """
    Una fila por evento_id de Google Calendar — NO por thread_id.

    Un mismo cliente puede tener N citas vigentes. El upsert por evento_id
    solo da idempotencia ante reintentos del MISMO evento (no duplica ni
    re-notifica). Un segundo booking del mismo thread crea fila nueva y
    notifica normalmente.
    """
    try:
        thread_id = normalize_thread_id(thread_id)
        b = booking or {}
        evento_id = _safe(b.get("evento_id"))

        with _get_engine().connect() as conn:
            ya_existia = False
            if evento_id:
                ya_existia = (
                    conn.execute(
                        text("SELECT 1 FROM bookings WHERE evento_id = :evento_id"),
                        {"evento_id": evento_id},
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
                     WHERE l.thread_id = :thread_id
                     ORDER BY l.created_at DESC LIMIT 1),
                    :evento_id, :fecha, :hora_inicio, :hora_fin, :modalidad,
                    :meet_link, :html_link, :cliente_nombre, :cliente_email,
                    CAST(:inicio_cita AS timestamptz), now()
                ON CONFLICT (evento_id) DO UPDATE SET
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
                    "evento_id": evento_id,
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

        if not ya_existia:
            from integrations.whatsapp.notify import notify_booking
            notify_booking(b)
    except Exception as e:
        logger.exception("[db] insert_booking falló: %s", e)
