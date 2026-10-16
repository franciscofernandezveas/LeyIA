import logging
import os

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert

from .connection import async_session
from .models import MessageRecord
from ..normalizer import WhatsAppMessage
from core.threads import wa_thread_id, normalize_thread_id

logger = logging.getLogger(__name__)

# TTL de sesión: si pasan estas horas sin actividad, la próxima interacción
# abre una conversación NUEVA (la anterior queda cerrada con closed_at).
SESSION_TTL_HORAS = int(os.getenv("CONVERSATION_TTL_HORAS", "24"))


async def ensure_conversation(thread_id: str, phone: str) -> None:
    """
    Garantiza que exista una conversación ABIERTA para este thread_id:
      1. Cierra la sesión activa si expiró por TTL (> SESSION_TTL_HORAS).
      2. Crea una sesión nueva si no hay ninguna activa.
      3. Actualiza el heartbeat (updated_at) de la sesión activa.

    Idempotente y seguro de llamar en cada mensaje entrante.
    """
    thread_id = normalize_thread_id(thread_id)

    async with async_session() as session:
        # 1. Cerrar sesión activa si expiró por inactividad
        await session.execute(
            text("""
                UPDATE conversations
                SET closed_at = updated_at,
                    status = 'session_expired'
                WHERE thread_id = :tid
                  AND closed_at IS NULL
                  AND updated_at < now() - make_interval(hours => :ttl)
            """),
            {"tid": thread_id, "ttl": SESSION_TTL_HORAS},
        )

        # 2. Crear sesión nueva si no hay ninguna activa
        await session.execute(
            text("""
                INSERT INTO conversations
                    (thread_id, channel, status, metadata,
                     created_at, updated_at)
                SELECT
                    CAST(:tid AS text), 'wa', 'open',
                    jsonb_build_object('origen', 'wa_webhook',
                                       'telefono', CAST(:phone AS text)),
                    now(), now()
                WHERE NOT EXISTS (
                    SELECT 1 FROM conversations
                    WHERE thread_id = :tid AND closed_at IS NULL
                )
            """),
            {"tid": thread_id, "phone": phone or ""},
        )

        # 3. Heartbeat: marcar la sesión activa como 'recién usada'
        await session.execute(
            text("""
                UPDATE conversations
                SET updated_at = now(),
                    channel = CASE
                        WHEN channel IS NULL
                          OR channel IN ('desconocido', 'sin_canal')
                        THEN 'wa'
                        ELSE channel
                    END
                WHERE thread_id = :tid AND closed_at IS NULL
            """),
            {"tid": thread_id},
        )

        await session.commit()


async def register_message(message: WhatsAppMessage) -> str:
    tid = wa_thread_id(message.customer_phone)

    # 1) Conversación primero (con rotación de sesión por TTL).
    #    Sin esto insert_message() del agente podría insertar 0 filas.
    await ensure_conversation(tid, message.customer_phone)

    # 2) Registrar el mensaje entrante.
    stmt = (
        insert(MessageRecord)
        .values(
            whatsapp_message_id=message.message_id,
            phone_number_id=message.phone_number_id,
            customer_phone=message.customer_phone,
            message_type=message.message_type,
            content=message.text,
            thread_id=tid,
        )
        .on_conflict_do_nothing(index_elements=["whatsapp_message_id"])
        .returning(MessageRecord.id)
    )

    async with async_session() as session:
        inserted = (await session.execute(stmt)).scalar_one_or_none()
        await session.commit()

        if inserted is not None:
            logger.info("Mensaje nuevo registrado: %s", message.message_id)
            return "new"

        result = await session.execute(
            select(MessageRecord.status).where(
                MessageRecord.whatsapp_message_id == message.message_id
            )
        )
        return result.scalar_one()


async def mark_processed(whatsapp_message_id: str) -> None:
    async with async_session() as session:
        await session.execute(
            update(MessageRecord)
            .where(MessageRecord.whatsapp_message_id == whatsapp_message_id)
            .values(status="processed")
        )
        await session.commit()
