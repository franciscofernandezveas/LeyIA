import logging

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert

from .connection import async_session
from .models import MessageRecord
from ..normalizer import WhatsAppMessage
from core.threads import wa_thread_id  # import absoluto; alternativa relativa: from ....core.threads import wa_thread_id

logger = logging.getLogger(__name__)


async def ensure_conversation(thread_id: str, phone: str) -> None:
    """
    Crea la conversación ANTES de registrar el mensaje.
    Repara también conversaciones históricas con channel='desconocido'.
    """
    async with async_session() as session:
        await session.execute(
            text("""
                INSERT INTO conversations (thread_id, channel, status, metadata, created_at, updated_at)
                VALUES (:tid, 'wa', 'open',
                        jsonb_build_object('origen', 'wa_webhook', 'telefono', :phone),
                        now(), now())
                ON CONFLICT (thread_id) DO UPDATE SET
                    channel = CASE
                        WHEN conversations.channel IS NULL
                          OR conversations.channel IN ('desconocido', 'sin_canal')
                        THEN 'wa'
                        ELSE conversations.channel
                    END,
                    updated_at = now()
            """),
            {"tid": thread_id, "phone": phone},
        )
        await session.commit()


async def register_message(message: WhatsAppMessage) -> str:
    tid = wa_thread_id(message.customer_phone)  # 'wa-<teléfono>'

    # 1) Conversación primero: sin esto insert_message() del agente
    #    podría insertar 0 filas silenciosamente.
    await ensure_conversation(tid, message.customer_phone)

    # 2) Registrar el mensaje con su thread_id.
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
