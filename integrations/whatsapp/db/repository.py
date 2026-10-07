import logging

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from whatsapp.db.connection import async_session
from whatsapp.db.models import MessageRecord
from whatsapp.normalizer import WhatsAppMessage

logger = logging.getLogger(__name__)


async def register_message(message: WhatsAppMessage) -> str:
    """
    Inserta el mensaje de forma atómica (seguro bajo reintentos de Meta
    y concurrencia entre réplicas de la API).

    Retorna el estado resultante:
      "new"       → insertado ahora. El webhook debe encolarlo.
      "pending"   → ya existía pero nunca se procesó.
                    El webhook debe re-encolarlo.
      "processed" → ya existía y el worker lo atendió. Ignorar.
    """
    stmt = (
        insert(MessageRecord)
        .values(
            whatsapp_message_id=message.message_id,
            phone_number_id=message.phone_number_id,
            customer_phone=message.customer_phone,
            message_type=message.message_type,
            content=message.text,
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
    """
    Llamado por el handler DESPUÉS de procesar exitosamente el mensaje,
        dentro del lock de conversación.
    """
    async with async_session() as session:
        await session.execute(
            update(MessageRecord)
            .where(MessageRecord.whatsapp_message_id == whatsapp_message_id)
            .values(status="processed")
        )
        await session.commit()
