"""Notificación de escalamientos vía WhatsApp (Twilio) a la ejecutiva."""
import logging

logger = logging.getLogger(__name__)

WHATSAPP_EJECUTIVA = "+56942313989"  # número designado por Manzzo y Cía


def notificar_escalamiento(*, thread_id: str, resumen: str,
                           telefono_cliente: str | None) -> bool:
    """Envía el resumen del caso al WhatsApp de la ejecutiva.

    Producción: Twilio WhatsApp API (mismo canal del bot).
    """
    body = (
        f"🔔 *Escalamiento LeyIA*\n"
        f"Caso: {thread_id}\n"
        f"Cliente: {telefono_cliente or 'no identificado'}\n\n"
        f"{resumen}"
    )
    # from twilio.rest import Client
    # Client(SID, TOKEN).messages.create(
    #     from_="whatsapp:+14155238886",
    #     to=f"whatsapp:{WHATSAPP_EJECUTIVA}", body=body)
    logger.info("ESCALAMIENTO notificado a %s | thread=%s", WHATSAPP_EJECUTIVA, thread_id)
    print(f"📤 [WhatsApp ejecutiva]\n{body}\n")  # stub para consola de pruebas
    return True
