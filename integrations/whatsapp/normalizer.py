from pydantic import BaseModel


class WhatsAppMessage(BaseModel):
    message_id: str
    phone_number_id: str
    customer_phone: str
    customer_name: str | None = None
    message_type: str
    text: str | None = None
    timestamp: str
