from pydantic import BaseModel

import re


class WhatsAppMessage(BaseModel):
    message_id: str
    phone_number_id: str
    customer_phone: str
    customer_name: str | None = None
    message_type: str
    text: str | None = None
    interactive_id: str | None = None  # id del botón/lista tocado (analítica/logs)
    timestamp: str


def _normalize_phone(raw: str | None) -> str:
    """Normaliza un número de teléfono a dígitos E.164 sin '+'.
    Si ya viene con prefijo de país (Chile: 56), lo conserva.
    Si es un móvil chileno de 9 dígitos sin prefijo, le antepone 56.
    """
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("56") and len(digits) == 11:
        return digits
    if len(digits) == 9 and digits.startswith("9"):
        return f"56{digits}"
    return digits
