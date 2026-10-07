import httpx

from whatsapp.config import ACCESS_TOKEN, API_VERSION, PHONE_NUMBER_ID

API_BASE = "https://graph.facebook.com"

MAX_WA_LEN = 4000  # WhatsApp corta ~4096 caracteres


def _chunks(text: str):
    text = text.strip()
    while len(text) > MAX_WA_LEN:
        cut = text.rfind("\n", 0, MAX_WA_LEN) or text.rfind(" ", 0, MAX_WA_LEN) or MAX_WA_LEN
        yield text[:cut]
        text = text[cut:].lstrip()
    if text:
        yield text


async def _post(payload: dict) -> dict:
    url = f"{API_BASE}/{API_VERSION}/{PHONE_NUMBER_ID}/messages"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            json=payload,
        )
        if response.status_code >= 400:
            logger = __import__("logging").getLogger("leyia-whatsapp")
            logger.error(f"WA API {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json()


async def send_text(to: str, body: str) -> None:
    for chunk in _chunks(body):
        await _post({
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": chunk},
        })


async def mark_as_read(wamid: str) -> None:
    try:
        await _post({
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": wamid,
        })
    except Exception:
        logger = __import__("logging").getLogger("leyia-whatsapp")
        logger.warning(f"mark_as_read falló para {wamid}", exc_info=True)
