import logging
import re

import httpx

from .config import ACCESS_TOKEN, API_VERSION, PHONE_NUMBER_ID

logger = logging.getLogger("leyia-whatsapp")
API_BASE = "https://graph.facebook.com"
MAX_WA_LEN = 4000

MAX_INTERACTIVE_BODY = 1024   # body de un interactivo Meta
MAX_BUTTON_TITLE = 20
MAX_ROW_TITLE = 24

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _wa_clean(text: str) -> str:
    """WhatsApp no renderiza HTML: convierte <br> en saltos de línea reales."""
    return _BR_RE.sub("\n", text or "").strip()


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _chunks(text: str):
    text = text.strip()
    while len(text) > MAX_WA_LEN:
        cut = text.rfind("\n", 0, MAX_WA_LEN) or text.rfind(" ", 0, MAX_WA_LEN) or MAX_WA_LEN
        yield text[:cut]
        text = text[cut:].lstrip()
    if text:
        yield text


def _build_interactive(interactive: dict, body: str) -> dict:
    """Convierte el formato interno del grafo al de Meta Cloud API."""
    wa: dict = {"body": {"text": _clip(body, MAX_INTERACTIVE_BODY)}}
    kind = interactive.get("kind")

    if kind == "buttons":
        buttons = [
            {"type": "reply", "reply": {"id": str(b["id"]),
                                        "title": _clip(b["title"], MAX_BUTTON_TITLE)}}
            for b in interactive.get("buttons", [])[:3]
        ]
        if not buttons:
            raise ValueError("interactive 'buttons' sin botones")
        wa["type"] = "button"
        wa["action"] = {"buttons": buttons}

    elif kind == "list":
        rows = [
            {"id": str(o["id"]), "title": _clip(o["title"], MAX_ROW_TITLE)}
            for o in interactive.get("options", [])[:10]
        ]
        if not rows:
            raise ValueError("interactive 'list' sin opciones")
        wa["type"] = "list"
        wa["action"] = {
            "button": _clip(interactive.get("cta") or "Ver opciones", 20),
            "sections": [{
                "title": _clip(interactive.get("title") or "Opciones", 24),
                "rows": rows,
            }],
        }

    else:
        raise ValueError(f"kind de interactive no soportado: {kind!r}")

    return wa


async def _post(payload: dict) -> dict:
    url = f"{API_BASE}/{API_VERSION}/{PHONE_NUMBER_ID}/messages"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            json=payload,
        )
        if response.status_code >= 400:
            logger.error(f"WA API {response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json()


async def send_text(to: str, body: str) -> None:
    for chunk in _chunks(_wa_clean(body)):
        await _post({
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": chunk},
        })


async def send_interactive(to: str, body: str, interactive: dict) -> dict:
    return await _post({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "interactive",
        "interactive": _build_interactive(interactive, _wa_clean(body)),
    })


async def mark_as_read(wamid: str) -> None:
    try:
        await _post({
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": wamid,
        })
    except Exception:
        logger.warning(f"mark_as_read falló para {wamid}", exc_info=True)
