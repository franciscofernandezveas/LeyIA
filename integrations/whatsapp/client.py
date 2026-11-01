"""integraciones/whatsapp/client.py — Adaptador Meta Cloud API.

v1.2 — FIX CTA robusto + fallback a texto:
  - _coerce_https_url(): normaliza URLs que llegan envueltas en markdown
    [texto](https://...) o con puntuación colgante.
  - _normalize_cta_payload(): extrae url del markdown y usa el texto del
    enlace como display_text si éste viene vacío o es markdown.
  - send_interactive(): si _build_interactive() falla por payload inválido,
    se degrada el mensaje a send_text() con el body + URL rescatada, en
    lugar de perderlo (era el bug de "interactive inválido ignorado").

Restricciones Meta vigentes:
  - buttons: ≤3 botones, title ≤ 20.
  - list: ≤10 filas totales, title ≤ 24, desc ≤ 72, CTA ≤ 20.
  - cta_url: 1 botón por mensaje, display_text ≤ 25, https obligatorio,
    NO combinable con botones de respuesta en el mismo mensaje.
"""

import logging
import re

import httpx

from .config import ACCESS_TOKEN, API_VERSION, PHONE_NUMBER_ID

logger = logging.getLogger("leyia-whatsapp")
API_BASE = "https://graph.facebook.com"
MAX_WA_LEN = 4000

MAX_INTERACTIVE_BODY = 1024
MAX_BUTTON_TITLE = 20
MAX_ROW_TITLE = 24
MAX_ROW_DESC = 72
MAX_LIST_ROWS = 10
MAX_LIST_CTA = 20
MAX_SECTION_TITLE = 24
MAX_CTA_DISPLAY = 25
MAX_URL_LEN = 2000

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_MD_LINK_RE = re.compile(
    r"\[\s*(?P<text>[^\]]*?)\s*\]\(\s*(?P<url>https?://[^\s)]+?)\s*\)",
    re.IGNORECASE,
)
_BARE_URL_RE = re.compile(r"https?://[^\s)\]}]+", re.IGNORECASE)
_URL_TRIM = "<>\"'`()[] \t\r\n"


def _wa_clean(text: str) -> str:
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


def _row(o: dict) -> dict:
    row = {"id": str(o["id"]), "title": _clip(o["title"], MAX_ROW_TITLE)}
    if o.get("description"):
        row["description"] = _clip(o["description"], MAX_ROW_DESC)
    return row


def _coerce_https_url(*candidates: str) -> str | None:
    """Extrae y limpia una URL https válida de cualquier candidato.

    Acepta URLs desnudas o envueltas en markdown [texto](https://...).
    Devuelve None si no encuentra nada usable.
    """
    for raw in candidates:
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue

        m = _MD_LINK_RE.search(text)
        url = m.group("url") if m else None
        if not url:
            m2 = _BARE_URL_RE.search(text)
            url = m2.group(0) if m2 else None
        if not url:
            continue

        url = url.strip(_URL_TRIM).rstrip(".,;:!?")
        if url.lower().startswith("http://"):
            url = "https://" + url[len("http://"):]
        if not url.lower().startswith("https://"):
            continue
        if len(url) > MAX_URL_LEN:
            continue
        return url
    return None


def _normalize_cta_payload(interactive: dict) -> dict | None:
    """Normaliza {text, url} para cta_url. Devuelve None si no es recuperable."""
    raw_url = str(interactive.get("url") or "")
    raw_text = str(interactive.get("text") or "").strip()

    url = _coerce_https_url(raw_url, raw_text)
    if not url:
        return None

    display_text = raw_text
    if (not display_text
            or _MD_LINK_RE.search(display_text)
            or "http" in display_text.lower()):
        md = _MD_LINK_RE.search(raw_url) or _MD_LINK_RE.search(raw_text)
        display_text = (md.group("text").strip() if md else "") or "Abrir"

    return {"text": _clip(display_text, MAX_CTA_DISPLAY), "url": url}


def _salvage_url_for_fallback(interactive: dict) -> str | None:
    """Best-effort: rescata una URL https del payload roto para adjuntarla al texto."""
    return _coerce_https_url(
        str(interactive.get("url") or ""),
        str(interactive.get("text") or ""),
        str(interactive.get("link") or ""),
        str(interactive.get("cta") or ""),
    )


def _build_interactive(interactive: dict, body: str) -> dict:
    wa: dict = {"body": {"text": _clip(body, MAX_INTERACTIVE_BODY)}}

    kind = interactive.get("kind")
    if kind is None and interactive.get("type") is not None:
        logger.warning("interactive usa clave legacy 'type'; corrige el nodo emisor: %r", interactive)
        kind = interactive.get("type")

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
        if interactive.get("sections"):
            sections = [
                {"title": _clip(sec.get("title") or "Opciones", MAX_SECTION_TITLE),
                 "rows": [_row(r) for r in sec.get("rows", [])]}
                for sec in interactive["sections"]
            ]
        else:
            sections = [{
                "title": _clip(interactive.get("title") or "Opciones", MAX_SECTION_TITLE),
                "rows": [_row(o) for o in interactive.get("options", [])],
            }]

        cupos = MAX_LIST_ROWS
        final = []
        for s in sections:
            rows = s["rows"][:cupos]
            if rows:
                final.append({"title": s["title"], "rows": rows})
                cupos -= len(rows)
            if cupos <= 0:
                break
        if not final:
            raise ValueError("interactive 'list' sin opciones")
        wa["type"] = "list"
        wa["action"] = {
            "button": _clip(interactive.get("cta") or "Ver opciones", MAX_LIST_CTA),
            "sections": final,
        }

    elif kind == "cta":
        cta = _normalize_cta_payload(interactive)
        if cta is None:
            raise ValueError(
                f"cta requiere URL https válida; url={interactive.get('url')!r}"
            )
        if cta["url"] != str(interactive.get("url") or "").strip():
            logger.info(
                "[whatsapp] cta URL normalizada: %r -> %r",
                interactive.get("url"), cta["url"],
            )
        wa["type"] = "cta_url"
        wa["action"] = {
            "name": "cta_url",
            "parameters": {
                "display_text": cta["text"],
                "url": cta["url"],
            },
        }
    else:
        raise ValueError(f"kind de interactive no soportado: {kind!r}")

    return wa


async def _post(payload: dict) -> dict:
    url = f"{API_BASE}/{API_VERSION}/{PHONE_NUMBER_ID}/messages"
    to = payload.get("to", "unknown")
    msg_type = payload.get("type", "status" if payload.get("status") else "unknown")

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            json=payload,
        )

        try:
            data = response.json()
        except Exception:
            data = {}

        msg_id = (
            (data.get("messages") or [{}])[0].get("id")
            or data.get("message_id")
            or "n/a"
        )

        if response.status_code >= 400:
            logger.error(
                "[whatsapp] WA API %s para %s (type=%s, msg_id=%s): %s",
                response.status_code, to, msg_type, msg_id, response.text[:300]
            )
        else:
            logger.info(
                "[whatsapp] Meta aceptó mensaje %s para %s (type=%s)",
                msg_id, to, msg_type
            )

        response.raise_for_status()
        return data


async def send_text(to: str, body: str) -> None:
    chunks = list(_chunks(_wa_clean(body)))
    for i, chunk in enumerate(chunks, 1):
        logger.info(
            "[whatsapp] texto a %s chunk %d/%d: %.120s%s",
            to, i, len(chunks), chunk, "..." if len(chunk) > 120 else ""
        )
        await _post({
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": chunk},
        })


async def send_interactive(to: str, body: str, interactive: dict) -> dict:
    kind = interactive.get("kind") or interactive.get("type") or "unknown"
    clean_body = _wa_clean(body)
    logger.info(
        "[whatsapp] interactivo a %s | kind=%s | body=%.120s%s",
        to, kind, clean_body, "..." if len(clean_body) > 120 else ""
    )

    try:
        wa_payload = _build_interactive(interactive, clean_body)
    except (ValueError, KeyError, TypeError) as exc:
        # Regla de oro: NUNCA perder el mensaje principal por un CTA mal formado.
        logger.warning(
            "[whatsapp] interactive inválido (%s); fallback a texto a %s | %r",
            exc, to, interactive,
        )
        fallback = clean_body
        url = _salvage_url_for_fallback(interactive)
        if url and url not in fallback:
            fallback = f"{fallback}\n\n{url}".strip()
        if not fallback:
            fallback = "No pude completar la acción solicitada."
        await send_text(to, fallback)
        return {"fallback": "text", "reason": str(exc), "messages": []}

    return await _post({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "interactive",
        "interactive": wa_payload,
    })


async def mark_as_read(wamid: str) -> None:
    try:
        logger.info("[whatsapp] marcando como leído: %s", wamid)
        await _post({
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": wamid,
        })
    except Exception:
        logger.warning("mark_as_read falló para %s", wamid, exc_info=True)
