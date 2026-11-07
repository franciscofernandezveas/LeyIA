from .normalizer import WhatsAppMessage, _normalize_phone


def _extract_content(msg: dict) -> tuple[str | None, str | None, str | None]:
    """Devuelve (texto_equivalente, media_id, interactive_id).

    - Interactivas: se normaliza al `title` visible (el grafo lo valida).
    - Audios: en Cloud API las notas de voz llegan como `type: "audio"` con
      `audio.voice: true`; el tipo legado `"voice"` también se acepta. Lo único
      relevante aquí es el `id` del adjunto: se descarga y transcribe en el handler.
    """
    msg_type = msg.get("type")

    if msg_type == "text":
        return msg.get("text", {}).get("body"), None, None

    if msg_type in ("audio", "voice"):
        media = msg.get("audio") or msg.get("voice") or {}
        return None, media.get("id"), None

    if msg_type == "interactive":
        it = msg.get("interactive") or {}
        reply = it.get("button_reply") or it.get("list_reply") or {}
        text = (reply.get("title") or reply.get("id") or "").strip() or None
        return text, None, reply.get("id")

    return None, None, None


def parse_webhook_payload(payload: dict) -> list[WhatsAppMessage]:
    messages: list[WhatsAppMessage] = []

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            if "messages" not in value:
                continue

            phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
            contacts = {
                _normalize_phone(c["wa_id"]): c
                for c in value.get("contacts", [])
            }

            for msg in value["messages"]:
                msg_type = msg.get("type", "unknown")
                from_raw = msg.get("from", "")
                from_norm = _normalize_phone(from_raw)
                contact = contacts.get(from_norm, {})
                text, media_id, interactive_id = _extract_content(msg)

                messages.append(WhatsAppMessage(
                    message_id=msg["id"],
                    phone_number_id=phone_number_id,
                    customer_phone=from_norm,
                    customer_name=contact.get("profile", {}).get("name"),
                    message_type=msg_type,
                    text=text,
                    media_id=media_id,
                    interactive_id=interactive_id,
                    timestamp=msg.get("timestamp", ""),
                ))

    return messages
