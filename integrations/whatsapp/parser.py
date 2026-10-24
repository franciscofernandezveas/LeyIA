from .normalizer import WhatsAppMessage, _normalize_phone


def _extract_content(msg: dict) -> tuple[str | None, str | None]:
    """Devuelve (texto_equivalente, interactive_id).

    Para respuestas interactivas se normaliza al `title` del botón/fila
    (no al `id`): el grafo ya valida esos textos visibles directamente.
    """
    msg_type = msg.get("type")

    if msg_type == "text":
        return msg.get("text", {}).get("body"), None

    if msg_type == "interactive":
        it = msg.get("interactive") or {}
        reply = it.get("button_reply") or it.get("list_reply") or {}
        text = (reply.get("title") or reply.get("id") or "").strip() or None
        return text, reply.get("id")

    return None, None


def parse_webhook_payload(payload: dict) -> list[WhatsAppMessage]:
    messages: list[WhatsAppMessage] = []

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            if "messages" not in value:
                continue

            phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
            # Normalizamos wa_id también, por si Meta envía formatos distintos
            contacts = {
                _normalize_phone(c["wa_id"]): c
                for c in value.get("contacts", [])
            }

            for msg in value["messages"]:
                msg_type = msg.get("type", "unknown")
                from_raw = msg.get("from", "")
                from_norm = _normalize_phone(from_raw)
                contact = contacts.get(from_norm, {})
                text, interactive_id = _extract_content(msg)

                messages.append(WhatsAppMessage(
                    message_id=msg["id"],
                    phone_number_id=phone_number_id,
                    customer_phone=from_norm,
                    customer_name=contact.get("profile", {}).get("name"),
                    message_type=msg_type,
                    text=text,
                    interactive_id=interactive_id,
                    timestamp=msg.get("timestamp", ""),
                ))

    return messages
