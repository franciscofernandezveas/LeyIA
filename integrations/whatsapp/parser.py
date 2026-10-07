from integrations.whatsapp.normalizer import WhatsAppMessage


def parse_webhook_payload(payload: dict) -> list[WhatsAppMessage]:
    """
    El webhook de Meta puede traer varios entry/change/messages.
    Los 'statuses' (entregado, leído) se ignoran.
    """
    messages: list[WhatsAppMessage] = []

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            if "messages" not in value:
                continue

            phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
            contacts = {c["wa_id"]: c for c in value.get("contacts", [])}

            for msg in value["messages"]:
                msg_type = msg.get("type", "unknown")
                contact = contacts.get(msg.get("from", ""), {})

                messages.append(WhatsAppMessage(
                    message_id=msg["id"],
                    phone_number_id=phone_number_id,
                    customer_phone=msg.get("from", ""),
                    customer_name=contact.get("profile", {}).get("name"),
                    message_type=msg_type,
                    text=msg.get("text", {}).get("body") if msg_type == "text" else None,
                    timestamp=msg.get("timestamp", ""),
                ))

    return messages
