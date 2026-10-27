"""integrations/whatsapp/notify.py — Notificaciones internas de LeyIA."""

import logging
import os

import httpx

from .config import ACCESS_TOKEN, API_VERSION, PHONE_NUMBER_ID

logger = logging.getLogger("leyia-notify")
API_BASE = "https://graph.facebook.com"
MAX_LEN = 3500

_TEMPLATE_FALLBACK_CODES = {131047, 131026, 131030}

_MOTIVOS = {
    "ficha_completada":              ("📋", "Ficha completada — esperando decisión"),
    "hablar_abogada":                ("🚨", "Cliente quiere hablar contigo"),
    "contacto_directo":              ("📞", "Cliente pidió contacto directo"),
    "derivacion_parcial":            ("⚠️", "Derivación parcial por reintentos"),
    "crisis_riesgo_vital":           ("🆘", "RIESGO VITAL — revisar de inmediato"),
    "aprobacion_agendamiento_rechazada": ("⛔", "Agenda rechazada — contactar al cliente"),
}


def lawyer_number() -> str:
    raw = (os.getenv("LAWYER_WHATSAPP_TO")
           or os.getenv("WHATSAPP_OPERATOR_NUMBER") or "")
    return raw.replace("+", "").replace(" ", "").strip()


def _clip(text, limit: int = 900) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _template_param(text, limit: int = 900) -> str:
    """Sanitiza un parámetro de plantilla WhatsApp.

    Meta rechaza newlines, tabs y más de 4 espacios consecutivos en los
    parámetros de tipo text (error #132018).
    """
    text = (text or "").strip()
    text = text.replace("\n", " ").replace("\t", " ")
    # Colapsar espacios múltiples
    while "  " in text:
        text = text.replace("  ", " ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _wa_cliente(telefono: str | None) -> str:
    t = (telefono or "").replace("+", "").strip()
    return f"\n💬 Contactar: https://wa.me/{t}" if t else ""


def _post(payload: dict) -> tuple[bool, int | None]:
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        logger.warning("[notify] WhatsApp no configurado; skip")
        return False, None

    to = payload.get("to", "unknown")
    msg_type = payload.get("type", "unknown")

    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{API_BASE}/{API_VERSION}/{PHONE_NUMBER_ID}/messages",
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                json=payload,
            )

        try:
            data = r.json()
        except Exception:
            data = {}

        msg_id = (
            (data.get("messages") or [{}])[0].get("id")
            or data.get("message_id")
            or "n/a"
        )

        if r.status_code < 400:
            logger.info("[notify] enviado a %s (type=%s, msg_id=%s)", to, msg_type, msg_id)
            return True, None

        code = None
        try:
            code = data.get("error", {}).get("code")
        except Exception:
            pass
        logger.error("[notify] WA %s (code=%s, msg_id=%s): %s",
                     r.status_code, code, msg_id, r.text[:300])
        return False, code
    except Exception:
        logger.exception("[notify] falló el POST a %s (type=%s)", to, msg_type)
        return False, None


def _build_template_payload(to: str, template: str, lang: str,
                            titulo: str, detalle: str) -> dict:
    return {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template,
            "language": {"code": lang},
            "components": [{
                "type": "body",
                "parameters": [
                    {"type": "text", "text": _template_param(titulo, 60)},
                    {"type": "text", "text": _template_param(detalle, 900)},
                ],
            }],
        },
    }


def _send_template(to: str, titulo: str, detalle: str) -> bool:
    template = os.getenv("NOTIFY_TEMPLATE", "").strip()
    if not template:
        logger.error("[notify] rechazado por ventana 24h y NOTIFY_TEMPLATE no configurada")
        return False

    # FIX: tu template leyia_alerta_interna está creado en Meta como
    # "Spanish (CHL)" → language code = es_CL. Muchos .env tenían "es"
    # heredado, lo que provocaba 404 (#132001). Se fuerza es_CL.
    lang = "es_CL"
    logger.info("[notify] enviando plantilla '%s' a %s | lang=%s | titulo=%s",
                template, to, lang, titulo)

    ok, code = _post(_build_template_payload(to, template, lang, titulo, detalle))

    # Fallback defensivo: si es_CL no existe, intentar con es genérico.
    if not ok and code == 132001:
        logger.warning("[notify] %s no existe en %s; reintentando con es", template, lang)
        ok, _ = _post(_build_template_payload(to, template, "es", titulo, detalle))

    return ok


def _send_alert(titulo: str, body: str) -> bool:
    to = lawyer_number()
    if not to:
        logger.warning("[notify] destinatario vacío")
        return False

    logger.info("[notify] alerta '%s' → %s", titulo, to)
    return _send_template(to, titulo, body)



def notify_new_lead(lead: dict) -> bool:
    d = lead or {}
    logger.info("[notify] evento: nuevo lead | %s", d.get("telefono"))
    body = (
        "📋 *Nuevo lead completo*\n"
        f"👤 Nombre: {_clip(d.get('nombre'), 120)}\n"
        f"📞 Teléfono: {_clip(d.get('telefono'), 30)}\n"
        f"📧 Email: {_clip(d.get('email'), 120)}\n"
        f"🗂️ Categoría: {_clip(d.get('categoria'), 60)}\n"
        f"📍 Situación: {_clip(d.get('situacion'), 300)}\n"
        f"🔖 Etapa: {_clip(d.get('etapa_proceso'), 60)}"
        + _wa_cliente(d.get("telefono"))
    )
    return _send_alert("Nuevo lead completo", body)


def notify_escalation(thread_id, summary, telefono=None, motivo="hablar_abogada") -> bool:
    logger.info("[notify] evento: escalación %s | %s", motivo, telefono)
    icon, titulo = _MOTIVOS.get(motivo, ("🚨", "Escalamiento"))
    body = (
        f"{icon} *{titulo}*\n"
        f"🧵 Hilo: {thread_id}\n"
        f"📞 Cliente: {_clip(telefono or 'sin teléfono', 30)}\n"
        f"📝 Resumen: {_clip(summary, 1200)}"
        + _wa_cliente(telefono)
    )
    return _send_alert(titulo, body)


def notify_booking(booking: dict) -> bool:
    b = booking or {}
    logger.info("[notify] evento: booking | %s | %s %s",
                b.get("cliente_nombre"), b.get("fecha"), b.get("hora_inicio"))
    link = b.get("html_link") or b.get("meet_link") or ""
    body = (
        "📅 *Nueva cita agendada*\n"
        f"👤 Cliente: {_clip(b.get('cliente_nombre'), 120)}\n"
        f"🗓️ Fecha: {_clip(b.get('fecha'), 20)}\n"
        f"⏰ Horario: {_clip(b.get('hora_inicio'), 10)} - {_clip(b.get('hora_fin'), 10)}\n"
        f"📍 Modalidad: {_clip(b.get('modalidad_label') or b.get('modalidad'), 60)}"
        + (f"\n🔗 {_clip(link, 200)}" if link else "")
    )
    return _send_alert("Nueva cita agendada", body)
