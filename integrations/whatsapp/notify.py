"""integrations/whatsapp/notify.py — Notificaciones internas de LeyIA.

Envía WhatsApp al número de la abogada cuando ocurren eventos de negocio:
  - Nuevo lead completo
  - Escalamiento
  - Cita agendada

Best-effort: nunca lanza excepción al grafo.
Módulo SINCRÓNICO: pensado para ser llamado desde core.db_client (psycopg3).
"""
import logging

import httpx

from .config import ACCESS_TOKEN, API_VERSION, LAWYER_WHATSAPP_TO, PHONE_NUMBER_ID

logger = logging.getLogger("leyia-notify")
API_BASE = "https://graph.facebook.com"
MAX_LEN = 3500


def _clip(text: str, limit: int = 900) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def send_text_sync(to: str, body: str) -> bool:
    """Envía un texto por WhatsApp Cloud API. Devuelve True si Meta aceptó."""
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        logger.warning("[notify] WhatsApp no configurado; skip")
        return False
    if not to:
        logger.warning("[notify] destinatario vacío; skip")
        return False

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": _clip(body, MAX_LEN)},
    }
    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{API_BASE}/{API_VERSION}/{PHONE_NUMBER_ID}/messages",
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                json=payload,
            )
            if r.status_code >= 400:
                logger.error("[notify] WA %s: %s", r.status_code, r.text)
                return False
            logger.info("[notify] enviado a %s", to)
            return True
    except Exception:
        logger.exception("[notify] falló envío a %s", to)
        return False


# ------------------------------------------------------------------ #
# Eventos de negocio                                                   #
# ------------------------------------------------------------------ #
def notify_new_lead(lead: dict) -> bool:
    d = lead or {}
    body = (
        "📋 *Nuevo lead completo*\n"
        f"👤 Nombre: {_clip(d.get('nombre'), 120)}\n"
        f"📞 Teléfono: {_clip(d.get('telefono'), 30)}\n"
        f"📧 Email: {_clip(d.get('email'), 120)}\n"
        f"🗂️ Categoría: {_clip(d.get('categoria'), 60)}\n"
        f"📍 Situación: {_clip(d.get('situacion'), 300)}\n"
        f"🔖 Etapa: {_clip(d.get('etapa_proceso'), 60)}"
    )
    return send_text_sync(LAWYER_WHATSAPP_TO, body)


def notify_escalation(thread_id: str, summary: str, telefono: str | None = None) -> bool:
    body = (
        "🚨 *Escalamiento a ejecutiva*\n"
        f"🧵 Hilo: {thread_id}\n"
        f"📞 Cliente: {_clip(telefono or 'sin teléfono', 30)}\n"
        f"📝 Resumen: {_clip(summary, 1200)}"
    )
    return send_text_sync(LAWYER_WHATSAPP_TO, body)


def notify_booking(booking: dict) -> bool:
    b = booking or {}
    link = b.get("html_link") or b.get("meet_link") or ""
    body = (
        "📅 *Nueva cita agendada*\n"
        f"👤 Cliente: {_clip(b.get('cliente_nombre'), 120)}\n"
        f"🗓️ Fecha: {_clip(b.get('fecha'), 20)}\n"
        f"⏰ Horario: {_clip(b.get('hora_inicio'), 10)} - {_clip(b.get('hora_fin'), 10)}\n"
        f"📍 Modalidad: {_clip(b.get('modalidad_label') or b.get('modalidad'), 60)}"
        + (f"\n🔗 {_clip(link, 200)}" if link else "")
    )
    return send_text_sync(LAWYER_WHATSAPP_TO, body)
