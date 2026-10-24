"""integrations/whatsapp/notify.py — Notificaciones internas de LeyIA.

Eventos de negocio → WhatsApp a la abogada:
  - Nuevo lead completo      (notify_new_lead)
  - Escalamiento / handoff   (notify_escalation)
  - Cita agendada            (notify_booking)

Reglas:
  - Best-effort: NUNCA lanza excepción al grafo.
  - Sincrónico: corre dentro del executor del grafo (no bloquea el loop).
  - Ventana 24h: si Meta rechaza el texto libre, reintenta con la
    plantilla utility definida en NOTIFY_TEMPLATE.
"""
import logging
import os

import httpx

from .config import ACCESS_TOKEN, API_VERSION, PHONE_NUMBER_ID

logger = logging.getLogger("leyia-notify")
API_BASE = "https://graph.facebook.com"
MAX_LEN = 3500

# Códigos de Meta que significan "no puedes usar texto libre aquí":
#   131047 = fuera de ventana 24h   131026 = undeliverable
#   131030 = destinatario no está en lista permitida (app en desarrollo)
_TEMPLATE_FALLBACK_CODES = {131047, 131026, 131030}

# v2: 'contacto_directo' y 'crisis_riesgo_vital' ya no caen al default genérico.
_MOTIVOS = {
    "ficha_completada":              ("📋", "Ficha completada — esperando decisión"),
    "hablar_abogada":                ("🚨", "Cliente quiere hablar contigo"),
    "contacto_directo":              ("📞", "Cliente pidió contacto directo"),
    "derivacion_parcial":            ("⚠️", "Derivación parcial por reintentos"),
    "crisis_riesgo_vital":           ("🆘", "RIESGO VITAL — revisar de inmediato"),
    "aprobacion_agendamiento_rechazada": ("⛔", "Agenda rechazada — contactar al cliente"),
}



def lawyer_number() -> str:
    """Lazy: lee el entorno en cada llamada para no depender del orden de
    carga de dotenv/uvicorn en el arranque."""
    raw = (os.getenv("LAWYER_WHATSAPP_TO")
           or os.getenv("WHATSAPP_OPERATOR_NUMBER") or "")
    return raw.replace("+", "").replace(" ", "").strip()


def _clip(text, limit: int = 900) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _wa_cliente(telefono: str | None) -> str:
    t = (telefono or "").replace("+", "").strip()
    return f"\n💬 Contactar: https://wa.me/{t}" if t else ""


def _post(payload: dict) -> tuple[bool, int | None]:
    """Devuelve (ok, error_code_de_Meta)."""
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        logger.warning("[notify] WhatsApp no configurado; skip")
        return False, None
    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{API_BASE}/{API_VERSION}/{PHONE_NUMBER_ID}/messages",
                headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
                json=payload,
            )
        if r.status_code < 400:
            return True, None
        code = None
        try:
            code = r.json().get("error", {}).get("code")
        except Exception:
            pass
        logger.error("[notify] WA %s (code=%s): %s",
                     r.status_code, code, r.text[:300])
        return False, code
    except Exception:
        logger.exception("[notify] falló el POST a Meta")
        return False, None


def _send_template(to: str, titulo: str, detalle: str) -> bool:
    """Plantilla utility con 2 variables de cuerpo: {{1}} título, {{2}} detalle."""
    template = os.getenv("NOTIFY_TEMPLATE", "").strip()
    if not template:
        logger.error("[notify] rechazado por ventana 24h y NOTIFY_TEMPLATE "
                     "no configurada — aviso perdido")
        return False
    ok, _ = _post({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": template,
            "language": {"code": os.getenv("NOTIFY_TEMPLATE_LANG", "es")},
            "components": [{
                "type": "body",
                "parameters": [
                    {"type": "text", "text": _clip(titulo, 60)},
                    {"type": "text", "text": _clip(detalle, 900)},
                ],
            }],
        },
    })
    return ok


def _send_alert(titulo: str, body: str) -> bool:
    to = lawyer_number()
    if not to:
        logger.warning("[notify] destinatario vacío "
                       "(LAWYER_WHATSAPP_TO / WHATSAPP_OPERATOR_NUMBER); skip")
        return False

    ok, code = _post({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": _clip(body, MAX_LEN)},
    })
    if ok:
        logger.info("[notify] enviado a %s", to)
        return True
    if code in _TEMPLATE_FALLBACK_CODES:
        logger.warning("[notify] fuera de ventana 24h (code=%s) → plantilla", code)
        return _send_template(to, titulo, body)
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
        + _wa_cliente(d.get("telefono"))
    )
    return _send_alert("Nuevo lead completo", body)


def notify_escalation(thread_id, summary, telefono=None,
                      motivo="hablar_abogada") -> bool:
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
