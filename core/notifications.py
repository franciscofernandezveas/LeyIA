"""core/notifications.py — Notificación simple por email a ejecutiva."""
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
EJECUTIVA_EMAIL = os.getenv("EJECUTIVA_EMAIL")


def notificar_email(subject: str, body_html: str, to: str | None = None) -> bool:
    """Envía un email HTML. Si SMTP no está configurado, loguea warning y
    retorna False sin romper el grafo."""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, EJECUTIVA_EMAIL]):
        logger.warning("SMTP no configurado; email no enviado")
        return False

    dest = to or EJECUTIVA_EMAIL
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = dest
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [dest], msg.as_string())
        logger.info("Email enviado a %s", dest)
        return True
    except Exception as e:
        logger.exception("Fallo envío email: %s", e)
        return False
