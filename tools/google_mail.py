"""tools/google_mail.py — Envío de correos vía Gmail API.

Reutiliza las mismas credenciales OAuth que tools/google_calendar.py
(CREDENTIALS_FILE / TOKEN_FILE), pero con un scope adicional para Gmail.

REQUISITO: token.json debe haberse autorizado con BOTH scopes.
Si fue generado solo con calendar, users().messages().send() devuelve
403 "Insufficient Permission". Re-autorizar:

    rm token.json
    python -c "from tools.google_mail import check_auth_preflight; check_auth_preflight()"

y volver a copiar token.json a la env var GOOGLE_CALENDAR_TOKEN_JSON en
Railway.

Ningún método lanza excepción hacia el llamador (mismo patrón best-effort
que tools/google_calendar.py).
"""
import base64
import logging
import os
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from functools import lru_cache

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE = os.getenv("GOOGLE_CALENDAR_TOKEN_FILE", "token.json")

# Debe coincidir con los scopes con que se generó token.json.
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.compose",   # era gmail.send
]


SENDER_EMAIL = os.getenv("GMAIL_SENDER_EMAIL", "manzzo.mkt@gmail.com")
SENDER_NAME = os.getenv("GMAIL_SENDER_NAME", "Agenda Manzzo y Cía")


@lru_cache(maxsize=1)
def _gmail_service():
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=_oauth_credentials(),
                 cache_discovery=False)


def _oauth_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = OAuthCredentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if os.getenv("RAILWAY_ENVIRONMENT") or not os.path.exists(CREDENTIALS_FILE):
            raise RuntimeError(
                "Gmail sin credenciales válidas en entorno headless. "
                "Regenera token.json con scope gmail.send y configura "
                "GOOGLE_CALENDAR_TOKEN_JSON en Railway."
            )
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return creds


def check_auth_preflight() -> bool:
    """Smoke test. Uso: python -c 'from tools.google_mail import check_auth_preflight; check_auth_preflight()'"""
    try:
        profile = _gmail_service().users().getProfile(userId="me").execute()
        logger.info("✅ Gmail auth OK (user=%s)", profile.get("emailAddress"))
        return True
    except Exception as e:
        logger.exception("❌ Gmail auth preflight falló: %s", e)
        return False


def diagnosticar_scopes() -> list[str]:
    """Devuelve los scopes REALES del token.json actual. Útil para
    diagnosticar 'Insufficient Permission'."""
    if not os.path.exists(TOKEN_FILE):
        return []
    try:
        from google.oauth2.credentials import Credentials as OAuthCredentials
        creds = OAuthCredentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        return list(creds.scopes or [])
    except Exception as e:
        logger.exception("diagnosticar_scopes falló: %s", e)
        return []


# ---------------------------------------------------------------------------
# API PÚBLICA
# ---------------------------------------------------------------------------
def enviar_correo(to: str | list[str], asunto: str, cuerpo_html: str,
                  cuerpo_texto: str | None = None,
                  reply_to: str | None = None) -> bool:
    """Envía un correo HTML + fallback texto. Devuelve True si Gmail aceptó,
    False ante cualquier fallo (nunca lanza). Idempotencia y reintentos son
    responsabilidad del llamador."""
    if isinstance(to, str):
        to = [to]
    to = [t.strip() for t in to if t and t.strip()]
    if not to:
        logger.warning("enviar_correo: sin destinatarios")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = formataddr((SENDER_NAME, SENDER_EMAIL))
    msg["To"] = ", ".join(to)
    msg["Subject"] = asunto
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=SENDER_EMAIL.split("@")[-1])
    if reply_to:
        msg["Reply-To"] = reply_to

    if not cuerpo_texto:
        cuerpo_texto = _html_a_texto(cuerpo_html)
    msg.attach(MIMEText(cuerpo_texto, "plain", "utf-8"))
    msg.attach(MIMEText(cuerpo_html, "html", "utf-8"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")
    try:
        _gmail_service().users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()
        logger.info("correo enviado | %s | %s", asunto, to)
        return True
    except Exception as e:
        logger.exception("enviar_correo falló (%s): %s", to, e)
        return False


def _html_a_texto(html: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    t = re.sub(r"</p>", "\n\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    return t.strip()


# ---------------------------------------------------------------------------
# PLANTILLAS
# ---------------------------------------------------------------------------
def plantilla_cita_studio(*, lead_nombre: str, lead_email: str,
                          lead_telefono: str | None, modalidad_label: str,
                          fecha: str, hora_inicio: str, hora_fin: str,
                          meet_link: str | None, motivo: str | None,
                          html_link: str | None) -> tuple[str, str]:
    """Devuelve (asunto, cuerpo_html) para la alerta al estudio."""
    asunto = f"Nueva cita agendada — {lead_nombre} — {fecha} {hora_inicio}"
    meet_row = (f'<tr><td><b>Meet</b></td>'
                f'<td><a href="{meet_link}">{meet_link}</a></td></tr>'
                ) if meet_link else ""
    motivo_row = (f'<tr><td><b>Motivo</b></td><td>{motivo}</td></tr>'
                  ) if motivo else ""
    tel_row = (f'<tr><td><b>Teléfono</b></td><td>{lead_telefono}</td></tr>'
               ) if lead_telefono else ""
    ver_cal = (f'<p><a href="{html_link}">Ver en Google Calendar →</a></p>'
               ) if html_link else ""
    html = f"""\
<html><body style="font-family:Arial,sans-serif;color:#222;max-width:600px">
  <h2 style="margin:0 0 12px 0">📅 Nueva cita agendada</h2>
  <table cellpadding="6" style="border-collapse:collapse;width:100%">
    <tr><td><b>Cliente</b></td><td>{lead_nombre}</td></tr>
    <tr><td><b>Email</b></td><td>{lead_email}</td></tr>
    {tel_row}
    <tr><td><b>Modalidad</b></td><td>{modalidad_label}</td></tr>
    <tr><td><b>Fecha</b></td><td>{fecha}, {hora_inicio}–{hora_fin}</td></tr>
    {meet_row}
    {motivo_row}
  </table>
  {ver_cal}
  <p style="color:#888;font-size:12px;margin-top:24px">
    Enviado automáticamente por LeyIA — Manzzo y Cía.
  </p>
</body></html>"""
    return asunto, html
