"""tools/google_calendar.py — Agendamiento DIRECTO en Google Calendar.

El agente crea / reagenda / cancela el evento al tiro en el calendario del
estudio. Si la modalidad es online, el evento lleva Google Meet incluido
(conferenceData) y el cliente recibe la invitación por correo automáticamente
(sendUpdates='all').

Auth (OAuth Desktop app con manzzo.mkt@gmail.com):
  credentials.json — OAuth client "Desktop app" descargado de Google Cloud.
  token.json       — se genera UNA vez en local (abre el browser) y se
                     autorrefresca. En Railway se escriben desde las env vars
                     GOOGLE_CALENDAR_CREDENTIALS_JSON / GOOGLE_CALENDAR_TOKEN_JSON.
                     La app OAuth debe estar PUBLICADA ("In production"); si
                     queda en Testing el refresh token expira en 7 días.
                     NUNCA commitear ninguno de los dos.

Dependencias (solo libs oficiales de Google):
  pip install google-api-python-client google-auth-oauthlib google-auth

Ninguna función lanza excepción hacia el grafo: devuelven None/[]/False y
loguean — quién decide el fallback es el nodo agendar_asesoria (mismo patrón
"best-effort" que la sync con SheetDB).

Changelog:
  v5 — Guard headless en _oauth_credentials (Railway no cuelga esperando
       un browser inexistente; falla rápido con instrucciones) +
       check_auth_preflight() como smoke test post-deploy / post-rotación.
  v4 — _ensure_files() para Railway + proximos_slots() pública para nodes.py.
  v3 — Se ELIMINA langchain_google_community (arrastra TF/Keras y explota).
  v2 — DIRECCION_OFICINA como constante pública de nivel módulo.
"""
import logging
import os
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG — todo sobreescribible por .env
# ---------------------------------------------------------------------------
CREDENTIALS_FILE = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE = os.getenv("GOOGLE_CALENDAR_TOKEN_FILE", "token.json")
CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
TZ = ZoneInfo(os.getenv("GOOGLE_CALENDAR_TIMEZONE", "America/Santiago"))

SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Datos del estudio
DIRECCION_OFICINA = os.getenv("DIRECCION_OFICINA", "Oficina Manzzo y Cía")

# Política de agenda del estudio
DURACION_MIN = int(os.getenv("ASESORIA_DURACION_MIN", "30"))
HORA_APERTURA = int(os.getenv("AGENDA_HORA_APERTURA", "9"))
HORA_CIERRE = int(os.getenv("AGENDA_HORA_CIERRE", "18"))
DIAS_HABILES = (0, 1, 2, 3, 4)                               # 0 = lunes
ANTICIPACION_MIN_HORAS = int(os.getenv("AGENDA_ANTICIPACION_HORAS", "2"))


def _ensure_files():
    """Railway no tiene filesystem persistente entre deploys: si las env vars
    traen los JSON, se escriben a disco al arrancar el contenedor."""
    credentials_json = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON")
    if credentials_json and not os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "w", encoding="utf-8") as f:
            f.write(credentials_json)
        logger.info("credentials.json escrito desde env var")

    token_json = os.getenv("GOOGLE_CALENDAR_TOKEN_JSON")
    if token_json and not os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(token_json)
        logger.info("token.json escrito desde env var")


_ensure_files()


class LeadCalendar(BaseModel):
    """Datos mínimos del lead para agendar una asesoría."""
    nombre: str
    email: str
    telefono: str | None = None
    modalidad: str = "online"                              # online|presencial
    categoria: str | None = None
    motivo: str | None = None


class EventoAsesoria(BaseModel):
    """Resultado de crear/reagendar: lo que el nodo necesita para responder."""
    evento_id: str
    html_link: str                            # ver el evento en Calendar
    meet_link: str | None = None              # solo modalidad online
    fecha: str                                # "18/11/2025"
    hora_inicio: str                          # "15:00"
    hora_fin: str
    modalidad_label: str


# ---------------------------------------------------------------------------
# AUTH — solo libs oficiales de Google. Lazy a propósito: sin token.json la
# primera corrida abre el browser, y eso JAMÁS debe gatillarse al importar
# el módulo.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _api_resource():
    from googleapiclient.discovery import build
    return build("calendar", "v3", credentials=_oauth_credentials())


def _oauth_credentials():
    """1) token.json válido → usarlo. 2) expirado con refresh_token → refrescar.
    3) no existe → flujo OAuth en browser UNA vez (SOLO local) y persistir.
    Tras cualquier refresh se re-escribe token.json para no perder la sesión."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = OAuthCredentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())                    # autorrefresh silencioso
    else:
        # v5 — GUARD HEADLESS: en Railway no hay navegador. run_local_server
        # colgaría la request del usuario esperando una autorización que
        # nunca llegará. Se falla rápido con instrucciones concretas; el
        # RuntimeError queda capturado por el try/except de cada función
        # pública (patrón best-effort del módulo), no llega al grafo.
        # Nota: lru_cache NO cachea excepciones → tras arreglar las env vars
        # y redeployar, el próximo intento reintenta limpio.
        if os.getenv("RAILWAY_ENVIRONMENT") or not os.path.exists(CREDENTIALS_FILE):
            raise RuntimeError(
                "Calendar sin credenciales válidas en entorno headless. "
                "Configura GOOGLE_CALENDAR_TOKEN_JSON (y "
                "GOOGLE_CALENDAR_CREDENTIALS_JSON) en Railway — y recuerda "
                "que la app OAuth debe estar PUBLICADA ('In production') o "
                "el refresh token expira en 7 días."
            )
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)       # abre el browser una sola vez
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return creds


def check_auth_preflight() -> bool:
    """Smoke test de auth para post-deploy o tras rotar credenciales.

    Fuerza la construcción de credenciales y consulta la lista de
    calendarios. Nunca lanza excepción hacia el llamador.

    Uso:
        Local:   python -c "from tools.google_calendar import check_auth_preflight; check_auth_preflight()"
        Railway: pestaña Shell del servicio → mismo comando

    Si falla, el traceback en logs dice exactamente qué pasó (token
    revocado, refresh expirado, credenciales ausentes, etc.).
    """
    try:
        _api_resource().calendarList().list(maxResults=1).execute()
        logger.info("✅ Calendar auth OK (CALENDAR_ID=%s)", CALENDAR_ID)
        return True
    except Exception as e:
        logger.exception("❌ Calendar auth preflight falló: %s", e)
        return False


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _meet_link(event: dict) -> str | None:
    for ep in (event.get("conferenceData", {}).get("entryPoints") or []):
        if ep.get("entryPointType") == "video":
            return ep.get("uri")
    return event.get("hangoutLink")


def _a_evento(event: dict) -> EventoAsesoria:
    ini = _iso(event["start"]["dateTime"]).astimezone(TZ)
    fin = _iso(event["end"]["dateTime"]).astimezone(TZ)
    meet = _meet_link(event)
    return EventoAsesoria(
        evento_id=event["id"],
        html_link=event.get("htmlLink", ""),
        meet_link=meet,
        fecha=ini.strftime("%d/%m/%Y"),
        hora_inicio=ini.strftime("%H:%M"),
        hora_fin=fin.strftime("%H:%M"),
        modalidad_label=("💻 Online (Google Meet)" if meet
                         else "📍 Presencial (oficina)"),
    )


def _descripcion(lead: LeadCalendar) -> str:
    lineas = [f"Lead captado por WhatsApp: {lead.nombre} <{lead.email}>"]
    if lead.telefono:
        lineas.append(f"Teléfono: {lead.telefono}")
    if lead.categoria:
        lineas.append(f"Materia: {lead.categoria}")
    if lead.motivo:
        lineas.append(f"Motivo: {lead.motivo}")
    lineas.append("\nAgendado automáticamente por el asistente virtual.")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# API PÚBLICA
# ---------------------------------------------------------------------------
def horarios_disponibles(dia: date) -> list[datetime]:
    """Slots libres de DURACION_MIN dentro del horario hábil de `dia`
    (API freebusy). [] si es fin de semana o si freebusy falla."""
    if dia.weekday() not in DIAS_HABILES:
        return []
    apertura = datetime.combine(dia, time(HORA_APERTURA), TZ)
    cierre = datetime.combine(dia, time(HORA_CIERRE), TZ)
    try:
        fb = _api_resource().freebusy().query(body={
            "timeMin": apertura.isoformat(),
            "timeMax": cierre.isoformat(),
            "timeZone": str(TZ),
            "items": [{"id": CALENDAR_ID}],
        }).execute()
        busy = [(_iso(b["start"]), _iso(b["end"]))
                for b in fb["calendars"][CALENDAR_ID].get("busy", [])]
    except Exception as e:
        logger.exception("freebusy falló: %s", e)
        return []

    minimo = datetime.now(TZ) + timedelta(hours=ANTICIPACION_MIN_HORAS)
    paso = timedelta(minutes=DURACION_MIN)
    slots, t = [], apertura
    while t + paso <= cierre:
        if t >= minimo and not any(t < be and t + paso > bs for bs, be in busy):
            slots.append(t)
        t += paso
    return slots


def proximos_slots(desde_dias: int = 0, hasta_dias: int = 7,
                   total_max: int = 3) -> list[datetime]:
    """Próximos slots libres: 1 por día hábil, alternando mañana/tarde
    para dar variedad. Lo usa nodes.py al proponer horarios al cliente."""
    hoy = datetime.now(TZ).date()
    slots: list[datetime] = []
    for delta in range(desde_dias, desde_dias + hasta_dias):
        if len(slots) >= total_max:
            break
        libres = horarios_disponibles(hoy + timedelta(days=delta))
        if libres:
            slots.append(libres[0] if len(slots) % 2 == 0 else libres[-1])
    return slots


def crear_evento_asesoria(lead: LeadCalendar,
                          inicio: datetime) -> EventoAsesoria | None:
    """Crea el evento e invita al cliente por correo (sendUpdates='all').
    Si `inicio` viene naive se asume hora local del estudio (TZ)."""
    if inicio.tzinfo is None:
        inicio = inicio.replace(tzinfo=TZ)
    fin = inicio + timedelta(minutes=DURACION_MIN)

    body: dict = {
        "summary": f"Asesoría legal — {lead.nombre}",
        "description": _descripcion(lead),
        "start": {"dateTime": inicio.isoformat(), "timeZone": str(TZ)},
        "end": {"dateTime": fin.isoformat(), "timeZone": str(TZ)},
        "attendees": [{"email": lead.email, "displayName": lead.nombre}],
        "reminders": {"useDefault": False, "overrides": [
            {"method": "email", "minutes": 24 * 60},
            {"method": "popup", "minutes": 60}]},
        "colorId": "5",
    }
    if lead.modalidad == "online":
        body["conferenceData"] = {"createRequest": {
            "requestId": uuid4().hex,
            "conferenceSolutionKey": {"type": "hangoutsMeet"}}}
    else:
        body["location"] = DIRECCION_OFICINA

    try:
        event = (_api_resource().events()
                 .insert(calendarId=CALENDAR_ID, body=body,
                         conferenceDataVersion=1, sendUpdates="all")
                 .execute())
    except Exception as e:
        logger.exception("crear evento falló (%s <%s>): %s",
                         lead.nombre, lead.email, e)
        return None
    logger.info("evento creado: %s", event.get("htmlLink"))
    return _a_evento(event)


def reagendar_evento(evento_id: str,
                     nuevo_inicio: datetime) -> EventoAsesoria | None:
    if nuevo_inicio.tzinfo is None:
        nuevo_inicio = nuevo_inicio.replace(tzinfo=TZ)
    fin = nuevo_inicio + timedelta(minutes=DURACION_MIN)
    body = {"start": {"dateTime": nuevo_inicio.isoformat(), "timeZone": str(TZ)},
            "end": {"dateTime": fin.isoformat(), "timeZone": str(TZ)}}
    try:
        event = (_api_resource().events()
                 .patch(calendarId=CALENDAR_ID, eventId=evento_id,
                        body=body, sendUpdates="all")
                 .execute())
    except Exception as e:
        logger.exception("reagendar %s falló: %s", evento_id, e)
        return None
    return _a_evento(event)


def cancelar_evento(evento_id: str) -> bool:
    try:
        (_api_resource().events()
         .delete(calendarId=CALENDAR_ID, eventId=evento_id,
                 sendUpdates="all")
         .execute())
        return True
    except Exception as e:
        logger.exception("cancelar %s falló: %s", evento_id, e)
        return False


def buscar_eventos_lead(email: str, solo_futuros: bool = True) -> list[dict]:
    """Eventos donde el lead aparece (asistente o texto del evento)."""
    params = {"calendarId": CALENDAR_ID, "q": email, "singleEvents": True,
              "orderBy": "startTime", "maxResults": 10}
    if solo_futuros:
        params["timeMin"] = datetime.now(TZ).isoformat()
    try:
        return (_api_resource().events().list(**params).execute()
                .get("items", []))
    except Exception as e:
        logger.exception("buscar eventos de %s falló: %s", email, e)
        return []
