"""tools/calendly.py — Integración Calendly para LeyIA (Manzzo y Cía).

v3 — Todo en uno:
  • v2 (plan Free, activo): link prellenado (name/email/location/aN dinámicos
    según el formulario real del Event Type) + VERIFICACIÓN de booking vía
    GET /scheduled_events (polling, funciona en Free).
  • v3 (plan pago, flag): Scheduling API — listar horarios + POST /invitees
    sin salir del chat. Activar con CALENDLY_USAR_BOOKING_API=true.
  • Auto-descubrimiento: duration, custom_questions[position] y locations
    leídos del propio Event Type (GET /event_types/{uuid}).
"""
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel, Field, field_validator

from core.config import (
    CALENDLY_API_TOKEN, CALENDLY_EVENT_TYPE_URI, CALENDLY_PUBLIC_LINK,
    CALENDLY_USAR_BOOKING_API,
)
from core.contracts import ModalidadLabel

logger = logging.getLogger(__name__)

API = "https://api.calendly.com"
TZ_IANA = "America/Santiago"
DIRECCION_OFICINA = "Psje. Doctor Sotero del Río #508, Of. 420, Santiago Centro"
MINUTOS_DEFAULT = 45  # si la introspección falla (FAQ: consultas de 30-45 min)

CATEGORIA_LABELS = {
    "pension_alimentos": "Pensión de alimentos",
    "rebaja_pension": "Rebaja de pensión",
    "regimen_visitas": "Relación directa y regular",
    "medidas_apremio": "Medidas de apremio",
    "terminacion_pension": "Término de pensión",
    "divorcio": "Divorcio",
    "compensacion_economica": "Compensación económica",
    "consulta_general": "Consulta general",
    "otro": "Otro",
}
MODALIDAD_LABELS = {
    "online": "Online (Google Meet)",
    "presencial": f"Presencial ({DIRECCION_OFICINA})",
}
KEYWORDS = {  # mapea pregunta del formulario → campo del lead, por su nombre
    "tipo_consulta": ["tipo de consulta", "servicio", "materia", "área"],
    "detalle_caso": ["caso", "detalle", "cuéntanos", "describe", "comentarios"],
    "modalidad": ["modalidad", "presencial", "online", "atención"],
}
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


# --------------------------------------------------------------------------
# EXCEPCIONES + MODELOS
# --------------------------------------------------------------------------
class ScopeError(Exception): ...        # token sin permiso / plan Free en endpoint pago
class AuthError(Exception): ...         # token inválido o expirado
class SlotCaducoError(Exception): ...   # horario ya tomado


class LeadData(BaseModel):
    """Datos del cliente capturados durante la conversación."""
    nombre: str = Field(min_length=3)
    email: str
    telefono: str | None = None                    # E.164 (+569...)
    categoria: str | None = None
    motivo: str | None = None
    modalidad: ModalidadLabel | None = None

    @field_validator("email")
    @classmethod
    def _email_ok(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("email inválido")
        return v


class BookingInfo(BaseModel):
    """Cita confirmada leída desde Calendly."""
    nombre_evento: str
    fecha: str                    # "viernes 28/08"
    hora_inicio: str              # "12:00"
    hora_fin: str                 # "13:00"
    modalidad_linea: str
    reschedule_url: str | None = None
    cancel_url: str | None = None


# --------------------------------------------------------------------------
# AUTO-DESCUBRIMIENTO DEL EVENT TYPE (scope: event_types:read — plan Free OK)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def obtener_event_type_info() -> dict | None:
    """GET /event_types/{uuid} → {duracion_min, preguntas[position], locations}.

    position de custom_questions = orden en la página de reserva; el mismo valor
    se usa en questions_and_answers[position] (v3) y en el param aN (v2).
    """
    if not CALENDLY_API_TOKEN:
        return None
    resp = requests.get(CALENDLY_EVENT_TYPE_URI,
                        headers={"Authorization": f"Bearer {CALENDLY_API_TOKEN}"},
                        timeout=15)
    if resp.status_code == 403:
        raise ScopeError(
            "403 en GET event_type → aprueba el scope 'event_types:read' en tu token.")
    resp.raise_for_status()
    r = resp.json()["resource"]
    return {
        "nombre": r["name"],
        "duracion_min": r.get("duration", MINUTOS_DEFAULT),
        "preguntas": sorted(
            [{"position": q["position"], "name": q["name"],
              "required": q.get("required", False)}
             for q in r.get("custom_questions", []) if q.get("enabled")],
            key=lambda q: q["position"]),
        "locations": r.get("locations", []),
    }


# --------------------------------------------------------------------------
# MAPEO DE RESPUESTAS DEL FORMULARIO
# --------------------------------------------------------------------------
def _resumen_compuesto(lead: LeadData) -> str:
    """Formulario genérico (sin preguntas mapeables): todo el contexto en un campo."""
    partes = []
    if lead.categoria:
        partes.append(f"Tipo de consulta: {CATEGORIA_LABELS.get(lead.categoria, 'Consulta general')}")
    if lead.modalidad:
        partes.append(f"Modalidad preferida: {MODALIDAD_LABELS[lead.modalidad]}")
    if lead.motivo:
        partes.append(f"Caso: {lead.motivo[:200]}")
    return " | ".join(partes)[:800] or "Consulta agendada vía asistente virtual"


def _respuestas_lead(lead: LeadData, preguntas: list[dict]) -> dict[int, str]:
    """position → respuesta del lead, por keyword-matching del nombre de la pregunta."""
    out: dict[int, str] = {}
    for q in preguntas:
        nombre_q = q["name"].lower()
        if any(k in nombre_q for k in KEYWORDS["tipo_consulta"]):
            out[q["position"]] = CATEGORIA_LABELS.get(lead.categoria or "", "Consulta general")
        elif any(k in nombre_q for k in KEYWORDS["detalle_caso"]):
            out[q["position"]] = (lead.motivo or "—")[:400]
        elif any(k in nombre_q for k in KEYWORDS["modalidad"]) and lead.modalidad:
            out[q["position"]] = MODALIDAD_LABELS[lead.modalidad]

    # Formulario default Calendly (1 pregunta genérica) → resumen compuesto
    if not out and preguntas:
        out[preguntas[0]["position"]] = _resumen_compuesto(lead)
    return out


# --------------------------------------------------------------------------
# V2 — LINK PRELLENADO (camino principal en plan Free)
# --------------------------------------------------------------------------
def _prefill_query(lead: LeadData) -> str:
    params = {"name": lead.nombre, "email": lead.email}
    if lead.telefono:
        params["location"] = lead.telefono        # solo si el form pide teléfono/SMS
    try:
        info = obtener_event_type_info()
        respuestas = _respuestas_lead(lead, info["preguntas"]) if info else {}
    except Exception as exc:                      # introspección nunca rompe el flujo
        logger.warning("introspección de formulario falló (%s); fallback estático.", exc)
        respuestas = {}
    for pos, answer in respuestas.items():
        params[f"a{pos + 1}"] = answer            # ⚠️ verificar una vez: a1 = pos 0
    return urlencode(params)                      # escapa tildes/ñ/paréntesis


def _con_prefill(url: str, lead: LeadData | None) -> str:
    if not lead:
        return url
    return f"{url}{'&' if '?' in url else '?'}{_prefill_query(lead)}"


def crear_link_agendamiento(lead: LeadData | None = None,
                            max_event_count: int = 1) -> str:
    """Single-use link + prefill (con scope scheduling_links:write) o público + prefill."""
    if CALENDLY_API_TOKEN and CALENDLY_EVENT_TYPE_URI:
        try:
            resp = requests.post(
                f"{API}/scheduling_links",
                headers={"Authorization": f"Bearer {CALENDLY_API_TOKEN}",
                         "Content-Type": "application/json"},
                json={"max_event_count": max_event_count,
                      "owner": CALENDLY_EVENT_TYPE_URI, "owner_type": "EventType"},
                timeout=15,
            )
            if resp.status_code == 403:
                raise ScopeError(
                    "403 en scheduling_links → aprueba 'scheduling_links:write' "
                    "en tu token. Mientras, se usa el link público.")
            resp.raise_for_status()
            return _con_prefill(resp.json()["resource"]["booking_url"], lead)
        except ScopeError as e:
            logger.warning("%s", e)
        except Exception as exc:
            logger.warning("Calendly API caída (%s); link público.", exc)
    return _con_prefill(CALENDLY_PUBLIC_LINK, lead)


# --------------------------------------------------------------------------
# VERIFICACIÓN DE BOOKING (polling — funciona en plan Free)
# scope: scheduled_events:read
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _org_uri() -> str:
    resp = requests.get(f"{API}/users/me",
                        headers={"Authorization": f"Bearer {CALENDLY_API_TOKEN}"},
                        timeout=15)
    resp.raise_for_status()
    return resp.json()["resource"]["current_organization"]


def _fmt_modalidad(loc: dict | None) -> str:
    kind = (loc or {}).get("kind")
    if kind == "google_conference":
        extra = f" ({loc.get('join_url')})" if loc.get("join_url") else " (el link llega a tu correo)"
        return f"💻 Online — Google Meet{extra}"
    if kind in ("physical", "custom"):
        return f"📍 {loc.get('location', DIRECCION_OFICINA)}"
    if kind == "zoom":
        return "💻 Online — Zoom (el link llega a tu correo)"
    return ""


def buscar_booking(email: str, desde: datetime | None = None) -> BookingInfo | None:
    """Devuelve la cita activa más próxima del lead, o None si aún no agenda."""
    if not (CALENDLY_API_TOKEN and email):
        return None
    try:
        desde = desde or datetime.now(timezone.utc)
        r = requests.get(
            f"{API}/scheduled_events",
            headers={"Authorization": f"Bearer {CALENDLY_API_TOKEN}"},
            params={
                "organization": _org_uri(),
                "invitee_email": email,
                "status": "active",
                "min_start_time": desde.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sort": "start_time:asc",
            },
            timeout=15,
        )
        r.raise_for_status()
        eventos = r.json().get("collection", [])
        if not eventos:
            return None
        ev = eventos[0]

        # Links de gestión viven en el invitee
        uuid = ev["uri"].rsplit("/", 1)[-1]
        inv = requests.get(f"{API}/scheduled_events/{uuid}/invitees",
                           headers={"Authorization": f"Bearer {CALENDLY_API_TOKEN}"},
                           timeout=15)
        inv.raise_for_status()
        invitee = (inv.json().get("collection") or [{}])[0]

        tz = ZoneInfo(TZ_IANA)
        ini = datetime.fromisoformat(ev["start_time"].replace("Z", "+00:00")).astimezone(tz)
        fin = datetime.fromisoformat(ev["end_time"].replace("Z", "+00:00")).astimezone(tz)

        return BookingInfo(
            nombre_evento=ev.get("name", "Consulta"),
            fecha=f"{DIAS[ini.weekday()]} {ini.day:02d}/{ini.month:02d}",
            hora_inicio=ini.strftime("%H:%M"),
            hora_fin=fin.strftime("%H:%M"),
            modalidad_linea=_fmt_modalidad(ev.get("location")),
            reschedule_url=invitee.get("reschedule_url"),
            cancel_url=invitee.get("cancel_url"),
        )
    except requests.RequestException as exc:
        logger.warning("buscar_booking falló (%s)", exc)
        return None


# --------------------------------------------------------------------------
# V3 — SCHEDULING API (booking total sin salir del chat; REQUIERE PLAN PAGO)
# Activar: CALENDLY_USAR_BOOKING_API=true + scopes availability:read + scheduled_events:write
# --------------------------------------------------------------------------
def listar_horarios(dias: int = 7, max_slots: int = 3) -> list[dict]:
    """GET /event_type_available_times → [{'utc': ..., 'local': 'lunes 03/03 · 10:00'}]."""
    if not CALENDLY_USAR_BOOKING_API:
        raise ScopeError("Scheduling API requiere plan pago y CALENDLY_USAR_BOOKING_API=true")
    ahora = datetime.now(timezone.utc)
    resp = requests.get(
        f"{API}/event_type_available_times",
        headers={"Authorization": f"Bearer {CALENDLY_API_TOKEN}"},
        params={"event_type": CALENDLY_EVENT_TYPE_URI,
                "start_time": ahora.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_time": (ahora + timedelta(days=dias)).strftime("%Y-%m-%dT%H:%M:%SZ")},
        timeout=15,
    )
    if resp.status_code == 429:                       # rate limit → respetar Retry-After
        import time
        time.sleep(int(resp.headers.get("Retry-After", 5)))
        return listar_horarios(dias, max_slots)
    resp.raise_for_status()

    tz = ZoneInfo(TZ_IANA)
    slots = []
    for s in resp.json().get("collection", []):
        if s.get("status") != "available":
            continue
        local = datetime.fromisoformat(s["start_time"].replace("Z", "+00:00")).astimezone(tz)
        slots.append({"utc": s["start_time"],
                      "local": f"{DIAS[local.weekday()]} {local.day:02d}/{local.month:02d} "
                               f"· {local.strftime('%H:%M')} hrs"})
        if len(slots) >= max_slots:
            break
    return slots


def agendar_cita(lead: LeadData, start_time_utc: str) -> dict:
    """POST /invitees — booking real sin UI de Calendly (plan pago)."""
    if not CALENDLY_USAR_BOOKING_API:
        raise ScopeError("Scheduling API requiere plan pago y CALENDLY_USAR_BOOKING_API=true")
    location = ({"kind": "physical", "location": DIRECCION_OFICINA}
                if lead.modalidad == "presencial"
                else {"kind": "google_conference"})
    payload = {
        "event_type": CALENDLY_EVENT_TYPE_URI,
        "start_time": start_time_utc,
        "invitee": {"name": lead.nombre, "email": lead.email, "timezone": TZ_IANA,
                    **({"text_reminder_number": lead.telefono} if lead.telefono else {})},
        "location": location,
        "questions_and_answers": [
            {"question": q["name"], "answer": ans, "position": q["position"]}
            for q in obtener_event_type_info()["preguntas"]
            if (ans := _respuestas_lead(lead, [q]).get(q["position"]))
        ],
        "tracking": {"utm_source": "leyia", "utm_medium": "whatsapp",
                     "utm_campaign": "agente_virtual"},
    }
    resp = requests.post(f"{API}/invitees",
                         headers={"Authorization": f"Bearer {CALENDLY_API_TOKEN}",
                                  "Content-Type": "application/json"},
                         json=payload, timeout=15)
    if resp.status_code == 404:
        raise SlotCaducoError("El horario ya no está disponible; re-consultar.")
    if resp.status_code == 401:
        raise AuthError("Token Calendly inválido/expirado.")
    if resp.status_code == 403:
        raise ScopeError("Falta scope scheduled_events:write o plan pago.")
    if resp.status_code == 400:
        raise ValueError(f"Datos inválidos: {resp.json().get('details')}")
    resp.raise_for_status()

    r = resp.json()["resource"]
    logger.info("cita agendada para %s***", lead.email[:2])   # PII enmascarada
    return {"event_uri": r["scheduled_event"]["uri"],
            "reschedule_url": r.get("reschedule_url"),
            "cancel_url": r.get("cancel_url")}
