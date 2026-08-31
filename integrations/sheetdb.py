"""integrations/sheetdb.py — Guarda/actualiza leads en Google Sheets vía SheetDB.

Endpoint configurado: https://sheetdb.io/api/v1/oxl3h0nanwlhh

SheetDB expone la hoja de Google Sheets como una API REST:
  - POST  /api/v1/{id}              → crear fila(s). Body: {"data": [{...}]}
  - GET   /api/v1/{id}/search       → buscar por columna
  - PATCH /api/v1/{id}/{col}/{val}  → actualizar fila(s) por columna

Las columnas deben llamarse EXACTAMENTE como en la hoja de Google Sheets
(incluyendo espacios y caracteres especiales), porque SheetDB usa los headers
como keys del JSON.

Nota importante: SheetDB rechaza payloads donde falte alguna columna de la hoja
o donde los valores superen el límite de celda (~50.000 caracteres).
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

SHEETDB_BASE_URL = "https://sheetdb.io/api/v1/oxl3h0nanwlhh"
MAX_JSON_BACKUP_CHARS = 30_000  # margen seguro bajo límite de Google Sheets

# Nombres EXACTOS de tus columnas en Google Sheets
COL = {
    "id_lead": "ID Lead",
    "created_at": "Fecha creación",
    "closed_at": "Fecha cierre",
    "source": "Canal origen",
    "lead_status": "Estado lead",
    "nombre": "Nombre completo",
    "email": "Correo electrónico",
    "telefono": "Teléfono",
    "situacion": "Situación / Caso",
    "category": "Categoría legal",
    "etapa": "Etapa del proceso",
    "urgency": "Urgencia",
    "sentiment": "Sentimiento",
    "hijos": "Hijos menores",
    "comuna": "Comuna",
    "horario": "Horario contacto",
    "como_nos_conocio": "Cómo nos conoció",
    "consentimiento": "Consentimiento datos",
    "summary": "Resumen ejecutiva",
    "agenda_link": "Link agendamiento",
    "booking_fecha": "Fecha cita agendada",
    "modalidad": "Modalidad asesoría",
    "asignado_a": "Asignado a",
    "notas": "Notas ejecutiva",
    "thread_id": "Thread ID",
    "intent": "Intent detectado",
    "route": "Route",
    "clf_reason": "Razón clasificación",
    "json_backup": "JSON backup",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _telefono_cliente(thread_id: str | None) -> str | None:
    import re
    tid = thread_id or ""
    return tid if re.fullmatch(r"\+\d{8,15}", tid) else None


def _format_booking(booking: dict | None) -> str:
    if not booking:
        return ""
    fecha = booking.get("fecha") or ""
    hora = booking.get("hora_inicio") or ""
    return f"{fecha} {hora}".strip()


def _si_no(val: Any) -> str:
    if val is True:
        return "SÍ"
    if val is False:
        return "NO"
    return ""


def _detectar_canal(thread_id: str | None) -> str:
    tid = thread_id or ""
    if tid.startswith("cli-"):
        return "cli"
    if tid.startswith("web-"):
        return "web"
    if _telefono_cliente(tid):
        return "whatsapp"
    return "desconocido"


def _build_payload(state: dict, canal: str | None = None) -> dict:
    """Mapea el estado del grafo al payload de SheetDB."""
    r = state.get("intake_respuestas") or {}
    booking = state.get("booking") or {}

    if canal is None:
        canal = _detectar_canal(state.get("thread_id"))

    created = state.get("intake_started_en") or _now_iso()
    closed = _now_iso()

    payload = {
        COL["id_lead"]: state.get("thread_id", ""),
        COL["created_at"]: created,
        COL["closed_at"]: closed,
        COL["source"]: canal,
        COL["lead_status"]: "nuevo",
        COL["nombre"]: r.get("nombre", ""),
        COL["email"]: r.get("email", ""),
        COL["telefono"]: r.get("telefono") or _telefono_cliente(state.get("thread_id")) or "",
        COL["situacion"]: r.get("situacion_actual", ""),
        COL["category"]: state.get("category", ""),
        COL["etapa"]: r.get("etapa_proceso", ""),
        COL["urgency"]: state.get("urgency", ""),
        COL["sentiment"]: state.get("sentiment", ""),
        COL["hijos"]: _si_no(r.get("hijos_menores")),
        COL["comuna"]: r.get("comuna", ""),
        COL["horario"]: r.get("horario_contacto", ""),
        COL["como_nos_conocio"]: r.get("como_nos_conocio", ""),
        COL["consentimiento"]: _si_no(r.get("consentimiento_datos")),
        COL["summary"]: state.get("summary", ""),
        COL["agenda_link"]: state.get("agenda_link", ""),
        COL["booking_fecha"]: _format_booking(booking),
        COL["modalidad"]: state.get("lead_modalidad", ""),
        COL["asignado_a"]: "",
        COL["notas"]: "",
        COL["thread_id"]: state.get("thread_id", ""),
        COL["intent"]: state.get("intent", ""),
        COL["route"]: state.get("route", ""),
        COL["clf_reason"]: state.get("clf_reason", ""),
    }

    # JSON backup: serializar solo campos esenciales, no todo el state
    # (incluir todo el state con messages puede superar el límite de celda)
    backup = {
        "thread_id": state.get("thread_id"),
        "clasificacion": {
            "sentiment": state.get("sentiment"),
            "urgency": state.get("urgency"),
            "intent": state.get("intent"),
            "category": state.get("category"),
            "clf_reason": state.get("clf_reason"),
        },
        "intake": r,
        "lead": {
            "nombre": state.get("lead_nombre"),
            "email": state.get("lead_email"),
            "modalidad": state.get("lead_modalidad"),
        },
        "booking": booking,
    }
    json_str = json.dumps(backup, ensure_ascii=False, default=str)
    payload[COL["json_backup"]] = json_str[:MAX_JSON_BACKUP_CHARS]

    return payload


def _safe_url(value: str) -> str:
    return quote(str(value), safe="")


def _buscar_por_thread_id(thread_id: str | None) -> dict | None:
    """Busca si ya existe una fila con ese Thread ID. Array vacío = no existe."""
    if not thread_id:
        return None
    url = f"{SHEETDB_BASE_URL}/search"
    try:
        r = requests.get(url, params={COL["thread_id"]: thread_id}, timeout=10)
        if r.status_code == 404:
            return None  # no existe (SheetDB a veces devuelve 404 si no hay coincidencias)
        r.raise_for_status()
        rows = r.json()
        if rows and isinstance(rows, list) and len(rows) > 0:
            return rows[0]
    except Exception as e:
        logger.warning("Error buscando lead en SheetDB: %s", e)
    return None


# ---------------------------------------------------------------------------
# OPERACIONES CRUD
# ---------------------------------------------------------------------------
def crear_lead(state: dict, canal: str | None = None) -> dict:
    """Crea una nueva fila en la hoja. SheetDB espera {"data": [payload]}."""
    payload = _build_payload(state, canal)
    body = {"data": [payload]}  # ← FORMATO CORRECTO DE SHEETDB

    try:
        r = requests.post(SHEETDB_BASE_URL, json=body, timeout=15)
        if r.status_code >= 400:
            # Loguear el cuerpo de la respuesta para depurar
            logger.error("SheetDB POST error %s: %s", r.status_code, r.text)
        r.raise_for_status()
        logger.info("Lead creado en SheetDB: %s", state.get("thread_id"))
        return {"ok": True, "action": "created", "response": r.json()}
    except Exception as e:
        logger.exception("Error creando lead en SheetDB: %s", e)
        raise


def actualizar_lead_por_thread_id(thread_id: str | None, data: dict) -> dict:
    """Actualiza campos parciales de la fila cuyo 'Thread ID' coincida."""
    if not thread_id:
        raise ValueError("thread_id requerido para actualizar lead")
    url = f"{SHEETDB_BASE_URL}/{_safe_url(COL['thread_id'])}/{_safe_url(thread_id)}"

    body = {"data": data}  # SheetDB también espera {"data": {...}} para PATCH
    try:
        r = requests.patch(url, json=body, timeout=15)
        if r.status_code >= 400:
            logger.error("SheetDB PATCH error %s: %s", r.status_code, r.text)
        r.raise_for_status()
        logger.info("Lead actualizado en SheetDB: %s", thread_id)
        return {"ok": True, "action": "updated", "response": r.json()}
    except Exception as e:
        logger.exception("Error actualizando lead en SheetDB: %s", e)
        raise


def sync_lead(state: dict, canal: str | None = None) -> dict:
    """Si existe (por Thread ID) actualiza; si no, crea."""
    thread_id = state.get("thread_id")
    if not thread_id:
        raise ValueError("state sin thread_id: no se puede guardar lead")

    existing = _buscar_por_thread_id(thread_id)
    payload = _build_payload(state, canal)

    if existing:
        update_data = {k: v for k, v in payload.items()
                       if k not in (COL["id_lead"], COL["thread_id"])}
        return actualizar_lead_por_thread_id(thread_id, update_data)

    return crear_lead(state, canal)


# ---------------------------------------------------------------------------
# ACTUALIZACIONES PARCIALES
# ---------------------------------------------------------------------------
def actualizar_booking(state: dict) -> dict:
    """Llamar cuando Calendly confirma una cita."""
    thread_id = state.get("thread_id")
    if not thread_id:
        return {"ok": False, "error": "sin thread_id"}

    booking = state.get("booking") or {}
    data = {
        COL["agenda_link"]: state.get("agenda_link", ""),
        COL["booking_fecha"]: _format_booking(booking),
        COL["modalidad"]: state.get("lead_modalidad", ""),
        COL["lead_status"]: "agendado",
        COL["closed_at"]: _now_iso(),
    }
    return actualizar_lead_por_thread_id(thread_id, data)
