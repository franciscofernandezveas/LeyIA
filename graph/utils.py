"""graph/utils.py — Helpers transversales del grafo padre e intake.

v1.9 — FIX saludos reiterados + bug riesgo + contrato intake_skipped:
  - Añade `_apertura_pres` para que solo la primera respuesta del hilo
    anteponga el disclosure.
  - Actualiza TEMPLATE_ARGS para validar los nuevos placeholders
    `{apertura}` e `{instruccion_apertura}` de prompts.yaml v4.2.3.
  - RESET_SESION_NUEVA incluye `asistente_presentado: False` (una
    presentación por sesión).
  - RESET_INTAKE incluye `intake_skipped: set()` (ahora canal declarado).
  - `_format_few_shots` ahora incluye la etiqueta `riesgo`.

v1.8 — FIX prefijo wa-:
  - _canal_desconocido reconoce "wa-" como canal whatsapp.
  - _telefono_cliente extrae el número de "wa-56912345678".
  - Unificación: única fuente del número de la abogada/esecutiva a través
    de _ejecutiva_number(), compartida con integrations/whatsapp/notify.py
    e integrations/whatsapp/handler.py.

v1.7 — FIX crisis:
  - RESET_SESION_NUEVA incluye "crisis": False para que el flag de crisis no
    sobreviva entre sesiones de conversación distintas (gatekeeper de
    receive_message).
"""

import json
import logging
import os
import re
import unicodedata

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import yaml
from langchain_core.messages import AIMessage

from core.db_client import insert_message, upsert_conversation

logger = logging.getLogger(__name__)

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "core" / "prompts.yaml"
ESCALATIONS_DIR = Path("escalations")
WA_TEXTO_MAX = 900

AGENDA_CAPTURA_TTL_HORAS = 24
INTAKE_TTL_HORAS = 24


# ---------------------------------------------------------------------------
# Resets de estado transitorio
# ---------------------------------------------------------------------------
RESET_INTAKE = {
    "intake_activo": False,
    "intake_idx": 0,
    "intake_respuestas": {},
    "intake_attempts": 0,
    "intake_exit": None,
    "intake_resume": False,
    "intake_decision": None,
    "intake_stage": None,
    "intake_started_en": None,
    "intake_completado": False,
    "intake_oferta_qid": None,
    "intake_skipped": set(),      # ← v1.9: canal ahora declarado en AgentState
}

RESET_HANDOFF_CHOICE = {
    "handoff_choice_pending": False,
    "handoff_choice_attempts": 0,
    "handoff_choice_en": None,
}

RESET_BOOKING = {
    "booking_stage": None,
    "slots_propuestos": [],
    "booking_match": None,
    "booking_match_candidatos": [],
    "booking_signal": None,
    "booking_attempts": 0,
    "booking_decision": None,
    "agenda_started_en": None,
    "agenda_ventana_desde": 0,
    "booking_franja": None,
    "agenda_dia_sin_cupos": None,
    "lead_modalidad": None,
}

# Nueva sesión de conversación → el "caso" anterior quedó atrás.
# Se CONSERVA a propósito: lead_nombre, lead_email, booking_privacidad_dicha
# (política lead_*: memoria larga del cliente).
RESET_SESION_NUEVA = {
    "closed": False,
    "summary": None,
    "notificacion_pendiente": False,
    "case_category": None,
    "cta_agenda_ofrecida": False,
    "crisis": False,                   # v1.7: no sobrevive entre sesiones
    "asistente_presentado": False,     # ← v1.9: una presentación por sesión
    **RESET_INTAKE,
    **RESET_HANDOFF_CHOICE,
    **RESET_BOOKING,
}


# --------------------------------------------------------------------------
# Config / prompts.yaml
# --------------------------------------------------------------------------
REQUIRED_KEYS = {
    "atencion": {"disclosure", "cta_agendar", "faq_system_prompt", "tonos",
                 "summary_system_prompt", "fuera_dominio_message",
                 "handoff_message", "hilo_cerrado_message",
                 "agenda_inicio_aviso",
                 "agenda_pedir_nombre", "agenda_pedir_email",
                 "agenda_pedir_modalidad", "agenda_proponer_slots",
                 "agenda_reintento_slots", "agenda_sin_horarios",
                 "agenda_confirmada", "agenda_slot_ocupado",
                 "agenda_eleccion_ambigua", "agenda_abortado",
                 "agenda_lateral_system"},
    "classification": {"system_prompt", "few_shot_examples"},
    "intake": {"aviso_saltar", "sin_consentimiento", "reanudar",
               "derivacion_parcial", "cierre_abandono"},
}

TEMPLATE_ARGS = {
    "atencion.faq_system_prompt": {"tono", "context",
                                  "instruccion_apertura"},      # ← v1.9
    "atencion.fuera_dominio_message": {"apertura"},             # ← v1.9
    "atencion.handoff_message": {"apertura", "whatsapp_ejecutiva",
                                 "thread_id"},                  # ← v1.9
    "atencion.handoff_fallback_message": {"apertura", "whatsapp_ejecutiva",
                                          "thread_id"},          # ← v1.9
    "atencion.handoff_choice_message": {"apertura"},              # ← v1.9
    "atencion.handoff_choice_retry_message": set(),               # ← v1.9
    "atencion.hilo_cerrado_message": {"whatsapp_ejecutiva", "thread_id"},
    "atencion.cta_agendar": set(),
    "atencion.agenda_inicio_aviso": set(),
    "atencion.agenda_pedir_nombre": set(),
    "atencion.agenda_pedir_email": set(),
    "atencion.agenda_pedir_modalidad": set(),
    "atencion.agenda_proponer_slots": {"nombre", "opciones"},
    "atencion.agenda_reintento_slots": {"opciones"},
    "atencion.agenda_sin_horarios": set(),
    "atencion.agenda_confirmada": {"nombre", "fecha", "hora_inicio",
                                   "hora_fin", "modalidad_linea", "html_link"},
    "atencion.agenda_slot_ocupado": {"opciones"},
    "atencion.agenda_eleccion_ambigua": {"opciones"},
    "atencion.agenda_abortado": set(),
    "atencion.agenda_lateral_system": {"opciones"},
    "intake.apertura_ia": set(),           # ← v1.9
    "intake.apertura_expectativa": set(),
    "intake.aviso_saltar": set(),
    "intake.sin_consentimiento": {"wa_link"},
    "intake.reanudar": set(),
    "intake.reanudar_duda": {"duda"},
    "intake.oferta_salida": {"nombre"},
    "intake.error_no_saltar": set(),
    "intake.cierre_listo": {"nombre"},
    "intake.cierre_sla": set(),
    "intake.cierre_ctas": set(),            # ← v1.9: plantilla sin placeholders
    "intake.derivacion_parcial": {"wa_link", "nombre"},
    "intake.cierre_abandono": {"wa_link"},
}

# Nota v1.7 (opcional/pendiente): si se implementa contador de ofertas de
# agendamiento en FAQ, registrar aquí el placeholder correspondiente, p.ej.
# "atencion.faq_system_prompt": {"disclosure", "tono", "context", "oferta_estado"}
# y pasarlo desde graph/faq/nodes.py::responder_pregunta.

_PLACEHOLDER_RE = re.compile(r"{(\w+)}")


@lru_cache(maxsize=1)
def _cfg() -> dict:
    with open(PROMPTS_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [f"{sec}.{key}" for sec, keys in REQUIRED_KEYS.items()
               for key in keys if key not in (cfg.get(sec) or {})]
    if missing:
        raise KeyError(f"prompts.yaml incompleto — faltan: {missing}")
    bad = []
    for dotted, esperados in TEMPLATE_ARGS.items():
        seccion, key = dotted.split(".", 1)
        tpl = (cfg.get(seccion) or {}).get(key) or ""
        encontrados = set(_PLACEHOLDER_RE.findall(tpl))
        if not encontrados.issubset(esperados):
            bad.append(f"{dotted}: plantilla usa "
                       f"{sorted(encontrados - esperados)} que el nodo no "
                       f"entrega (disponibles: {sorted(esperados)})")
    if bad:
        raise ValueError("prompts.yaml — placeholders desalineados:\n" + "\n".join(bad))
    return cfg


def _apertura_pres(state: dict) -> str:
    """Texto de apertura (disclosure) si el asistente aún no se presentó.

    Usa el flag asistente_presentado; como defensa adicional, si ya hay
    mensajes AI en el historial y el flag no está, no saluda.
    """
    if state.get("asistente_presentado"):
        return ""
    msgs = state.get("messages", [])
    if any(getattr(m, "type", None) == "ai" for m in msgs):
        return ""
    disclosure = _cfg()["atencion"].get("disclosure", "")
    return (disclosure + " ").strip() if disclosure else ""


def _truncar_burbuja(texto: str, max_len: int = WA_TEXTO_MAX) -> str:
    if len(texto) <= max_len:
        return texto
    return texto[:max_len - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# Número de la ejecutiva / abogada (única fuente de verdad)
# --------------------------------------------------------------------------
def _ejecutiva_number() -> str:
    """Número E.164 sin '+'. Usado para:
      - links wa.me que ve el cliente (contacto directo)
      - alertas HITL del handler.py (vía import)
      - notificaciones de negocio (notify.py, vía import)
    Lazy: lee en cada uso para no depender del orden de carga de dotenv."""
    raw = (os.getenv("LAWYER_WHATSAPP_TO")
           or os.getenv("WHATSAPP_EJECUTIVA")
           or os.getenv("WHATSAPP_OPERATOR_NUMBER") or "")
    return re.sub(r"\D", "", raw)


# --------------------------------------------------------------------------
# WhatsApp / links
# --------------------------------------------------------------------------
def _wa_link(texto: str = "") -> str:
    base = f"https://wa.me/{_ejecutiva_number()}"
    return f"{base}?text={quote(texto)}" if texto else base


def _wa_link_display(texto: str = "", display: str = "Hablar con un humano") -> str:
    return f"[{display}]({_wa_link(texto)})"


def _wa_texto_intake(state: dict) -> str:
    r = state.get("intake_respuestas") or {}
    nombre = r.get("nombre")

    lineas = [
        f"Hola, soy {nombre}." if nombre
        else "Hola, vengo del asistente virtual de Manzzo y Cía.",
        ("Acabo de completar mi registro. Resumen de mi caso:"
         if state.get("intake_completado") else
         "Prefiero hablar directamente con una ejecutiva. Datos que alcancé a registrar:"),
    ]
    area = state.get("case_category") or state.get("category")
    if area:
        lineas.append(f"• Área: {area}")
    if r.get("email"):
        lineas.append(f"• Correo: {r['email']}")
    if r.get("situacion_actual"):
        sit = str(r["situacion_actual"])
        if len(sit) > 280:
            sit = sit[:277].rstrip() + "…"
        lineas.append(f"• Mi situación: {sit}")
    if r.get("etapa_proceso"):
        lineas.append(f"• Etapa de mi caso: {r['etapa_proceso']}")
    if r.get("comuna"):
        lineas.append(f"• Comuna: {r['comuna']}")
    if r.get("hijos_menores") is True:
        lineas.append("• Tengo hijos menores de edad")
    if r.get("horario_contacto"):
        lineas.append(f"• Horario preferido: {r['horario_contacto']}")

    tid = state.get("thread_id")
    if tid:
        lineas.append(f"(ID de mi atención: {tid})")

    txt = "\n".join(lineas)
    return _truncar_burbuja(txt, WA_TEXTO_MAX)


def _wa_link_diagnostico(state: dict) -> str:
    return _wa_link_display(_wa_texto_intake(state), display="Habler con un humano")


def _wa_link_cliente(state: dict) -> str:
    if state.get("intake_respuestas"):
        return _wa_link_diagnostico(state)
    return _wa_link_display(
        "Hola, vengo del asistente virtual de Manzzo y Cía.",
        display="Hablar con un humano",
    )


# --------------------------------------------------------------------------
# Fechas
# --------------------------------------------------------------------------
def _ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(iso_ts: str | None) -> datetime | None:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _expirado(iso_ts: str | None, ttl_horas: int) -> bool:
    dt = _parse_dt(iso_ts)
    return bool(dt) and (datetime.now(timezone.utc) - dt > timedelta(hours=ttl_horas))


# --------------------------------------------------------------------------
# Mensajes / historial
# --------------------------------------------------------------------------
def _recent_messages(state: dict, k: int = 6) -> list:
    return state.get("messages", [])[-k:]


def _format_few_shots(examples: list[dict]) -> str:
    return "\n\n".join(
        f'ENTRADA: "{ex["input"]}"\n'
        f"sentiment={ex['sentiment']} | urgency={ex['urgency']} | "
        f"intent={ex['intent']} | category={ex['category']} | "
        f"riesgo={ex.get('riesgo', 'false')} | "     # ← v1.9
        f"reason={ex['reason']}"
        for ex in examples
    )


def _primer_nombre(nombre: str | None) -> str:
    partes = (nombre or "").split()
    return partes[0] if partes else ""


def _norm_simple(s: str) -> str:
    return (s or "").strip().lower().rstrip(".")


def _transcript(state: dict) -> list[dict]:
    return [{"rol": m.type, "contenido": m.content}
            for m in state.get("messages", [])]


def _transcript_resumen(state: dict, max_chars: int = 6000) -> str:
    txt = "\n".join(f"{t['rol']}: {t['contenido']}" for t in _transcript(state))
    if len(txt) > max_chars:
        txt = "…[inicio de la conversación omitido]\n" + txt[-max_chars:]
    return txt


def _ficha_intake(state: dict) -> str:
    r = state.get("intake_respuestas") or {}
    if not r:
        return ""
    filas = "\n".join(f"- {k}: {v}" for k, v in r.items() if v not in (None, ""))
    return f"FICHA DE INTAKE:\n{filas}\n\n"


def _telefono_cliente(state: dict) -> str | None:
    """Extrae el número del cliente desde el thread_id.
    Soporta '56912345678' y 'wa-56912345678'."""
    tid = (state.get("thread_id") or "")
    if tid.startswith("wa-"):
        tid = tid[3:]
    tid = tid.strip()
    return tid if re.fullmatch(r"\+?\d{8,15}", tid) else None


def _canal_desconocido(thread_id: str | None) -> str:
    """Detecta el canal desde el prefijo del thread_id."""
    tid = thread_id or ""
    if tid.startswith("wa-"):
        return "whatsapp"
    if tid.startswith("cli-"):
        return "cli"
    if tid.startswith("web-"):
        return "web"
    if re.fullmatch(r"\+?\d{8,15}", tid):
        return "whatsapp"
    return "desconocido"


def _guardar_ai(state: dict, content: str) -> None:
    insert_message(
        thread_id=state.get("thread_id", ""),
        role="ai", content=content,
        route=state.get("route"), sentiment=state.get("sentiment"),
        urgency=state.get("urgency"), intent=state.get("intent"),
        category=state.get("category"),
    )


def _persist_escalation(state: dict, summary: str) -> Path:
    payload = {
        "thread_id": state.get("thread_id"),
        "cerrado_en": _ahora_iso(),
        "clasificacion": {
            "sentiment": state.get("sentiment"), "urgency": state.get("urgency"),
            "intent": state.get("intent"), "category": state.get("category"),
        },
        "lead": {"nombre": state.get("lead_nombre"),
                 "email": state.get("lead_email"),
                 "modalidad": state.get("lead_modalidad")},
        "intake": state.get("intake_respuestas"),
        "booking": state.get("booking"),
        "resumen_ejecutiva": summary,
        "transcript": _transcript(state),
    }
    ESCALATIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(state.get("thread_id") or "sin-id"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = ESCALATIONS_DIR / f"{safe_id}_{timestamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Abort explícito de booking (v1.4 — fix #6, auditoría)
# --------------------------------------------------------------------------
_ABORT_BOOKING_PHRASES = (
    "cancelar la hora", "cancela la hora", "cancelo la hora",
    "cancelar la cita", "cancela la cita", "cancelo la cita",
    "cancelar el agendamiento", "cancela el agendamiento",
    "cancelar mi hora", "cancela mi hora",
    "cancelar mi cita", "cancela mi cita",
    "anular la hora", "anula la hora", "anular la cita", "anula la cita",
    "anular mi hora", "anula mi hora",
    "cancelalo", "cancelala", "cancelalo todo",
    "ya no quiero agendar", "ya no quiero la hora", "ya no quiero una cita",
    "ya no quiero la asesoria",
    "ya no me interesa agendar", "ya no me interesa la hora",
    "no quiero agendar", "no quiero agendar mas",
    "no quiero seguir con esto", "no quiero continuar con esto",
    "no me interesa agendar", "no me interesa la hora",
    "olvidalo", "olvidalo todo", "olvidate del tema",
    "dejalo asi", "dejemoslo", "dejemoslo asi",
    "mejor lo dejamos", "mejor no agendo", "mejor no agendamos",
)

_ABORT_BOOKING_RX = tuple(
    re.compile(r"\b" + re.escape(frase) + r"\b", re.IGNORECASE)
    for frase in _ABORT_BOOKING_PHRASES
)


def _parece_abort(query: str) -> bool:
    """Detección de abort EXPLÍCITO de booking."""
    q = (query or "").strip().lower()
    if not q:
        return False
    q_norm = "".join(c for c in unicodedata.normalize("NFD", q)
                     if unicodedata.category(c) != "Mn")
    return any(rx.search(q_norm) for rx in _ABORT_BOOKING_RX)


# --------------------------------------------------------------------------
# Upsert inicial de conversación abierta (usado por receive_message)
# --------------------------------------------------------------------------
def _upsert_conversacion_abierta(thread_id: str) -> None:
    upsert_conversation(
        thread_id=thread_id,
        channel=_canal_desconocido(thread_id),
        status="open",
    )
