"""graph/utils.py — Helpers transversales del grafo padre e intake.

v1.3 — Añade RESET_INTAKE, RESET_HANDOFF_CHOICE, RESET_BOOKING y
        RESET_SESION_NUEVA centralizados, consumidos por los nodos del grafo.
v1.2 — Registra atencion.agenda_inicio_aviso (emisión one-shot en booking).
v1.1 — Añade WA_TEXTO_MAX y _truncar_burbuja para reutilizar en subgrafos.
"""

import json
import logging
import re
import unicodedata           # ← NUEVO (fix #6)

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import yaml
from langchain_core.messages import AIMessage

from core.db_client import insert_message, upsert_conversation
from tools.notify_whatsapp import WHATSAPP_EJECUTIVA

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
    "cta_agenda_ofrecida": False,      # ← v1.6: nueva sesión = nueva oferta permitida
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
                 "agenda_inicio_aviso",                       # v1.2 — aviso one-shot booking
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
    "atencion.faq_system_prompt": {"disclosure", "tono", "context"},
    "atencion.fuera_dominio_message": {"disclosure"},
    "atencion.handoff_message": {"disclosure", "whatsapp_ejecutiva", "thread_id"},
    "atencion.hilo_cerrado_message": {"whatsapp_ejecutiva", "thread_id"},
    "atencion.cta_agendar": set(),
    "atencion.agenda_inicio_aviso": set(),                   # v1.2 — sin placeholders
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
    "intake.apertura_ia": set(),
    "intake.apertura_expectativa": set(),
    "intake.aviso_saltar": set(),
    "intake.sin_consentimiento": {"wa_link"},
    "intake.reanudar": set(),
    "intake.reanudar_duda": {"duda"},
    "intake.oferta_salida": {"nombre"},
    "intake.error_no_saltar": set(),
    "intake.cierre_listo": {"nombre"},
    "intake.cierre_sla": set(),
    "intake.cierre_ctas": {"wa_link"},
    "intake.derivacion_parcial": {"wa_link", "nombre"},
    "intake.cierre_abandono": {"wa_link"},
}

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


def _truncar_burbuja(texto: str, max_len: int = WA_TEXTO_MAX) -> str:
    if len(texto) <= max_len:
        return texto
    return texto[:max_len - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# WhatsApp / links
# --------------------------------------------------------------------------
def _wa_link(texto: str = "") -> str:
    num = re.sub(r"\D", "", WHATSAPP_EJECUTIVA or "")
    base = f"https://wa.me/{num}"
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
    return _wa_link_display(_wa_texto_intake(state), display="Hablar con un humano")


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
        f"intent={ex['intent']} | category={ex['category']} | reason={ex['reason']}"
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
    tid = state.get("thread_id") or ""
    return tid if re.fullmatch(r"\+?\d{8,15}", tid) else None


def _canal_desconocido(thread_id: str | None) -> str:
    tid = thread_id or ""
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
    path = ESCALATIONS_DIR / f"{safe_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _upsert_conversacion_abierta(thread_id: str) -> None:
    upsert_conversation(
        thread_id=thread_id,
        channel=_canal_desconocido(thread_id),
        status="abierto",
    )


_ABORT_BOOKING = ("no quiero", "mejor no", "olvídalo", "olvidalo",
                  "dejalo", "déjalo", "ya no me interesa")


# --------------------------------------------------------------------------
# Abort explícito de booking (v1.4 — fix #6, auditoría)
#
# Solo frases INEQUÍVOCAS de cancelación/renuncia. Se eliminaron
# deliberadamente "no quiero" y "mejor no" sueltos porque son substring de
# mensajes legítimos:
#   "no quiero online, mejor presencial"  → abortaba (mal)
#   "no quiero el lunes, mejor el martes" → abortaba (mal)
#   "mejor no sé qué hacer"               → abortaba (mal)
#
# Frases ambiguas quedan a cargo del LLM del planner de booking
# (BookingDecision.accion="abortar" con contexto), que es la red de
# seguridad real.
#
# Detección: normalización NFD (tildes fuera) + match por límite de palabra
# (\b...\b), para no matchear "cancelar" dentro de "cancelaron" ni
# "olvidalo" dentro de "no me olvides".
# --------------------------------------------------------------------------
_ABORT_BOOKING_PHRASES = (
    # Cancelación explícita con objeto (cita / hora / agendamiento)
    "cancelar la hora", "cancela la hora", "cancelo la hora",
    "cancelar la cita", "cancela la cita", "cancelo la cita",
    "cancelar el agendamiento", "cancela el agendamiento",
    "cancelar mi hora", "cancela mi hora",
    "cancelar mi cita", "cancela mi cita",
    "anular la hora", "anula la hora", "anular la cita", "anula la cita",
    "anular mi hora", "anula mi hora",
    # Cancelación imperativa
    "cancelalo", "cancelala", "cancelalo todo",
    # Renuncia explícita al agendamiento
    "ya no quiero agendar", "ya no quiero la hora", "ya no quiero una cita",
    "ya no quiero la asesoria",
    "ya no me interesa agendar", "ya no me interesa la hora",
    "no quiero agendar", "no quiero agendar mas",
    "no quiero seguir con esto", "no quiero continuar con esto",
    "no me interesa agendar", "no me interesa la hora",
    # Abandono coloquial (inequívoco)
    "olvidalo", "olvidalo todo", "olvidate del tema",
    "dejalo asi", "dejemoslo", "dejemoslo asi",
    "mejor lo dejamos", "mejor no agendo", "mejor no agendamos",
)

_ABORT_BOOKING_RX = tuple(
    re.compile(r"\b" + re.escape(frase) + r"\b", re.IGNORECASE)
    for frase in _ABORT_BOOKING_PHRASES
)

def _parece_abort(query: str) -> bool:
    """Detección de abort EXPLÍCITO de booking.

    v1.4 (fix #6): normaliza tildes (NFD) y usa frase completa con
    límites de palabra. Solo frases inequívocas — ver bloque de
    _ABORT_BOOKING_PHRASES. Los casos ambiguos los resuelve el LLM del
    planner de booking (accion="abortar").
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    q_norm = "".join(c for c in unicodedata.normalize("NFD", q)
                     if unicodedata.category(c) != "Mn")
    return any(rx.search(q_norm) for rx in _ABORT_BOOKING_RX)

