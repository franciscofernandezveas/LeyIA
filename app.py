# app.py
"""Dashboard Streamlit para el agente Manzzo y Cía (LeyIA).

Versión: v6.3.2 — import limpio de Command y corrección de sintaxis.
Alineada con grafo padre v16.1, intake v9 y booking por booking_stage.
"""
from dotenv import load_dotenv
load_dotenv()

import importlib
import json
import logging
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from core.contracts import TipoHITL, make_config
from graph.builder import agent_graph

logging.basicConfig(level=logging.INFO,
                    format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ESCALATIONS_DIR = Path("escalations")
GOLDEN_PATH = Path("tests/golden_sentiment.json")

MODULOS_CON_CFG = ("graph.nodes", "graph.faq.nodes",
                   "graph.intake.nodes", "graph.booking.nodes")

st.set_page_config(page_title="LeyIA · Manzzo y Cía", page_icon="⚖️",
                   layout="wide")


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("thread_id", f"web-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("chat_log", [])
    st.session_state.setdefault("pending_interrupt", None)
    st.session_state.setdefault("last_meta", {})


def new_thread() -> None:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.chat_log = []
    st.session_state.pending_interrupt = None
    st.session_state.last_meta = {}


def log(rol: str, content: str, meta: dict | None = None) -> None:
    st.session_state.chat_log.append({"rol": rol, "content": content,
                                      "meta": meta})


META_KEYS = ("sentiment", "urgency", "intent", "category", "case_category",
             "route", "clf_reason", "closed",
             "booking_stage", "intake_activo", "intake_exit",
             "intake_completado", "notificacion_pendiente")


def extract_meta(result: dict) -> dict:
    return {k: result.get(k) for k in META_KEYS
            if result.get(k) not in (None, False)}


def _config() -> dict:
    return make_config(st.session_state.thread_id)


# ---------------------------------------------------------------------------
# Helpers de checkpoint
# ---------------------------------------------------------------------------
def _interrupts_de_snap(snap) -> list:
    if not snap or not snap.tasks:
        return []
    return [i for task in snap.tasks for i in (task.interrupts or [])]


def _interrupts_pendientes() -> list:
    return _interrupts_de_snap(agent_graph.get_state(_config()))


def _estado_hilo() -> dict:
    snap = agent_graph.get_state(_config())
    return (snap.values or {}) if snap else {}


def _resumen_subflujo(valores: dict | None = None) -> str | None:
    st_ = valores if valores is not None else _estado_hilo()
    if st_.get("closed"):
        return "🗂️ Caso cerrado/derivado a la ejecutiva (seguimiento → handoff)"
    if st_.get("booking_stage"):
        if st_.get("slots_propuestos"):
            return "🕐 Eligiendo un horario disponible"
        return "🗓️ Capturando datos de agendamiento"
    if st_.get("intake_activo") and not st_.get("intake_completado"):
        if st_.get("intake_exit") == "pausa":
            return ("⏸️ Intake en pausa — el próximo mensaje del cliente "
                    "reanuda la ficha solo")
        return "📋 Ficha de intake en curso"
    return None


def cargar_hilo(thread_id: str) -> None:
    snap = agent_graph.get_state(make_config(thread_id))
    valores = (snap.values or {}) if snap else {}
    pendientes = _interrupts_de_snap(snap)

    st.session_state.thread_id = thread_id
    st.session_state.last_meta = {}
    st.session_state.chat_log = []

    if not valores:
        log("sistema",
            f"⚠️ `{thread_id}` no existe en el checkpointer; "
            "se creará como hilo nuevo al primer mensaje.")

    for m in valores.get("messages", []):
        if m.type == "human":
            log("cliente", m.content)
        elif m.type == "ai" and m.content:
            log("agente", m.content)

    st.session_state.pending_interrupt = pendientes[0].value if pendientes else None

    aviso = _resumen_subflujo(valores)
    if aviso:
        log("sistema", aviso)


# ---------------------------------------------------------------------------
# Ciclo del grafo
# ---------------------------------------------------------------------------
def process_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.pending_interrupt = result["__interrupt__"][0].value
        return

    st.session_state.pending_interrupt = None
    meta = extract_meta(result)
    st.session_state.last_meta = meta

    if result.get("response"):
        log("agente", result["response"], meta=meta)
    else:
        log("sistema", "⚠️ El grafo terminó sin `response`. "
                       "Revisa la terminal o el inspector de estado.")


def send_client_message(query: str) -> None:
    log("cliente", query)
    with st.spinner("🤖 pensando…"):
        try:
            result = agent_graph.invoke(
                {"messages": [("human", query)],
                 "thread_id": st.session_state.thread_id},
                config=_config(),
            )
            process_result(result)
        except Exception as e:
            logger.exception("Error invocando el grafo")
            log("sistema", f"⚠️ Error del agente: `{type(e).__name__}: {e}`")


def resolve_hitl(decision: dict) -> None:
    with st.spinner("Procesando decisión del operador…"):
        try:
            result = agent_graph.invoke(Command(resume=decision),
                                        config=_config())
            process_result(result)
        except Exception as e:
            logger.exception("Error resolviendo HITL")
            log("sistema", f"⚠️ Error tras decisión del operador: "
                           f"`{type(e).__name__}: {e}`. "
                           "El HITL sigue pendiente — puedes reintentarlo.")
            pendientes = _interrupts_pendientes()
            st.session_state.pending_interrupt = (
                pendientes[0].value if pendientes else None
            )


# ---------------------------------------------------------------------------
# Panel del OPERADOR
# ---------------------------------------------------------------------------
_CLAVES_HITL_CONOCIDAS = {"tipo", "query", "detalle", "lead",
                          "urgencia", "categoria"}


def render_operator_panel() -> None:
    payload = st.session_state.pending_interrupt
    tipo = str(payload.get("tipo", ""))

    st.warning("⏸️  **INTERVENCIÓN HUMANA REQUERIDA** (eres el operador)")

    st.markdown(f"**Tipo:** `{tipo}`")
    st.markdown(f"**Cliente:** {payload.get('query', '—')}")
    st.caption(payload.get("detalle", ""))

    lead = payload.get("lead") or {}
    if lead:
        st.markdown(
            f"**Lead:** {lead.get('nombre', '—')} · "
            f"{lead.get('email', '—')} · {lead.get('modalidad', '—')}")
    extras = {k: payload.get(k) for k in ("urgencia", "categoria")
              if payload.get(k) is not None}
    if extras:
        st.write(extras)

    desconocidas = {k: v for k, v in payload.items()
                    if k not in _CLAVES_HITL_CONOCIDAS}
    if desconocidas:
        with st.expander("Payload adicional del HITL"):
            st.json(json.loads(json.dumps(desconocidas, default=str)))

    if tipo == TipoHITL.APROBACION_AGENDAMIENTO.value:
        col1, col2 = st.columns(2)
        nota = st.text_input("Nota para el cliente (si rechazas):",
                             key="hitl_nota")
        if col1.button("✅ Aprobar y crear evento en Google Calendar",
                       type="primary", use_container_width=True):
            log("operador", "Aprobó la creación del evento en Calendar.")
            resolve_hitl({"aprobado": True})
            st.rerun()
        if col2.button("❌ Rechazar", use_container_width=True):
            log("operador",
                f"Rechazó el agendamiento. Nota: {nota or '(sin nota)'}")
            resolve_hitl({"aprobado": False, "nota": nota})
            st.rerun()
    else:
        nota = st.text_input("Nota:", key="hitl_nota_generica")
        if st.button("Rechazar HITL desconocido"):
            log("operador", f"Rechazó HITL desconocido ({tipo}).")
            resolve_hitl({"aprobado": False,
                          "nota": nota or "HITL no reconocido"})
            st.rerun()


# ---------------------------------------------------------------------------
# Render del chat
# ---------------------------------------------------------------------------
AVATARES = {"cliente": "🧑", "agente": "⚖️", "operador": "👷", "sistema": "🛠️"}


def render_chat() -> None:
    for msg in st.session_state.chat_log:
        role = "assistant" if msg["rol"] in ("agente", "sistema") else "user"
        with st.chat_message(role, avatar=AVATARES.get(msg["rol"])):
            if msg["rol"] == "sistema" and msg["content"].startswith("⚠️"):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(" · ".join(f"{k}: {v}"
                                      for k, v in msg["meta"].items()))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("🧵 Sesión")
        st.code(st.session_state.thread_id, language=None)

        st.button("🔄 Hilo nuevo", on_click=new_thread,
                  use_container_width=True)

        with st.expander("📂 Cargar hilo existente"):
            tid = st.text_input("thread_id", key="cargar_id",
                                placeholder="web-xxxxxxxx o cli-xxxxxxxx")
            if st.button("Cargar", use_container_width=True) and tid.strip():
                cargar_hilo(tid.strip())
                st.rerun()

        aviso = _resumen_subflujo()
        if aviso:
            st.info(aviso)

        if st.button("🔁 Recargar cfg", use_container_width=True,
                     help="Limpia caches de prompts.yaml y auth de Calendar"):
            limpiados = []
            for nombre_mod in MODULOS_CON_CFG:
                try:
                    mod = importlib.import_module(nombre_mod)
                except ImportError:
                    continue
                cfg_fn = getattr(mod, "_cfg", None)
                if hasattr(cfg_fn, "cache_clear"):
                    cfg_fn.cache_clear()
                    limpiados.append(nombre_mod)
            try:
                from tools.google_calendar import _api_resource
                _api_resource.cache_clear()
                limpiados.append("tools.google_calendar")
            except (ImportError, AttributeError):
                pass
            st.success(f"Caches limpiados: {', '.join(limpiados) or 'ninguno'}")

        st.divider()
        st.header("📊 Último turno")
        meta = st.session_state.last_meta
        if meta:
            for campo, valor in meta.items():
                st.markdown(f"- **{campo}**: `{valor}`")
        else:
            st.caption("Aún no hay turnos clasificados.")

        st.divider()
        st.header("🧪 Quick-test (golden)")
        if GOLDEN_PATH.exists():
            golden = json.loads(
                GOLDEN_PATH.read_text(encoding="utf-8"))["examples"]
            opciones = {f"#{ex['id']} · {ex['query'][:52]}…": ex["query"]
                        for ex in golden}
            sel = st.selectbox("Ejemplo anotado:", list(opciones.keys()))
            if st.button("▶️ Enviar como cliente", use_container_width=True):
                send_client_message(opciones[sel])
                st.rerun()
            with st.expander("Etiquetas esperadas"):
                ex = golden[list(opciones).index(sel)]
                st.json(ex["expected"])
        else:
            st.caption(f"No se encontró {GOLDEN_PATH}")

        st.divider()
        st.header("🗂️ Casos cerrados")
        if ESCALATIONS_DIR.exists():
            archivos = sorted(ESCALATIONS_DIR.glob("*.json"))
            if archivos:
                sel_f = st.selectbox("Escalamiento guardado:",
                                     [f.name for f in archivos])
                with st.expander("Ver expediente"):
                    st.json(json.loads(
                        (ESCALATIONS_DIR / sel_f).read_text(encoding="utf-8")))
            else:
                st.caption("Sin escalamientos aún.")

        st.divider()
        with st.expander("🔬 Estado del grafo (debug)"):
            try:
                valores = {k: v for k, v in _estado_hilo().items()
                           if k != "messages"}
                st.json(json.loads(json.dumps(valores, default=str)))
            except Exception as e:
                st.caption(f"Sin estado disponible: {e}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
init_state()

st.title("LeyIA — Manzzo y Cía")

render_sidebar()
render_chat()

if st.session_state.pending_interrupt is not None:
    render_operator_panel()
    st.chat_input("Esperando decisión del operador…", disabled=True)
else:
    if query := st.chat_input("Escribe como cliente…"):
        send_client_message(query)
        st.rerun()
