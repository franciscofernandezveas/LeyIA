# app.py
"""Dashboard Streamlit para probar el agente Manzzo y Cía (LeyIA).

Roles:
  🧑 CLIENTE  → chat inferior.
  👷 OPERADOR → panel que aparece cuando el grafo interrumpe (HITL).

v2 — Fix "error silencioso":
  · Los errores del grafo van al chat_log (session_state) → sobreviven al rerun.
  · Todo camino sin `response` queda registrado en el chat.
  · logging activado: verás los logger.info/warning de nodes/calendly en terminal.
  · Botón de recarga de prompts.yaml + caches de Calendly.
  · Inspector del estado del grafo (debug) en el sidebar.

Uso:
    streamlit run app.py
"""
import json
import logging
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from core.contracts import make_config
from graph.builder import agent_graph

logging.basicConfig(level=logging.INFO,
                    format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ESCALATIONS_DIR = Path("escalations")
GOLDEN_PATH = Path("tests/golden_sentiment.json")

st.set_page_config(page_title="LeyIA · Manzzo y Cía", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("thread_id", f"web-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("chat_log", [])          # [{rol, content, meta}]
    st.session_state.setdefault("pending_interrupt", None)
    st.session_state.setdefault("last_meta", {})


def new_thread() -> None:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:8]}"
    st.session_state.chat_log = []
    st.session_state.pending_interrupt = None
    st.session_state.last_meta = {}


def log(rol: str, content: str, meta: dict | None = None) -> None:
    st.session_state.chat_log.append({"rol": rol, "content": content, "meta": meta})


def extract_meta(result: dict) -> dict:
    """Trazabilidad del turno: clasificación + flags de sub-flujos activos."""
    keys = ("sentiment", "urgency", "intent", "category", "route",
            "escalated", "closed", "recolectando_datos_agenda",
            "esperando_confirmacion_booking", "esperando_slot")
    return {k: result.get(k) for k in keys if result.get(k) not in (None, False)}


def _config() -> dict:
    return make_config(st.session_state.thread_id)


# ---------------------------------------------------------------------------
# Ciclo del grafo — NADA se pierde: todo va al chat_log
# ---------------------------------------------------------------------------
def process_result(result: dict) -> None:
    """Qué hacer tras un invoke: ¿hay interrupción, respuesta, o nada?"""
    if "__interrupt__" in result:
        st.session_state.pending_interrupt = result["__interrupt__"][0].value
        return  # el panel de operador se renderiza abajo

    st.session_state.pending_interrupt = None
    meta = extract_meta(result)
    st.session_state.last_meta = meta

    if result.get("response"):
        log("agente", result["response"], meta=meta)
    else:
        # Camino que termina sin respuesta → visible, no silencioso
        log("sistema", "⚠️ El grafo terminó sin `response`. "
                       "Revisa la terminal (logging INFO activo) o el inspector de estado.")


def send_client_message(query: str) -> None:
    log("cliente", query)
    with st.spinner("🤖 pensando…"):
        try:
            result = agent_graph.invoke(
                {"messages": [("human", query)],
                 "query": query,
                 "thread_id": st.session_state.thread_id},
                config=_config(),
            )
            process_result(result)
        except Exception as e:
            # Antes: st.error() aquí → el rerun lo borraba = error invisible.
            # Ahora: va al chat_log (persiste) + traceback completo en terminal.
            logger.exception("Error invocando el grafo")
            log("sistema", f"⚠️ Error del agente: `{type(e).__name__}: {e}`")


def resolve_hitl(decision: dict) -> None:
    try:
        result = agent_graph.invoke(Command(resume=decision), config=_config())
        process_result(result)  # puede encadenar otra interrupción
    except Exception as e:
        logger.exception("Error resolviendo HITL")
        log("sistema", f"⚠️ Error tras decisión del operador: `{type(e).__name__}: {e}`")
        st.session_state.pending_interrupt = None  # no quedar atascado en el panel


# ---------------------------------------------------------------------------
# Panel del OPERADOR (aparece solo cuando hay __interrupt__ pendiente)
# ---------------------------------------------------------------------------
def render_operator_panel() -> None:
    payload = st.session_state.pending_interrupt
    tipo = str(payload.get("tipo", ""))

    with st.container(border=True):
        st.warning("⏸️  **INTERVENCIÓN HUMANA REQUERIDA** (eres el operador)")
        st.markdown(f"**Tipo:** `{tipo}`")
        st.markdown(f"**Cliente:** {payload.get('query', '—')}")
        st.caption(payload.get("detalle", ""))

        extras = {k: payload.get(k) for k in
                  ("urgencia", "categoria", "sentimiento", "prioritario")
                  if payload.get(k) is not None}
        if extras:
            st.write(extras)

        if "agendamiento" in tipo:
            col1, col2 = st.columns(2)
            nota = st.text_input("Nota para el cliente (si rechazas):",
                                 key="hitl_nota")
            if col1.button("✅ Aprobar y enviar link", type="primary",
                           use_container_width=True):
                log("operador", "Aprobó el envío del link de agendamiento.")
                resolve_hitl({"aprobado": True})
                st.rerun()
            if col2.button("❌ Rechazar", use_container_width=True):
                log("operador", f"Rechazó el agendamiento. Nota: {nota or '(sin nota)'}")
                resolve_hitl({"aprobado": False, "nota": nota})
                st.rerun()
        else:  # escalamiento
            modo = st.radio("Respuesta al cliente:",
                            ["📨 Mensaje por defecto", "✉️ Personalizado"],
                            horizontal=True, key="hitl_modo")
            custom = ""
            if "Personalizado" in modo:
                custom = st.text_area("Mensaje del operador:", key="hitl_msg")
            if st.button("Enviar al cliente", type="primary"):
                txt = custom.strip() or "(mensaje por defecto)"
                log("operador", f"Escalamiento resuelto: {txt}")
                resolve_hitl({"aprobado": True, "mensaje": custom.strip() or None})
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
                st.error(msg["content"])      # errores en rojo, DENTRO del chat
            else:
                st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(" · ".join(f"{k}: {v}" for k, v in msg["meta"].items()))


# ---------------------------------------------------------------------------
# Sidebar: sesión, clasificación, debug y QA
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("🧵 Sesión")
        st.code(st.session_state.thread_id, language=None)
        c1, c2 = st.columns(2)
        c1.button("🔄 Hilo nuevo", on_click=new_thread, use_container_width=True)
        if c2.button("🔁 Recargar cfg", use_container_width=True,
                     help="Limpia caches de prompts.yaml y Calendly"):
            from graph.nodes import _cfg
            _cfg.cache_clear()
            try:
                from tools.calendly import _org_uri, obtener_event_type_info
                _org_uri.cache_clear()
                obtener_event_type_info.cache_clear()
            except ImportError:
                pass
            st.success("Caches limpiados")

        st.divider()
        st.header("📊 Último turno")
        meta = st.session_state.last_meta
        if meta:
            for campo, valor in meta.items():
                st.markdown(f"- **{campo}**: `{valor}`")
        else:
            st.caption("Aún no hay turnos clasificados.")

        # Quick-tests desde el golden set (QA del clasificador)
        st.divider()
        st.header("🧪 Quick-test (golden)")
        if GOLDEN_PATH.exists():
            golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["examples"]
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

        # Visor de escalamientos cerrados
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

        # Inspector del estado real del grafo (checkpoint) — clave para debug
        st.divider()
        with st.expander("🔬 Estado del grafo (debug)"):
            try:
                snap = agent_graph.get_state(_config())
                valores = {k: v for k, v in snap.values.items() if k != "messages"}
                st.json(json.loads(json.dumps(valores, default=str)))
            except Exception as e:
                st.caption(f"Sin estado disponible: {e}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
init_state()

st.title("⚖️ LeyIA — Agente Manzzo y Cía")


render_sidebar()
render_chat()

if st.session_state.pending_interrupt is not None:
    render_operator_panel()
    st.chat_input("Esperando decisión del operador…", disabled=True)
else:
    if query := st.chat_input("Escribe como cliente…"):
        send_client_message(query)
        st.rerun()
