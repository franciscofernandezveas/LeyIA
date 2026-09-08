# app.py
"""Dashboard Streamlit para el agente Manzzo y Cía (LeyIA).

Versión: v6.2 — Alineada con main.py v6.1 (CLI). Mismo contrato con agent_graph.

Cambios v6.1 → v6.2:
  a) load_dotenv() al inicio, ANTES de importar core.*/graph.* — sin esto,
     db_client.py lee DATABASE_URL=None al importarse y todos los upserts
     fallan en silencio (los '[db] ... falló' solo salen en la terminal).
     En Streamlit Cloud es un no-op inofensivo (los Secrets ya son env vars).
  b) resolve_hitl(): si el resume falla, el panel ya NO se limpia a ciegas.
     El interrupt sigue vivo en el checkpoint → borrarlo solo de la UI
     habilitaba el chat sobre un hilo interrumpido → LangGraph lanzaba error
     al recibir input normal. Ahora se re-sincroniza desde el checkpoint
     (fuente de verdad): el panel persiste y el operador puede reintentar.
  c) cargar_hilo(): valida que el hilo exista en el checkpointer (fix /cargar
     de la CLI aquí no portado) y hace UN solo get_state en vez de tres.

Alineación con la CLI (main.py v6.1):
  · Misma forma de invoke: {"messages": [("human", q)], "thread_id": ...}
    sin "query" — receive_message la deriva del último mensaje humano.
  · Mismo resume HITL: Command(resume={"aprobado": ..., "nota": ...}).
  · thread_id estable en session_state (prefijo "web-" para distinguir en BD).
  · Historial reconstruido desde el checkpointer al cargar un hilo.
  · Chat deshabilitado mientras haya HITL pendiente (≡ drenar-antes-de-enviar
    de la CLI: mismo efecto, UI bloqueada en vez de drenado automático).

Roles:
  🧑 CLIENTE  → chat inferior.
  👷 OPERADOR → panel que aparece cuando el grafo interrumpe (HITL).

Uso:
    streamlit run app.py
"""
# ⚠️ ORDEN CRÍTICO: load_dotenv() antes de importar core.*/graph.*,
# porque db_client.py captura DATABASE_URL en tiempo de importación.
from dotenv import load_dotenv
load_dotenv()

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

st.set_page_config(page_title="LeyIA · Manzzo y Cía", page_icon="⚖️",
                   layout="wide")


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("thread_id", f"web-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("chat_log", [])         # [{rol, content, meta}]
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


def extract_meta(result: dict) -> dict:
    """Trazabilidad del turno: clasificación + flags de sub-flujos (estado v8)."""
    keys = ("sentiment", "urgency", "intent", "category", "route",
            "closed", "recolectando_datos_agenda",
            "esperando_eleccion_horario", "intake_activo")
    return {k: result.get(k) for k in keys if result.get(k) not in (None, False)}


def _config() -> dict:
    return make_config(st.session_state.thread_id)


# ---------------------------------------------------------------------------
# Helpers de checkpoint (≡ _pending_interrupts/_estado_hilo de main.py)
# ---------------------------------------------------------------------------
def _interrupts_de_snap(snap) -> list:
    """Extrae los interrupts de un snapshot ya obtenido (sin get_state extra)."""
    if not snap or not snap.tasks:
        return []
    return [i for task in snap.tasks for i in (task.interrupts or [])]


def _interrupts_pendientes() -> list:
    """Interrupts pendientes en el checkpoint del hilo actual."""
    return _interrupts_de_snap(agent_graph.get_state(_config()))


def _estado_hilo() -> dict:
    snap = agent_graph.get_state(_config())
    return (snap.values or {}) if snap else {}


def _resumen_subflujo(valores: dict | None = None) -> str | None:
    """≡ los ℹ️ de la CLI al hacer /cargar.

    Si se pasan `valores` (snapshot ya consultado), no hace get_state extra;
    si no, consulta el checkpoint del hilo actual (uso desde el sidebar).
    """
    st_ = valores if valores is not None else _estado_hilo()
    if st_.get("closed"):
        return "🗂️ Caso cerrado/derivado a la ejecutiva"
    if st_.get("intake_activo"):
        return "📋 Ficha de intake en curso"
    if st_.get("recolectando_datos_agenda"):
        return "🗓️ Capturando datos de agendamiento"
    if st_.get("esperando_eleccion_horario"):
        return "🕐 Eligiendo un horario disponible"
    return None


def cargar_hilo(thread_id: str) -> None:
    """≡ /cargar de la CLI v6.1: valida existencia, cambia de hilo,
    reconstruye el chat desde el checkpoint y levanta el panel del
    operador si quedó un HITL pendiente. UN solo get_state."""
    snap = agent_graph.get_state(make_config(thread_id))
    valores = (snap.values or {}) if snap else {}
    pendientes = _interrupts_de_snap(snap)

    st.session_state.thread_id = thread_id
    st.session_state.last_meta = {}
    st.session_state.chat_log = []

    # Validación (≡ fix /cargar de la CLI): typo o id inexistente → avisar
    # en el chat en vez de conversar a ciegas bajo un hilo basura.
    if not valores:
        log("sistema",
            f"⚠️ `{thread_id}` no existe en el checkpointer; "
            "se creará como hilo nuevo al primer mensaje.")

    # Reconstruir conversación visible desde los mensajes persistidos
    for m in valores.get("messages", []):
        if m.type == "human":
            log("cliente", m.content)
        elif m.type == "ai" and m.content:
            log("agente", m.content)

    # HITL pendiente de una sesión anterior → operador decide ahora
    st.session_state.pending_interrupt = pendientes[0].value if pendientes else None

    aviso = _resumen_subflujo(valores)
    if aviso:
        log("sistema", aviso)


# ---------------------------------------------------------------------------
# Ciclo del grafo — NADA se pierde: todo va al chat_log
# ---------------------------------------------------------------------------
def process_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.pending_interrupt = result["__interrupt__"][0].value
        return                                       # el panel se renderiza abajo

    st.session_state.pending_interrupt = None
    meta = extract_meta(result)
    st.session_state.last_meta = meta

    if result.get("response"):
        log("agente", result["response"], meta=meta)
    else:
        log("sistema", "⚠️ El grafo terminó sin `response`. "
                       "Revisa la terminal o el inspector de estado.")


def send_client_message(query: str) -> None:
    """≡ paso 2 de la CLI: solo messages + thread_id. La derivación de query
    la hace receive_message dentro del grafo."""
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
            # Antes: st.error() aquí → el rerun lo borraba = error invisible.
            # Ahora: va al chat_log (persiste) + traceback en terminal.
            logger.exception("Error invocando el grafo")
            log("sistema", f"⚠️ Error del agente: `{type(e).__name__}: {e}`")


def resolve_hitl(decision: dict) -> None:
    """Resume el HITL pendiente. Si el resume falla (ej. Calendar API caída),
    el checkpoint sigue interrumpido → re-sincronizamos desde él (fuente de
    verdad) en vez de limpiar la UI a ciegas. ≡ CLI v6.1: hilo en pausa."""
    with st.spinner("Procesando decisión del operador…"):
        try:
            result = agent_graph.invoke(Command(resume=decision),
                                        config=_config())
            process_result(result)      # puede encadenar otra interrupción
        except Exception as e:
            # Antes: pending_interrupt = None → chat habilitado sobre un hilo
            # AÚN interrumpido → el siguiente mensaje del cliente hacía que
            # LangGraph lanzara error por input normal en vez de Command(resume).
            logger.exception("Error resolviendo HITL")
            log("sistema", f"⚠️ Error tras decisión del operador: "
                           f"`{type(e).__name__}: {e}`. "
                           "El HITL sigue pendiente — puedes reintentarlo.")
            pendientes = _interrupts_pendientes()
            st.session_state.pending_interrupt = (
                pendientes[0].value if pendientes else None
            )


# ---------------------------------------------------------------------------
# Panel del OPERADOR (solo cuando hay __interrupt__ pendiente)
# ---------------------------------------------------------------------------
def render_operator_panel() -> None:
    payload = st.session_state.pending_interrupt
    tipo = str(payload.get("tipo", ""))

    st.warning("⏸️  **INTERVENCIÓN HUMANA REQUERIDA** (eres el operador)")

    st.markdown(f"**Tipo:** `{tipo}`")
    st.markdown(f"**Cliente:** {payload.get('query', '—')}")
    st.caption(payload.get("detalle", ""))

    # Lead + clasificación (≡ el print de la CLI)
    lead = payload.get("lead") or {}
    if lead:
        st.markdown(
            f"**Lead:** {lead.get('nombre', '—')} · "
            f"{lead.get('email', '—')} · {lead.get('modalidad', '—')}")
    extras = {k: payload.get(k) for k in ("urgencia", "categoria")
              if payload.get(k) is not None}
    if extras:
        st.write(extras)

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
        # Fallback defensivo (≡ CLI): HITL de tipo desconocido → rechazo
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
                st.error(msg["content"])     # errores en rojo, DENTRO del chat
            else:
                st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(" · ".join(f"{k}: {v}"
                                      for k, v in msg["meta"].items()))


# ---------------------------------------------------------------------------
# Sidebar: sesión, clasificación, debug y QA
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("🧵 Sesión")
        st.code(st.session_state.thread_id, language=None)

        st.button("🔄 Hilo nuevo", on_click=new_thread,
                  use_container_width=True)

        # ≡ /cargar <id> de la CLI
        with st.expander("📂 Cargar hilo existente"):
            tid = st.text_input("thread_id", key="cargar_id",
                                placeholder="web-xxxxxxxx o cli-xxxxxxxx")
            if st.button("Cargar", use_container_width=True) and tid.strip():
                cargar_hilo(tid.strip())
                st.rerun()

        # Estado de sub-flujo del hilo actual (≡ los ℹ️ de la CLI)
        aviso = _resumen_subflujo()
        if aviso:
            st.info(aviso)

        if st.button("🔁 Recargar cfg", use_container_width=True,
                     help="Limpia caches de prompts.yaml y auth de Calendar"):
            from graph.nodes import _cfg
            _cfg.cache_clear()
            from tools.google_calendar import _api_resource
            _api_resource.cache_clear()      # re-auth en la próxima llamada
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
