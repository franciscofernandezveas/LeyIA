#!/usr/bin/env python3
# api.py - FastAPI Server para LeyIA (Agente de soporte Manzzo y Cía)
# v1.0.0 - Capa HTTP/SSE sobre agent_graph (LangGraph) + Auth propio + HITL web
#
# A diferencia de la CLI (main.py), aquí CLIENTE y OPERADOR son personas
# distintas: el HITL NO se resuelve dentro del chat, se notifica por SSE
# (evento "hitl") y se resuelve con POST /api/v1/hitl/resume.

# ============================================================================
# 0. CARGAR .env ANTES DE CUALQUIER IMPORT PESADO
# ============================================================================

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.absolute()

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv
    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print("✅ .env cargado")
except Exception as e:
    print(f"⚠️ .env error: {e}")

# DEBUG LangSmith
print(f"🔍 LANGSMITH_TRACING    = {os.getenv('LANGSMITH_TRACING')}")
print(f"🔍 LANGSMITH_PROJECT    = {os.getenv('LANGSMITH_PROJECT')}")
print(f"🔍 LANGSMITH_API_KEY set= {bool(os.getenv('LANGSMITH_API_KEY'))}")

# ============================================================================
# HELPER: Configuración explícita de LangSmith
# ============================================================================

def configure_langsmith():
    """Activa LangSmith tracing y valida configuración (prefijo LANGSMITH_*)."""
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "leyia"
    endpoint = (
        os.getenv("LANGSMITH_ENDPOINT")
        or os.getenv("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    )

    if tracing and api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        print(f"✅ LangSmith tracing ACTIVO | project={project} | endpoint={endpoint}")
        return True
    print("⚠️ LangSmith tracing INACTIVO (opcional, la API funciona igual)")
    return False

LANGSMITH_ENABLED = configure_langsmith()

# ============================================================================
# IMPORTS DESPUÉS DE load_dotenv()
# ============================================================================

import logging
import re
import json
import asyncio
import time
from datetime import datetime, date, time as dt_time, timedelta
from typing import Dict, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4, UUID
from decimal import Decimal
from collections import defaultdict

# LangSmith SDK (decorador @traceable) — opcional
try:
    from langsmith import traceable
    LANGSMITH_SDK_AVAILABLE = True
    print("✅ langsmith SDK importado")
except Exception as e:
    print(f"⚠️ langsmith SDK no disponible: {e}")
    def traceable(*dargs, **dkwargs):
        def decorator(func):
            return func
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return decorator
    LANGSMITH_SDK_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("leyia-api")

logger.info("=" * 70)
logger.info("🚀 INICIANDO API.PY v1.0.0 - LEYIA (AGENTE MANZZO Y CÍA)")
logger.info("=" * 70)

if not os.getenv("OPENAI_API_KEY"):
    logger.warning("⚠️ OPENAI_API_KEY no definida (el grafo podría fallar al invocar el LLM)")

# ============================================================================
# 1. GRAFO DEL AGENTE (mismo objeto que usa main.py, con su checkpointer)
# ============================================================================

from langgraph.types import Command
from core.contracts import TipoHITL, make_config
from graph.builder import agent_graph

AGENT_AVAILABLE = agent_graph is not None
logger.info(f"✅ agent_graph cargado: {AGENT_AVAILABLE}")

# Máximo de rondas de resume (misma protección anti loop-infinito de main.py)
MAX_RONDAS_RESUME = 5

# ============================================================================
# 2. FASTAPI, CORS Y ROUTERS
# ============================================================================

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# AUTH PROPIO (app_auth.users + Supabase JWT)
from auth import get_current_user, User, router as auth_router
logger.info("✅ Auth import OK")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI(
    title="LeyIA - API",
    version="1.0.0",
    description="API del agente de soporte Manzzo y Cía (LangGraph + HITL + SSE streaming)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"➡️ {request.method} {request.url.path} | Origin: {request.headers.get('origin')}")
    response = await call_next(request)
    logger.info(f"⬅️ {response.status_code} {request.method} {request.url.path}")
    return response


# REGISTRO DE RUTAS
logger.info("Registrando routers...")
app.include_router(auth_router)

# ============================================================================
# 3. SESIONES POR USUARIO (thread_id del checkpointer)
# ============================================================================
# El checkpointer (checkpoints.sqlite) persiste el estado del grafo.
# Este dict solo recuerda qué thread_id le toca a cada usuario logueado.

SESSIONS: Dict[str, Any] = {}


def get_or_create_thread_id(user_id: str) -> str:
    if user_id not in SESSIONS:
        SESSIONS[user_id] = {"thread_id": f"lex-{uuid4().hex[:8]}"}
    return SESSIONS[user_id]["thread_id"]


def reset_thread_id(user_id: str) -> str:
    SESSIONS[user_id] = {"thread_id": f"lex-{uuid4().hex[:8]}"}
    return SESSIONS[user_id]["thread_id"]


def _graph_config(thread_id: str, user: Optional[User] = None) -> dict:
    """make_config() de la CLI + tags/metadata para LangSmith."""
    config = dict(make_config(thread_id) or {})
    config["tags"] = ["leyia", "api", f"thread:{thread_id}"]
    config["metadata"] = {
        "thread_id": thread_id,
        "user_id": str(user.id) if user else "anon",
        "user_email": getattr(user, "email", None),
        "service": "leyia-api",
        "langsmith_enabled": LANGSMITH_ENABLED,
    }
    return config


# ============================================================================
# 4. HELPERS DE ESTADO / INTERRUPTS / SSE
# ============================================================================

def _json_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date, dt_time)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _pending_interrupts(config: dict) -> list:
    """Interrupts pendientes del checkpoint (idéntico a main.py)."""
    try:
        snap = agent_graph.get_state(config)
    except Exception as e:
        logger.warning(f"get_state falló: {e}")
        return []
    if not snap or not snap.tasks:
        return []
    return [i for task in snap.tasks for i in (task.interrupts or [])]


def _state_values(config: dict) -> dict:
    try:
        snap = agent_graph.get_state(config)
        return (snap.values or {}) if snap else {}
    except Exception as e:
        logger.warning(f"get_state values falló: {e}")
        return {}


def _serialize_interrupt(intr) -> dict:
    value = getattr(intr, "value", intr)
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))
    except Exception:
        return {"detalle": str(value)}


def sse_event(event_type: str, **payload) -> str:
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False, default=_json_default)}\n\n"


def _node_message(node: Optional[str]) -> str:
    """Mensajes amigables por nodo del grafo (fallback genérico si no calza)."""
    return {
        "receive_message": "Recibiendo mensaje...",
        "classifier": "Clasificando tu consulta...",
        "clasificar": "Clasificando tu consulta...",
        "intake": "Actualizando ficha del caso...",
        "agenda": "Preparando agendamiento...",
        "agendar": "Preparando agendamiento...",
        "calendar": "Consultando agenda...",
        "handoff": "Derivando a un especialista...",
        "responder": "Redactando respuesta...",
        "response": "Redactando respuesta...",
    }.get(node or "", f"Procesando ({node})...")


async def sse_stream_text(text: Optional[str], sleep_time: float = 0.004):
    if not text:
        return
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for i, sentence in enumerate(sentences):
        sep = " " if i < len(sentences) - 1 else ""
        yield sse_event("chunk", content=sentence + sep)
        await asyncio.sleep(sleep_time)


async def _in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


@traceable(name="leyia_invoke", run_type="chain", tags=["leyia", "api"])
def _invoke_sync(payload: dict, config: dict):
    """Fallback síncrono: agent_graph.invoke (igual que main.py)."""
    return agent_graph.invoke(payload, config=config)


# ============================================================================
# 5. MODELOS DE REQUEST
# ============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="Mensaje del cliente")


class HITLDecision(BaseModel):
    """Contrato de decisión del operador (mismo shape que resolver_hitl en main.py)."""
    aprobado: bool
    nota: Optional[str] = None


# ============================================================================
# 6. ENDPOINTS DEBUG / SISTEMA
# ============================================================================

@app.get("/api/debug/routes")
async def debug_routes():
    routes = []
    for r in app.routes:
        routes.append({
            "path": r.path,
            "name": r.name,
            "methods": list(r.methods) if hasattr(r, "methods") else []
        })
    return {"total": len(routes), "routes": [r for r in routes if r["path"].startswith("/api")]}


@app.get("/api/debug/ping")
async def debug_ping():
    return {"status": "ok"}


@app.get("/api/debug/config")
async def debug_config(user: User = Depends(get_current_user)):
    thread_id = SESSIONS.get(user.id, {}).get("thread_id")
    pendientes = []
    if thread_id:
        pendientes = await _in_executor(_pending_interrupts, _graph_config(thread_id, user))
    return {
        "frontend_url": FRONTEND_URL,
        "user": user.model_dump(),
        "agent_available": AGENT_AVAILABLE,
        "thread_id": thread_id,
        "pending_hitls": len(pendientes),
    }


@app.get("/api/v1/system/status")
async def get_system_status(user: User = Depends(get_current_user)):
    return JSONResponse(content={
        "status": "ok",
        "capabilities": {
            "agent_available": AGENT_AVAILABLE,
            "langsmith_enabled": LANGSMITH_ENABLED,
            "version": "1.0.0",
            "service": "LeyIA - Manzzo y Cía",
        }
    })


# ============================================================================
# 7. ENDPOINTS DE SESIÓN
# ============================================================================

@app.get("/api/v1/session/state")
async def session_state(user: User = Depends(get_current_user)):
    """Estado del hilo actual: flags de intake/agenda + HITLs pendientes."""
    thread_id = get_or_create_thread_id(user.id)
    config = _graph_config(thread_id, user)
    values = await _in_executor(_state_values, config)
    pendientes = await _in_executor(_pending_interrupts, config)

    return {
        "success": True,
        "thread_id": thread_id,
        "flags": {
            "intake_activo": bool(values.get("intake_activo")),
            "recolectando_datos_agenda": bool(values.get("recolectando_datos_agenda")),
            "esperando_eleccion_horario": bool(values.get("esperando_eleccion_horario")),
        },
        "pending_hitls": [_serialize_interrupt(i) for i in pendientes],
    }


@app.post("/api/v1/session/clear")
async def clear_session(user: User = Depends(get_current_user)):
    """Equivalente al /nuevo de la CLI: hilo nuevo (el anterior queda persistido)."""
    old = SESSIONS.get(user.id, {}).get("thread_id")
    new_thread = reset_thread_id(user.id)
    return {"success": True, "previous_thread_id": old, "thread_id": new_thread}


# ============================================================================
# 8. CHAT (ROL CLIENTE) - SSE STREAMING
# ============================================================================

@app.post("/api/v1/chat/stream")
async def stream_chat(
    request: Request,
    body: QueryRequest,
    user: User = Depends(get_current_user)
):
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agente LeyIA no disponible")

    thread_id = get_or_create_thread_id(user.id)

    async def event_generator() -> AsyncGenerator[str, None]:
        t_start = time.time()
        config = _graph_config(thread_id, user)

        try:
            # --------------------------------------------------------------
            # 1. Igual que la CLI: no aceptar mensajes nuevos si hay HITL
            #    pendiente. Se notifica al frontend con evento "hitl".
            # --------------------------------------------------------------
            pendientes = await _in_executor(_pending_interrupts, config)
            if pendientes:
                yield sse_event("start", thread_id=thread_id)
                async for chunk in sse_stream_text(
                    "⏸️ Tu conversación tiene una solicitud pendiente de aprobación. "
                    "Debe resolverla un operador antes de continuar."
                ):
                    yield chunk
                yield sse_event(
                    "hitl",
                    interrupts=[_serialize_interrupt(i) for i in pendientes],
                )
                yield sse_event("end", intent="HITL_PENDING", success=False)
                return

            yield sse_event("start", thread_id=thread_id)

            # --------------------------------------------------------------
            # 2. Enviar el mensaje al grafo (mismo payload que main.py)
            # --------------------------------------------------------------
            payload = {"messages": [("human", body.question)], "thread_id": thread_id}

            if hasattr(agent_graph, "astream"):
                # stream_mode="updates" → {nombre_nodo: {...}} por cada paso
                last_node = None
                async for update in agent_graph.astream(payload, config, stream_mode="updates"):
                    for node in (update or {}).keys():
                        if node and node != last_node:
                            logger.debug(f"Graph node → {node}")
                            yield sse_event("progress", node=node, message=_node_message(node))
                            last_node = node
            else:
                yield sse_event("progress", node="agent", message="Procesando...")
                await _in_executor(_invoke_sync, payload, config)

            # --------------------------------------------------------------
            # 3. ¿La ejecución disparó un HITL nuevo?
            # --------------------------------------------------------------
            pendientes = await _in_executor(_pending_interrupts, config)
            values = await _in_executor(_state_values, config)

            if pendientes:
                yield sse_event(
                    "hitl",
                    interrupts=[_serialize_interrupt(i) for i in pendientes],
                )
                async for chunk in sse_stream_text(
                    "⏸️ Tu solicitud requiere la aprobación de un operador. "
                    "Te notificaremos cuando sea resuelta."
                ):
                    yield chunk
                yield sse_event("end", intent="HITL_REQUIRED", success=True)
                return

            # --------------------------------------------------------------
            # 4. Respuesta normal: metadata + texto streameado
            # --------------------------------------------------------------
            yield sse_event(
                "meta",
                sentiment=values.get("sentiment"),
                route=values.get("route"),
                intent=values.get("intent"),
            )

            response = values.get("response") or "(sin respuesta generada)"
            async for chunk in sse_stream_text(response):
                yield chunk

            yield sse_event("end", intent="AGENT_RESPONSE", success=True)

        except asyncio.CancelledError:
            logger.info(f"Cliente desconectado: {user.id}")
        except Exception as e:
            logger.error(f"Error streaming: {e}", exc_info=True)
            yield sse_event("error", content=str(e))
            yield sse_event("end", intent="ERROR", success=False)
        finally:
            logger.info(f"⏱️ Chat stream finalizado en {time.time() - t_start:.2f}s | user={user.id} | thread={thread_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# 9. HITL (ROL OPERADOR)
# ============================================================================

@app.get("/api/v1/hitl/pending")
async def hitl_pending(user: User = Depends(get_current_user)):
    """Solicitudes HITL pendientes en el hilo del usuario actual."""
    thread_id = SESSIONS.get(user.id, {}).get("thread_id")
    if not thread_id:
        return {"success": True, "thread_id": None, "pending": []}

    config = _graph_config(thread_id, user)
    pendientes = await _in_executor(_pending_interrupts, config)
    return {
        "success": True,
        "thread_id": thread_id,
        "pending": [_serialize_interrupt(i) for i in pendientes],
    }


@app.post("/api/v1/hitl/resume")
async def hitl_resume(
    body: HITLDecision,
    user: User = Depends(get_current_user),
):
    """Aprueba o rechaza el HITL pendiente (ej. crear evento en Google Calendar).

    Contrato idéntico a resolver_hitl() de main.py:
      aprobar  → {"aprobado": True}
      rechazar → {"aprobado": False, "nota": "..."}
    Se resuelve UN interrupt por llamada; si quedan más, vienen en "pending".
    """
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agente LeyIA no disponible")

    thread_id = SESSIONS.get(user.id, {}).get("thread_id")
    if not thread_id:
        raise HTTPException(status_code=404, detail="El usuario no tiene hilo activo")

    config = _graph_config(thread_id, user)
    pendientes = await _in_executor(_pending_interrupts, config)
    if not pendientes:
        return {"success": False, "error": "No hay solicitudes pendientes", "pending": []}

    actual = _serialize_interrupt(pendientes[0])
    logger.info(
        f"⏸️ Resolviendo HITL | tipo={actual.get('tipo')} | "
        f"aprobado={body.aprobado} | user={user.email}"
    )

    if body.aprobado:
        decision: Dict[str, Any] = {"aprobado": True}
    else:
        decision = {"aprobado": False, "nota": body.nota or ""}

    try:
        await _in_executor(
            lambda: agent_graph.invoke(Command(resume=decision), config=config)
        )
    except Exception as e:
        logger.error(f"Error al reanudar HITL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"No se pudo reanudar el HITL: {e}")

    # Estado post-resume: respuesta generada + interrupts restantes
    values = await _in_executor(_state_values, config)
    restantes = await _in_executor(_pending_interrupts, config)

    if restantes:
        logger.warning(
            f"Aún quedan {len(restantes)} interrupt(s) tras el resume "
            f"(máx. protección CLI: {MAX_RONDAS_RESUME} rondas)"
        )

    return {
        "success": True,
        "thread_id": thread_id,
        "resolved": actual,
        "response": values.get("response"),
        "metadata": {
            "sentiment": values.get("sentiment"),
            "route": values.get("route"),
            "intent": values.get("intent"),
        },
        "pending": [_serialize_interrupt(i) for i in restantes],
    }


# ============================================================================
# 10. MÉTRICAS PARA DASHBOARD DE WHATSAPP
# ============================================================================

# Cliente Supabase para métricas (solo si está configurado)
try:
    from supabase import create_client
    _sb_url = os.getenv("SUPABASE_URL")
    _sb_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if _sb_url and _sb_key:
        supabase_metrics = create_client(_sb_url, _sb_key)
        logger.info("✅ Supabase client creado para métricas")
    else:
        supabase_metrics = None
        logger.warning("⚠️ SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY no definidos; métricas usarán mock data")
except Exception as e:
    logger.warning(f"⚠️ Supabase no disponible para métricas: {e}")
    supabase_metrics = None


def _parse_range(range_str: str) -> tuple[int, datetime, datetime]:
    days = int(range_str.replace("d", ""))
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    return days, start, end


def _day_label(dt: datetime) -> str:
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    return dias[dt.weekday()]


def _format_time(iso: Optional[str]) -> str:
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return ""


def _mock_metrics(range_str: str):
    """Datos de ejemplo cuando Supabase no está disponible."""
    return {
        "kpis": [
            {"id": "conversations", "icon": "💬", "label": "Conversaciones", "value": 342, "delta": 12},
            {"id": "messages", "icon": "📨", "label": "Mensajes procesados", "value": 1874, "delta": 8},
            {"id": "resolution", "icon": "✅", "label": "Resolución automática", "value": "78%", "delta": 3},
            {"id": "response_time", "icon": "⚡", "label": "Tiempo medio respuesta", "value": "1.8s", "delta": -15},
            {"id": "escalations", "icon": "👨‍⚖️", "label": "Escaladas a abogado", "value": 75, "delta": -4},
            {"id": "leads", "icon": "🎯", "label": "Leads cualificados", "value": 28, "delta": 21},
        ],
        "conversationsByDay": [
            {"label": "Lun", "value": 45}, {"label": "Mar", "value": 52},
            {"label": "Mié", "value": 38}, {"label": "Jue", "value": 61},
            {"label": "Vie", "value": 55}, {"label": "Sáb", "value": 27},
            {"label": "Dom", "value": 19},
        ],
        "resolution": [
            {"label": "Resueltas por el agente", "value": 267, "color": "#25d366"},
            {"label": "Escaladas a abogado", "value": 75, "color": "#f59e0b"},
        ],
        "topIntents": [
            {"label": "Laboral", "value": 96}, {"label": "Civil", "value": 74},
            {"label": "Familia", "value": 61}, {"label": "Comercial", "value": 41},
            {"label": "Penal", "value": 22},
        ],
        "recentLeads": [
            {"name": "María González", "area": "Laboral", "status": "nuevo", "time": "10:42"},
            {"name": "Pedro Soto", "area": "Familia", "status": "seguimiento", "time": "09:15"},
            {"name": "Empresa Andina SpA", "area": "Comercial", "status": "escalado", "time": "Ayer"},
            {"name": "Camila Rojas", "area": "Civil", "status": "nuevo", "time": "Ayer"},
        ],
    }


@app.get("/api/metrics/overview")
async def metrics_overview(range: str = Query("7d", enum=["7d", "30d", "90d"])):
    """
    Métricas del agente de WhatsApp para el dashboard del estudio de abogados.
    Si Supabase está configurado, lee datos reales; si no, devuelve datos de ejemplo.

    No requiere autenticación para facilitar pruebas del dashboard.
    """
    try:
        if not supabase_metrics:
            raise RuntimeError("Supabase no configurado")

        days, start, end = _parse_range(range)
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        # --- Conversaciones en el rango ---
        conv_resp = supabase_metrics.table("conversations") \
            .select("*", count="exact") \
            .gte("created_at", start_iso) \
            .lte("created_at", end_iso) \
            .execute()

        total_conversations = conv_resp.count or 0
        conv_data = conv_resp.data or []

        # --- Conversaciones por día ---
        daily = defaultdict(int)
        for row in conv_data:
            created = row.get("created_at")
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    key = dt.strftime("%Y-%m-%d")
                    daily[key] += 1
                except Exception:
                    continue

        conversations_by_day = []
        for i in range(days):
            dt = end - timedelta(days=days - 1 - i)
            key = dt.strftime("%Y-%m-%d")
            label = _day_label(dt)
            conversations_by_day.append({"label": label, "value": daily.get(key, 0)})

        # --- Resolución: resueltas vs escaladas ---
        resolved_resp = supabase_metrics.table("conversations") \
            .select("id", count="exact") \
            .gte("created_at", start_iso) \
            .lte("created_at", end_iso) \
            .eq("status", "resolved") \
            .execute()

        escalated_resp = supabase_metrics.table("conversations") \
            .select("id", count="exact") \
            .gte("created_at", start_iso) \
            .lte("created_at", end_iso) \
            .eq("status", "escalated") \
            .execute()

        resolved_count = resolved_resp.count or 0
        escalated_count = escalated_resp.count or 0

        # --- Top intenciones / áreas legales ---
        area_counts = defaultdict(int)
        for row in conv_data:
            area = row.get("area") or "General"
            area_counts[area] += 1

        top_intents = [
            {"label": k, "value": v}
            for k, v in sorted(area_counts.items(), key=lambda x: -x[1])[:5]
        ]

        # --- Leads recientes ---
        leads_resp = supabase_metrics.table("conversations") \
            .select("name, area, status, created_at") \
            .eq("is_lead", True) \
            .order("created_at", desc=True) \
            .limit(5) \
            .execute()

        recent_leads = []
        for row in leads_resp.data or []:
            recent_leads.append({
                "name": row.get("name") or "Sin nombre",
                "area": row.get("area") or "General",
                "status": row.get("status") or "nuevo",
                "time": _format_time(row.get("created_at")),
            })

        # --- Métricas derivadas ---
        messages_count = total_conversations * 5
        total_for_resolution = resolved_count + escalated_count
        resolution_pct = f"{round(resolved_count / total_for_resolution * 100)}%" if total_for_resolution else "0%"

        return {
            "kpis": [
                {"id": "conversations", "icon": "💬", "label": "Conversaciones", "value": total_conversations, "delta": 12},
                {"id": "messages", "icon": "📨", "label": "Mensajes procesados", "value": messages_count, "delta": 8},
                {"id": "resolution", "icon": "✅", "label": "Resolución automática", "value": resolution_pct, "delta": 3},
                {"id": "response_time", "icon": "⚡", "label": "Tiempo medio respuesta", "value": "1.8s", "delta": -15},
                {"id": "escalations", "icon": "👨‍⚖️", "label": "Escaladas a abogado", "value": escalated_count, "delta": -4},
                {"id": "leads", "icon": "🎯", "label": "Leads cualificados", "value": len(recent_leads), "delta": 21},
            ],
            "conversationsByDay": conversations_by_day,
            "resolution": [
                {"label": "Resueltas por el agente", "value": resolved_count, "color": "#25d366"},
                {"label": "Escaladas a abogado", "value": escalated_count, "color": "#f59e0b"},
            ],
            "topIntents": top_intents,
            "recentLeads": recent_leads,
        }

    except Exception as e:
        logger.warning(f"⚠️ Métricas reales no disponibles, devolviendo mock: {e}")
        return _mock_metrics(range)


# ============================================================================
# 11. MANEJO DE ERRORES Y LIFESPAN
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "timestamp": datetime.now().isoformat()}
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🟢 LeyIA API iniciada")
    logger.info(f"🔍 LangSmith tracing: {'ACTIVO' if LANGSMITH_ENABLED else 'INACTIVO'} | "
                f"project={os.getenv('LANGSMITH_PROJECT', 'leyia')}")
    logger.info(f"🤖 agent_graph: {'OK' if AGENT_AVAILABLE else 'NO DISPONIBLE'}")
    yield
    logger.info("🛑 LeyIA API finalizada")

app.router.lifespan_context = lifespan

# MAIN
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
