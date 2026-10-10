"""database/connection.py — Pool psycopg3 para componentes que necesitan
conexión cruda a PostgreSQL/Supabase (RAG, utilidades, etc.).

IMPORTANTE: este pool apunta al pooler transaccional de Supabase (puerto 6543).
psycopg3 prepara statements server-side por defecto (nombres _pg3_N); con el
pooler transaccional los backends se comparten entre clientes, por lo que
desactivamos prepare_threshold=None en cada conexión del pool.

Si un componente necesita sesión persistente (checkpointer de LangGraph,
LISTEN/NOTIFY, advisory locks o temp tables), NO debe usar este pool:
debe crear una conexión dedicada vía psycopg.connect(..., autocommit=True,
prepare_threshold=None).
"""
import os
import threading

import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# 1) Construcción de DATABASE_URL (única fuente de verdad: entorno)
# --------------------------------------------------------------------------
def _build_database_url() -> str:
    """Prioridad: DEMO_DATABASE_URL → DATABASE_URL → variables DB_*.
    Rechaza defaults de secretos: si falta, falla explícito.
    """
    url = os.getenv("DEMO_DATABASE_URL") or os.getenv("DATABASE_URL", "")

    if not url or "tu-database-url" in url.lower():
        DB_USER = os.getenv("DB_USER", "")
        DB_PASSWORD = os.getenv("DB_PASSWORD", "")
        DB_HOST = os.getenv("DB_HOST", "")
        DB_NAME = os.getenv("DB_NAME", "")
        DB_PORT = os.getenv("DB_PORT", "6543")

        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            raise RuntimeError(
                "❌ Faltan variables de entorno de base de datos. "
                "Se requiere DATABASE_URL (o DEMO_DATABASE_URL) o bien "
                "DB_USER, DB_PASSWORD, DB_HOST y DB_NAME."
            )

        url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Eliminar pgbouncer=true (flag del pooler, no parte de la URI de psycopg3)
    for flag in ["?pgbouncer=true", "&pgbouncer=true"]:
        url = url.replace(flag, "")
    if url.endswith("?"):
        url = url[:-1]

    return url


DATABASE_URL = _build_database_url()


# --------------------------------------------------------------------------
# 2) Pool singleton, thread-safe y compatible con el pooler transaccional
# --------------------------------------------------------------------------
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=DATABASE_URL,
                    min_size=1,
                    max_size=5,
                    kwargs={
                        "prepare_threshold": None,  # ← fix DuplicatePreparedStatement
                        "autocommit": False,          # transacciones manuales por default
                    },
                    timeout=30,
                    open=True,
                )
    return _pool


def get_connection() -> psycopg.Connection:
    return _get_pool().getconn()


def release_connection(conn: psycopg.Connection) -> None:
    _get_pool().putconn(conn)


def close_all_connections() -> None:
    global _pool
    if _pool:
        _pool.close()
        _pool = None


# --------------------------------------------------------------------------
# 3) Helper transaccional (opcional pero recomendado)
# --------------------------------------------------------------------------
def with_connection(func):
    """Decorador: obtiene conexión del pool, la libera, loggea errores."""
    def wrapper(*args, **kwargs):
        conn = get_connection()
        try:
            return func(conn, *args, **kwargs)
        except Exception as e:
            raise e
        finally:
            release_connection(conn)
    return wrapper
