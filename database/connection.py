import os
import re

import psycopg
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv

load_dotenv()

# Prioridad: DEMO_DATABASE_URL (usado por el MVP) → DATABASE_URL
DATABASE_URL = os.getenv("DEMO_DATABASE_URL") or os.getenv("DATABASE_URL", "")

# Si no hay URL completa, armarla desde variables individuales
if not DATABASE_URL or "tu-database-url" in DATABASE_URL.lower():
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "")
    DB_NAME = os.getenv("DB_NAME", "")
    DB_PORT = os.getenv("DB_PORT", "6543")

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise Exception("❌ Faltan variables de entorno de base de datos (DATABASE_URL, DEMO_DATABASE_URL o DB_*)")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Normalizar prefijo
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Eliminar pgbouncer=true si viene (no compatible con psycopg3 directo)
if "?pgbouncer=true" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("?pgbouncer=true", "")
if "&pgbouncer=true" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("&pgbouncer=true", "")
if DATABASE_URL.endswith("?"):
    DATABASE_URL = DATABASE_URL[:-1]

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=20,
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
