"""core/persistence.py — Checkpointer: el estado de cada thread_id sobrevive reinicios.

Desarrollo: SQLite (SqliteSaver), activo por defecto.
Producción: PostgresSaver vía build_postgres_checkpointer(), que ya trae
prepare_threshold=None — psycopg3 + pooler en modo transacción colisiona
(DuplicatePreparedStatement), mismo criterio aplicado en core.db_client.

Para activar Postgres basta definir la variable de entorno
CHECKPOINT_DATABASE_URL; sin tocar código ni graph/builder.py.
"""
import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import CHECKPOINT_DB


def build_postgres_checkpointer(database_url: str):
    """Producción: PostgresSaver inmune al pooler en modo transacción."""
    from psycopg_pool import ConnectionPool
    from langgraph.checkpoint.postgres import PostgresSaver

    # Supabase añade ?pgbouncer=true en la URL del pooler; psycopg no lo entiende.
    cleaned = database_url.replace("?pgbouncer=true", "").replace("&pgbouncer=true", "")
    if cleaned.endswith("?"):
        cleaned = cleaned[:-1]

    pool = ConnectionPool(
        cleaned,
        kwargs={"autocommit": True, "prepare_threshold": None},
        max_size=5,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver


_checkpoint_url = os.getenv("CHECKPOINT_DATABASE_URL")

if _checkpoint_url:
    # Producción
    checkpointer = build_postgres_checkpointer(_checkpoint_url)
else:
    # Desarrollo local
    _conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    checkpointer = SqliteSaver(_conn)
