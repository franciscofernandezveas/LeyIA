"""Checkpointer SQLite: el estado de cada thread_id sobrevive reinicios."""
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import CHECKPOINT_DB

_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
checkpointer = SqliteSaver(_conn)

# Producción: cambiar a PostgresSaver (langgraph-checkpoint-postgres)
