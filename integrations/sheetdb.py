"""integrations/sheetdb.py — Stub temporal: inserciones a Google Sheets desactivadas.

La persistencia oficial de leads ahora es Supabase.  Este módulo mantiene las
funciones públicas anteriores como no-ops para no romper los importadores
existentes (graph/nodes.py, tools/booking, etc.) hasta que se migren las llamadas
directamente al cliente de Supabase.

CUANDO MIGRES:
- Reemplaza las llamadas a crear_lead / actualizar_lead_por_thread_id /
  sync_lead / actualizar_booking por inserciones/updates en Supabase.
- Una vez ningún otro archivo importe desde integrations.sheetdb, elimina este
  archivo por completo.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _log_noop(fn_name: str, thread_id: str | None = None) -> None:
    logger.debug(
        "[sheetdb:%s] no-op: persistencia en Google Sheets desactivada (%s)",
        fn_name,
        thread_id or "<sin thread_id>",
    )


def _noop_state(state: dict | None = None, canal: str | None = None, **kwargs: Any) -> dict:
    thread_id = (state or {}).get("thread_id")
    _log_noop("crear/sync", thread_id)
    return {"ok": True, "action": "noop", "storage": "supabase_pending"}


def _noop_thread(
    thread_id: str | None = None,
    data: dict | None = None,
    **kwargs: Any,
) -> dict:
    _log_noop("actualizar", thread_id)
    return {"ok": True, "action": "noop", "storage": "supabase_pending"}


# API pública anterior.  Ahora solo logea y devuelve éxito.
crear_lead = _noop_state
actualizar_lead_por_thread_id = _noop_thread
sync_lead = _noop_state
actualizar_booking = _noop_state
