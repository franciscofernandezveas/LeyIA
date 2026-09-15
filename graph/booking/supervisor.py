"""graph/booking/supervisor.py — Routing interno del sub-flujo.

v3.1 — Guard defensivo: si en propuesta faltan datos de contacto, se
  reinicia el wizard de captura en lugar de confirmar con datos vacíos.
"""
import logging

from core.contracts import AgentState

from .contracts import AGENDA_VENTANA_MAX_DIAS, BOOKING_MAX_ATTEMPTS

logger = logging.getLogger(__name__)


def _datos_completos(state: AgentState) -> bool:
    return bool(state.get("lead_nombre") and state.get("lead_email")
                and state.get("lead_modalidad"))


def _siguiente_captura(state: AgentState) -> str:
    """Devuelve la etapa de captura que falta."""
    if not state.get("lead_nombre"):
        return "pedir_nombre"
    if not state.get("lead_email"):
        return "pedir_email"
    if not state.get("lead_modalidad"):
        return "pedir_modalidad"
    return "procesar_captura"


def route_booking(state: AgentState) -> str:
    d = state.get("booking_decision") or {}
    accion = d.get("accion")
    stage = state.get("booking_stage")

    # 1) Anti-loop
    if (state.get("booking_attempts") or 0) >= BOOKING_MAX_ATTEMPTS:
        logger.warning("[booking] max reintentos → ofrecer ejecutiva")
        return "ofrecer_ejecutiva"

    # 2) Abort
    if accion == "abortar":
        return "cerrar_booking"

    # 3) Duda lateral
    if accion == "consultar":
        return "responder_lateral"

    # 4) Wizard de captura: la etapa activa manda
    if stage in ("captura_nombre", "captura_email", "captura_modalidad"):
        return "procesar_captura"

    # 5) Si estamos en propuesta pero faltan datos (estado residual),
    #    reiniciar captura antes de cualquier otra cosa.
    if stage == "propuesta" and not _datos_completos(state):
        logger.warning("[booking] propuesta sin datos completos → reiniciar captura")
        return _siguiente_captura(state)

    # 6) Ventana agotada pidiendo más fechas
    if accion == "pedir_otra_fecha" and \
            (state.get("agenda_ventana_desde") or 0) >= AGENDA_VENTANA_MAX_DIAS:
        return "ofrecer_ejecutiva"

    # 7) Nueva disponibilidad
    if accion in ("entregar_datos", "pedir_otra_fecha", "filtrar_franja"):
        return "consultar_disponibilidad"

    # 8) Elección sobre propuesta vigente
    if accion == "elegir_horario":
        if not state.get("slots_propuestos"):
            return "consultar_disponibilidad"
        if not _datos_completos(state):
            return _siguiente_captura(state)
        if state.get("booking_match") is not None:
            return "confirmar_y_crear"
        if state.get("booking_match_candidatos"):
            return "aclarar_eleccion"
        return "reintentar_eleccion"

    # 9) Fallback
    return "consultar_disponibilidad"


def despues_de_consultar(state: AgentState) -> str:
    señal = state.get("booking_signal")
    if señal == "ventana_agotada":
        return "ofrecer_ejecutiva"
    if señal == "tanda_repetida":
        return "responder_lateral"
    return "proponer_slots"


def despues_de_confirmar(state: AgentState) -> str:
    return ("consultar_disponibilidad"
            if state.get("booking_signal") == "slot_stale"
            else END)


from langgraph.graph import END  # noqa: E402
