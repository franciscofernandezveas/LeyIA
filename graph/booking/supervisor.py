"""graph/booking/supervisor.py — Routing interno del sub-flujo.

Ídem supervisor_node: lectura defensiva, guardias anti-loop primero,
prioridades después. Diferencia con el BI: aquí TODA ruta termina en un
nodo que responde al cliente (no hay ciclos intra-turno).

v2 — despues_de_consultar enruta "tanda_repetida" → responder_lateral:
consultar_disponibilidad detectó que la tanda nueva es IDÉNTICA a la
vigente (el LLM clasificó una duda como navegación). Re-proponer lo mismo
era un bucle invisible del anti-loop (proponer_slots resetea attempts);
el lateral contesta la duda con las opciones vigentes.
"slot_stale" se enruta explícito a proponer_slots por legibilidad: el
mismo nodo decide el template ("se acaba de ocupar") leyendo la señal.
ATENCIÓN builder.py: "responder_lateral" debe estar en el path_map del
add_conditional_edges de consultar_disponibilidad o explota en runtime.
"""
import logging

from core.contracts import AgentState

from .contracts import AGENDA_VENTANA_MAX_DIAS, BOOKING_MAX_ATTEMPTS

logger = logging.getLogger(__name__)


def _datos_completos(state: AgentState) -> bool:
    return bool(state.get("lead_nombre") and state.get("lead_email")
                and state.get("lead_modalidad"))


def route_booking(state: AgentState) -> str:
    """Post-planner. Orden de prioridad (≡ supervisor BI):
    anti-loop → abort → duda lateral → captura → ventana → consulta → elección."""
    d = state.get("booking_decision") or {}
    accion = d.get("accion")

    # 1) Anti-loop: demasiados reintentos de elección → humano
    if (state.get("booking_attempts") or 0) >= BOOKING_MAX_ATTEMPTS:
        logger.warning("[booking] max reintentos → ofrecer ejecutiva")
        return "ofrecer_ejecutiva"

    # 2) Abort explícito
    if accion == "abortar":
        return "cerrar_booking"

    # 3) Duda lateral sin perder el stage (responde y re-muestra opciones)
    if accion == "consultar":
        return "responder_lateral"

    # 4) Datos incompletos priman sobre todo lo demás (≡ needs_followup)
    if not _datos_completos(state):
        return "pedir_datos"

    # 5) Ventana agotada pidiendo más fechas
    if accion == "pedir_otra_fecha" and \
            (state.get("agenda_ventana_desde") or 0) >= AGENDA_VENTANA_MAX_DIAS:
        return "ofrecer_ejecutiva"

    # 6) Nueva disponibilidad: entrada con datos / otra fecha / franja
    if accion in ("entregar_datos", "pedir_otra_fecha", "filtrar_franja"):
        return "consultar_disponibilidad"

    # 7) Elección sobre la propuesta vigente
    if accion == "elegir_horario":
        if state.get("slots_propuestos") is None:
            return "consultar_disponibilidad"      # defensivo: nada propuesto aún
        if state.get("booking_match") is not None:
            return "confirmar_y_crear"
        if state.get("booking_match_candidatos"):
            return "aclarar_eleccion"              # ambigüedad real (subset)
        return "reintentar_eleccion"

    # 8) Fallback: datos completos pero nada interpretable → proponer
    return "consultar_disponibilidad"


def despues_de_consultar(state: AgentState) -> str:
    """consultar_disponibilidad solo trae slots, agota la ventana o detecta
    que la tanda nueva es idéntica a la vigente.

      ventana_agotada  → humano (agenda/franja llena o freebusy caído)
      tanda_repetida   → lateral: era una DUDA mal clasificada; re-proponer
                         idéntico es un bucle invisible del anti-loop
      slot_stale       → proponer con template "se acaba de ocupar" (replan
                         en caliente: el slot elegido se ocupó post-propuesta)
    """
    señal = state.get("booking_signal")
    if señal == "ventana_agotada":
        return "ofrecer_ejecutiva"
    if señal == "tanda_repetida":
        return "responder_lateral"
    if señal == "slot_stale":
        return "proponer_slots"          # explícito: proponer lee la señal
    return "proponer_slots"


def despues_de_confirmar(state: AgentState) -> str:
    """Replan en caliente (≡ replan con memoria): el slot elegido se ocupó
    entre la propuesta y la elección → regenerar tanda del MISMO día/franja."""
    return ("consultar_disponibilidad"
            if state.get("booking_signal") == "slot_stale"
            else END)


from langgraph.graph import END  # noqa: E402  (al final para legibilidad del routing)
