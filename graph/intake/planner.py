"""graph/intake/planner.py — Decide la acción del turno en el sub-flujo de intake.

El intake es secuencial, así que este planner es mayormente determinista:
su trabajo es mirar el ledger y decidir qué nodo de acción debe ejecutarse.
La única llamada LLM (extracción de campos) sigue viviendo en el nodo
`procesar_respuesta`, no aquí: el planner solo decide SI se invoca.
"""
import logging

from core.contracts import AgentState

from .contracts import IntakeDecision

logger = logging.getLogger(__name__)


def intake_planner(state: AgentState) -> AgentState:
    """0 LLM. Mira el ledger y elige la acción del turno."""
    activo = bool(state.get("intake_activo"))
    completado = bool(state.get("intake_completado"))
    consentimiento_rechazado = (
        (state.get("intake_respuestas") or {}).get("consentimiento_datos") is False
    )

    # Guard: si la ficha ya se completó, forzar cierre (no repetir)
    if completado:
        logger.info("[intake] ficha ya completada → forzar completar_ficha")
        return {
            "intake_decision": IntakeDecision(
                accion="completar_ficha",
                razon="ficha ya estaba marcada como completada",
            ).model_dump(),
            "intake_stage": "completado",
        }

    # Guard: consentimiento rechazado en paso anterior
    if consentimiento_rechazado:
        return {
            "intake_decision": IntakeDecision(
                accion="sin_consentimiento",
                razon="consentimiento de datos rechazado",
            ).model_dump(),
            "intake_stage": "sin_consentimiento",
        }

    # Primera vez: activar ficha
    if not activo:
        return {
            "intake_decision": IntakeDecision(
                accion="iniciar_ficha",
                razon="ledger de intake no activo",
            ).model_dump(),
            "intake_stage": "apertura",
        }

    # Turno normal: el nodo procesar_respuesta hará toda la validación y
    # decidirá internamente si avanzar, repetir, completar o rechazar.
    idx = state.get("intake_idx", 0)
    from .nodes import QUESTIONS  # import local para evitar ciclo
    campo = QUESTIONS[idx]["id"] if 0 <= idx < len(QUESTIONS) else None

    return {
        "intake_decision": IntakeDecision(
            accion="procesar_respuesta",
            campo_actual=campo,
            razon="ledger activo: procesar respuesta del turno",
        ).model_dump(),
        "intake_stage": "preguntando",
    }
