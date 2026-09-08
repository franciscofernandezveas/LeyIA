"""graph/intake/contracts.py — Contratos del sub-agente INTAKE.

v7 — Contrato limpio: solo las 3 acciones reales del subgrafo.
"""
from typing import Literal

from pydantic import BaseModel, Field

IntakeStageLabel = Literal["apertura", "preguntando", "completado", "sin_consentimiento"]

IntakeActionLabel = Literal[
    "iniciar_ficha",
    "procesar_respuesta",
    "sin_consentimiento",
]


class IntakeDecision(BaseModel):
    """Decisión del planner de intake para este turno."""
    accion: IntakeActionLabel
    campo_actual: str | None = Field(
        default=None,
        description="ID de la pregunta activa (informativo; índice real en intake_idx)."
    )
    razon: str = Field(default="", description="Justificación breve.")
    confianza: float = Field(default=1.0, description="Siempre 1.0; planner determinista.")
