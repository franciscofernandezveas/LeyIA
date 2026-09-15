"""graph/booking/contracts.py — Contratos del sub-agente de agendamiento.

Wizard de 3 pasos: captura_nombre → captura_email → captura_modalidad → propuesta.
El LLM PROPONE (BookingDecision), los validadores DISPONEN.
"""
from typing import Literal

from pydantic import BaseModel, Field

from core.contracts import (
    BookingSignalLabel,
    BookingStageLabel,
    FranjaLabel,
    ModalidadLabel,
)

# Franjas coherentes con horario_contacto del intake (core/contracts.py)
FRANJA_HORAS: dict[FranjaLabel, tuple[int, int]] = {
    "manana": (9, 13),
    "tarde": (14, 18),
}

# Políticas del sub-flujo
AGENDA_TTL_HORAS = 24
AGENDA_DIAS_POR_TANDA = 1
AGENDA_HORAS_POR_DIA = 7
AGENDA_SLOTS_VENTANA_DIAS = 3
AGENDA_VENTANA_MAX_DIAS = 10
BOOKING_MAX_ATTEMPTS = 2

# False = crear directo; True = interrupt() HITL de aprobación operador
REQUIERE_APROBACION_AGENDAMIENTO = False


class BookingDecision(BaseModel):
    """Decisión del planner para ESTE turno."""
    accion: Literal[
        "entregar_nombre",
        "entregar_email",
        "entregar_modalidad",
        "entregar_datos",       # compatibilidad: aporte múltiple (no usado en wizard)
        "elegir_horario",
        "pedir_otra_fecha",
        "filtrar_franja",
        "consultar",
        "abortar",
    ]
    nombre: str | None = None
    email: str | None = None
    modalidad: ModalidadLabel | None = None
    franja: FranjaLabel | None = None
    dias_offset: int | None = Field(
        default=None,
        description="0=hoy, 1=mañana, 7=la próxima semana",
    )
    eleccion: int | None = Field(
        default=None,
        description="Índice 1-based SOLO si se refiere inequívocamente a una opción vigente.",
    )
    confianza: float = 1.0
