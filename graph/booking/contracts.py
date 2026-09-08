"""graph/booking/contracts.py — Contratos del sub-agente de agendamiento.

El LLM PROPONE (BookingDecision), los validadores DISPONEN:
  - email:    EMAIL_RE determinista
  - horario:  slot_match + set de slots REALES propuestos
  - creación: revalidación freebusy inmediatamente antes de insertar
El LLM jamás alucina un horario porque el único "catálogo" que existe
es la disponibilidad devuelta por horarios_disponibles().
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

# Políticas del sub-flujo (vivían dispersas en nodes.py)
AGENDA_TTL_HORAS = 24            # TTL único del sub-flujo (short-circuit)
AGENDA_DIAS_POR_TANDA = 1        # días mostrados por propuesta
AGENDA_HORAS_POR_DIA = 7         # horas representativas por día (¡fix del 1-slot!)
AGENDA_SLOTS_VENTANA_DIAS = 3    # avance de "otro día"
AGENDA_VENTANA_MAX_DIAS = 10     # límite → ejecutiva
BOOKING_MAX_ATTEMPTS = 2         # reintentos de elección fallidos (anti-loop)

# False = crear directo; True = interrupt() HITL de aprobación operador
REQUIERE_APROBACION_AGENDAMIENTO = False


class BookingDecision(BaseModel):
    """Decisión del planner para ESTE turno. Solo puede referirse a los
    horarios VIGENTES o pedir nueva disponibilidad — nunca fabricar uno."""
    accion: Literal[
        "entregar_datos",    # aporta/corrige nombre/email/modalidad
        "elegir_horario",    # apunta a una opción/hora de las propuestas
        "pedir_otra_fecha",  # "otro día", "más adelante", "la próxima semana"
        "filtrar_franja",    # "por la mañana", "después de las 15"
        "consultar",         # duda lateral sin perder el flujo ("¿dura 30 min?")
        "abortar",           # desiste del agendamiento
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
        description=(
            "Índice 1-based SOLO si se refiere inequívocamente a una "
            "opción vigente (respaldo acotado de match_slot)."
        ),
    )
    confianza: float = 1.0
