"""graph/faq/contracts.py — Contratos del sub-agente FAQ.

El FAQ agent es especialista en responder desde RAG. Puede:
- Responder la pregunta normalmente
- Detectar que el usuario quiere agendar en medio de la conversación
- Detectar que quiere hablar con un humano
- Detectar que la pregunta está fuera de dominio
"""
from typing import Literal

from pydantic import BaseModel, Field


FAQStageLabel = Literal["respondiendo", "clarificando"]


class FAQDecision(BaseModel):
    """Decisión del planner FAQ para este turno."""
    accion: Literal[
        "responder_pregunta",   # RAG + LLM con contexto
        "sugerir_agendar",      # el usuario muestra intención de agendar
        "hablar_humano",        # pide ejecutiva/persona
        "fuera_de_dominio",     # no relacionado con servicios legales
        "pedir_clarificacion",  # pregunta ambigua, falta contexto
    ]
    razon: str = Field(default="", description="Justificación breve de la acción.")
    confianza: float = 1.0
