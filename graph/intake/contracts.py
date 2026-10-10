"""graph/intake/contracts.py — Contratos del sub-agente INTAKE.

v8.2 — Refactor:
  - Quita hijos_menores del esquema IntakeDecision: no es una pregunta activa
    del flujo, confundía al LLM y podía romper validación en JSON mode.
  - Mantiene normalización de centinelas nulos.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from core.contracts import (
    ComoConocioLabel, EtapaProcesoLabel, HorarioContactoLabel,
)

IntakeStageLabel = Literal["apertura", "preguntando", "pausado",
                           "completado", "sin_consentimiento", "abandonado"]

IntakeActionLabel = Literal[
    "iniciar_ficha",
    "procesar_respuesta",
    "reanudar",
    "pausar_para_faq",
    "pausar_ficha",
    "derivar_parcial",
    "abandonar_ficha",
    "sin_consentimiento",
]

IntakeTurnoLabel = Literal[
    "respuesta_formulario",
    "duda_o_consulta",
    "pausar",
    "insiste_humano",
    "abandonar",
]

_CENTINELAS_NULAS = frozenset({
    "", "null", "none", "nil", "n/a", "na", "-", "--",
    "sin dato", "no aplica", "no especificado", "no especifica",
})

_ETAPA_DESDE_NUMERO = {
    1: "sin_inicio",
    2: "demandado",
    3: "causa_en_curso",
    4: "sentencia_previa",
}


def _a_none(v):
    if isinstance(v, str) and v.strip().lower() in _CENTINELAS_NULAS:
        return None
    return v


class IntakeDecision(BaseModel):
    """Salida del planner semántico de intake (1 LLM/turno)."""
    tipo: IntakeTurnoLabel = Field(
        description=(
            "'respuesta_formulario': el cliente entrega datos o responde la "
            "pregunta activa (aunque sea breve: un número, un sí, una frase); "
            "'duda_o_consulta': hace una pregunta (precios, proceso, plazos) o "
            "comenta algo ajeno a la ficha SIN cancelarla; "
            "'pausar': pide pausa/momento/más tarde; "
            "'insiste_humano': pide hablar con una persona/ejecutiva o lo "
            "repite por segunda vez; "
            "'abandonar': ya no quiere seguir con el registro."
        )
    )
    nombre: str | None = Field(
        default=None,
        description="Nombre de pila y apellido(s) tal como los declara el "
                    "cliente. Quita fórmulas ('me llamo', 'soy'). Jamás una "
                    "frase que no sea un nombre propio.")
    email: str | None = Field(
        default=None,
        description="Correo normalizado user@dominio.cl. Si viene dictado "
                    "('francisco arroba gmail punto com'), conviértelo.")
    situacion_actual: str | None = Field(
        default=None,
        description="Resumen fiel de los hechos declarados (sin agregar nada).")
    etapa_proceso: EtapaProcesoLabel | None = Field(
        default=None,
        description="'sin_inicio' | 'demandado' | 'causa_en_curso' | "
                    "'sentencia_previa'. Un número solo (1-4) mapea a la "
                    "opción correspondiente de la pregunta activa.")
    comuna: str | None = Field(default=None, description="Comuna/ciudad.")
    horario_contacto: HorarioContactoLabel | None = Field(default=None)
    como_nos_conocio: ComoConocioLabel | None = Field(default=None)
    confianza: float = Field(default=0.0, ge=0, le=1)
    razon: str = Field(default="", description="Justificación breve.")

    @field_validator(
        "nombre", "email", "situacion_actual", "comuna",
        "etapa_proceso", "horario_contacto", "como_nos_conocio",
        mode="before",
    )
    @classmethod
    def _centinelas_a_none(cls, v):
        return _a_none(v)

    @field_validator("tipo", mode="before")
    @classmethod
    def _tipo_por_defecto(cls, v):
        v = _a_none(v)
        return v if v is not None else "respuesta_formulario"

    @field_validator("confianza", mode="before")
    @classmethod
    def _confianza_por_defecto(cls, v):
        v = _a_none(v)
        if v is None:
            return 0.0
        if isinstance(v, str):
            try:
                return float(v.strip().replace(",", "."))
            except ValueError:
                return 0.0
        return v

    @field_validator("razon", mode="before")
    @classmethod
    def _razon_por_defecto(cls, v):
        v = _a_none(v)
        return "" if v is None else v

    @field_validator("etapa_proceso", mode="before")
    @classmethod
    def _etapa_numero_a_literal(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return _ETAPA_DESDE_NUMERO.get(int(v), v)
        if isinstance(v, str) and v.strip().isdigit():
            return _ETAPA_DESDE_NUMERO.get(int(v.strip()), v)
        return v
