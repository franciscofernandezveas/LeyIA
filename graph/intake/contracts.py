"""graph/intake/contracts.py — Contratos del sub-agente INTAKE.

v8.1 — IntakeDecision = decisión semántica + extracción embebida (patrón
BookingDecision del subgrafo de booking). UNA llamada LLM por turno.
consentimiento_datos SIGUE fuera del esquema (Ley 21.719): se valida solo
con el validador determinista sí/no.

Fix nulos stringificados: el LLM (function calling / JSON mode) a veces
serializa la ausencia de un dato como la CADENA "null"/"none"/"" en vez
de null JSON. Eso rompía la validación de los campos Literal|None → el
planner caía en fallback todos los turnos, y en campos str|None (nombre,
email, comuna) llegaba a validar como dato legítimo ("Null" como nombre).
Los validators mode="before" normalizan esos centinelas ANTES de validar.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from core.contracts import (
    ComoConocioLabel, EtapaProcesoLabel, HorarioContactoLabel,
)

IntakeStageLabel = Literal["apertura", "preguntando", "pausado",
                           "completado", "sin_consentimiento", "abandonado"]

IntakeActionLabel = Literal[
    "iniciar_ficha",        # apertura (0 LLM)
    "procesar_respuesta",   # validar + avanzar/completar
    "reanudar",             # vuelve de FAQ lateral: re-anexa pendiente (0 LLM)
    "pausar_para_faq",      # el cliente hizo una duda: FAQ responde, luego reanudar
    "derivar_parcial",      # insiste humano o reintentos agotados → handoff con lo que haya
    "abandonar_ficha",      # cancela el registro: cierre amable + link
    "sin_consentimiento",
]

IntakeTurnoLabel = Literal[
    "respuesta_formulario",  # está respondiendo la ficha (puede traer varios campos)
    "duda_o_consulta",       # hace una pregunta/cambia de tema sin cancelar
    "insiste_humano",        # quiere una persona YA (2ª vez o explícito)
    "abandonar",             # no quiere seguir con el registro
]

# Cadenas que el LLM emite como "ausencia de dato" (no son None real).
_CENTINELAS_NULAS = frozenset({
    "", "null", "none", "nil", "n/a", "na", "-", "--",
    "sin dato", "no aplica", "no especificado", "no especifica",
})

# Defensivo: si el modelo devuelve el número de la opción sin mapearlo.
_ETAPA_DESDE_NUMERO = {
    1: "sin_inicio",
    2: "demandado",
    3: "causa_en_curso",
    4: "sentencia_previa",
}


def _a_none(v):
    """Normaliza centinelas de ausencia emitidos como cadena por el LLM."""
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
            "'insiste_humano': pide hablar con una persona/ejecutiva o lo "
            "repite por segunda vez; "
            "'abandonar': ya no quiere seguir con el registro."
        )
    )
    # --- extracción multi-campo: SOLO datos EXPLÍCITOS en el mensaje ---
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
    hijos_menores: bool | None = Field(
        default=None, description="True si menciona hijos menores de edad.")
    comuna: str | None = Field(default=None, description="Comuna/ciudad.")
    horario_contacto: HorarioContactoLabel | None = Field(default=None)
    como_nos_conocio: ComoConocioLabel | None = Field(default=None)
    # --- meta ---
    confianza: float = Field(default=0.0, ge=0, le=1)
    razon: str = Field(default="", description="Justificación breve.")

    # ------------------------------------------------------------------
    # VALIDATORS (mode="before"): corren ANTES de validar Literal/str.
    # ------------------------------------------------------------------
    @field_validator(
        "nombre", "email", "situacion_actual", "comuna",
        "etapa_proceso", "horario_contacto", "como_nos_conocio",
        "hijos_menores",
        mode="before",
    )
    @classmethod
    def _centinelas_a_none(cls, v):
        """'null'/'none'/''/etc. (cadena) → None real."""
        return _a_none(v)

    @field_validator("tipo", mode="before")
    @classmethod
    def _tipo_por_defecto(cls, v):
        """Si 'tipo' viene ausente/centinela → conservador: respuesta."""
        v = _a_none(v)
        return v if v is not None else "respuesta_formulario"

    @field_validator("confianza", mode="before")
    @classmethod
    def _confianza_por_defecto(cls, v):
        """'null' → 0.0; acepta '0,8' (coma decimal)."""
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
        """razon es str obligatorio: centinela → cadena vacía."""
        v = _a_none(v)
        return "" if v is None else v

    @field_validator("etapa_proceso", mode="before")
    @classmethod
    def _etapa_numero_a_literal(cls, v):
        """1-4 (int o str) → literal correspondiente."""
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return _ETAPA_DESDE_NUMERO.get(int(v), v)
        if isinstance(v, str) and v.strip().isdigit():
            return _ETAPA_DESDE_NUMERO.get(int(v.strip()), v)
        return v
