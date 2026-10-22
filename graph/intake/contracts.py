"""graph/intake/contracts.py — Contratos del sub-agente INTAKE.

v17 — Reorganización por ramas temáticas:
  - IntakeDecision ahora solo contiene campos que tienen una PREGUNTA real
    en nodes.QUESTIONS (se elimina la extracción fantasma de horario_contacto
    y como_nos_conocio: el LLM los extraía y _aplicar_extraccion los
    descartaba silenciosamente).
  - Nuevos campos extraíbles con Literales cerrados: area_caso, rol_penal,
    ingresos_rango, medidas_apremio, hijos_menores, audiencia_proxima,
    detenido (bool). hijos_menores VUELVE como extracción pasiva: el
    problema v8.2 no era el dato sino que no existía pregunta activa; ahora
    sí existe (módulo familia).
  - Nueva acción "derivar_urgente": detención vigente / orden de arresto
    deriva a humano de inmediato. El padre ve intake_exit="handoff"
    (IntakeExitLabel no se toca).
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from core.contracts import EtapaProcesoLabel

# ---------------------------------------------------------------------------
# Labels del sub-agente (propiedad del intake; no tocan core.contracts)
# ---------------------------------------------------------------------------

IntakeStageLabel = Literal["apertura", "preguntando", "pausado",
                           "completado", "sin_consentimiento", "abandonado"]

IntakeActionLabel = Literal[
    "iniciar_ficha",
    "procesar_respuesta",
    "reanudar",
    "pausar_para_faq",
    "pausar_ficha",
    "derivar_parcial",
    "derivar_urgente",
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

# Labels cerrados de las nuevas preguntas. Las CLAVES de estos mapeos son lo
# que el LLM puede devolver; nodes.QUESTIONS las traduce al texto canónico
# almacenado en intake_respuestas (vía label_map de cada pregunta).
AreaCasoLabel = Literal[
    "pension_alimentos",   # me demandan por pensión / quiero rebajarla / terminarla
    "regimen_visitas",     # ver a mis hijos
    "divorcio",            # divorcio o compensación económica
    "medidas_apremio",     # embargo, licencia retenida, registro de deudores, arresto
    "penal",               # materia penal (la category del padre lo guarda como "otro")
    "laboral_civil",       # despido, empleador, contrato, arriendo, deuda civil
    "otro_tema",
]

RolPenalLabel = Literal["imputado", "victima", "familiar"]

IngresosLabel = Literal[
    "dependiente_estable",     # trabajo dependiente, sueldo estable
    "independiente_variable",  # independiente / ingresos variables
    "cesante_reducidos",       # cesante o ingresos muy reducidos
    "jubilado",                # jubilado / pensionado
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

_AREA_DESDE_NUMERO = {
    1: "pension_alimentos",
    2: "regimen_visitas",
    3: "divorcio",
    4: "medidas_apremio",
    5: "penal",
    6: "laboral_civil",
    7: "otro_tema",
}

_ROL_DESDE_NUMERO = {1: "imputado", 2: "victima", 3: "familiar"}

_INGRESOS_DESDE_NUMERO = {
    1: "dependiente_estable",
    2: "independiente_variable",
    3: "cesante_reducidos",
    4: "jubilado",
}

_BOOL_TRUE = frozenset({"si", "sí", "sipo", "sip", "yes", "true", "1",
                        "afirmativo", "claro", "así es", "correcto"})
_BOOL_FALSE = frozenset({"no", "nop", "nope", "false", "0", "negativo",
                         "para nada", "no gracias"})


def _a_none(v):
    if isinstance(v, str) and v.strip().lower() in _CENTINELAS_NULAS:
        return None
    return v


def _numero_a_literal(v, mapa: dict):
    """'2' / 2 → literal correspondiente (respuestas numéricas a listas WA)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return mapa.get(int(v), v)
    if isinstance(v, str) and v.strip().isdigit():
        return mapa.get(int(v.strip()), v)
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

    # ---- Tronco común ----
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
    comuna: str | None = Field(
        default=None,
        description="Comuna o ciudad de Chile mencionada por el cliente.")

    # ---- Clasificación de rama ----
    area_caso: AreaCasoLabel | None = Field(
        default=None,
        description=(
            "EXTENDER SIEMPRE que el mensaje revele el área, aunque no se "
            "haya preguntado: demanda/pago de pensión → pension_alimentos; "
            "no me dejan ver a mis hijos → regimen_visitas; divorcio o "
            "compensación → divorcio; embargo / licencia retenida / registro "
            "de deudores / orden de arresto → medidas_apremio; detención, "
            "fiscalía, carabineros, PDI, querella, formalización → penal; "
            "despido / empleador / contrato / arriendo → laboral_civil. Un "
            "número solo (1-7) respondiendo la pregunta de área mapea a la "
            "opción correspondiente."))

    # ---- Módulo penal ----
    rol_penal: RolPenalLabel | None = Field(
        default=None,
        description="Solo si el caso es penal: 'imputado' (lo acusan), "
                    "'victima' (él sufrió el delito), 'familiar' (un cercano, "
                    "típicamente el detenido). Número 1-3 mapea a la opción.")
    detenido: bool | None = Field(
        default=None,
        description=(
            "True SOLO si hay una detención VIGENTE declarada como hecho "
            "('me detuvieron', 'mi hijo está detenido', 'control de detención "
            "mañana'); False si declara expresamente que no; None si no se "
            "sabe o es una pregunta hipotética ('¿me pueden detener?')."))
    audiencia_proxima: str | None = Field(
        default=None,
        description="Fecha o descripción de audiencia/citación próxima, tal "
                    "cual la declara ('la formalización es el martes').")

    # ---- Módulo familia ----
    hijos_menores: str | None = Field(
        default=None,
        description="Cantidad de hijos menores involucrados y sus edades, "
                    "si el cliente las declara ('2, de 5 y 8 años', 'una "
                    "hija de 12'). None si no lo menciona.")
    ingresos_rango: IngresosLabel | None = Field(
        default=None,
        description="Situación laboral/ingresos declarada. Número 1-4 mapea "
                    "a la opción de la lista.")
    medidas_apremio: str | None = Field(
        default=None,
        description="Medidas de apremio vigentes declaradas (retención de "
                    "licencia, prohibición de salir del país, embargo, "
                    "retención de sueldo/AFP, orden de arresto) o 'ninguna'.")

    # ---- Meta ----
    confianza: float = Field(default=0.0, ge=0, le=1)
    razon: str = Field(default="", description="Justificación breve.")

    # ------------------------------------------------------------------
    @field_validator(
        "nombre", "email", "situacion_actual", "comuna", "etapa_proceso",
        "area_caso", "rol_penal", "audiencia_proxima", "hijos_menores",
        "ingresos_rango", "medidas_apremio",
        mode="before",
    )
    @classmethod
    def _centinelas_a_none(cls, v):
        return _a_none(v)

    @field_validator("detenido", mode="before")
    @classmethod
    def _detenido_a_bool(cls, v):
        v = _a_none(v)
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        t = str(v).strip().lower()
        if t in _BOOL_TRUE:
            return True
        if t in _BOOL_FALSE:
            return False
        return None

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
        return _numero_a_literal(v, _ETAPA_DESDE_NUMERO)

    @field_validator("area_caso", mode="before")
    @classmethod
    def _area_numero_a_literal(cls, v):
        return _numero_a_literal(v, _AREA_DESDE_NUMERO)

    @field_validator("rol_penal", mode="before")
    @classmethod
    def _rol_numero_a_literal(cls, v):
        return _numero_a_literal(v, _ROL_DESDE_NUMERO)

    @field_validator("ingresos_rango", mode="before")
    @classmethod
    def _ingresos_numero_a_literal(cls, v):
        return _numero_a_literal(v, _INGRESOS_DESDE_NUMERO)
