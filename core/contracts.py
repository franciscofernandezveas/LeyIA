"""Contratos centrales: estado del grafo, thread_id y esquemas estructurados.

v6.7 — FIX crisis:
  - Agrega ROUTE_CRISIS = "crisis" y se incluye en VALID_ROUTES.
  - Agrega campo riesgo: bool en AnalisisResult para detección semántica
    de crisis (violencia actual, peligro inminente, riesgo de menores,
    autolesión). El backstop por keywords sigue en graph/nodes.py para el
    caso de fallback silencioso del LLM.

v6.6 — FIX #11 (auditoría): tipado de intake_exit + eliminación de faq_stage.
  ...
[conservar el resto del changelog existente]
"""

import re
from enum import Enum
from typing import Annotated, List, Literal, TypedDict, get_args

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, NotRequired


# --------------------------------------------------------------------------
# 1) ETIQUETAS CERRADAS DE CLASIFICACIÓN  (fuente única de verdad)
# --------------------------------------------------------------------------
SentimentLabel = Literal["positivo", "neutro", "negativo"]
UrgencyLabel = Literal["alta", "media", "baja"]
IntentLabel = Literal[
    "agendar_asesoria",   # pide explícitamente agendar/hora o acepta oferta
    "respuestas_faq",     # pregunta info general O describe su caso buscando orientación
    "hablar_humano",      # pide hablar con persona/ejecutiva, o ACEPTA la oferta
    "fuera_de_dominio",
]
CategoryLabel = Literal[
    "pension_alimentos",
    "rebaja_pension",
    "regimen_visitas",
    "medidas_apremio",
    "terminacion_pension",
    "divorcio",
    "compensacion_economica",
    "consulta_general",
    "otro",
]

VALID_LABELS = {
    "sentiment": set(get_args(SentimentLabel)),
    "urgency": set(get_args(UrgencyLabel)),
    "intent": set(get_args(IntentLabel)),
    "category": set(get_args(CategoryLabel)),
}


# --------------------------------------------------------------------------
# 2) RUTAS OFICIALES DEL GRAFO (las emite compute_route en graph/nodes.py)
#
# Regla v6.7:
#   intent=fuera_de_dominio → respuesta_fuera_dominio
#   intent=hablar_humano    → handoff_choice (preguntar ficha breve o contacto directo)
#                             → intake_lead (ficha) → handoff_humano
#                             → handoff_humano (contacto directo)
#   intent=agendar_asesoria → SUBGRAFO de booking (graph/booking/) en ROUTE_AGENDAR
#   riesgo=True o crisis backstop → ROUTE_CRISIS → handoff_humano con crisis_message
#   resto                   → respuestas_faq (atención/orientación RAG)
# ⚠️ El sentimiento NO routea: solo modula el tono de respuestas_faq.
# ⚠️ Las rutas DEBEN estar registradas como nodos en graph/builder.py
#    (test de contrato: test_toda_ruta_tiene_nodo). ROUTE_AGENDAR es un
#    subgrafo compilado registrado como nodo — el invariante no se toca.
# --------------------------------------------------------------------------
ROUTE_HANDOFF: str = "handoff_humano"
ROUTE_HANDOFF_CHOICE: str = "handoff_choice"
ROUTE_AGENDAR: str = "agendar_asesoria"
ROUTE_FAQ: str = "respuestas_faq"
ROUTE_FUERA_DOMINIO: str = "respuesta_fuera_dominio"
ROUTE_INTAKE: str = "intake_lead"
ROUTE_CRISIS: str = "crisis"  # v6.7

VALID_ROUTES = {ROUTE_HANDOFF, ROUTE_HANDOFF_CHOICE, ROUTE_AGENDAR, ROUTE_FAQ,
                ROUTE_FUERA_DOMINIO, ROUTE_INTAKE, ROUTE_CRISIS}  # v6.7


# --------------------------------------------------------------------------
# 3) SUB-FLUJO DE BOOKING (graph/booking/) + CONTRATO HITL
# --------------------------------------------------------------------------
ModalidadLabel = Literal["online", "presencial"]

FranjaLabel = Literal["manana", "tarde"]

BookingStageLabel = Literal["captura_nombre", "captura_email", "captura_modalidad", "propuesta"]

BookingSignalLabel = Literal[
    "slot_stale",
    "ventana_agotada",
    "creacion_fallida",
    "tanda_repetida",
]

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)*\.[a-zA-Z]{2,}$")


class TipoHITL(str, Enum):
    APROBACION_AGENDAMIENTO = "aprobacion_agendamiento"


class HitlPayload(TypedDict):
    tipo: str
    thread_id: str
    query: str
    detalle: str
    lead: dict
    categoria: str
    urgencia: str


# --------------------------------------------------------------------------
# 3bis) LABELS DEL INTAKE
# --------------------------------------------------------------------------
EtapaProcesoLabel = Literal["sin_inicio", "demandado",
                            "causa_en_curso", "sentencia_previa"]
HorarioContactoLabel = Literal["manana", "tarde", "indiferente"]
ComoConocioLabel = Literal["google", "instagram", "tiktok",
                           "recomendacion", "otro"]

IntakeExitLabel = Literal["faq", "pausa", "handoff", "end", "cta_cierre"]


# --------------------------------------------------------------------------
# 4) ESTADO COMPARTIDO DEL GRAFO
# --------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    thread_id: str

    session_id: str | None

    sentiment: SentimentLabel
    urgency: UrgencyLabel
    intent: IntentLabel
    category: CategoryLabel
    case_category: CategoryLabel
    clf_reason: str
    route: str

    context: List[str]
    response: str

    faq_decision: dict | None
    cta_agenda_ofrecida: bool

    lead_nombre: str
    lead_email: str
    lead_modalidad: ModalidadLabel
    booking_stage: BookingStageLabel | None
    booking_decision: dict
    booking_match: int | None
    booking_match_candidatos: List[int]
    booking_signal: BookingSignalLabel | None
    booking_franja: FranjaLabel | None
    booking_attempts: int
    slots_propuestos: List[str]
    agenda_ventana_desde: int
    agenda_started_en: str
    agenda_dia_sin_cupos: str | None
    booking_privacidad_dicha: bool
    booking: dict

    intake_activo: bool
    intake_idx: int
    intake_respuestas: dict
    intake_completado: bool
    intake_started_en: str
    intake_decision: dict | None
    intake_stage: str | None
    intake_attempts: int
    intake_exit: IntakeExitLabel | None
    intake_resume: bool
    intake_oferta_qid: NotRequired[str | None]

    handoff_choice_pending: bool
    handoff_choice_attempts: int
    handoff_choice_en: str | None

    closed: bool
    summary: str
    notificacion_pendiente: bool

    # v6.7: flag de crisis de sesión (levantado por backstop keyword o por
    # clasificador con riesgo=True). Se limpia en RESET_SESION_NUEVA.
    crisis: bool | None

    response_bubbles: NotRequired[list[str]]
    response_interactive: NotRequired[dict | None]
    response_cta: NotRequired[dict | None]


# --------------------------------------------------------------------------
# 5) ESQUEMAS DE SALIDA ESTRUCTURADA DEL LLM
# --------------------------------------------------------------------------
class AnalisisResult(BaseModel):
    """Salida del clasificador unificado (nodo analyze_sentiment).

    v7.1 — Campo riesgo añadido (v6.7). El campo reason sigue primero.
    """
    reason: str = Field(
        description=(
            "Análisis en 1 línea, ANTES de clasificar: qué pide explícitamente "
            "el cliente, qué emoción expresa y qué hecho legal vigente menciona. "
            "Si hay violencia actual, peligro inminente, riesgo para menores "
            "o ideas de autolesión, empieza el reason con 'RIESGO:'."
        )
    )
    sentiment: SentimentLabel = Field(...)
    urgency: UrgencyLabel = Field(...)
    intent: IntentLabel = Field(...)
    category: CategoryLabel = Field(...)
    riesgo: bool = Field(
        default=False,
        description=(
            "True SOLO si el mensaje menciona violencia actual, peligro "
            "inminente, riesgo para menores o ideas de autolesión. En caso "
            "de duda, preferir False. Este campo activa la ruta ROUTE_CRISIS."
        )
    )


class IntakeExtract(BaseModel):
    nombre: str | None = Field(default=None, description="Nombre completo.")
    email: str | None = Field(default=None, description="Correo electrónico.")
    telefono: str | None = Field(default=None)
    situacion_actual: str | None = Field(default=None)
    etapa_proceso: EtapaProcesoLabel | None = Field(default=None)
    hijos_menores: bool | None = Field(default=None)
    comuna: str | None = Field(default=None)
    horario_contacto: HorarioContactoLabel | None = Field(default=None)
    como_nos_conocio: ComoConocioLabel | None = Field(default=None)


def fallback_analisis(motivo: str = "error de parseo") -> AnalisisResult:
    """Fallback conservador: no rompe el grafo y deja rastro en clf_reason."""
    return AnalisisResult(
        sentiment="neutro", urgency="media",
        intent="respuestas_faq", category="otro",
        reason=f"fallback por {motivo}",
        riesgo=False,
    )


# --------------------------------------------------------------------------
# 6) Config de ejecución
# --------------------------------------------------------------------------
def make_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
    }
