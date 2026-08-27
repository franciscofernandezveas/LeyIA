"""Contratos centrales: estado del grafo, thread_id y esquemas estructurados.

v3 — Clasificación unificada (1 LLM: sentiment + urgency + intent + category)
     + sub-flujo de agendamiento (extracción de lead: nombre/email/modalidad)
     + escalamiento por oferta aceptada (closed/summary para la ejecutiva).

Changelog:
  v2 - Etiquetas cerradas (Literal/Pydantic), rutas oficiales, fallback.
  v3 - Intent 'hablar_humano'; LeadExtract; estado de sub-flujo de agenda;
       descripciones de campos alineadas a la filosofía v4 (el sentimiento
       modula tono, NUNCA routea).
"""
from typing import Annotated, List, Literal, TypedDict, get_args

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# 1) ETIQUETAS CERRADAS DE CLASIFICACIÓN  (fuente única de verdad)
# --------------------------------------------------------------------------
SentimentLabel = Literal["positivo", "neutro", "negativo"]
UrgencyLabel = Literal["alta", "media", "baja"]
IntentLabel = Literal[
    "agendar_asesoria",   # pide explícitamente agendar/hora o que un abogado tome su caso
    "respuestas_faq",     # pregunta info general O describe su caso buscando orientación
    "hablar_humano",      # pide hablar con persona/ejecutiva, o ACEPTA la oferta
    "fuera_de_dominio",
]
CategoryLabel = Literal[
    "pension_alimentos",      # defensa en demandas de alimentos
    "rebaja_pension",         # rebaja por cambio de ingresos
    "regimen_visitas",        # relación directa y regular
    "medidas_apremio",        # embargo, retención licencia, registro deudores...
    "terminacion_pension",    # hijos ya independientes
    "divorcio",
    "compensacion_economica",
    "consulta_general",       # precios, horarios, ubicación, proceso...
    "otro",
]

# Derivado automáticamente de los tipos → nunca se desincroniza del esquema
VALID_LABELS = {
    "sentiment": set(get_args(SentimentLabel)),
    "urgency": set(get_args(UrgencyLabel)),
    "intent": set(get_args(IntentLabel)),
    "category": set(get_args(CategoryLabel)),
}


# --------------------------------------------------------------------------
# 2) RUTAS OFICIALES DEL GRAFO (las emite compute_route en graph/nodes.py)
#
# Regla v4:
#   intent=fuera_de_dominio → respuesta_fuera_dominio
#   intent=hablar_humano    → handoff_humano   (SOLO si el cliente lo pide/acepta)
#   intent=agendar_asesoria → agendar_asesoria (sub-flujo de captura + link)
#   resto                   → respuestas_faq   (atención/orientación RAG)
# ⚠️ El sentimiento NO routea desde v4: solo modula el tono de respuestas_faq.
# --------------------------------------------------------------------------
ROUTE_HANDOFF: str = "handoff_humano"
ROUTE_AGENDAR: str = "agendar_asesoria"
ROUTE_FAQ: str = "respuestas_faq"
ROUTE_FUERA_DOMINIO: str = "respuesta_fuera_dominio"

VALID_ROUTES = {ROUTE_HANDOFF, ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO}


# --------------------------------------------------------------------------
# 3) ETIQUETAS DEL SUB-FLUJO DE AGENDAMIENTO
# --------------------------------------------------------------------------
ModalidadLabel = Literal["online", "presencial"]


# --------------------------------------------------------------------------
# 4) ESTADO COMPARTIDO DEL GRAFO
# --------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # --- Entrada / persistencia ---
    messages: Annotated[list, add_messages]   # historial acumulado
    query: str                                # entrada normalizada
    thread_id: str                            # en prod = teléfono Twilio

    # --- Clasificación unificada (nodo analyze_sentiment) ---
    sentiment: SentimentLabel
    urgency: UrgencyLabel
    intent: IntentLabel
    category: CategoryLabel
    clf_reason: str                           # auditoría/debug
    route: str                                # decisión de routing (VALID_ROUTES)

    # --- Atención RAG (nodo respuestas_faq) ---
    context: List[str]                        # fragmentos recuperados
    response: str                             # salida al cliente

    # --- Sub-flujo de agendamiento (nodo agendar_asesoria) ---
    lead_nombre: str
    lead_email: str
    lead_modalidad: ModalidadLabel
    recolectando_datos_agenda: bool           # short-circuit: capturando datos
    slots_propuestos: list                    # (v3 Scheduling API — inactivo en plan Free)
    esperando_slot: bool                      # short-circuit: eligiendo horario

    # --- Escalamiento / cierre (nodo handoff_humano) ---
    escalated: bool
    closed: bool
    summary: str                              # resumen del caso para la ejecutiva

    esperando_confirmacion_booking: bool   # link enviado, reserva aún no confirmada
    agenda_enviada_en: str                 # ISO timestamp (para min_start_time)
    booking: dict  
# --------------------------------------------------------------------------
# 5) ESQUEMAS DE SALIDA ESTRUCTURADA DEL LLM
# (etiqueta fuera del Literal → falla el parseo → NUNCA llega al grafo)
# --------------------------------------------------------------------------
class AnalisisResult(BaseModel):
    """Salida del clasificador unificado (nodo analyze_sentiment).

    ⚠️ Estas descriptions viajan al LLM junto al esquema: deben calzar
    con classification.system_prompt de core/prompts.yaml (v4).
    """
    sentiment: SentimentLabel = Field(
        description="Sentimiento dominante del mensaje del cliente."
    )
    urgency: UrgencyLabel = Field(
        description=(
            "alta: demanda notificada, embargo vigente, medidas de apremio o "
            "plazo judicial; media: quiere avanzar pronto sin coacción vigente; "
            "baja: información general sin apuro explícito."
        )
    )
    intent: IntentLabel = Field(
        description=(
            "'respuestas_faq': pregunta info general O describe su caso "
            "buscando orientación (el agente orienta primero, siempre); "
            "'agendar_asesoria': pide EXPLÍCITAMENTE agendar/reservar hora o "
            "que un abogado tome su caso; "
            "'hablar_humano': pide hablar con una persona/ejecutiva o ACEPTA "
            "la oferta de contacto humano previa; "
            "'fuera_de_dominio': no se relaciona con servicios legales."
        )
    )
    category: CategoryLabel = Field(
        description="Tema legal principal del mensaje (ver CategoryLabel)."
    )
    reason: str = Field(
        description="Justificación breve (1 línea) de la clasificación."
    )


class LeadExtract(BaseModel):
    """Extracción de datos del lead durante el sub-flujo de agendamiento.
    Campos faltantes en el mensaje → None (el nodo re-pregunta solo esos).
    """
    nombre: str | None = Field(
        default=None, description="Nombre completo del cliente."
    )
    email: str | None = Field(
        default=None, description="Correo electrónico del cliente."
    )
    modalidad: ModalidadLabel | None = Field(
        default=None,
        description=(
            "'online' si prefiere videollamada/Google Meet/virtual; "
            "'presencial' si prefiere oficina/en persona."
        ),
    )


def fallback_analisis(motivo: str = "error de parseo") -> AnalisisResult:
    """Fallback conservador: no rompe el grafo y deja rastro en clf_reason."""
    return AnalisisResult(
        sentiment="neutro", urgency="media",
        intent="respuestas_faq", category="otro",
        reason=f"fallback por {motivo}",
    )


# --------------------------------------------------------------------------
# 6) Helper oficial para la config de ejecución (usa thread_id)
# --------------------------------------------------------------------------
def make_config(thread_id: str) -> dict:
    """LangGraph persiste el estado por thread_id vía checkpointer."""
    return {"configurable": {"thread_id": thread_id}}
