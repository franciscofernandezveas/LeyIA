"""Contratos centrales: estado del grafo, thread_id y esquemas estructurados.

v5 — Integración del nodo de INTAKE (ficha proactiva de lead + caso):
     + ROUTE_INTAKE en el set oficial de rutas.
     + Labels de intake (etapa del proceso / horario / canal de origen).
     + IntakeExtract: esquema del extractor LLM (la validación queda en
       los validadores deterministas de graph/intake.py).
     + Estado: intake_activo / intake_idx / intake_respuestas /
       intake_completado / intake_started_en.
     ⚠️ consentimiento_datos NO está en IntakeExtract a propósito:
        debe responderse directamente (cumplimiento Ley 21.719).

Changelog anterior:
  v4 - State de agenda completo (link idempotente, TTLs, contador de
       verificaciones); contrato HITL tipado (TipoHITL/HitlPayload);
       EMAIL_RE; recursion_limit en make_config; eliminados
       'escalated' / 'slots_propuestos' / 'esperando_slot'.
  v3 - Intent 'hablar_humano'; LeadExtract; sub-flujo de agenda.
  v2 - Etiquetas cerradas (Literal/Pydantic), rutas oficiales, fallback.
"""
import re
from enum import Enum
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
# Regla v5:
#   intent=fuera_de_dominio → respuesta_fuera_dominio
#   intent=hablar_humano    → intake_lead (ficha) → handoff_humano
#   intent=agendar_asesoria → agendar_asesoria (sub-flujo captura + link)
#   resto                   → respuestas_faq (atención/orientación RAG)
# ⚠️ El sentimiento NO routea: solo modula el tono de respuestas_faq.
# ⚠️ Las 5 rutas DEBEN estar registradas como nodos en graph/builder.py
#    (test de contrato: test_toda_ruta_tiene_nodo).
# --------------------------------------------------------------------------
ROUTE_HANDOFF: str = "handoff_humano"
ROUTE_AGENDAR: str = "agendar_asesoria"
ROUTE_FAQ: str = "respuestas_faq"
ROUTE_FUERA_DOMINIO: str = "respuesta_fuera_dominio"
ROUTE_INTAKE: str = "intake_lead"

VALID_ROUTES = {ROUTE_HANDOFF, ROUTE_AGENDAR, ROUTE_FAQ,
                ROUTE_FUERA_DOMINIO, ROUTE_INTAKE}


# --------------------------------------------------------------------------
# 3) SUB-FLUJO DE AGENDAMIENTO + CONTRATO HITL
# --------------------------------------------------------------------------
ModalidadLabel = Literal["online", "presencial"]

# Formato razonable de email (validación práctica, no RFC 5322 completa).
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


class TipoHITL(str, Enum):
    """Tipos cerrados de interrupción humana. El CLI/app compara contra
    estos valores — nunca con substrings sueltos."""
    APROBACION_AGENDAMIENTO = "aprobacion_agendamiento"
    # ESCALAMIENTO = "escalamiento"  # reservado: HITL por sentimiento quedó
    # fuera por filosofía (derivar solo si el cliente lo pide/acepta).


class HitlPayload(TypedDict):
    """Payload del interrupt() hacia el operador (contrato firme nodo↔CLI)."""
    tipo: str            # TipoHITL.value
    thread_id: str
    query: str           # último mensaje del cliente
    detalle: str         # instrucción legible para el operador
    lead: dict           # {nombre, email, modalidad}
    categoria: str
    urgencia: str


# --------------------------------------------------------------------------
# 3bis) LABELS DEL INTAKE  (los consume IntakeExtract y graph/intake.py)
# --------------------------------------------------------------------------
EtapaProcesoLabel = Literal["sin_inicio", "demandado",
                            "causa_en_curso", "sentencia_previa"]
HorarioContactoLabel = Literal["manana", "tarde", "indiferente"]
ComoConocioLabel = Literal["google", "instagram", "tiktok",
                           "recomendacion", "otro"]


# --------------------------------------------------------------------------
# 4) ESTADO COMPARTIDO DEL GRAFO
# --------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # --- Entrada / persistencia ---
    messages: Annotated[list, add_messages]   # historial acumulado
    query: str                                # derivado SIEMPRE del último humano
    thread_id: str                            # en prod = teléfono (E.164)

    # --- Clasificación unificada (nodo analyze_sentiment) ---
    sentiment: SentimentLabel
    urgency: UrgencyLabel
    intent: IntentLabel
    category: CategoryLabel
    clf_reason: str                           # auditoría/debug
    route: str                                # decisión de routing (VALID_ROUTES)

    # --- Atención RAG (nodo respuestas_faq) ---
    context: List[str]
    response: str

    # --- Sub-flujo de agendamiento (nodo agendar_asesoria) ---
    lead_nombre: str
    lead_email: str
    lead_modalidad: ModalidadLabel
    recolectando_datos_agenda: bool           # short-circuit: capturando datos
    agenda_started_en: str                    # ISO: inicio de captura (TTL 24 h)
    esperando_confirmacion_booking: bool      # link enviado, reserva sin confirmar
    agenda_enviada_en: str                    # ISO: envío del link (TTL 48 h)
    agenda_link: str                          # link enviado (reusar, NO regenerar)
    verificaciones_fallidas: int              # contador con escape a humano
    booking: dict                             # cita confirmada (BookingInfo dump)

    # --- Sub-flujo de intake / ficha de lead (nodo intake_lead) ---
    intake_activo: bool                       # short-circuit: ficha en curso
    intake_idx: int                           # índice de la pregunta actual
    intake_respuestas: dict                   # {campo: valor validado}
    intake_completado: bool                   # ficha lista → handoff directo
    intake_started_en: str                    # ISO: inicio de ficha (TTL 24 h)

    # --- Escalamiento / cierre (nodo handoff_humano) ---
    closed: bool                              # hilo derivado/cerrado
    summary: str                              # resumen del caso para la ejecutiva
    notificacion_pendiente: bool              # WhatsApp falló; reintentar por job


# --------------------------------------------------------------------------
# 5) ESQUEMAS DE SALIDA ESTRUCTURADA DEL LLM
# --------------------------------------------------------------------------
class AnalisisResult(BaseModel):
    """Salida del clasificador unificado (nodo analyze_sentiment).

    ⚠️ Estas descriptions viajan al LLM: deben calzar con
    classification.system_prompt de core/prompts.yaml.
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
    reason: str = Field(description="Justificación breve (1 línea).")


class LeadExtract(BaseModel):
    """Extracción de datos para agendamiento. Campos faltantes → None.
    La validez del email NUNCA se delega al LLM: se verifica con EMAIL_RE."""
    nombre: str | None = Field(default=None, description="Nombre completo.")
    email: str | None = Field(default=None, description="Correo electrónico.")
    modalidad: ModalidadLabel | None = Field(
        default=None,
        description=(
            "'online' si prefiere videollamada/Google Meet/virtual; "
            "'presencial' si prefiere oficina/en persona."
        ),
    )


class IntakeExtract(BaseModel):
    """Mapeo texto libre → campos de la ficha de intake (nodo intake_lead).
    Los validadores deterministas de graph/intake.py son la fuente de verdad;
    este esquema es solo la costura de lenguaje natural.

    ⚠️ consentimiento_datos queda FUERA a propósito: debe responderse
    directamente (sí/no), nunca inferirse con el LLM (Ley 21.719).
    """
    nombre: str | None = Field(default=None, description="Nombre completo.")
    email: str | None = Field(default=None, description="Correo electrónico.")
    telefono: str | None = Field(
        default=None, description="Móvil chileno, normalizado a +569XXXXXXXX."
    )
    situacion_actual: str | None = Field(
        default=None,
        description="Descripción breve del problema legal y desde cuándo.",
    )
    etapa_proceso: EtapaProcesoLabel | None = Field(
        default=None,
        description=(
            "'sin_inicio': no ha hecho nada; 'demandado': lo demandaron/"
            "notificaron; 'causa_en_curso': hay juicio activo; "
            "'sentencia_previa': ya existe fallo o acuerdo."
        ),
    )
    hijos_menores: bool | None = Field(
        default=None,
        description="True si menciona hijos menores de edad de por medio.",
    )
    comuna: str | None = Field(default=None, description="Comuna/ciudad.")
    horario_contacto: HorarioContactoLabel | None = Field(
        default=None, description="Horario preferido para llamada de la ejecutiva."
    )
    como_nos_conocio: ComoConocioLabel | None = Field(
        default=None, description="Canal por el que conoció el estudio."
    )


def fallback_analisis(motivo: str = "error de parseo") -> AnalisisResult:
    """Fallback conservador: no rompe el grafo y deja rastro en clf_reason."""
    return AnalisisResult(
        sentiment="neutro", urgency="media",
        intent="respuestas_faq", category="otro",
        reason=f"fallback por {motivo}",
    )


# --------------------------------------------------------------------------
# 6) Config de ejecución
# --------------------------------------------------------------------------
def make_config(thread_id: str) -> dict:
    """LangGraph persiste el estado por thread_id vía checkpointer.
    recursion_limit defensivo (el grafo es casi lineal; cubre el edge
    intake → handoff y futuros ciclos)."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
    }
