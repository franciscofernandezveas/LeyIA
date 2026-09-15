"""Contratos centrales: estado del grafo, thread_id y esquemas estructurados.

v6.1 — Booking signal/estado extendidos:
       + BookingSignalLabel incluye "tanda_repetida" (consultar_disponibilidad
         detectó propuesta idéntica → duda mal clasificada como navegación).
       + AgentState incluye agenda_dia_sin_cupos (etiqueta legible del día
         pedido explícitamente que quedó sin cupo; lo muestra proponer_slots
         antes de las alternativas).

v6 — BOOKING migrado a subgrafo (graph/booking/):
     + Estado: booking_stage / booking_decision / booking_match /
       booking_match_candidatos / booking_signal / booking_franja /
       booking_attempts (ledger del sub-flujo, persistido por checkpointer).
     + Labels nuevos: FranjaLabel, BookingStageLabel, BookingSignalLabel.
       Viven AQUÍ y no en graph/booking/contracts.py para respetar la
       dirección de imports (core ← graph, jamás al revés: se evita la
       circularidad, ya que booking/*.py importa AgentState de aquí).
     − ELIMINADOS recolectando_datos_agenda / esperando_eleccion_horario:
       el short-circuit de analyze_sentiment ahora es UN solo flag
       (booking_stage: "captura" | "propuesta").
     − ELIMINADO LeadExtract: la extracción del lead la hace el planner de
       booking vía BookingDecision (graph/booking/contracts.py); la validez
       del email sigue siendo EMAIL_RE determinista, aplicada allá.
       ⚠️ Si algún test/script legado importa LeadExtract, debe migrarse.

Changelog anterior:
  v5 - Integración del nodo de INTAKE (ficha proactiva + IntakeExtract).
       ⚠️ consentimiento_datos NUNCA está en los esquemas LLM (Ley 21.719).
  v4 - Contrato HITL tipado (TipoHITL/HitlPayload); EMAIL_RE;
       recursion_limit en make_config.
  v3 - Intent 'hablar_humano'; sub-flujo de agenda.
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
# Regla v6:
#   intent=fuera_de_dominio → respuesta_fuera_dominio
#   intent=hablar_humano    → intake_lead (ficha) → handoff_humano
#   intent=agendar_asesoria → SUBGRAFO de booking (graph/booking/) en ROUTE_AGENDAR
#   resto                   → respuestas_faq (atención/orientación RAG)
# ⚠️ El sentimiento NO routea: solo modula el tono de respuestas_faq.
# ⚠️ Las 5 rutas DEBEN estar registradas como nodos en graph/builder.py
#    (test de contrato: test_toda_ruta_tiene_nodo). ROUTE_AGENDAR es un
#    subgrafo compilado registrado como nodo — el invariante no se toca.
# --------------------------------------------------------------------------
ROUTE_HANDOFF: str = "handoff_humano"
ROUTE_AGENDAR: str = "agendar_asesoria"
ROUTE_FAQ: str = "respuestas_faq"
ROUTE_FUERA_DOMINIO: str = "respuesta_fuera_dominio"
ROUTE_INTAKE: str = "intake_lead"

VALID_ROUTES = {ROUTE_HANDOFF, ROUTE_AGENDAR, ROUTE_FAQ,
                ROUTE_FUERA_DOMINIO, ROUTE_INTAKE}


# --------------------------------------------------------------------------
# 3) SUB-FLUJO DE BOOKING (graph/booking/) + CONTRATO HITL
# --------------------------------------------------------------------------
ModalidadLabel = Literal["online", "presencial"]

# Franja horaria demandada por el cliente. Coherente (mismos límites) con
# HorarioContactoLabel del intake; los rangos exactos viven en
# graph/booking/contracts.py → FRANJA_HORAS.
FranjaLabel = Literal["manana", "tarde"]

# Étapa del sub-flujo de booking. None = fuera del sub-flujo.
# El short-circuit de analyze_sentiment lee SOLO este flag (v9 nodes.py).
BookingStageLabel = Literal["captura_nombre", "captura_email", "captura_modalidad", "propuesta"]


# Señal consumible de replan interno (≡ replan_errors del núcleo BI):
# la emite la acción, la consume/limpia el planner de booking, nunca cruza
# turnos sin consumirse.
BookingSignalLabel = Literal[
    "slot_stale",        # el slot elegido se ocupó tras la propuesta → replan
    "ventana_agotada",   # sin cupo en ventana/franja → ofrecer ejecutiva
    "creacion_fallida",  # la API rechazó el insert → ofrecer ejecutiva
    "tanda_repetida",    # tanda nueva idéntica a la vigente → duda mal clasificada
]

# Formato razonable de email (validación práctica, no RFC 5322 completa).
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


class TipoHITL(str, Enum):
    """Tipos cerrados de interrupción humana. El CLI/app compara contra
    estos valores — nunca con substrings sueltos."""
    APROBACION_AGENDAMIENTO = "aprobacion_agendamiento"
    # ESCALAMIENTO = "escalamiento"  # reservado: HITL por sentimiento quedó
    # fuera por filosofía (derivar solo si el cliente lo pide/acepta).


class HitlPayload(TypedDict):
    """Payload del interrupt() hacia el operador (contrato firme nodo↔CLI).
    Lo emite confirmar_y_crear en graph/booking/nodes.py cuando
    REQUIERE_APROBACION_AGENDAMIENTO=True."""
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
    # --- Sub-flujo de FAQ (subgrafo graph/faq/) ---
    faq_decision: dict | None        # dump de FAQDecision (turno actual)
    faq_stage: str | None            # "respondiendo" | "clarificando"




    # --- Sub-flujo de BOOKING (subgrafo graph/booking/) ---
    # Ledger del sub-flujo: lo escriben/leen los nodos de graph/booking/;
    # analyze_sentiment solo lee booking_stage (short-circuit) y hace
    # reset completo por abort/TTL. lead_* se conservan a propósito tras
    # abortar: si el cliente retoma, no se le vuelve a pedir su correo.
    lead_nombre: str
    lead_email: str
    lead_modalidad: ModalidadLabel
    booking_stage: BookingStageLabel | None   # None | "captura" | "propuesta"
    booking_decision: dict                    # dump de BookingDecision (turno actual)
    booking_match: int | None                 # índice en slots_propuestos (validado)
    booking_match_candidatos: List[int]       # ambigüedad real (subset a aclarar)
    booking_signal: BookingSignalLabel | None # señal de replan consumible
    booking_franja: FranjaLabel | None        # preferencia "por la mañana/tarde"
    booking_attempts: int                     # anti-loop de reintentos de elección
    slots_propuestos: List[str]               # ISO datetimes (fuente de verdad del match)
    agenda_ventana_desde: int                 # offset de días para "otro día"
    agenda_started_en: str                    # ISO: inicio del sub-flujo (TTL 24 h, único)
    agenda_dia_sin_cupos: str | None          # "miércoles 09/09" si el día pedido está lleno
    booking: dict                             # cita creada (dump de EventoAsesoria)

    # --- Sub-flujo de intake / ficha de lead (nodo intake_lead) ---
    
    # Agregar dentro de AgentState, en el bloque "Sub-flujo de intake":

    # --- Sub-flujo de intake / ficha de lead (nodo intake_lead) ---
    intake_activo: bool                       # ficha en curso
    intake_idx: int                           # índice de la pregunta actual (telemetría;
                                              # la fuente de verdad es _pendientes)
    intake_respuestas: dict                   # {campo: valor validado}
    intake_completado: bool                   # ficha lista → handoff directo
    intake_started_en: str                    # ISO: inicio de ficha (TTL 24 h)
    intake_decision: dict | None              # dump de IntakeDecision (turno actual):
                                              # {accion, tipo, confianza, razon, campos...}
    intake_stage: str | None                  # apertura|preguntando|pausado|completado|...
    intake_attempts: int                      # fallos de validación en la pregunta activa
    intake_exit: str | None                   # None | "faq" (pausa) | "handoff" (parcial)
    intake_resume: bool                       # tras FAQ lateral, intake re-pregunta pendiente
               # ISO: inicio de ficha (TTL 24 h)

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


class IntakeExtract(BaseModel):
    """Mapeo texto libre → campos de la ficha de intake (nodo intake_lead).
    Los validadores deterministas de graph/intake.py son la fuente de verdad;
    este esquema es solo la costura de lenguaje natural.

    ⚠️ consentimiento_datos queda FUERA a propósito: debe responderse
    directamente (sí/no), nunca inferirse con el LLM (Ley 21.719).

    (La extracción del LEAD de agendamiento ya no vive aquí: es
    BookingDecision en graph/booking/contracts.py, validada por
    graph/booking/planner.py.)
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
    recursion_limit defensivo: cubre el edge intake → handoff y las cadenas
    internas del subgrafo de booking (planner → consultar → proponer, y el
    replan confirmar → consultar por slot_stale), que cuentan como super-pasos
    dentro de la misma invocación del grafo padre."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
    }
