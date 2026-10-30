"""Contratos centrales: estado del grafo, thread_id y esquemas estructurados.

v6.8 — PLAN A: limpieza de crisis + alineación intent con prompts v5.0.0:
  - Elimina ROUTE_CRISIS y su entrada en VALID_ROUTES (nadie la emite:
    graph/nodes.py v22.4 ya no contiene backstops ni guardas de crisis, y
    handoff_humano perdió su bloque de crisis).
  - Elimina `crisis` del AgentState: ya ningún nodo lo lee ni lo escribe.
    Checkpoints antiguos pueden conservar la clave sin efecto (TypedDict
    total=False; los lectores la ignoran).
  - `riesgo` PERMANECE en AnalisisResult: es contrato con el clasificador
    (prompts v5.0.0). Ya no activa ruta alguna; nodes.py v22.4 solo lo
    loguea como warning y deriva por el flujo estándar (hablar_humano +
    urgency alta).
  - FIX descripción de `intent`: el texto heredado (pre-v4.1) definía
    agendar_asesoria como "(b) necesito abogado / (c) angustia aguda /
    (d) lead no-familia", contradiciendo la política vigente (agendar SOLO
    ante pedido explícito o aceptación de oferta). Como el esquema viaja
    al LLM vía with_structured_output, la descripción vieja sobre-disparaba
    agendamientos. Ahora calza con classification.system_prompt v5.0.0.
  - Pendiente fuera de este archivo: quitar el mapping de ROUTE_CRISIS del
    conditional edge en graph/builder.py y del reset en graph/utils.py si
    lo incluye; test_toda_ruta_tiene_nodo sigue verde al iterar
    VALID_ROUTES ya depurado.

v6.7 — FIX saludos reiterados + bug riesgo:
  - Añade `asistente_presentado` e `intake_skipped` al AgentState.
  - Añade `riesgo: bool` a AnalisisResult; el fallback conserva False.

v6.6 — FIX #11 (auditoría): IntakeExitLabel tipado; faq_stage eliminado.
v6.5 — FIX CRÍTICO sesión: session_id dentro del TypedDict AgentState.
v6.4 — Handoff choice: ROUTE_HANDOFF_CHOICE + estado handoff_choice_*.
v6.3 — Cierre dual post-intake + fix EMAIL_RE.
v6.2 — Refactor intake.
v6.1 — Booking signal/estado extendidos.
v6   — BOOKING migrado a subgrafo.

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
from typing_extensions import TypedDict, NotRequired


# --------------------------------------------------------------------------
# 1) ETIQUETAS CERRADAS DE CLASIFICACIÓN  (fuente única de verdad)
# --------------------------------------------------------------------------
SentimentLabel = Literal["positivo", "neutro", "negativo"]
UrgencyLabel = Literal["alta", "media", "baja"]
IntentLabel = Literal[
    "agendar_asesoria",   # pide explícitamente agendar/hora/cita o acepta la oferta
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
# Regla v6.4:
#   intent=fuera_de_dominio → respuesta_fuera_dominio
#   intent=hablar_humano    → handoff_choice (preguntar ficha breve o contacto directo)
#                             → intake_lead (ficha) → handoff_humano
#                             → handoff_humano (contacto directo)
#   intent=agendar_asesoria → SUBGRAFO de booking (graph/booking/) en ROUTE_AGENDAR
#   resto                   → respuestas_faq (atención/orientación RAG)
# ⚠️ El sentimiento NO routea: solo modula el tono de respuestas_faq.
# ⚠️ Las rutas DEBEN estar registradas como nodos en graph/builder.py
#    (test de contrato: test_toda_ruta_tiene_nodo). ROUTE_AGENDAR es un
#    subgrafo compilado registrado como nodo — el invariante no se toca.
# ⚠️ El intake emite intake_exit="cta_cierre" cuando la ficha se completa
#    con el cierre dual; route_post_intake lo deriva a ROUTE_HANDOFF (rama
#    explícita desde v6.6).
# v6.8 (Plan A): no existe ruta de crisis; un mensaje con riesgo entra por
# hablar_humano → handoff estándar con urgencia alta (sin contención).
# --------------------------------------------------------------------------
ROUTE_HANDOFF: str = "handoff_humano"
ROUTE_HANDOFF_CHOICE: str = "handoff_choice"
ROUTE_AGENDAR: str = "agendar_asesoria"
ROUTE_FAQ: str = "respuestas_faq"
ROUTE_FUERA_DOMINIO: str = "respuesta_fuera_dominio"
ROUTE_INTAKE: str = "intake_lead"

VALID_ROUTES = {ROUTE_HANDOFF, ROUTE_HANDOFF_CHOICE, ROUTE_AGENDAR, ROUTE_FAQ,
                ROUTE_FUERA_DOMINIO, ROUTE_INTAKE}


# --------------------------------------------------------------------------
# 3) SUB-FLUJO DE BOOKING (graph/booking/) + CONTRATO HITL
# --------------------------------------------------------------------------
ModalidadLabel = Literal["online", "presencial"]

# Franja horaria demandada por el cliente. Coherente (mismos límites) con
# HorarioContactoLabel del intake; los rangos exactos viven en
# graph/booking/contracts.py → FRANJA_HORAS.
FranjaLabel = Literal["manana", "tarde"]

# Etapa del sub-flujo de booking. None = fuera del sub-flujo.
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
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)*\.[a-zA-Z]{2,}$")


class TipoHITL(str, Enum):
    """Tipos cerrados de interrupción humana."""
    APROBACION_AGENDAMIENTO = "aprobacion_agendamiento"


class HitlPayload(TypedDict):
    """Payload del interrupt() hacia el operador (contrato firme nodo↔CLI)."""
    tipo: str            # TipoHITL.value
    thread_id: str
    query: str
    detalle: str
    lead: dict           # {nombre, email, modalidad}
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

# v6.6 — FIX #11 (auditoría): IntakeExitLabel tipado y con valores muertos
# eliminados. Cada valor corresponde a un emisor real verificado:
#   "faq"        → pausar_para_faq
#   "pausa"      → pausar_ficha
#   "handoff"    → _derivar_parcial
#   "end"        → abandonar_ficha / sin_consentimiento
#   "cta_cierre" → _completar_ficha (cierre dual: Agendar / Hablar con abogada)
# Agregar un valor requiere: (1) incluirlo aquí, (2) emitirlo en algún nodo,
# (3) manejar la rama en route_post_intake — el test
# test_route_post_intake_cobertura_exhaustiva verifica el punto (3).
IntakeExitLabel = Literal["faq", "pausa", "handoff", "end", "cta_cierre"]


# --------------------------------------------------------------------------
# 4) ESTADO COMPARTIDO DEL GRAFO
# --------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # --- Entrada / persistencia ---
    messages: Annotated[list, add_messages]
    query: str
    thread_id: str

    # --- Sesión (v6.5: canal del estado; lo escribe receive_message) ---
    session_id: str | None

    # --- Presentación del asistente ---
    asistente_presentado: bool

    # --- Clasificación unificada (nodo analyze_sentiment) ---
    sentiment: SentimentLabel
    urgency: UrgencyLabel
    intent: IntentLabel
    category: CategoryLabel
    case_category: CategoryLabel
    clf_reason: str
    route: str

    # --- Atención RAG (nodo respuestas_faq) ---
    context: List[str]
    response: str

    # --- Sub-flujo de FAQ (subgrafo graph/faq/) ---
    faq_decision: dict | None        # dump de FAQDecision (turno actual)
    cta_agenda_ofrecida: bool

    # --- Sub-flujo de BOOKING (subgrafo graph/booking/) ---
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

    # --- Sub-flujo de intake / ficha de lead (nodo intake_lead) ---
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
    intake_skipped: NotRequired[set]

    # --- Bifurcación handoff: ficha breve vs contacto directo (v6.4) ---
    handoff_choice_pending: bool
    handoff_choice_attempts: int
    handoff_choice_en: str | None

    # --- Escalamiento / cierre (nodo handoff_humano) ---
    closed: bool
    summary: str
    notificacion_pendiente: bool

    response_bubbles: NotRequired[list[str]]
    response_interactive: NotRequired[dict | None]
    # Canal CTA preexistente (cierre dual del intake). Si tu dispatcher ya
    # lo envía como cta_url, úsalo en vez de response_interactive kind="cta"
    # — pero nunca ambos en el mismo turno (duplicarías el mensaje).
    response_cta: NotRequired[dict | None]


# --------------------------------------------------------------------------
# 5) ESQUEMAS DE SALIDA ESTRUCTURADA DEL LLM
# --------------------------------------------------------------------------
class AnalisisResult(BaseModel):
    """Salida del clasificador unificado (nodo analyze_sentiment).

    v6.8 — FIX descripción de `intent`: alineada con la política vigente
    (prompts v4.1+/v5.0.0): agendar_asesoria SOLO ante pedido explícito de
    hora/cita o aceptación de oferta. La descripción heredada pre-v4.1
    ("necesito abogado", "angustia aguda", "lead no-familia" → agendar)
    viajaba al LLM en el esquema y sobre-disparaba agendamientos.
    `riesgo` se conserva (contrato con el prompt) pero no activa rutas.

    v7 — 'reason' ANTES de las etiquetas (el LLM genera en orden de campos:
    razona primero, clasifica después) + descriptions operativas calibradas
    con el golden set (umbrales explícitos del dominio Manzzo).
    """
    reason: str = Field(
        description=(
            "Análisis en 1 línea, ANTES de clasificar: qué pide explícitamente "
            "el cliente, qué emoción expresa y qué hecho legal vigente menciona."
        )
    )
    sentiment: SentimentLabel = Field(
        description=(
            "Emoción dominante hacia su situación o el servicio. "
            "negativo: sufrimiento, frustración o enojo EXPRESADO — explícito "
            "('desesperado', 'chato', 'injusto', garabatos) o implícito "
            "(sarcasmo, 'esperaba algo mejor de ustedes'). La cortesía de "
            "envoltura ('hola buenas, disculpa la hora') NO lo atenúa. "
            "neutro: consulta tranquila; INCLUYE preocupación hipotética o "
            "moderada sin sufrimiento expresado ('¿me pueden embargar altiro?', "
            "'ando justo de plata'). "
            "positivo: gratitud o satisfacción explícita ('muchas gracias', "
            "'excelente atención')."
        )
    )
    urgency: UrgencyLabel = Field(
        description=(
            "La urgencia es del HECHO, no del tono: apremio vigente narrado "
            "con calma sigue siendo alta. "
            "alta: coacción judicial VIGENTE o plazo — demanda notificada, "
            "embargo, medidas de apremio (licencia retenida, registro de "
            "deudores) — o inmediatez exclamada ('YA'). "
            "media: pregunta sobre su propio caso (proceso, escenarios, "
            "hipótesis legales), quiere avanzar pronto ('lo antes posible', "
            "'esta semana', pide contacto humano), o ya ocurrió un evento "
            "legal adverso sin coacción vigente (despido, no me deja ver a "
            "mis hijos). "
            "baja: preguntas sobre la FIRMA sin caso propio de por medio "
            "(precios, contacto, cobertura, horarios, proceso de contratación, "
            "servicios ofrecidos en abstracto)."
        )
    )
    intent: IntentLabel = Field(
        description=(
            "La emoción NO cambia la intención: una pregunta factual con "
            "garabatos sigue siendo respuestas_faq. "
            "'fuera_de_dominio': nada relacionado con servicios legales "
            "(contador, psicólogo, celulares). "
            "'hablar_humano': pide ser atendido por una persona/ejecutiva "
            "('quiero hablar con alguien', 'que me atienda un abogado "
            "directamente') o ACEPTA la oferta humana previa ('sí, que me "
            "contacte la ejecutiva'). También aplica a mensajes con riesgo "
            "vital (violencia actual, peligro, autolesión): se derivan por "
            "este mismo flujo con urgency alta. No confundir con pedir que "
            "un abogado tome su caso. "
            "'agendar_asesoria': SOLO (a) pide explícitamente agendar/"
            "reservar una hora/cita/asesoría, o (b) ACEPTA una oferta de "
            "agendamiento hecha en el turno anterior ('sí, agendemos', "
            "'me interesa la asesoría, ¿para cuándo hay hora?'). Jamás "
            "inferirlo del relato del caso, del tono ni de la angustia. "
            "'respuestas_faq': pregunta informativa O describe/pregunta por "
            "su caso buscando orientación — con calma o con angustia — SIN "
            "pedido explícito de hora. Incluye 'necesito un abogado' o "
            "'quiero que tomen mi caso' sin pedido de hora: el FAQ orienta "
            "y sugiere la asesoría; si el cliente acepta, el turno siguiente "
            "ya entra como agendar_asesoria."
        )
    )
    category: CategoryLabel = Field(
        description=(
            "Tema principal, INCLUSO si solo pide info de precios. "
            "pension_alimentos: demanda, monto o pago de pensión (default en "
            "causas de alimentos). rebaja_pension: bajar el monto vigente. "
            "terminacion_pension: dejar de pagar. regimen_visitas: ver a los "
            "hijos. medidas_apremio: embargo, licencia retenida, registro de "
            "deudores. divorcio / compensacion_economica: según materia. "
            "consulta_general: preguntas sobre la FIRMA (precios, contacto, "
            "cobertura, horarios, proceso de contratación). otro: temas "
            "no-familia sin categoría propia (laboral, civil, penal), "
            "violencia intrafamiliar, o sin tema identificable. "
            "Agradecimientos: la categoría del caso mencionado; si no hay "
            "tema claro → consulta_general."
        )
    )
    riesgo: bool = Field(
        default=False,
        description=(
            "true si el mensaje menciona violencia actual, peligro inmediato, "
            "riesgo para menores o ideas de autolesión; false en cualquier "
            "otro caso. En caso de duda, false. (v6.8: solo metadata para "
            "logs/alertas del equipo; no activa rutas del grafo.)"
        ),
    )


class IntakeExtract(BaseModel):
    """DEPRECATED — v6.2: la extracción activa del intake usa IntakeDecision en
    graph/intake/contracts.py. Se mantiene solo por compatibilidad.
    """
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
        riesgo=False,
        reason=f"fallback por {motivo}",
    )


# --------------------------------------------------------------------------
# 6) Config de ejecución
# --------------------------------------------------------------------------
def make_config(thread_id: str) -> dict:
    """LangGraph persiste el estado por thread_id vía checkpointer."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
    }
