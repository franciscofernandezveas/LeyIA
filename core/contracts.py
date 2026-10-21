"""Contratos centrales: estado del grafo, thread_id y esquemas estructurados.

v6.6 — FIX #11 (auditoría): tipado de intake_exit + eliminación de faq_stage.
  - Nuevo IntakeExitLabel = Literal["faq","pausa","handoff","end","cta_cierre"]
    reemplaza `intake_exit: str | None`. Elimina valores muertos que nunca
    fueron emitidos por ningún nodo ("agendar", "cta_escala", "booking") y
    que vivían solo en la documentación o en lectores defensivos.
  - route_post_intake en graph/nodes.py ahora hace matching exhaustivo con
    rama explícita para "cta_cierre" (antes caía por acoplamiento al
    fallback intake_completado=True).
  - Elimina faq_stage de AgentState: era dead code (siempre "respondiendo",
    ningún nodo lo leía, y su semántica era incorrecta cuando la acción era
    pedir_clarificacion). FAQStageLabel en graph/faq/contracts.py se
    mantiene como contrato interno del subagente.

v6.5 — FIX CRÍTICO sesión: session_id estaba declarado como anotación de
  módulo FUERA del TypedDict AgentState, por lo que no era un canal del
  estado. Ahora vive DENTRO de AgentState.

v6.4 — Handoff choice: bifurcación previa al intake que pregunta ficha breve
  vs contacto directo. Añade ROUTE_HANDOFF_CHOICE y estado handoff_choice_*
  en AgentState.

v6.3 — Cierre dual post-intake + fix EMAIL_RE:
  - EMAIL_RE acepta dominios con un solo punto (gmail.com, yahoo.cl).
  - intake_exit incluye nuevos valores "cta_cierre" y "cta_escala"
    para el cierre dual: Agendar / Hablar con abogada.
    (v6.6: "cta_escala" se elimina por ser dead code — nunca emitido.)

v6.2 — Refactor intake: ...
v6.1 — Booking signal/estado extendidos: ...
v6 — BOOKING migrado a subgrafo (graph/booking/): ...

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
# ⚠️ El intake v16+ emite intake_exit="cta_cierre" cuando la ficha se
#    completa con el cierre dual (Agendar / Hablar con la abogada); el
#    padre route_post_intake lo deriva a ROUTE_HANDOFF (v6.6 fija la rama
#    explícita — antes dependía del fallback intake_completado=True).
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
# Valores eliminados (nunca emitidos por ningún nodo):
#   "agendar"    → jamás se emitió; route_post_intake tenía rama muerta
#   "cta_escala" → documentado en v6.3 pero sin emisor
#   "booking"    → solo aparecía en handoff_humano.intake_ya_despidio (lectura)
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
    # v6.6: faq_stage ELIMINADO — era dead code (siempre "respondiendo",
    # ningún nodo lo leía, y su semántica era incorrecta cuando la acción
    # era pedir_clarificacion). FAQStageLabel vive en graph/faq/contracts.py
    # como contrato interno del subagente y no cruza al estado del padre.
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
    intake_exit: IntakeExitLabel | None    # v6.6: tipado (antes str | None)
    intake_resume: bool
    intake_oferta_qid: NotRequired[str | None]

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
    response_cta: NotRequired[dict | None]


# --------------------------------------------------------------------------
# 5) ESQUEMAS DE SALIDA ESTRUCTURADA DEL LLM
# --------------------------------------------------------------------------
class AnalisisResult(BaseModel):
    """Salida del clasificador unificado (nodo analyze_sentiment).

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
            "('desesperado', 'chato', 'injusto', garabatos, 'esperaba algo mejor de ustedes') o implícito "
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
            "deudores), o inmediatez exclamada ('YA'). "
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
            "(celulares, contador, psicólogo). "
            "'hablar_humano': pide ser atendido por una persona/ejecutiva "
            "('quiero hablar con alguien', 'que me atienda un abogado "
            "directamente') o ACEPTA la oferta humana previa ('sí, que me "
            "contacte la ejecutiva'). No confundir con pedir que un abogado "
            "tome su caso. "
            "'agendar_asesoria': (a) pide agendar/reservar hora; (b) pide "
            "explícitamente ayuda profesional ('necesito abogado', 'quiero "
            "que me ayuden', 'quiero que se termine'); (c) describe su caso "
            "con ANGUSTIA AGUDA pidiendo auxilio ('estoy desesperado, ¿qué "
            "hago?'); (d) pregunta por un servicio cubierto PERO no de "
            "familia (laboral/civil/penal) describiendo su caso → lead "
            "calificado. "
            "'respuestas_faq': pregunta informativa O describe/pregunta sobre "
            "su caso buscando orientación, SIN auxilio explícito ni angustia "
            "aguda (el agente orienta primero)."
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
            "no-familia sin categoría propia (laboral, civil, penal) o sin "
            "tema identificable. Agradecimientos: la categoría del caso "
            "mencionado; si no hay tema claro → consulta_general."
        )
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
