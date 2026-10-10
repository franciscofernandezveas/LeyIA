"""Contratos centrales: estado del grafo, thread_id y esquemas estructurados.

v6.2 — Refactor intake:
  - EMAIL_RE más estricto (rechaza TLD de 1 carácter y dominios mal formados).
  - IntakeExtract marcado como deprecated (la extracción activa del intake ahora
    usa IntakeDecision en graph/intake/contracts.py).
  - Limpieza de comentarios duplicados en AgentState; intake_exit documenta
    los nuevos valores "agendar" y "end".

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
# Regla v6:
#   intent=fuera_de_dominio → respuesta_fuera_dominio
#   intent=hablar_humano    → intake_lead (ficha) → handoff_humano
#   intent=agendar_asesoria → SUBGRAFO de booking (graph/booking/) en ROUTE_AGENDAR
#   resto                   → respuestas_faq (atención/orientación RAG)
# ⚠️ El sentimiento NO routea: solo modula el tono de respuestas_faq.
# ⚠️ Las 5 rutas DEBEN estar registradas como nodos en graph/builder.py
#    (test de contrato: test_toda_ruta_tiene_nodo). ROUTE_AGENDAR es un
#    subgrafo compilado registrado como nodo — el invariante no se toca.
# ⚠️ Intake v13+ emite intake_exit="agendar" cuando la ficha se cierra con
#    el CTA de agendamiento; el padre route_post_intake lo deriva a
#    ROUTE_AGENDAR.
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
# v6.2: más estricto — requiere al menos un subdominio y TLD de 2+ caracteres,
# rechazando casos como test@gmail.c o usuario@dominio-.
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+){1,}\.[a-zA-Z]{2,}$")


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
    case_category: CategoryLabel
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
    intake_exit: str | None                   # None | "faq" | "pausa" | "handoff" |
                                              # "agendar" | "end"
    intake_resume: bool                       # tras FAQ lateral, intake re-pregunta pendiente

    # --- Escalamiento / cierre (nodo handoff_humano) ---
    closed: bool                              # hilo derivado/cerrado
    summary: str                              # resumen del caso para la ejecutiva
    notificacion_pendiente: bool              # WhatsApp falló; reintentar por job

    response_bubbles: NotRequired[list[str]]        # burbujas separadas (fallback: response)
    response_interactive: NotRequired[dict | None]  # {"kind": "buttons"|"list", ...} one-shot
    intake_oferta_qid: NotRequired[str | None]      # campo para el que ya se ofreció salida

# --------------------------------------------------------------------------
# 5) ESQUEMAS DE SALIDA ESTRUCTURADA DEL LLM
# --------------------------------------------------------------------------
class AnalisisResult(BaseModel):
    """Salida del clasificador unificado (nodo analyze_sentiment).

    v7 — 'reason' ANTES de las etiquetas (el LLM genera en orden de campos:
    razona primero, clasifica después) + descriptions operativas calibradas
    con el golden set (umbrales explícitos del dominio Manzzo).

    ⚠️ Estas descriptions viajan al LLM: deben calzar con
    classification.system_prompt de core/prompts.yaml (v3.0.0+).
    ⚠️ El clasificador NO routea: la ruta la deriva compute_route()
    (regla determinista). Aquí solo se etiqueta.
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
            "('desesperado', 'chato', 'injusto', garabatos, 'ayuda') o implícito "
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
    graph/intake/contracts.py. Se mantiene solo por compatibilidad con tests o
    scripts legacy que aún importen este esquema; no usar en nodos nuevos.

    Mapeo texto libre → campos de la ficha de intake (nodo intake_lead).
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
