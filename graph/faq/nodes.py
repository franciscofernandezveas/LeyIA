"""graph/faq/nodes.py — Acciones del sub-agente FAQ.

v3.4 — Fix saludos reiterados:
  - El FAQ ya no decide por sí mismo cuándo presentarse: el sistema le
    entrega una instrucción de apertura computada ({instruccion_apertura}).
  - Todos los emisores de burbujas levantan `asistente_presentado=True` para
    que el resto del grafo sepa que la identidad ya fue anunciada.
  - fuera_dominio_message usa {apertura} mediante `_apertura_pres`.

v3.3 — ledger one-shot de sugerir_agendar.

v3.2 — Fix #2 (auditoría): derivar_a_ejecutiva señala ROUTE_HANDOFF_CHOICE
       (antes ROUTE_INTAKE), alineando la política con analyze_sentiment
       para intent=hablar_humano. Antes, el padre no honraba la señal y el
       cliente quedaba sin respuesta (route_post_faq caía en END por
       intake_resume=False). Ahora el nodo preguntar_preferencia_handoff
       emite la elección "ficha breve vs contacto directo" tras el END del
       subgrafo FAQ.

v3.1 — Importa helpers desde graph/utils.py (rompe ciclo graph.nodes↔faq).
v3 — Integración con intake v11 (multi-burbuja / one-shot):
  - Todos los emisores devuelven response_bubbles + response_interactive
    (None), para que ningún payload interactivo del intake sobreviva de un
    turno a otro por el canal.
  - sugerir_agendar suma botón nativo "Agendar una hora": mismo título que
    reconoce la excepción 0-LLM del padre en hilos cerrados y clasifica
    limpio como agendar_asesoria en hilos abiertos.
  - derivar_a_ejecutiva limpia response_interactive (no emite texto: la
    respuesta del turno la da el subgrafo de intake).

v2 — Integridad y consistencia: persistencia vía _guardar_ai; derivación sin
mentiras (señala ROUTE_INTAKE, el handoff real vive solo en handoff_humano);
pedir_clarificacion/faq_error en prompts.yaml; sugerir_agendar no re-rutea.
"""
import logging

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.contracts import ROUTE_HANDOFF_CHOICE, AgentState
from core.llm import LLM
from core.rag import retrieve
from graph.utils import _apertura_pres, _cfg, _guardar_ai, _recent_messages, _wa_link

logger = logging.getLogger(__name__)

# Botón nativo del CTA de agendamiento. El title llega como query del turno
# siguiente: en hilo cerrado post-intake lo reconoce la excepción 0-LLM de
# analyze_sentiment (_AGENDAR_CTA); en hilo abierto clasifica como
# agendar_asesoria sin costo de ambigüedad.
_BOTON_AGENDAR = {"kind": "buttons",
                  "buttons": [{"id": "agendar", "title": "Agendar una hora"}]}

_INSTRUCCION_PRIMERA_VEZ = (
    "Esta es tu primera respuesta del hilo. Preséntate en una sola línea "
    "como 'Soy LeyIA, la asistente virtual de Manzzo y Cía.', indica que "
    "estás a disposición y luego responde la consulta."
)

_INSTRUCCION_YA_PRESENTADO = (
    "El cliente ya te conoce y está en conversación contigo. Queda "
    "PROHIBIDO saludar, presentarte o decir quién eres. Responde "
    "directamente a lo sustantivo, sin frases de apertura."
)


def responder_pregunta(state: AgentState) -> AgentState:
    """RAG + LLM con historial. Devuelve respuesta al cliente."""
    atn = _cfg()["atencion"]

    try:
        docs = retrieve(state["query"], k=3)
    except Exception as e:
        logger.warning("[faq] retrieve falló: %s", e)
        docs = []
    context = "\n\n---\n\n".join(d.page_content for d in docs) or "Sin contexto."

    tono = atn["tonos"].get(state.get("sentiment", "neutro"), atn["tonos"]["neutro"])

    # ¿Ya nos presentamos? Flag primero; como defensa, historial AI.
    hay_historial_ai = any(
        getattr(m, "type", None) == "ai" for m in state.get("messages", [])
    )
    ya_presentado = bool(state.get("asistente_presentado")) or hay_historial_ai

    history = _recent_messages(state, k=7)[:-1]

    prompt = ChatPromptTemplate.from_messages([
        ("system", atn["faq_system_prompt"]),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    try:
        response = (prompt | LLM).invoke({
            "context": context,
            "query": state["query"],
            "history": history,
            "tono": tono,
            "instruccion_apertura": (
                _INSTRUCCION_PRIMERA_VEZ if not ya_presentado
                else _INSTRUCCION_YA_PRESENTADO
            ),
        }).content
    except Exception as e:
        logger.exception("[faq] LLM falló: %s", e)
        response = atn["faq_error"].format(wa_link=_wa_link())

    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": None,
        "context": [d.page_content for d in docs],
        "messages": [AIMessage(content=response)],
        "asistente_presentado": True,
    }


def pedir_clarificacion(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["faq_clarificacion"]
    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": None,
        "messages": [AIMessage(content=response)],
        "asistente_presentado": True,
    }


def sugerir_agendar(state: AgentState) -> AgentState:
    """El usuario mostró intención de agendar dentro del FAQ. Se envía el CTA
    (pregunta + botón nativo) y se levanta el ledger one-shot: si el planner
    vuelve a proponerlo en esta sesión, el supervisor lo degrada a
    responder_pregunta (la oferta queda embebida en la respuesta RAG)."""
    atn = _cfg()["atencion"]
    response = atn["cta_agendar"]
    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": _BOTON_AGENDAR,
        "cta_agenda_ofrecida": True,
        "messages": [AIMessage(content=response)],
        "asistente_presentado": True,
    }



def derivar_a_ejecutiva(state: AgentState) -> AgentState:
    """El planner FAQ detectó pedido de humano dentro de la conversación.

    Emite ROUTE_HANDOFF_CHOICE (v3.2): misma política que analyze_sentiment
    para intent=hablar_humano. Tras el END de este subgrafo, route_post_faq
    reconoce la señal y enruta a preguntar_preferencia_handoff, que ofrece
    la bifurcación "ficha breve (2 min) vs contacto directo" con botones
    nativos y anti-loop (handoff_choice_attempts).

    No emite mensaje: la respuesta de ESTE turno la emite
    preguntar_preferencia_handoff inmediatamente después.

    Se limpia intake_resume: si el cliente venía de una pausa lateral del
    intake y aquí pidió humano, el flujo pasa por la bifurcación de handoff
    en vez de reanudar la ficha mecánicamente.
    """
    return {
        "route": ROUTE_HANDOFF_CHOICE,
        "intake_resume": False,
        "response_interactive": None,
    }


def derivar_a_fuera_dominio(state: AgentState) -> AgentState:
    """Redirección amable por tema no legal. No re-rutea: este nodo ya
    responde; si la ficha estaba pausada (duda lateral), el padre la
    reanudará después vía route_post_faq."""
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(apertura=_apertura_pres(state))
    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": None,
        "messages": [AIMessage(content=response)],
        "asistente_presentado": True,
    }
