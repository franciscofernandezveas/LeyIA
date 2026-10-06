"""graph/faq/nodes.py — Acciones del sub-agente FAQ.

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

from core.contracts import ROUTE_INTAKE, AgentState
from core.llm import LLM
from core.rag import retrieve
from graph.nodes import _cfg, _guardar_ai, _recent_messages, _wa_link

logger = logging.getLogger(__name__)

# Botón nativo del CTA de agendamiento. El title llega como query del turno
# siguiente: en hilo cerrado post-intake lo reconoce la excepción 0-LLM de
# analyze_sentiment (_AGENDAR_CTA); en hilo abierto clasifica como
# agendar_asesoria sin costo de ambigüedad.
_BOTON_AGENDAR = {"kind": "buttons",
                  "buttons": [{"id": "agendar", "title": "Agendar una hora"}]}


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
    es_primer_contacto = not any(m.type == "ai" for m in state.get("messages", []))
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
            "disclosure": (atn["disclosure"] if es_primer_contacto
                           else "Eres el asistente virtual de Manzzo y Cía (ya te presentaste)."),
            "tono": tono,
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
    }


def pedir_clarificacion(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["faq_clarificacion"]
    _guardar_ai(state, response)
    return {"response": response,
            "response_bubbles": [response],
            "response_interactive": None,
            "messages": [AIMessage(content=response)]}


def sugerir_agendar(state: AgentState) -> AgentState:
    """El usuario mostró intención de agendar dentro del FAQ. Se envía el CTA
    (pregunta + botón nativo) y se espera su respuesta: el próximo turno
    clasifica agendar_asesoria (o, en hilo cerrado post-intake, la excepción
    0-LLM del padre) y entra booking. NO se re-rutea en el mismo turno."""
    atn = _cfg()["atencion"]
    response = atn["cta_agendar"]
    _guardar_ai(state, response)
    return {"response": response,
            "response_bubbles": [response],
            "response_interactive": _BOTON_AGENDAR,
            "messages": [AIMessage(content=response)]}


def derivar_a_ejecutiva(state: AgentState) -> AgentState:
    """El planner FAQ detectó pedido de humano dentro de la conversación.

    Señala ROUTE_INTAKE para que el padre desvíe el turno al subgrafo de
    intake (embudo oficial: ficha → handoff). No emite mensaje: la respuesta
    de ESTE turno la emite el intake (apertura o, si el cliente insiste,
    derivación con ficha parcial).

    Se limpia intake_resume: si el cliente venía de una pausa lateral y aquí
    pidió humano, el planner de intake debe EVALUAR el mensaje (→
    insiste_humano → derivar_parcial), no reanudar la ficha mecánicamente.
    """
    return {
        "route": ROUTE_INTAKE,
        "intake_resume": False,
        "response_interactive": None,
    }


def derivar_a_fuera_dominio(state: AgentState) -> AgentState:
    """Redirección amable por tema no legal. No re-rutea: este nodo ya
    responde; si la ficha estaba pausada (duda lateral), el padre la
    reanudará después vía route_post_faq."""
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(disclosure=atn["disclosure"])
    _guardar_ai(state, response)
    return {"response": response,
            "response_bubbles": [response],
            "response_interactive": None,
            "messages": [AIMessage(content=response)]}
