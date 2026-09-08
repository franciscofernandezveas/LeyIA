"""graph/faq/nodes.py — Acciones del sub-agente FAQ."""
import logging

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.contracts import ROUTE_AGENDAR, ROUTE_HANDOFF, ROUTE_FUERA_DOMINIO, AgentState
from core.llm import LLM
from core.rag import retrieve
from graph.nodes import _cfg, _recent_messages

logger = logging.getLogger(__name__)


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
        response = ("Disculpa, tuve un problema técnico procesando tu mensaje 🙏. "
                    "¿Podrías repetírmelo en unos minutos?")

    return {
        "response": response,
        "context": [d.page_content for d in docs],
        "messages": [AIMessage(content=response)],
    }


def pedir_clarificacion(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = ("No estoy seguro de haber entendido bien tu consulta. "
                "¿Podrías darme un poco más de contexto? Por ejemplo, ¿se trata "
                "de pensiones de alimentos, divorcio, visitas o alguna otra "
                "materia familiar?")
    return {"response": response, "messages": [AIMessage(content=response)]}


def sugerir_agendar(state: AgentState) -> AgentState:
    """El usuario mostró intención de agendar dentro del FAQ. Se devuelve
    al grafo padre con la señal de que debe ir a booking."""
    atn = _cfg()["atencion"]
    response = atn["cta_agendar"]
    return {
        "response": response,
        "route": ROUTE_AGENDAR,  # señal para el grafo padre (opcional, según diseño)
        "messages": [AIMessage(content=response)],
    }


def derivar_a_ejecutiva(state: AgentState) -> AgentState:
    """Devuelve al grafo padre la señal de handoff."""
    atn = _cfg()["atencion"]
    response = atn["handoff_message"].format(
        disclosure=atn["disclosure"],
        whatsapp_ejecutiva="",  # se rellena en handoff_humano
        thread_id=state.get("thread_id", ""),
    )
    return {
        "response": response,
        "route": ROUTE_HANDOFF,
        "messages": [AIMessage(content=response)],
    }


def derivar_a_fuera_dominio(state: AgentState) -> AgentState:
    """Devuelve al grafo padre la señal de fuera de dominio."""
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(disclosure=atn["disclosure"])
    return {
        "response": response,
        "route": ROUTE_FUERA_DOMINIO,
        "messages": [AIMessage(content=response)],
    }
