from langgraph.graph import StateGraph, START, END

from core.contracts import AgentState
from core.persistence import checkpointer
from graph.nodes import (
    agendar_asesoria, analyze_sentiment, handoff_humano,
    receive_message, respuestas_faq, route_query,
)


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("receive_message", receive_message)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("agendar_asesoria", agendar_asesoria)
    workflow.add_node("respuestas_faq", respuestas_faq)
    workflow.add_node("handoff_humano", handoff_humano)

    workflow.add_edge(START, "receive_message")
    workflow.add_edge("receive_message", "analyze_sentiment")
    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_query,
        {
            "agendar_asesoria": "agendar_asesoria",
            "respuestas_faq": "respuestas_faq",
            "handoff_humano": "handoff_humano",
        },
    )
    workflow.add_edge("agendar_asesoria", END)
    workflow.add_edge("respuestas_faq", END)
    workflow.add_edge("handoff_humano", END)

    # El checkpointer habilita persistencia + interrupt/resume (HITL)
    return workflow.compile(checkpointer=checkpointer)


agent_graph = build_graph()

# Visualización (como en tu ejemplo):
# from IPython.display import display, Image
# display(Image(agent_graph.get_graph(xray=True).draw_mermaid_png()))
