"""graph/intake/utils.py — Helpers internos del subgrafo de intake."""

import logging

from langchain_core.messages import AIMessage

from graph.utils import WA_TEXTO_MAX

logger = logging.getLogger(__name__)


def _truncar_burbuja(texto: str, max_len: int = WA_TEXTO_MAX) -> str:
    if len(texto) <= max_len:
        return texto
    return texto[:max_len - 1].rstrip() + "…"


def _pack(burbujas: list[str], interactive: dict | None = None) -> dict:
    burbujas = [_truncar_burbuja(b) for b in burbujas if b]
    msg = "\n\n".join(burbujas)
    return {
        "response": msg,
        "response_bubbles": burbujas,
        "response_interactive": interactive,
        "messages": [AIMessage(content=msg)],
    }


def _evento_intake(evento: str, state: dict, **kwargs) -> None:
    logger.info(
        "intake.evento",
        extra={
            "thread_id": state.get("thread_id"),
            "evento": evento,
            **kwargs,
        },
    )
