# core/llm.py
"""Inicialización del LLM.

La API key se resuelve UNA sola vez en core/config.py (con load_dotenv()
incluido), y todos los módulos la toman de ahí. No leer os.environ aquí.
"""
import logging
import os
from typing import Type

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config import OPENAI_API_KEY  # ← fuente única de verdad

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Modelo configurable por variable de entorno
# ------------------------------------------------------------------
MODEL_NAME = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip().lower()

# Modelos de razonamiento (o-series) y GPT más recientes no aceptan
# 'max_tokens' ni 'temperature' → usan 'max_completion_tokens'.
IS_REASONING = any(name in MODEL_NAME for name in ["o1", "o3", "o4", "o5"])
IS_NEW_GPT = any(name in MODEL_NAME for name in ["gpt-4.1", "gpt-5"])

if IS_REASONING or IS_NEW_GPT:
    llm_kwargs = {
        "model": MODEL_NAME,
        "api_key": OPENAI_API_KEY,
        "max_completion_tokens": 2048,
    }
else:
    llm_kwargs = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 2048,
        "api_key": OPENAI_API_KEY,
    }


def _build_llm() -> ChatOpenAI:
    """Falla con un mensaje claro si falta la key, en vez del OpenAIError crudo."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "❌ No se encontró API key de OpenAI. "
            "Define DEMO_OPENAI_API_KEY u OPENAI_API_KEY en el archivo .env "
            "de la raíz del proyecto (junto a core/, scripts/, etc.)."
        )
    return ChatOpenAI(**llm_kwargs)


LLM = _build_llm()


def with_structured_output(model: Type[BaseModel], method: str = "function_calling") -> ChatOpenAI:
    """
    Crea un LLM con salida estructurada.
    Usa function_calling para evitar problemas con structured output de OpenAI.
    """
    return LLM.with_structured_output(model, method=method)
