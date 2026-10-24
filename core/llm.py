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


def _build_llm(max_tokens: int) -> ChatOpenAI:
    """Falla con un mensaje claro si falta la key, en vez del OpenAIError crudo."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "❌ No se encontró API key de OpenAI. "
            "Define DEMO_OPENAI_API_KEY u OPENAI_API_KEY en el archivo .env "
            "de la raíz del proyecto (junto a core/, scripts/, etc.)."
        )

    if IS_REASONING or IS_NEW_GPT:
        kwargs = {
            "model": MODEL_NAME,
            "api_key": OPENAI_API_KEY,
            "max_completion_tokens": max_tokens,
        }
    else:
        kwargs = {
            "model": MODEL_NAME,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "api_key": OPENAI_API_KEY,
        }
    return ChatOpenAI(**kwargs)


# ------------------------------------------------------------------
# Tres cortes de token según el uso
# ------------------------------------------------------------------

# 1) Respuestas al cliente por WhatsApp. 900 tokens ≈ 600-700 caracteres,
#    suficiente para 1-2 burbujas. Forzamos concisión por hard cap.
LLM = _build_llm(max_tokens=900)

# 2) Salidas estructuradas (JSON): AnalisisResult, BookingDecision, etc.
#    Son objetos pequeños; 512 tokens sobran y evitamos pagar por espacio
#    de generación libre.
LLM_ESTRUCTURADO = _build_llm(max_tokens=512)

# 3) Textos internos que NO ve el cliente: resúmenes de crisis/handoff
#    para la abogada. Aquí sí puede ser más extenso.
LLM_LARGO = _build_llm(max_tokens=2048)


def with_structured_output(model: Type[BaseModel], method: str = "function_calling") -> ChatOpenAI:
    """
    Crea un LLM con salida estructurada.
    Usa LLM_ESTRUCTURADO para no desperdiciar tokens de salida en JSON.
    """
    return LLM_ESTRUCTURADO.with_structured_output(model, method=method)
