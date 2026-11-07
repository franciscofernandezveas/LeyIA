"""integrations/whatsapp/stt.py — Descarga de audios desde Meta Graph API
y transcripción con OpenAI Whisper (entrada; la respuesta siempre sale como texto)."""

import logging

import httpx
import openai

from .config import ACCESS_TOKEN, API_VERSION, OPENAI_API_KEY

logger = logging.getLogger("leyia-whatsapp.stt")

GRAPH_BASE = "https://graph.facebook.com"
WHISPER_MODEL = "whisper-1"
WHISPER_LANGUAGE = "es"  # fuerza español: mejor precisión y evita autodetección

_EXT_BY_MIME = {
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/mpeg": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
}
DEFAULT_EXT = "ogg"

_openai: openai.AsyncOpenAI | None = None


def _get_openai() -> openai.AsyncOpenAI:
    global _openai
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada; no se puede transcribir audio")
    if _openai is None:
        _openai = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai


async def _download_media(media_id: str) -> tuple[bytes, str, str]:
    """Devuelve (binario, filename, mime) de un adjunto de WhatsApp."""
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        # 1) media_id → metadata + URL temporal de descarga (lookaside)
        meta = await client.get(f"{GRAPH_BASE}/{API_VERSION}/{media_id}", headers=headers)
        meta.raise_for_status()
        info = meta.json()

        url = info.get("url")
        if not url:
            raise ValueError(f"Meta no devolvió URL de descarga (media_id={media_id})")

        mime = (info.get("mime_type") or "audio/ogg").split(";")[0].strip().lower()
        ext = _EXT_BY_MIME.get(mime, DEFAULT_EXT)

        # 2) Descargar el binario con el mismo token
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    return resp.content, f"wa-{media_id}.{ext}", mime


async def transcribe_whatsapp_audio(media_id: str) -> str:
    """Descarga el audio desde Meta y lo transcribe. Devuelve texto en español."""
    audio_bytes, filename, mime = await _download_media(media_id)

    transcript = await _get_openai().audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=(filename, audio_bytes, mime),
        language=WHISPER_LANGUAGE,
        prompt="Nota de voz de WhatsApp en español de Chile sobre una consulta legal.",
    )
    text = (transcript.text or "").strip()
    logger.info("Audio %s transcrito (%d bytes → %d chars)", media_id, len(audio_bytes), len(text))
    return text
