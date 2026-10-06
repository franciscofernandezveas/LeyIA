import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION = "faqs"
CHECKPOINT_DB = BASE_DIR / "checkpoints.sqlite"
KNOWLEDGE_PATH = BASE_DIR / "data" / "knowledge.md"

# --------------------------------------------------------------------------
# OpenAI API Key (fuente única para todo el proyecto)
# --------------------------------------------------------------------------
OPENAI_API_KEY = (
    os.environ.get("DEMO_OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)
if OPENAI_API_KEY:
    OPENAI_API_KEY = OPENAI_API_KEY.strip().strip('"').strip("'")
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
else:
    logger.warning("⚠️ DEMO_OPENAI_API_KEY u OPENAI_API_KEY no están configuradas.")

# --------------------------------------------------------------------------
# Calendly
# --------------------------------------------------------------------------
CALENDLY_API_TOKEN: str = os.getenv("CALENDLY_API_TOKEN", "")
CALENDLY_EVENT_TYPE_URI: str = os.getenv("CALENDLY_EVENT_TYPE_URI", "")
CALENDLY_PUBLIC_LINK: str = os.getenv("CALENDLY_PUBLIC_LINK", "")
CALENDLY_USAR_BOOKING_API: bool = (
    os.getenv("CALENDLY_USAR_BOOKING_API", "false").lower() == "true"
)

# --------------------------------------------------------------------------
# Twilio + HITL
# --------------------------------------------------------------------------
TWILIO_ACCOUNT_SID: str = (os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
TWILIO_AUTH_TOKEN: str = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
TWILIO_WHATSAPP_FROM: str = (os.getenv("TWILIO_WHATSAPP_FROM") or "whatsapp:+14155238886").strip()
HITL_API_KEY: str = (os.getenv("HITL_API_KEY") or "").strip()

# Si está en "true"/"1"/"yes"/"on", el webhook validará la firma X-Twilio-Signature.
# En desarrollo local puedes desactivarlo, pero en producción debe estar activado.
TWILIO_VALIDATE_SIGNATURE: bool = (
    os.getenv("TWILIO_VALIDATE_SIGNATURE", "true").lower() in ("1", "true", "yes", "on")
)

# Advertencia si faltan credenciales críticas para WhatsApp/HITL
if not all((TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, HITL_API_KEY)):
    logger.warning(
        "⚠️ Faltan variables de entorno de Twilio o HITL_API_KEY. "
        "El webhook de WhatsApp y/o el endpoint HITL no funcionarán correctamente."
    )
