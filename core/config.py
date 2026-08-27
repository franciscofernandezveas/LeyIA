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
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)  # requerido por OpenAIEmbeddings
else:
    logger.warning("⚠️ DEMO_OPENAI_API_KEY u OPENAI_API_KEY no están configuradas.")

# ---- Calendly ----
CALENDLY_API_TOKEN: str = os.getenv("CALENDLY_API_TOKEN", "")
CALENDLY_EVENT_TYPE_URI: str = os.getenv("CALENDLY_EVENT_TYPE_URI", "")
CALENDLY_PUBLIC_LINK: str = os.getenv("CALENDLY_PUBLIC_LINK", "")
CALENDLY_USAR_BOOKING_API: bool = (
    os.getenv("CALENDLY_USAR_BOOKING_API", "false").lower() == "true"
)  

# Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
