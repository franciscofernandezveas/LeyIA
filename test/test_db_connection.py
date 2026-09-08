from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

from core.db_client import upsert_conversation, insert_message

upsert_conversation(thread_id="test-local", channel="cli", status="abierto")
insert_message(thread_id="test-local", role="human", content="mensaje de prueba")

print("✅ OK — revisa Supabase → Table Editor → conversations y messages")
