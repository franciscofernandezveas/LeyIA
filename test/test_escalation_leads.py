# test_escalation_leads.py
from dotenv import load_dotenv
load_dotenv()  # ← OBLIGATORIO antes de importar core.db_client

import logging
logging.basicConfig(level=logging.INFO)

from core.db_client import upsert_conversation, upsert_lead, insert_escalation

THREAD = "test-leads-001"

upsert_conversation(THREAD, channel="cli", status="abierto")

upsert_lead(
    THREAD,
    {
        "nombre": "Prueba Test",
        "email": "prueba@test.cl",
        "telefono": "+56912345678",
        "situacion_actual": "despido injustificado",
        "etapa_proceso": "sin iniciar",
        "consentimiento_datos": True,
    },
    category="laboral",
    completed=True,
)

insert_escalation(THREAD, summary="Lead de prueba completo", notificado_whatsapp=False)

print("Ejecutado. Revisa arriba si aparece '[db] ... falló'")
