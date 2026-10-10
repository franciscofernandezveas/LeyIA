# test/request_code_verify.py
import os, httpx
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
PHONE_ID = "1280521861820592"          # Phone Number ID de la WABA que vas a usar

payload = {
    "messaging_product": "whatsapp",
    "code_method": "SMS",   # Cambia a "VOICE" si prefieres llamada
    "language": "es"
}

resp = httpx.post(
    f"https://graph.facebook.com/v21.0/{PHONE_ID}/request_code",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json=payload,
    timeout=15,
)

print("Status:", resp.status_code)
print(resp.json())
