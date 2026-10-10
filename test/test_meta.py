import os
import httpx
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
DESTINO = "56920918225"  # tu número de prueba

if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
    print("❌ Faltan WHATSAPP_ACCESS_TOKEN o WHATSAPP_PHONE_NUMBER_ID")
    exit(1)

print(f"Token empieza con: {ACCESS_TOKEN[:10]}...")
print(f"Phone Number ID: {PHONE_NUMBER_ID}")

url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"

payload = {
    "messaging_product": "whatsapp",
    "recipient_type": "individual",
    "to": DESTINO,
    "type": "text",
    "text": {"preview_url": False, "body": "Prueba local LeyIA"},
}

resp = httpx.post(
    url,
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
    json=payload,
    timeout=15,
)

print(f"Status: {resp.status_code}")
print(resp.json())

if resp.status_code == 200:
    print("✅ Meta aceptó el envío")
elif resp.status_code == 401:
    print("❌ Token inválido o expirado. Genera uno nuevo.")
elif resp.status_code == 400:
    print("❌ Posible: destinatario no autorizado o phone_id incorrecto.")
else:
    print("❌ Error de Meta. Revisa el dashboard.")
