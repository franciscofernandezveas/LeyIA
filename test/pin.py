# test/set_2fa_pin.py
import os, httpx
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
PHONE_ID = "1280521861820592"   # Phone Number ID de la WABA nueva
PIN = "280100"                 # ⇨ <--- ¡TU PIN DE 6 DÍGITOS! (guárdalo)

payload = {
    "messaging_product": "whatsapp",
    "pin": PIN
}

resp = httpx.post(
    f"https://graph.facebook.com/v21.0/{PHONE_ID}/register",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json=payload,
    timeout=15,
)

print("Status:", resp.status_code)
print(resp.json())
