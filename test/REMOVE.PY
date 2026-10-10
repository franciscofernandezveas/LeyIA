# test/remove_from_old_waba.py
import os, httpx
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
OLD_PHONE_ID = "1360074010525604"          # Phone Number ID que pertenece a la WABA vieja

url = f"https://graph.facebook.com/v21.0/{OLD_PHONE_ID}/deregister"
resp = httpx.post(
    url,
    headers={"Authorization": f"Bearer {TOKEN}"},
    timeout=15,
)

print("Status:", resp.status_code)
print(resp.json())
