import os
import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
PHONE_NUMBER_ID = "1280521861820592"  # ← nuevo

r = httpx.get(
    f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}",
    params={
        "access_token": TOKEN,
        "fields": "id,display_phone_number,verified_name,status,quality_rating",
    },
    timeout=15,
)

data = r.json()
print(f"Estado: {data.get('status')}")
print(f"Quality: {data.get('quality_rating')}")
print(data)
