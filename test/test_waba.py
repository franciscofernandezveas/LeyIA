import os
import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()

IDS = [
    "1311154177707458",
    "1417067149882283",
]

for phone_id in IDS:
    url = f"https://graph.facebook.com/v21.0/{phone_id}"
    params = {
        "access_token": TOKEN,
        "fields": "id,display_phone_number,verified_name,status,quality_rating",
    }
    r = httpx.get(url, params=params, timeout=15)
    print(f"\n=== Phone ID: {phone_id} ===")
    print(f"Status HTTP: {r.status_code}")
    print(r.json())
