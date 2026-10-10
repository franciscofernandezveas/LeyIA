import os
import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()

WABA_IDS = [
    "1311154177707458",  # ONBOARDING
    "1417067149882283",  # ACTIVE
]

for waba_id in WABA_IDS:
    url = f"https://graph.facebook.com/v21.0/{waba_id}/phone_numbers"
    params = {
        "access_token": TOKEN,
        "fields": "id,display_phone_number,verified_name,status,quality_rating,account_id",
    }
    r = httpx.get(url, params=params, timeout=15)
    print(f"\n=== WABA {waba_id} /phone_numbers ===")
    print(f"Status HTTP: {r.status_code}")
    data = r.json()
    print(data)
    
    if r.status_code == 200 and data.get("data"):
        print("\n📞 Phone numbers encontrados:")
        for phone in data["data"]:
            print(f"  Phone Number ID: {phone.get('id')}")
            print(f"  Número: {phone.get('display_phone_number')}")
            print(f"  Nombre verificado: {phone.get('verified_name')}")
            print(f"  Estado: {phone.get('status')}")
            print(f"  Quality: {phone.get('quality_rating')}")
            print(f"  Account ID: {phone.get('account_id')}")
            print()
