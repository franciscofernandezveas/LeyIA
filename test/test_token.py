# test_token.py
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()

r = httpx.get(
    f"https://graph.facebook.com/v21.0/debug_token?input_token={TOKEN}&access_token={TOKEN}"
)
print(r.status_code)
print(r.json())
