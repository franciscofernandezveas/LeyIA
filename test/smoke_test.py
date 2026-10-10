import asyncio
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values, load_dotenv

# ============================================================
# CONFIGURACIÓN - edita solo esta línea:
TO = "56920918225"  # ← tu celular personal, sin + ni espacios
# ============================================================



# IDs ya confirmados en este proceso:
WABA_ID = "1311154177707458"   # WABA real (dueña del número)
APP_ID = "2018809342111648"    # la app nueva "Manzzo y cia IA"

# ── Carga robusta del .env.test ──────────────────────────────
ENV_PATH = Path(__file__).parent / ".env.test"
print("== archivo ==", ENV_PATH)
print("  existe:", ENV_PATH.exists())

vals = dotenv_values(ENV_PATH)
print("  claves encontradas:", list(vals.keys()))

load_dotenv(ENV_PATH)

ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "").strip().strip('"')
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "1360074010525604").strip().strip('"')
API_VERSION = os.environ.get("WHATSAPP_API_VERSION", "v23.0").strip().strip('"')
GRAPH = "https://graph.facebook.com"

# Validación temprana con mensajes claros
if not ACCESS_TOKEN:
    raise SystemExit("❌ WHATSAPP_ACCESS_TOKEN no se cargó. Revisa que .env.test contenga la línea WHATSAPP_ACCESS_TOKEN=EAA...")
if not ACCESS_TOKEN.startswith("EAA"):
    raise SystemExit("❌ El token no empieza con 'EAA' — pusiste otra cosa (¿quizás el App ID?). Genera el token del system user con la app 2018809342111648.")
if not TO or "X" in TO:
    raise SystemExit("❌ Falta tu número: edita TO al inicio del script (ej: \"56912345678\").")

print(f"  token recibido: {ACCESS_TOKEN[:12]}...{ACCESS_TOKEN[-6:]} (empieza con EAA ✔)\n")


async def main() -> None:
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    async with httpx.AsyncClient(base_url=GRAPH, timeout=15, headers=headers) as client:

        # ── 1. Verificar el token (qué es, qué scopes tiene) ────────────────
        r = await client.get(
            f"/{API_VERSION}/debug_token", params={"input_token": ACCESS_TOKEN}
        )
        d = r.json().get("data", {})
        print("== debug_token ==", r.status_code)
        print("   app_id esperado:", APP_ID, "| obtenido:", d.get("app_id"))
        print("   is_valid:", d.get("is_valid"))
        print("   scopes:", d.get("scopes"), "\n")

        # ── 2. Suscripción app ↔ WABA (necesaria para webhooks) ─────────────
        r = await client.get(f"/{API_VERSION}/{WABA_ID}/subscribed_apps")
        print("== subscribed_apps (GET) ==", r.status_code)
        print("  ", r.json(), "\n")

        apps = [a.get("id") for a in r.json().get("data", [])]
        if APP_ID not in apps:
            r = await client.post(f"/{API_VERSION}/{WABA_ID}/subscribed_apps")
            print("== subscribed_apps (POST suscribiendo) ==", r.status_code)
            print("  ", r.json(), "\n")
        else:
            print("   ✔ app ya suscrita a la WABA\n")

        # ── 3. ¿Vemos el número real? ───────────────────────────────────────
        r = await client.get(f"/{API_VERSION}/{WABA_ID}/phone_numbers")
        print("== phone_numbers ==", r.status_code)
        print("  ", r.json(), "\n")

        # ── 4. ¿Qué templates existen? ──────────────────────────────────────
        r = await client.get(
            f"/{API_VERSION}/{WABA_ID}/message_templates", params={"limit": 50}
        )
        print("== message_templates ==", r.status_code)
        data = r.json()
        names = [t.get("name") for t in data.get("data", [])]
        print("   templates:", names, "\n")

        # ── 5. Envío de prueba desde el número REAL ─────────────────────────
        if "hello_world" in names:
            template_name, language = "hello_world", "en_US"
        elif names:
            template_name, language = names[0], "es"
            print(f"   ⚠ hello_world no existe, usando '{template_name}'")
        else:
            print("   ⚠ No hay templates aprobados. Crea uno en:")
            print("     WhatsApp Manager → Herramientas → Plantillas → Crear")
            print("     (los de categoría 'Utilidad' se aprueban en minutos)\n")
            return

        r = await client.post(
            f"/{API_VERSION}/{PHONE_NUMBER_ID}/messages",
            json={
                "messaging_product": "whatsapp",
                "to": TO,
                "type": "template",
                "template": {"name": template_name, "language": {"code": language}},
            },
        )
        print("== send_template ==", r.status_code)
        print("  ", r.json())


if __name__ == "__main__":
    asyncio.run(main())
