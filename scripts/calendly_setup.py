# scripts/calendly_setup.py
"""Calendly doctor: valida token + scopes y descubre tu config real.

Uso:  python scripts/calendly_setup.py
"""
import base64
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://api.calendly.com"
TOKEN = os.environ.get("CALENDLY_API_TOKEN", "")

SCOPES_ESTE_SCRIPT = {"users:read", "event_types:read"}
SCOPES_ROADMAP = SCOPES_ESTE_SCRIPT | {
    "organizations:read",
    "scheduling_links:write",   # v2: links de un solo uso
    "availability:read",        # v3: horarios en el chat
    "scheduled_events:write",   # v3: POST /invitees (plan pago)
    "webhooks:write",           # fase 2: aviso invitee.created
}


def scopes_del_token(token: str) -> set[str]:
    """Decodifica el payload del PAT (JWT) localmente, sin verificar firma."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)          # padding base64url
        data = json.loads(base64.urlsafe_b64decode(payload.encode()))
        return set((data.get("scope") or "").split())
    except Exception:
        return set()


def main() -> None:
    if not TOKEN:
        raise SystemExit("❌ CALENDLY_API_TOKEN no está definido en .env")

    # ── Paso 0: diagnóstico del token (offline) ──────────────────────
    otorgados = scopes_del_token(TOKEN)
    print("🔑 Scopes del token:", ", ".join(sorted(otorgados)) or "(ilegible)")

    faltan = SCOPES_ESTE_SCRIPT - otorgados
    if faltan:
        raise SystemExit(
            f"\n❌ Al token le faltan scopes para este script: {sorted(faltan)}\n"
            "   Calendly → Integrations & Apps → API and Webhooks → nuevo token\n"
            f"   Incluye de una vez el roadmap completo:\n   {sorted(SCOPES_ROADMAP)}"
        )
    pendientes = SCOPES_ROADMAP - otorgados
    if pendientes:
        print(f"ℹ️  Para el roadmap completo faltarán: {sorted(pendientes)}\n")

    # ── Paso 1: identidad ────────────────────────────────────────────
    h = {"Authorization": f"Bearer {TOKEN}"}
    me = requests.get(f"{API}/users/me", headers=h, timeout=15)
    me.raise_for_status()
    u = me.json()["resource"]
    print(f"👤 {u['name']}  <{u['email']}>  ·  tz={u.get('timezone')}")

    # ── Paso 2: event types ──────────────────────────────────────────
    ets = requests.get(f"{API}/event_types", headers=h,
                       params={"user": u["uri"]}, timeout=15)
    ets.raise_for_status()
    collection = ets.json().get("collection", [])
    if not collection:
        raise SystemExit("⚠️ No hay Event Types. Crea uno en la web de Calendly primero.")

    for et in collection:
        print("\n" + "═" * 66)
        print(f"📅 {et['name']}  ({'activo' if et.get('active') else 'INACTIVO'})")
        print(f"   duración : {et.get('duration')} min")
        print(f"   URI API  : {et['uri']}")
        print(f"   link     : {et['scheduling_url']}")

        for loc in et.get("locations") or []:
            print(f"   📍 location → kind={loc.get('kind')}  {loc.get('location', '')}")

        preguntas = sorted(
            (q for q in (et.get("custom_questions") or []) if q.get("enabled")),
            key=lambda q: q["position"],
        )
        if preguntas:
            print("   📝 formulario (posición → param prefill aN):")
            for q in preguntas:
                req = "obligatoria" if q.get("required") else "opcional"
                print(f"      [pos={q['position']} → a{q['position'] + 1}?] {q['name']} ({req})")
        else:
            print("   📝 formulario sin preguntas personalizadas")

    print("\n" + "═" * 66)
    print("📋 Copia al .env el par del evento que usará el agente:\n")
    et = collection[0]
    print(f"CALENDLY_EVENT_TYPE_URI={et['uri']}")
    print(f"CALENDLY_PUBLIC_LINK={et['scheduling_url']}")
    print("CALENDLY_USAR_BOOKING_API=false")


if __name__ == "__main__":
    main()
