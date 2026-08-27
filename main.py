# main.py
"""CLI interactivo para probar el agente de soporte.

Roles que juegas en la terminal:
  1️⃣  CLIENTE  → escribes mensajes normalmente.
  2️⃣  OPERADOR → cuando el grafo se interrumpe (HITL), decides aprobar el
                 link de Calendly o redactar la respuesta del escalamiento.

Comandos:
  /nuevo   → crea un hilo nuevo (nueva conversación)
  /hilo    → muestra el thread_id actual
  /salir   → termina
"""
import uuid

from langgraph.types import Command

from core.contracts import make_config
from graph.builder import agent_graph

LINE = "─" * 64


# ---------------------------------------------------------------------------
# Rol: OPERADOR HUMANO
# ---------------------------------------------------------------------------
def resolver_hitl(payload: dict) -> dict:
    """Muestra la solicitud pendiente y pide la decisión del operador."""
    print(f"\n{LINE}")
    print("⏸️  INTERVENCIÓN HUMANA REQUERIDA (eres el operador)")
    print(f"   Tipo   : {payload.get('tipo')}")
    print(f"   Cliente: {payload.get('query')}")
    print(f"   Detalle: {payload.get('detalle')}")
    print(LINE)

    if "agendamiento" in str(payload.get("tipo", "")):
        print("  [A] Aprobar y enviar link de Calendly")
        print("  [R] Rechazar (puedes agregar una nota para el cliente)")
        op = input("operador> ").strip().lower()
        if op in ("a", "aprobar", ""):
            print("✅ Aprobado.\n")
            return {"aprobado": True}
        nota = input("Nota para el cliente (opcional): ").strip()
        print("❌ Rechazado.\n")
        return {"aprobado": False, "nota": nota}

    # escalamiento por sentimiento negativo
    print("  [D] Enviar mensaje por defecto")
    print("  [M] Escribir mensaje personalizado para el cliente")
    op = input("operador> ").strip().lower()
    if op in ("m", "mensaje"):
        mensaje = input("Mensaje del operador> ").strip()
        print("✉️  Enviando mensaje personalizado.\n")
        return {"aprobado": True, "mensaje": mensaje or None}
    print("📨 Enviando mensaje por defecto.\n")
    return {"aprobado": True, "mensaje": None}


# ---------------------------------------------------------------------------
# Rol: CLIENTE
# ---------------------------------------------------------------------------
def main():
    print(LINE)
    print("🤖  AGENTE MANZZO Y CÍA — consola de pruebas")
    print("    /nuevo (hilo nuevo) · /hilo (ver hilo) · /salir")
    print(LINE)

    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = make_config(thread_id)
    print(f"🧵 thread_id: {thread_id}")
    print("    (si vuelves a ingresar este id en otra sesión,")
    print("     recuperarás el historial gracias a la persistencia)\n")

    while True:
        try:
            query = input("tú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego.")
            break

        if not query:
            continue
        if query == "/salir":
            print("👋 Hasta luego.")
            break
        if query == "/hilo":
            print(f"🧵 thread_id actual: {thread_id}\n")
            continue
        if query == "/nuevo":
            thread_id = f"cli-{uuid.uuid4().hex[:8]}"
            config = make_config(thread_id)
            print(f"🔄 Hilo nuevo: {thread_id}\n")
            continue

        # ---- 1. Mensaje del cliente ----
        result = agent_graph.invoke(
            {"messages": [("human", query)], "query": query, "thread_id": thread_id},
            config=config,
        )

        # ---- 2. Resolver interrupciones HITL (pueden encadenarse) ----
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            decision = resolver_hitl(payload)
            result = agent_graph.invoke(Command(resume=decision), config=config)

        # ---- 3. Respuesta del agente + trazabilidad del routing ----
        print(f"\n🤖 {result['response']}")
        print(f"   [sentimiento: {result.get('sentiment')} · vía: {result.get('route')}]\n")


if __name__ == "__main__":
    main()
