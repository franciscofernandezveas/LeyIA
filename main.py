# main.py
"""CLI interactiva para probar el agente de soporte de Manzzo y Cía.

Versión: v6.2.1 — import robusto de langgraph.Command para entornos
con langgraph<0.2 o builds parciales. Alineada con grafo padre v16.1
e intake v9 / booking por booking_stage.

Comandos:
  /nuevo            → crea un hilo nuevo
  /cargar <id>      → retoma un hilo persistido por su thread_id
  /hilo             → muestra el thread_id actual
  /salir            → termina
"""
# ⚠️ ORDEN CRÍTICO: load_dotenv() antes de importar core.*/graph.*,
# porque db_client.py captura DATABASE_URL en tiempo de importación.
from dotenv import load_dotenv
load_dotenv()

import logging
import uuid

# langgraph>=0.2 expone Command en langgraph.types; algunas builds/versions
# anteriores lo tienen en langgraph.types.command. Pruebo ambas y, si fallan,
# doy un mensaje útil en vez de un import críptico.
from langgraph.types import Command


from core.contracts import TipoHITL, make_config
from graph.builder import agent_graph

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

LINE = "─" * 64
MAX_RONDAS_RESUME = 5


# ---------------------------------------------------------------------------
# Helpers de interrupt / estado
# ---------------------------------------------------------------------------
def _pending_interrupts(config: dict):
    snap = agent_graph.get_state(config)
    if not snap or not snap.tasks:
        return []
    return [i for task in snap.tasks for i in (task.interrupts or [])]


def _estado_hilo(config: dict) -> dict:
    snap = agent_graph.get_state(config)
    if not snap:
        return {}
    return snap.values or {}


def _resumen_subflujo(valores: dict) -> str | None:
    if valores.get("closed"):
        return ("Hilo cerrado/escalado — los próximos mensajes van a "
                "seguimiento (handoff), salvo el CTA de agendar.")
    if valores.get("booking_stage"):
        if valores.get("slots_propuestos"):
            return "Este hilo estaba eligiendo un horario propuesto."
        return "Este hilo estaba capturando datos de agendamiento."
    if valores.get("intake_activo") and not valores.get("intake_completado"):
        if valores.get("intake_exit") == "pausa":
            return ("Ficha de intake EN PAUSA — tu próximo mensaje la "
                    "reanuda automáticamente (no es un interrupt).")
        return "Este hilo tiene una ficha de intake en curso."
    return None


def _mostrar_respuesta(result: dict) -> None:
    response = result.get("response") or "(sin respuesta generada)"
    print(f"\n🤖 {response}")
    print(
        f"   [{result.get('sentiment')}/{result.get('urgency')} · "
        f"{result.get('intent')}/{result.get('category')} → {result.get('route')}]"
    )
    if reason := result.get("clf_reason"):
        print(f"   · {reason}")


# ---------------------------------------------------------------------------
# Rol: OPERADOR HUMANO
# ---------------------------------------------------------------------------
def resolver_hitl(payload: dict) -> dict:
    print(f"\n{LINE}")
    print("⏸️  INTERVENCIÓN HUMANA REQUERIDA (eres el operador)")
    print(f"   Tipo   : {payload.get('tipo')}")
    print(f"   Cliente: {payload.get('query')}")
    print(f"   Lead   : {payload.get('lead')}")
    print(f"   Categoría: {payload.get('categoria')} · Urgencia: {payload.get('urgencia')}")
    print(f"   Detalle: {payload.get('detalle')}")

    conocidas = {"tipo", "query", "lead", "categoria", "urgencia", "detalle"}
    extras = {k: v for k, v in payload.items() if k not in conocidas}
    if extras:
        print(f"   Extra  : {extras}")
    print(LINE)

    tipo = payload.get("tipo")

    if tipo == TipoHITL.APROBACION_AGENDAMIENTO.value:
        print("  [a] Aprobar y crear evento en Google Calendar")
        print("  [r] Rechazar (puedes agregar una nota para el cliente)")
        op = input("operador> [a/r]: ").strip().lower()

        if op == "a":
            print("✅ Aprobado.\n")
            return {"aprobado": True}

        nota = input("Nota para el cliente (opcional): ").strip()
        print("❌ Rechazado.\n")
        return {"aprobado": False, "nota": nota}

    print(f"  [r] Rechazar HITL de tipo desconocido: {tipo}")
    return {"aprobado": False, "nota": "HITL no reconocido por el operador"}


def drenar_interrupts(config: dict, pendientes: list | None = None) -> dict | None:
    ultimo_resultado = None
    rondas = 0
    interrupts = pendientes if pendientes is not None else _pending_interrupts(config)

    while interrupts:
        rondas += 1
        if rondas > MAX_RONDAS_RESUME:
            logging.warning(
                "drenar_interrupts abortó tras %d rondas; quedan %d interrupt(s) sin resolver",
                MAX_RONDAS_RESUME, len(interrupts),
            )
            print(
                "⚠️  No se pudo resolver un HITL tras varios intentos. "
                "El hilo queda en pausa — revisa logs y reintenta más tarde.\n"
            )
            break

        for intr in interrupts:
            decision = resolver_hitl(intr.value)
            try:
                ultimo_resultado = agent_graph.invoke(
                    Command(resume=decision), config=config
                )
            except Exception as e:
                logging.exception("Error al reanudar HITL: %s", e)
                print("⚠️  No se pudo reanudar el HITL (sigue pendiente). Revisa logs.")
                continue
            _mostrar_respuesta(ultimo_resultado)

        interrupts = _pending_interrupts(config)

    return ultimo_resultado


# ---------------------------------------------------------------------------
# Rol: CLIENTE
# ---------------------------------------------------------------------------
def main():
    print(LINE)
    print("🤖  AGENTE MANZZO Y CÍA — consola de pruebas")
    print("    /nuevo · /cargar <id> · /hilo · /salir")
    print(LINE)

    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = make_config(thread_id)
    print(f"🧵 thread_id: {thread_id}")
    print("    Usa /cargar <thread_id> para retomar una conversación.\n")

    drenar_interrupts(config)

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
        if query.startswith("/cargar"):
            partes = query.split(maxsplit=1)
            if len(partes) < 2:
                print("⚠️  Uso: /cargar <thread_id>\n")
                continue

            candidato_id = partes[1].strip()
            candidato_config = make_config(candidato_id)
            valores = _estado_hilo(candidato_config)

            if not valores:
                print(
                    f"⚠️  '{candidato_id}' no existe en el checkpointer. "
                    "Si continúas, se creará como hilo nuevo al primer mensaje."
                )
                confirma = input("¿Cargar de todos modos? [s/N]: ").strip().lower()
                if confirma != "s":
                    print("   Cancelado. Sigues en tu hilo actual.\n")
                    continue

            thread_id = candidato_id
            config = candidato_config

            print(f"🧵 Hilo cargado: {thread_id}")
            if aviso := _resumen_subflujo(valores):
                print(f"   ℹ️  {aviso}")

            drenar_interrupts(config)
            print()
            continue

        drenar_interrupts(config)

        try:
            result = agent_graph.invoke(
                {"messages": [("human", query)], "thread_id": thread_id},
                config=config,
            )
        except Exception as e:
            logging.exception("Error al invocar el grafo: %s", e)
            print(
                "\n⚠️  El agente tuvo un problema técnico procesando tu mensaje. "
                "Inténtalo de nuevo o escribe /nuevo para reiniciar.\n"
            )
            continue

        pendientes = _pending_interrupts(config)
        if pendientes:
            drenar_interrupts(config, pendientes=pendientes)
        else:
            _mostrar_respuesta(result)
        print()


if __name__ == "__main__":
    main()
