# main.py
"""CLI interactiva para probar el agente de soporte de Manzzo y Cía.

Versión: v5 (integra nodo intake_lead para ficha proactiva de lead + caso).

Roles que juegas en la terminal:
  1️⃣  CLIENTE  → escribes mensajes normalmente.
  2️⃣  OPERADOR → si el grafo se interrumpe (HITL), decides aprobar el
                 link de Calendly o redactar la respuesta de rechazo.

Comandos:
  /nuevo            → crea un hilo nuevo (nueva conversación)
  /cargar <id>      → retoma un hilo persistido por su thread_id
  /hilo             → muestra el thread_id actual
  /salir            → termina
"""
import logging
import uuid

from langgraph.types import Command

from core.contracts import TipoHITL, make_config
from graph.builder import agent_graph

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

LINE = "─" * 64


# ---------------------------------------------------------------------------
# Helpers de interrupt / estado
# ---------------------------------------------------------------------------
def _pending_interrupts(config: dict):
    """Devuelve los interrupts pendientes de un checkpoint dado."""
    snap = agent_graph.get_state(config)
    if not snap or not snap.tasks:
        return []
    return [i for task in snap.tasks for i in (task.interrupts or [])]


def _estado_hilo(config: dict) -> dict:
    """Snapshot rápido del estado para mensajes informativos al cargar."""
    snap = agent_graph.get_state(config)
    if not snap:
        return {}
    return snap.values or {}


def _mostrar_respuesta(result: dict) -> None:
    """Imprime la respuesta del agente + metadatos de routing."""
    response = result.get("response") or "(sin respuesta generada)"
    print(f"\n🤖 {response}")
    print(
        f"   [sentimiento: {result.get('sentiment')} · "
        f"vía: {result.get('route')} · "
        f"intent: {result.get('intent')}]"
    )


# ---------------------------------------------------------------------------
# Rol: OPERADOR HUMANO
# ---------------------------------------------------------------------------
def resolver_hitl(payload: dict) -> dict:
    """Muestra la solicitud pendiente y pide la decisión del operador.

    El único HITL activo en v5 es la aprobación del link de agendamiento
    (solo cuando REQUIERE_APROBACION_AGENDAMIENTO = True en nodes.py).
    """
    print(f"\n{LINE}")
    print("⏸️  INTERVENCIÓN HUMANA REQUERIDA (eres el operador)")
    print(f"   Tipo   : {payload.get('tipo')}")
    print(f"   Cliente: {payload.get('query')}")
    print(f"   Lead   : {payload.get('lead')}")
    print(f"   Categoría: {payload.get('categoria')} · Urgencia: {payload.get('urgencia')}")
    print(f"   Detalle: {payload.get('detalle')}")
    print(LINE)

    tipo = payload.get("tipo")

    if tipo == TipoHITL.APROBACION_AGENDAMIENTO.value:
        print("  [a] Aprobar y enviar link de Calendly")
        print("  [r] Rechazar (puedes agregar una nota para el cliente)")
        op = input("operador> [a/r]: ").strip().lower()

        if op == "a":
            print("✅ Aprobado.\n")
            return {"aprobado": True}

        nota = input("Nota para el cliente (opcional): ").strip()
        print("❌ Rechazado.\n")
        return {"aprobado": False, "nota": nota}

    # Fallback defensivo: si en el futuro se agrega otro tipo de HITL,
    # el operador puede verlo y responder sin romper el contrato.
    print(f"  [r] Rechazar HITL de tipo desconocido: {tipo}")
    return {"aprobado": False, "nota": "HITL no reconocido por el operador"}


def drenar_interrupts(config: dict, *, mostrar_si_vacio: bool = False) -> dict | None:
    """Resuelve todos los interrupts pendientes de un hilo.

    Devuelve el último resultado del grafo tras cerrar los HITLs, o None
    si no había interrupts pendientes. Cada resultado intermedio ya se
    imprime aquí (para que el operador vea la respuesta post-aprobación)."""
    ultimo_resultado = None
    vistos = False

    while True:
        interrupts = _pending_interrupts(config)
        if not interrupts:
            break
        vistos = True
        for intr in interrupts:
            decision = resolver_hitl(intr.value)
            try:
                ultimo_resultado = agent_graph.invoke(
                    Command(resume=decision), config=config
                )
            except Exception as e:
                logging.exception("Error al reanudar HITL: %s", e)
                print("⚠️  No se pudo reanudar el HITL. Revisa logs.")
                continue
            _mostrar_respuesta(ultimo_resultado)

    if mostrar_si_vacio and not vistos:
        return None

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

    # Si arrancamos con un thread que ya tenía un HITL pendiente, lo drenamos.
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
            thread_id = partes[1].strip()
            config = make_config(thread_id)

            print(f"🧵 Hilo cargado: {thread_id}")
            st = _estado_hilo(config)
            if st.get("intake_activo"):
                print("   ℹ️  Este hilo tiene una ficha de intake en curso.")
            elif st.get("recolectando_datos_agenda"):
                print("   ℹ️  Este hilo estaba capturando datos de agendamiento.")
            elif st.get("esperando_confirmacion_booking"):
                print("   ℹ️  Este hilo estaba esperando confirmación de Calendly.")

            drenar_interrupts(config)
            print()
            continue

        # ------------------------------------------------------------------
        # 1. Drenar HITLs viejos antes de meter un mensaje nuevo.
        #    Si el hilo estaba interrumpido e ignoramos esto, LangGraph
        #    lanzaría un error al recibir input normal en vez de Command(resume).
        # ------------------------------------------------------------------
        drenar_interrupts(config)

        # ------------------------------------------------------------------
        # 2. Enviar el mensaje del cliente.
        #    Nota: ya NO pasamos "query": query. receive_message lo deriva del
        #    último mensaje humano. thread_id sí se mantiene en el state porque
        #    varios nodos lo usan (handoff, link, expediente).
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 3. Si el último nodo disparó un HITL nuevo, lo resolvemos.
        #    Si no, mostramos la respuesta normal del grafo.
        #    Importante: drenar_interrupts() ya imprime la respuesta post-HITL,
        #    por lo que no volvemos a imprimir `result` en ese caso.
        # ------------------------------------------------------------------
        if _pending_interrupts(config):
            drenar_interrupts(config)
        else:
            _mostrar_respuesta(result)
        print()


if __name__ == "__main__":
    main()
