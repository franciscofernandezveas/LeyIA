# main.py
"""CLI interactiva para probar el agente de soporte de Manzzo y Cía.

Versión: v6.1 (Google Calendar: agenda directa, sin Calendly).

Cambios respecto a v6:
  a) drenar_interrupts(): se elimina el parámetro muerto `mostrar_si_vacio`.
  b) drenar_interrupts(): límite de rondas de reanudación — si el nodo que
     procesa el resume falla siempre (ej. Calendar API caída), el operador
     ya no queda atrapado en un loop infinito.
  c) /cargar valida que el hilo exista en el checkpointer y avisa si no.
  d) El chequeo de interrupts post-invoke se pasa como argumento a
     drenar_interrupts() para no golpear dos veces el checkpointer.

Roles que juegas en la terminal:
  1️⃣  CLIENTE  → escribes mensajes normalmente.
  2️⃣  OPERADOR → si el grafo se interrumpe (HITL), decides aprobar la
                 creación del evento en Google Calendar o redactar la
                 respuesta de rechazo.

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

# Máximo de rondas de reanudación por llamada a drenar_interrupts().
# Protege contra un nodo de resume que falle de forma permanente
# (ej. Google Calendar API caída): sin esto, el operador quedaría
# respondiendo el mismo HITL para siempre.
MAX_RONDAS_RESUME = 5


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

    El único HITL activo es la aprobación de agendamiento
    (solo cuando REQUIERE_APROBACION_AGENDAMIENTO = True en nodes.py).
    En la configuración por defecto el evento se crea directo y este HITL
    no se dispara.
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
        print("  [a] Aprobar y crear evento en Google Calendar")
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


def drenar_interrupts(config: dict, pendientes: list | None = None) -> dict | None:
    """Resuelve todos los interrupts pendientes de un hilo.

    Args:
        config:     config del hilo (con thread_id).
        pendientes: lista de interrupts ya consultada por el llamador, para
                    ahorrar un round-trip al checkpointer. Si es None, se
                    consulta aquí.

    Devuelve el último resultado del grafo tras cerrar los HITLs, o None
    si no había interrupts pendientes. Cada resultado intermedio ya se
    imprime aquí (para que el operador vea la respuesta post-aprobación).

    Protección anti loop-infinito: si tras MAX_RONDAS_RESUME rondas de
    reanudación sigue habiendo interrupts (un nodo de resume que falla
    siempre), se aborta y el hilo queda en pausa para revisión manual.
    """
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

    # Si arrancamos con un thread que ya tenía un HITL pendiente, lo drenamos.
    # (Con un UUID nuevo esto es un no-op, pero lo dejamos por robustez.)
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
            st = _estado_hilo(candidato_config)

            # Validación: typo o id inexistente → no cambiar de hilo a ciegas.
            if not st:
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
            if st.get("intake_activo"):
                print("   ℹ️  Este hilo tiene una ficha de intake en curso.")
            elif st.get("recolectando_datos_agenda"):
                print("   ℹ️  Este hilo estaba capturando datos de agendamiento.")
            elif st.get("esperando_eleccion_horario"):
                print("   ℹ️  Este hilo estaba eligiendo un horario disponible.")

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
        #    varios nodos lo usan (handoff, expediente).
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
        # 3. Si el último nodo disparó un HITL nuevo, lo resolvemos pasando la
        #    lista ya consultada (ahorra un segundo get_state al checkpointer).
        #    Si no, mostramos la respuesta normal del grafo.
        # ------------------------------------------------------------------
        pendientes = _pending_interrupts(config)
        if pendientes:
            drenar_interrupts(config, pendientes=pendientes)
        else:
            _mostrar_respuesta(result)
        print()


if __name__ == "__main__":
    main()
