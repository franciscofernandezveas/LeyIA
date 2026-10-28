"""graph/nodes.py — Nodos transversales del agente (Manzzo y Cía).

v22.3 — FIX: _hidratar_contacto devuelve updates completos
  - Corrige la regresión de v22.2: _hidratar_contacto vuelve a emitir
    lead_nombre/lead_email e intake_respuestas, preservando el contrato
    que consumen handoff_humano y preguntar_preferencia_handoff.
  - Los nodos llamadores usan el dict completo; handoff_humano extrae
    intake_respuestas desde pre["intake_respuestas"].

v22.2 — FIX reapertura de hilos cerrados + hidratación cruzada:
  - Añade _hidratar_contacto para reconstruir nombre/email desde el historial
    completo (no solo el turno actual), eliminando la "amnesia" al pasar de
    booking/intake a handoff y viceversa.
  - _hidratar_contacto alimenta tanto intake_respuestas como lead_nombre/
    lead_email, para que booking (que lee lead_*) no repregunte datos ya
    capturados en otros flujos.
  - analyze_sentiment reabre hilos cerrados explícitamente ante humano o CTA
    agendar, limpiando `closed`/`intake_exit` y evitando que handoff_humano
    caiga en la rama de hilo cerrado sin escalar.
  - Reapertura por clasificador para rutas accionables; FAQ/fuera_dominio NO
    reabren el hilo (evita re-escalamientos por pings pasivos).
  - handoff_humano hidrata y notifica con nombre/email reales, y persiste la
    hidratación en el checkpoint para turnos posteriores.

v22.1 — FIX salida de booking ante humano/crisis:
  - Agrega _pide_humano_explicito para detectar rechazo de la IA o pedido
    de persona en medio del flujo de agendamiento.
  - Si el cliente está en booking y pide humano explícitamente, o hay
    señales de crisis, el parent sale del sub-flujo booking (RESET_BOOKING)
    y reclasifica el mensaje, permitiendo derivación inmediata.

v22 — FIX "agendar forzado" + crisis + routing cerrado:
  - analyze_sentiment:
    • Backstop determinista de crisis por keywords al inicio (precedencia
      máxima).
    • Backstop semántico de crisis: si reason del clasificador empieza con
      "RIESGO:", se impone ROUTE_CRISIS por sobre intake/handoff_choice.
    • Guarda anti-ambigüedad: aceptaciones sueltas ("sí", "dale", "ok"...)
      solo rutear a agendamiento si el último mensaje del asistente ofreció
      agendar; si no, se reclasifican a FAQ.
    • Hilos cerrados: una intención explícita y accionable REABRE el hilo;
      FAQ / fuera_dominio no reabren (evita re-escalamientos por pings).
    • Fix merge de resets: TTL de booking conserva resets acumulados
      ({**reset, **RESET_BOOKING}).
  - handoff_humano:
    • Si state["crisis"] está activo, emite crisis_message, escala con
      motivo "crisis_riesgo_vital" y salta los branches closed /
      intake_ya_despidio para no quedarse sin respuesta.
  - preguntar_preferencia_handoff:
    • Fallback hardcodeado pasa a "usted" y sin emojis.

v21.2 — FIX #6 (auditoría): analyze_sentiment deja de interceptar aborts de
        booking. El parent SOLO aplica TTL por expiración; toda decisión
        de abort vive en el subgrafo (booking/planner.py). Antes, el
        substring "no quiero" en _parece_abort abortaba el booking con
        mensajes legítimos como "no quiero online, mejor presencial" y
        reemplazaba el cierre amable agenda_abortado por una respuesta
        FAQ aleatoria de RAG. Se elimina el import de _parece_abort
        (ya no se usa en este módulo).

v21.1 — FIX #2 (auditoría): route_post_faq honra las señales route de los
        nodos internos del FAQ (ROUTE_HANDOFF_CHOICE desde
        derivar_a_ejecutiva v3.2, y ROUTE_INTAKE legacy como defensa).
        Antes, hablar_humano detectado por el planner FAQ terminaba en END
        SIN emitir respuesta al cliente.

v21 — FIX handoff_choice: response_interactive usa el contrato del canal
       ("kind": "buttons", sin "body"); el body viaja por separado en
       send_interactive. Títulos de botones recortados a ≤ 20 chars
       (límite de Meta Cloud API, ver client.MAX_BUTTON_TITLE).

v20 — Gatekeeper de sesión en receive_message: detecta cambio de
       conversation_id y resetea el estado de flujo (intake, handoff_choice,
       booking) conservando la memoria larga del cliente.
       Resets centralizados en graph.utils (RESET_INTAKE,
       RESET_HANDOFF_CHOICE, RESET_BOOKING, RESET_SESION_NUEVA).

v19 — Añade handoff_choice: bifurcación previa a handoff/intake que pregunta
       al usuario si prefiere rellenar una ficha breve o recibir contacto
       directo, reduciendo la fricción de forzar siempre el intake completo.

v18 — FIX import: verificado, sin constantes de booking en core.contracts.
       Este archivo solo importa AgentState, AnalisisResult, rutas y
       fallback_analisis del módulo central.

v17 — Refactor crítico:
  - Helpers compartidos movidos a graph/utils.py; rompe ciclo de imports con
    graph/intake/nodes.py.
  - route_post_intake reconoce intake_exit == "agendar" y rutea a booking.
  - route_post_intake termina el turno (END) para abandono/sin_consentimiento.
  - _precaptura_contacto normaliza email sin depender de EMAIL_RE.
  - handoff_humano reconoce intake_exit="agendar" como "ya despidió".
"""

import logging
import os
import re

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from core.contracts import (
    AgentState, AnalisisResult,
    ROUTE_AGENDAR, ROUTE_FAQ, ROUTE_FUERA_DOMINIO, ROUTE_HANDOFF, ROUTE_INTAKE,
    ROUTE_HANDOFF_CHOICE, ROUTE_CRISIS,
    VALID_ROUTES, fallback_analisis,
)
from core.db_client import (
    get_active_conversation_id, insert_escalation, insert_message,
    upsert_conversation, upsert_lead,
)
from core.llm import LLM, with_structured_output
from core.notifications import notificar_email
from graph.utils import (
    RESET_BOOKING, RESET_HANDOFF_CHOICE, RESET_INTAKE, RESET_SESION_NUEVA,
    _ahora_iso, _apertura_pres, _cfg, _expirado, _ficha_intake, _format_few_shots,
    _guardar_ai, _norm_simple, _persist_escalation,
    _primer_nombre, _recent_messages, _telefono_cliente,
    _transcript_resumen, _upsert_conversacion_abierta, _wa_link_cliente,
    _wa_link_display,
)


logger = logging.getLogger(__name__)

WA_TEXTO_MAX = 900

INTAKE_TTL_HORAS = int(os.getenv("INTAKE_TTL_HORAS", "24"))
AGENDA_CAPTURA_TTL_HORAS = int(os.getenv("AGENDA_CAPTURA_TTL_HORAS", "2"))

_AGENDAR_CTA = ("agendar una hora", "agendar", "agendar ahora",
                "agendar hora", "quiero agendar", "quiero agendar una hora")

# Rutas que, si el hilo estaba cerrado, justifican reabrirlo (intención
# accionable). ROUTE_FAQ y ROUTE_FUERA_DOMINIO NO reabren: son pings pasivos
# y reabrir duplicaría notificaciones al equipo sin información nueva.
_RUTAS_REAPERTURA = (
    ROUTE_AGENDAR, ROUTE_INTAKE,
    ROUTE_HANDOFF, ROUTE_HANDOFF_CHOICE, ROUTE_CRISIS,
)

# Aceptaciones sueltas que, solas, no bastan para confirmar agendamiento.
_ACEPTACIONES_AMBIGUAS = (
    "si", "sí", "dale", "ya", "ok", "bueno", "listo", "me interesa",
    "me gustaria", "me gustaría", "perfecto",
)

# Backstop de crisis: señales de riesgo vital. Falso positivo = mostrar
# números de emergencia de más; falso negativo = peligro. Se prioriza sensibilidad.
_CRISIS_KEYWORDS = (
    "me golpea", "me golpeó", "me pegó", "me pega", "golpea a mis hijos",
    "golpea a los niños", "me va a matar", "amenaza con matarme",
    "me quiere matar", "me amenaza de muerte", "matarme",
    "quitarme la vida", "no quiero vivir", "quiero morir",
    "suicid", "hacerme daño", "hacerme dano",
    "no aguanto más", "no aguanto mas", "me quiere hacer daño",
    "no me deja salir", "me controla el celular", "me encierra",
    "me amenaza con un cuchillo", "cuchillo", "arma", "me apunta",
)


# v22.1: keywords de rechazo a la IA / pedido explícito de humano.
# Debe coincidir con _HUMANO_EXPLICITO de graph/faq/planner.py.
_HUMANO_EXPLICITO_KW = (
    "hablar con una ejecutiva", "hablar con una persona",
    "hablar con un humano", "persona real", "ejecutiva real",
    "contacto humano", "abogado real",
    "hablar directamente con alguien", "quiero una ejecutiva",
    "puedo hablar con alguien", "hablar con alguien de verdad",
    "quiero hablar con un abogado",
    "eres un bot", "eres un robot", "eres una ia",
    "no quiero un bot", "no quiero hablar con una maquina",
    "no quiero hablar con una máquina",
    "pásame con", "pasame con", "pásame a", "pasame a",
)


_EMAIL_BUSCA = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_NOMBRE_RE = re.compile(
    r"(?:soy|me llamo|mi nombre es)\s+"
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}"
    r"(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,}){0,3})",
    re.IGNORECASE,
)

# Normalizadas con _norm_simple para el parser 0-LLM.
_ELECCION_FICHA_RAW = (
    "ficha", "rellenar", "relleno", "formulario", "breve", "agilizar",
    "opcion 1", "hc_ficha",
)
_ELECCION_DIRECTO_RAW = (
    "directo", "directa", "contacto", "numero", "número", "telefono",
    "teléfono", "llamar", "llamada", "whatsapp", "ejecutiva", "abogada",
    "correo", "opcion 2", "hc_directo",
)
_ELECCION_FICHA = tuple(_norm_simple(k) for k in _ELECCION_FICHA_RAW)
_ELECCION_DIRECTO = tuple(_norm_simple(k) for k in _ELECCION_DIRECTO_RAW)
_DIGITO_RE = re.compile(r"(?<!\d)([12])(?!\d)")


def _es_crisis_keyword(query: str) -> bool:
    """Backstop 0-LLM para señales de riesgo vital."""
    t = _norm_simple(query)
    return any(k in t for k in _CRISIS_KEYWORDS)


def _pide_humano_explicito(query: str) -> bool:
    """v22.1: detecta rechazo de la IA o pedido explícito de persona."""
    t = _norm_simple(query)
    return any(k in t for k in _HUMANO_EXPLICITO_KW)


def _es_crisis_clasificador(result: AnalisisResult) -> bool:
    """Señal semántica de crisis emitida por el clasificador."""
    return result.reason.strip().upper().startswith("RIESGO:")


def _aceptacion_ambigua_agendar(query: str) -> bool:
    t = _norm_simple(query)
    return t in _ACEPTACIONES_AMBIGUAS


def _ultima_ai_ofrecio_agendar(state: AgentState) -> bool:
    """Verifica que el último mensaje del asistente haya ofrecido agendar."""
    for m in reversed(state.get("messages", [])):
        if m.type == "ai":
            txt = _norm_simple(m.content)
            return any(k in txt for k in (
                "agendar", "asesor", "cita", "hora", "le interesa"
            ))
    return False


def _interpretar_eleccion_handoff(q: str) -> str | None:
    """Interpreta la respuesta a '¿ficha breve o contacto directo?'.
    Devuelve 'ficha', 'directo' o None si es ambigua."""
    t = _norm_simple(q)
    if not t:
        return None
    if m := _DIGITO_RE.search(t):
        return "ficha" if m.group(1) == "1" else "directo"
    # "directo" primero: frases como "no quiero ficha, dame el número"
    # contienen ambas keywords y deben ir por la vía de menor fricción.
    if any(k in t for k in _ELECCION_DIRECTO):
        return "directo"
    if any(k in t for k in _ELECCION_FICHA):
        return "ficha"
    return None


def _precaptura_contacto(state: AgentState) -> dict | None:
    """Llena nombre/email del mensaje actual sin depender del routing.
    No persiste: solo alimenta intake_respuestas en memoria."""
    resp = dict(state.get("intake_respuestas") or {})
    q = state.get("query") or ""

    if "nombre" not in resp and (m := _NOMBRE_RE.search(q)):
        palabras = m.group(1).split()
        if len(palabras) >= 2:
            resp["nombre"] = " ".join(p.capitalize() for p in palabras)

    if "email" not in resp and (m := _EMAIL_BUSCA.search(q)):
        resp["email"] = m.group(0).lower().replace(" ", "")

    return resp if resp != (state.get("intake_respuestas") or {}) else None


def _hidratar_contacto(state: AgentState) -> dict | None:
    """Reconstruye datos de contacto desde TODAS las fuentes del hilo.

    Alimenta tanto intake_respuestas (handoff/intake) como lead_nombre/
    lead_email (booking), sin pisar datos válidos existentes.

    Orden de prioridad:
      1. intake_respuestas ya existentes
      2. canal canónico `contacto` / campos lead_* (memoria larga)
      3. regex (_NOMBRE_RE / _EMAIL_BUSCA) sobre el historial humano completo
         (state["messages"] sobrevive a RESET_BOOKING vía add_messages)

    Devuelve un dict de updates solo si hay novedades (checkpoint limpio).
    """
    resp = dict(state.get("intake_respuestas") or {})
    contacto = state.get("contacto") or {}
    updates: dict = {}

    # 1) intake_respuestas: memoria corta del flujo actual
    if "nombre" not in resp:
        if n := (contacto.get("nombre") or state.get("lead_nombre")):
            resp["nombre"] = n
    if "email" not in resp:
        if e := (contacto.get("email") or state.get("lead_email")):
            resp["email"] = e

    # 2) lead_*: memoria larga que booking lee directamente.
    #    Solo se escriben si faltan; nunca se sobrescriben.
    if not state.get("lead_nombre") and resp.get("nombre"):
        updates["lead_nombre"] = resp["nombre"]
    if not state.get("lead_email") and resp.get("email"):
        updates["lead_email"] = resp["email"]

    # 3) Fallback al historial humano si aún falta algo
    if "nombre" not in resp or "email" not in resp:
        for m in reversed(state.get("messages", [])):
            if m.type != "human":
                continue
            txt = m.content if isinstance(m.content, str) else str(m.content)
            if "nombre" not in resp and (mm := _NOMBRE_RE.search(txt)):
                palabras = mm.group(1).split()
                if len(palabras) >= 2:
                    resp["nombre"] = " ".join(p.capitalize() for p in palabras)
            if "email" not in resp and (mm := _EMAIL_BUSCA.search(txt)):
                resp["email"] = mm.group(0).lower().replace(" ", "")
            if "nombre" in resp and "email" in resp:
                break

    # Recalcular lead_* tras parsear historial
    if not state.get("lead_nombre") and resp.get("nombre"):
        updates["lead_nombre"] = resp["nombre"]
    if not state.get("lead_email") and resp.get("email"):
        updates["lead_email"] = resp["email"]

    if resp != (state.get("intake_respuestas") or {}):
        updates["intake_respuestas"] = resp

    return updates if updates else None


def receive_message(state: AgentState) -> AgentState:
    last_human = next(
        (m for m in reversed(state.get("messages", [])) if m.type == "human"), None
    )
    q = last_human.content if last_human else ""
    if not isinstance(q, str):
        q = str(q)

    tid = state.get("thread_id", "")
    _upsert_conversacion_abierta(tid)

    insert_message(thread_id=tid, role="human", content=q)

    updates: dict = {"query": q}

    # --- Gatekeeper de sesión -------------------------------------------
    # Si la conversación activa es OTRA distinta a la que veía el checkpoint,
    # el "caso" anterior quedó atrás: el cliente puede no haber podido nunca
    # salir de un hilo cerrado/escalado. Se resetea TODO el estado de flujo
    # (intake, handoff_choice, booking, closed) y se conserva la memoria
    # larga (lead_nombre / lead_email / booking_privacidad_dicha).
    sid_actual = get_active_conversation_id(tid)
    sid_prev = state.get("session_id")
    if sid_actual:
        if sid_prev and sid_prev != sid_actual:
            logger.info("[sesión] %s → %s: reset de flujo para thread=%s",
                        sid_prev, sid_actual, tid)
            updates.update(RESET_SESION_NUEVA)
        updates["session_id"] = sid_actual
    # sid_actual None → BD caída: best-effort, seguir con el checkpoint tal cual.

    return updates


def analyze_sentiment(state: AgentState) -> AgentState:
    query = state["query"]

    # 0) Crisis: precedencia máxima, 0-LLM. No espera a nada.
    if _es_crisis_keyword(query):
        logger.warning("[crisis] backstop keyword → ROUTE_CRISIS")
        return {
            "route": ROUTE_CRISIS,
            "sentiment": "negativo",
            "urgency": "alta",
            "crisis": True,
            "clf_reason": "crisis: señal de riesgo vital detectada (0-LLM)",
        }

    # Hilos cerrados: una intención explícita y accionable REABRE el hilo.
    # Bug corregido: antes se ruteaba sin limpiar `closed` → handoff_humano
    # caía en la rama "Hilo cerrado" (sin escalar) e intake repreguntaba
    # datos ya dictados en flujos previos.
    if state.get("closed"):
        if _pide_humano_explicito(query):
            pre = _hidratar_contacto(state)
            logger.info("[sesión] hilo cerrado reabierto: humano explícito")
            return {
                "route": ROUTE_HANDOFF_CHOICE,
                "closed": False,
                "intake_exit": None,
                "crisis": False,  # una crisis ya escalada no debe repetirse
                **({} if pre is None else pre),
                "clf_reason": "reapertura: humano explícito en hilo cerrado (0 LLM)",
            }
        if _norm_simple(query) in _AGENDAR_CTA:
            pre = _hidratar_contacto(state)
            logger.info("[sesión] hilo cerrado reabierto: CTA agendar")
            return {
                "route": ROUTE_AGENDAR,
                "closed": False,
                "intake_exit": None,
                **({} if pre is None else pre),
                "clf_reason": "reapertura: CTA agendar en hilo cerrado (0 LLM)",
            }
        # Otro mensaje: reclasificar. Si la ruta final es accionable, el
        # retorno general (más abajo) reabrirá; FAQ / fuera_dominio no reabren.

    en_booking = bool(state.get("booking_stage"))
    en_intake = bool(state.get("intake_activo")) and not state.get("intake_completado")
    en_intake_pausado = en_intake and state.get("intake_exit") == "pausa"
    reset: dict = {}
    forzar_intake = False
    choice_pendiente = False
    choice_attempts = int(state.get("handoff_choice_attempts") or 0)

    if en_intake_pausado:
        return {
            "route": ROUTE_INTAKE,
            "clf_reason": "intake pausado → reanudar",
            "intake_exit": None,
            "intake_resume": True,
        }

    # --- Bifurcación handoff: ofrecer ficha breve vs contacto directo ----
    if state.get("handoff_choice_pending"):
        if _expirado(state.get("handoff_choice_en"), INTAKE_TTL_HORAS):
            logger.info("elección handoff expirada (> %dh) → reclasificar",
                        INTAKE_TTL_HORAS)
            reset = {**reset, **RESET_HANDOFF_CHOICE}
        else:
            eleccion = _interpretar_eleccion_handoff(query)
            if eleccion == "ficha":
                return {
                    "route": ROUTE_INTAKE,
                    **RESET_HANDOFF_CHOICE,
                    "clf_reason": "handoff_choice → ficha breve (0 LLM)",
                }
            if eleccion == "directo":
                pre = _hidratar_contacto(state)
                return {
                    "route": ROUTE_HANDOFF,
                    **RESET_HANDOFF_CHOICE,
                    **({} if pre is None else pre),
                    "clf_reason": "handoff_choice → contacto directo (0 LLM)",
                }
            # Respuesta ambigua: dejamos que el LLM decida y re-preguntemos.
            choice_pendiente = True

    if en_booking:
        # v21.2 — FIX #6 (auditoría): el parent YA NO intercepta aborts.
        # Toda decisión de abort vive en el subgrafo (booking/planner.py).
        # v22.1 — Excepto pedido explícito de humano o crisis: ahí sí
        # salimos del booking para derivar inmediatamente.
        if _expirado(state.get("agenda_started_en"), AGENDA_CAPTURA_TTL_HORAS):
            logger.info("booking expirado por TTL → reclasificar")
            reset = {**reset, **RESET_BOOKING}
        elif _es_crisis_keyword(query) or _pide_humano_explicito(query):
            logger.info("booking: cliente pide humano o crisis → salir de booking")
            reset = {**reset, **RESET_BOOKING}
        else:
            return {"route": ROUTE_AGENDAR,
                    "clf_reason": f"sub-flujo booking/{state.get('booking_stage')} (0 LLM)"}

    if en_intake:
        if _expirado(state.get("intake_started_en"), INTAKE_TTL_HORAS):
            logger.info("ficha de intake expirada (> %dh) → reset total",
                        INTAKE_TTL_HORAS)
            reset = {**reset, **RESET_INTAKE}
        else:
            forzar_intake = True

    cfg = _cfg()["classification"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", cfg["system_prompt"]),
        ("system",
         "Historial reciente (clasifica el ÚLTIMO mensaje del cliente usando "
         "este contexto; los follow-ups breves heredan el tema del hilo):\n{history}"),
        ("system",
         "Ejemplos de etiquetado estándar (referencia obligatoria):\n{few_shots}"),
        ("human", "{query}"),
    ])
    history = "\n".join(
        f"- {m.type}: {m.content[:200]}" for m in _recent_messages(state)
    ) or "(sin historial)"

    chain = prompt | with_structured_output(AnalisisResult)
    try:
        result: AnalisisResult = chain.with_retry(stop_after_attempt=2).invoke({
            "query": query,
            "few_shots": _format_few_shots(cfg["few_shot_examples"]),
            "history": history,
        })
    except Exception as e:
        logger.warning("analyze_sentiment cayó a fallback: %s", e)
        result = fallback_analisis(str(e)[:120])

    route = compute_route(result)
    reason = result.reason
    crisis = _es_crisis_clasificador(result) or result.riesgo

    # 1) Crisis semántica gana sobre intake activo y handoff_choice pendiente.
    if crisis:
        route = ROUTE_CRISIS
        reason = f"crisis detectada por clasificador | {result.reason}"

    # 2) Guarda anti-ambigüedad: aceptación suelta sin oferta previa → FAQ.
    if route == ROUTE_AGENDAR and _aceptacion_ambigua_agendar(query) \
            and not _ultima_ai_ofrecio_agendar(state):
        route = ROUTE_FAQ
        result = result.model_copy(update={"intent": "respuestas_faq"})
        reason = f"aceptación ambigua sin oferta de agenda previa → FAQ | {result.reason}"

    # 3) Crisis nunca pasa por handoff_choice; va directo a handoff con flag.
    if route == ROUTE_HANDOFF_CHOICE and (crisis or state.get("crisis")):
        route = ROUTE_HANDOFF
        crisis = True
        reason = f"crisis: bypass handoff_choice → handoff directo | {reason}"

    if choice_pendiente:
        if route == ROUTE_HANDOFF_CHOICE:
            if choice_attempts >= 1:
                # Segunda respuesta ambigua: evitar bucle, ir por menor fricción.
                route = ROUTE_HANDOFF
                reason = f"handoff_choice ambigua x2 → directo | {result.reason}"
                reset = {**reset, **RESET_HANDOFF_CHOICE}
            else:
                reason = f"handoff_choice ambigua → re-preguntar | {result.reason}"
                reset = {**reset, "handoff_choice_attempts": choice_attempts + 1}
        else:
            # El usuario cambió de tema (faq/agendar): limpiar elección pendiente.
            reset = {**reset, **RESET_HANDOFF_CHOICE}
    elif forzar_intake and not crisis:
        route = ROUTE_INTAKE
        reason = f"intake activo (clasificación solo metadata) | {result.reason}"

    current_case_category = state.get("case_category")
    new_category = result.category
    case_category = current_case_category
    if not current_case_category or (
        current_case_category in ("otro", "consulta_general")
        and new_category not in ("otro", "consulta_general")
    ):
        case_category = new_category

    logger.info("clf | %s/%s/%s/%s → %s | crisis=%s | %s",
                result.sentiment, result.urgency, result.intent,
                result.category, route, crisis, reason)

    upsert_conversation(thread_id=state.get("thread_id", ""),
                        intent=result.intent, category=result.category)

    contacto_updates = {} if (pre := _precaptura_contacto(state)) is None else {"intake_respuestas": pre}

    was_closed = bool(state.get("closed"))
    reabrir = was_closed and route in _RUTAS_REAPERTURA
    if reabrir:
        logger.info("[sesión] hilo cerrado reabierto por clasificador → %s",
                    route)

    return {
        "sentiment": result.sentiment, "urgency": result.urgency,
        "intent": result.intent, "category": result.category,
        "case_category": case_category,
        "clf_reason": reason, "route": route,
        "crisis": crisis or state.get("crisis"),
        "closed": False if reabrir else state.get("closed", False),
        **contacto_updates,
        **reset,
    }


def compute_route(result: AnalisisResult) -> str:
    if result.intent == "fuera_de_dominio":
        return ROUTE_FUERA_DOMINIO
    if result.intent == "hablar_humano":
        return ROUTE_HANDOFF_CHOICE
    return ROUTE_AGENDAR if result.intent == "agendar_asesoria" else ROUTE_FAQ


def route_query(state: AgentState) -> str:
    route = state.get("route", ROUTE_FAQ)
    if route == ROUTE_INTAKE and state.get("intake_completado"):
        return ROUTE_HANDOFF
    if route not in VALID_ROUTES:
        logger.warning("route inesperado '%s' → %s", route, ROUTE_FAQ)
        return ROUTE_FAQ
    return route


def route_post_intake(state: AgentState) -> str:
    """Router de salida del subgrafo intake.

    v21.3 — FIX #11 (auditoría): matching exhaustivo contra IntakeExitLabel.
    Cada valor del Literal tiene su rama explícita (el test
    test_route_post_intake_cobertura_exhaustiva lo verifica):

      "faq"        → ROUTE_FAQ       (duda lateral; se reanudará después)
      "pausa"      → END             (espera próximo mensaje del cliente)
      "handoff"    → ROUTE_HANDOFF   (derivación parcial)
      "cta_cierre" → ROUTE_HANDOFF   (cierre dual con CTA; ficha completa)
      "end"        → END             (abandono / sin consentimiento)

    Fallback defensivo (None u otro): si intake_completado → ROUTE_HANDOFF,
    si no → END.
    """
    exit_ = state.get("intake_exit")

    if exit_ == "faq":
        return ROUTE_FAQ
    if exit_ == "pausa":
        return END
    if exit_ == "handoff":
        return ROUTE_HANDOFF
    if exit_ == "cta_cierre":
        # v21.3: rama explícita. Antes dependía del fallback
        # (intake_completado=True) — funcionaba por acoplamiento.
        return ROUTE_HANDOFF
    if exit_ == "end":
        return END

    # Fallback defensivo: cubre None (turno sin exit) y futuras extensiones.
    if state.get("intake_completado"):
        return ROUTE_HANDOFF
    return END



def route_post_faq(state: AgentState) -> str:
    """Router de salida del subgrafo FAQ.

    v21.1 — FIX #2 (auditoría): honra la señal route emitida por nodos
    internos del FAQ (derivar_a_ejecutiva). Antes solo miraba intake_resume,
    por lo que hablar_humano dentro de FAQ terminaba en END SIN emitir
    respuesta al cliente.

    Reglas:
      - route == ROUTE_HANDOFF_CHOICE → preguntar_preferencia_handoff
        (ruta emitida por derivar_a_ejecutiva v3.2; política unificada con
        analyze_sentiment para intent=hablar_humano).
      - route == ROUTE_INTAKE        → intake_lead (o handoff si la ficha
        ya estaba completa). Rama defensiva: cubre señales legacy.
      - intake_resume + intake_activo + !completado → intake_lead
        (round-trip lateral de duda dentro de la ficha).
      - resto → END.
    """
    route = state.get("route")
    if route == ROUTE_HANDOFF_CHOICE:
        return ROUTE_HANDOFF_CHOICE
    if route == ROUTE_INTAKE:
        return ROUTE_HANDOFF if state.get("intake_completado") else ROUTE_INTAKE
    if state.get("intake_resume") and state.get("intake_activo") \
            and not state.get("intake_completado"):
        return ROUTE_INTAKE
    return END


def preguntar_preferencia_handoff(state: AgentState) -> AgentState:
    """Nodo intermedio que reduce fricción: ficha breve vs contacto directo.
    Es defensivo: si faltan las plantillas en config, usa un fallback
    hardcodeado para evitar crash en producción."""
    atn = _cfg()["atencion"]
    pre = _hidratar_contacto(state)

    # v22: fallback de usted, sin emojis, coherente con la voz del sistema.
    fallback_default = (
        "Para derivarle con nuestro equipo, ¿cómo prefiere continuar?\n\n"
        "1) *Ficha breve* (2 min): nuestro equipo legal revisa su caso y le\n"
        "   contacta con todo el contexto.\n"
        "2) *Contacto directo*: le entrego ahora el WhatsApp del equipo.\n\n"
        "Responda *1* o *2*."
    )

    choice_attempts = state.get("handoff_choice_attempts") or 0
    key = ("handoff_choice_retry_message"
           if choice_attempts > 0
           else "handoff_choice_message")

    # Defensivo: resuelve en cascata para nunca crashear.
    template = atn.get(key) or atn.get("handoff_choice_message") or fallback_default

    if template == fallback_default:
        logger.warning(
            "handoff_choice: no se encontró '%s' en config; usando fallback por defecto",
            key,
        )

    response = template.format(apertura=_apertura_pres(state))

    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": {
            # v21: contrato del canal = "kind" (no "type"); el body viaja
            # aparte en send_interactive. Títulos ≤ 20 chars (Meta).
            "kind": "buttons",
            "buttons": [
                {"id": "hc_ficha",   "title": "1 Rellenar ficha"},
                {"id": "hc_directo", "title": "2 Contacto directo"},
            ],
        },
        "handoff_choice_pending": True,
        "handoff_choice_en": _ahora_iso(),
        "asistente_presentado": True,
        **({} if pre is None else pre),
        "messages": [AIMessage(content=response)],
    }


def respuesta_fuera_dominio(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    response = atn["fuera_dominio_message"].format(apertura=_apertura_pres(state))
    _guardar_ai(state, response)
    return {
        "response": response,
        "response_bubbles": [response],
        "response_interactive": None,
        "messages": [AIMessage(content=response)],
        "asistente_presentado": True,
    }


def handoff_humano(state: AgentState) -> AgentState:
    atn = _cfg()["atencion"]
    tid = state.get("thread_id", "")
    en_crisis = bool(state.get("crisis"))

    # ------------------------------------------------------------------
    # Crisis: mensaje de contención + escalamiento prioritario. Tiene
    # precedencia sobre hilo cerrado y sobre intake_ya_despidio para no
    # quedarse sin respuesta ante una emergencia.
    # ------------------------------------------------------------------
    if en_crisis:
        response = atn["crisis_message"].strip()
        _guardar_ai(state, response)

        try:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", atn["summary_system_prompt"]),
                ("human", "{transcript}"),
            ])
            summary = (summary_prompt | LLM).invoke(
                {"transcript": _ficha_intake(state) + "CONVERSACIÓN:\n"
                             + _transcript_resumen(state)}
            ).content
        except Exception as e:
            logger.exception("LLM de resumen de crisis falló: %s", e)
            summary = f"RIESGO: {state.get('query', '')[:300]}"

        upsert_conversation(thread_id=tid, status="escalado",
                            summary=summary, closed=True)

        notificacion_ok = insert_escalation(
            thread_id=tid,
            summary=summary,
            motivo="crisis_riesgo_vital",
            notificar=True,
        )

        _persist_escalation(state, summary)

        telefono = _telefono_cliente(state)
        wa_link = _wa_link_cliente(state)
        email_body = f"""
        <h3>Nuevo lead derivado — RIESGO VITAL — Manzzo y Cía</h3>
        <p><b>Thread ID:</b> {tid}</p>
        <p><b>Teléfono:</b> {telefono or 'No disponible'}</p>
        <p><b>Urgencia:</b> alta</p>
        <p><b>Resumen:</b> {summary}</p>
        <p><b>Link WhatsApp cliente:</b> <a href="{wa_link}">{wa_link}</a></p>
        """
        email_ok = notificar_email(
            subject=f"[Manzzo Bot] RIESGO VITAL: {tid}",
            body_html=email_body,
        )

        return {
            "response": response,
            "response_bubbles": [response],
            "response_interactive": {
                "kind": "buttons",
                "buttons": [{"id": "hc_directo", "title": "Contacto directo"}],
            },
            "summary": summary, "closed": True,
            "notificacion_pendiente": not (notificacion_ok or email_ok),
            "messages": [AIMessage(content=response)],
        }

    # Hilo cerrado sin crisis: mensaje de seguimiento.
    if state.get("closed"):
        response = atn["hilo_cerrado_message"].format(
            whatsapp_ejecutiva=_wa_link_cliente(state), thread_id=tid,
        )
        _guardar_ai(state, response)
        return {"response": response, "response_bubbles": [response],
                "response_interactive": None,
                "messages": [AIMessage(content=response)]}

    # Consolidar datos ANTES de resumir y notificar: el lead pudo venir de
    # booking, de una sesión anterior o solo existir en el historial. El
    # resumen LLM también debe verlos.
    pre = _hidratar_contacto(state)
    respuestas = dict(state.get("intake_respuestas") or {})
    if pre is not None:
        respuestas.update(pre.get("intake_respuestas", {}))
    state_para_resumen = {**state, "intake_respuestas": respuestas}

    try:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", atn["summary_system_prompt"]),
            ("human", "{transcript}"),
        ])
        summary = (summary_prompt | LLM).invoke(
            {"transcript": _ficha_intake(state_para_resumen) + "CONVERSACIÓN:\n"
                         + _transcript_resumen(state_para_resumen)}
        ).content
    except Exception as e:
        logger.exception("LLM de resumen falló: %s", e)
        summary = "(resumen automático no disponible — revisar transcript adjunto)"

    upsert_conversation(thread_id=tid, status="escalado",
                        summary=summary, closed=True)

    if respuestas.get("consentimiento_datos") is True:
        upsert_lead(
            thread_id=tid,
            intake_respuestas=respuestas,
            category=state.get("case_category") or state.get("category"),
            completed=bool(state.get("intake_completado")),
            notificar_crm=False,   # handoff_humano notifica vía insert_escalation
        )

    motivo = "hablar_abogada" if state.get("intake_completado") else "contacto_directo"
    notificacion_ok = insert_escalation(
        thread_id=tid,
        summary=summary,
        motivo=motivo,
        notificar=True,
    )

    _persist_escalation(state, summary)

    telefono = _telefono_cliente(state)
    wa_link = _wa_link_cliente(state)
    contacto = state.get("contacto") or {}
    nombre = respuestas.get("nombre") or state.get("lead_nombre") or "Sin nombre"
    email = respuestas.get("email") or contacto.get("email") or "No disponible"
    modalidad = (contacto.get("modalidad_preferida")
                 or respuestas.get("modalidad_preferida") or "—")
    categoria_final = state.get("case_category") or state.get("category") or "otro"
    email_body = f"""
    <h3>Nuevo lead derivado — Manzzo y Cía</h3>
    <p><b>Thread ID:</b> {tid}</p>
    <p><b>Nombre:</b> {nombre}</p>
    <p><b>Email:</b> {email}</p>
    <p><b>Modalidad preferida:</b> {modalidad}</p>
    <p><b>Teléfono:</b> {telefono or 'No disponible'}</p>
    <p><b>Categoría:</b> {categoria_final}</p>
    <p><b>Urgencia:</b> {state.get('urgency', 'media')}</p>
    <p><b>Resumen:</b> {summary}</p>
    <p><b>Link WhatsApp cliente:</b> <a href="{wa_link}">{wa_link}</a></p>
    <p><b>Ficha completa:</b> {respuestas}</p>
    """
    email_ok = notificar_email(
        subject=f"[Manzzo Bot] Nuevo lead: {nombre} | {categoria_final}",
        body_html=email_body,
    )

    try:
        from integrations.sheetdb import sync_lead
        if respuestas.get("consentimiento_datos") is True:
            sync_lead(state, modo="completo")
        else:
            sync_lead(state, modo="minimo")
    except Exception as e:
        logger.exception("SheetDB (fallback) falló: %s", e)

    # v21.3 — FIX #11 (auditoría): matching contra IntakeExitLabel.
    # Se eliminan "booking" y "agendar" (nunca emitidos; eran código
    # defensivo sin origen). "cta_cierre" se añade explícitamente: hoy
    # queda cubierto también por intake_completado=True, pero hacerlo
    # explícito documenta la intención y sobrevive a un refactor que
    # desacople ambos.
    intake_ya_despidio = state.get("intake_exit") in ("handoff", "cta_cierre") \
        or bool(state.get("intake_completado"))

    if intake_ya_despidio:
        return {
            "summary": summary, "closed": True,
            "notificacion_pendiente": not (notificacion_ok or email_ok),
            "intake_exit": None,
            **({} if pre is None else pre),
        }

    response = atn["handoff_message"].format(
        apertura=_apertura_pres(state),
        whatsapp_ejecutiva=_wa_link_cliente(state),
        thread_id=tid,
    )
    _guardar_ai(state, response)
    return {
        "response": response, "response_bubbles": [response],
        "response_interactive": None,
        "summary": summary, "closed": True,
        "notificacion_pendiente": not (notificacion_ok or email_ok),
        "messages": [AIMessage(content=response)],
        "asistente_presentado": True,
        **({} if pre is None else pre),
    }
