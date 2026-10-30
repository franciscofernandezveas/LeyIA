"""graph/nodes.py — Nodos transversales del agente (Manzzo y Cía).

v22.4 — PLAN A: eliminación de la ruta de crisis + enlaces como botón CTA:
  - Ruta de crisis eliminada del grafo padre (coherente con prompts v4.5.0,
    que ya no contiene crisis_message ni contención):
    • analyze_sentiment: fuera backstop de keywords (_CRISIS_KEYWORDS,
      _es_crisis_keyword) y guardas semánticas (_es_crisis_clasificador).
      El campo de estado `crisis` queda en desuso: checkpoints antiguos lo
      conservan, pero ya nadie lo lee.
    • Un mensaje con riesgo sigue clasificándose igual (el clasificador lo
      etiqueta intent=hablar_humano, urgency=alta, riesgo=true) y se deriva
      por el flujo estándar (handoff_choice → equipo), sin mensaje de
      contención ni números de emergencia. Se loguea como warning.
    • handoff_humano: eliminado el bloque de crisis (crisis_message, email
      "RIESGO VITAL", motivo crisis_riesgo_vital).
    • ROUTE_CRISIS ya no se importa ni se emite; permanece en contracts.py
      para compatibilidad de VALID_ROUTES (limpieza pendiente allí).
  - Enlaces estéticos: WhatsApp NO permite ocultar URLs en texto plano.
    hilo_cerrado y handoff adjuntan response_interactive kind="cta"
    (contrato ya soportado por client.py): botón "Escribir al equipo" que
    abre el wa.me sin mostrar la URL. Requiere plantillas v4.5.1 (sin la
    URL en el cuerpo); str.format ignora los kwargs extra, así que las
    firmas no cambian.
  - Email al equipo: el anchor del wa.me pasa a "Abrir chat con el cliente".
  - Fallback de handoff_choice alineado a la voz vigente ("tú"; en v22
    quedó en "usted" antes del cambio de voz de prompts v4.4.0).
  - Import limpio: se retira _wa_link_display (importado y nunca usado).

v22.3 — FIX: _hidratar_contacto devuelve updates completos.
v22.2 — FIX reapertura de hilos cerrados + hidratación cruzada.
v22.1 — FIX salida de booking ante humano explícito.
v22   — FIX "agendar forzado" + routing cerrado.
v21.2 — FIX #6: parent deja de interceptar aborts de booking.
v21.1 — FIX #2: route_post_faq honra señales route de nodos internos FAQ.
v21   — FIX handoff_choice: contrato del canal "kind" (no "type").
v20   — Gatekeeper de sesión en receive_message.
v19   — handoff_choice: ficha breve vs contacto directo.
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
    ROUTE_HANDOFF_CHOICE,
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
)


logger = logging.getLogger(__name__)

WA_TEXTO_MAX = 900

INTAKE_TTL_HORAS = int(os.getenv("INTAKE_TTL_HORAS", "24"))
AGENDA_CAPTURA_TTL_HORAS = int(os.getenv("AGENDA_CAPTURA_TTL_HORAS", "2"))

# v22.4: texto del botón CTA (contrato kind="cta" de client.py;
# display_text ≤ 25 caracteres, límite Meta).
CTA_TEXTO_EQUIPO = "Escribir al equipo"

_AGENDAR_CTA = ("agendar una hora", "agendar", "agendar ahora",
                "agendar hora", "quiero agendar", "quiero agendar una hora")

# Rutas que, si el hilo estaba cerrado, justifican reabrirlo (intención
# accionable). ROUTE_FAQ y ROUTE_FUERA_DOMINIO NO reabren: son pings pasivos
# y reabrir duplicaría notificaciones al equipo sin información nueva.
_RUTAS_REAPERTURA = (
    ROUTE_AGENDAR, ROUTE_INTAKE,
    ROUTE_HANDOFF, ROUTE_HANDOFF_CHOICE,
)

# Aceptaciones sueltas que, solas, no bastan para confirmar agendamiento.
_ACEPTACIONES_AMBIGUAS = (
    "si", "sí", "dale", "ya", "ok", "bueno", "listo", "me interesa",
    "me gustaria", "me gustaría", "perfecto",
)

# Keywords de rechazo a la IA / pedido explícito de humano.
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


def _pide_humano_explicito(query: str) -> bool:
    """Detecta rechazo de la IA o pedido explícito de persona."""
    t = _norm_simple(query)
    return any(k in t for k in _HUMANO_EXPLICITO_KW)


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


def _cta_equipo(state: AgentState) -> dict:
    """v22.4: bloque interactivo CTA-URL (kind="cta", contrato de client.py).

    WhatsApp no permite enmascarar enlaces en texto plano: el botón es la
    única forma nativa de mostrar "Escribir al equipo" en vez de la URL.
    """
    return {
        "kind": "cta",
        "text": CTA_TEXTO_EQUIPO,
        "url": _wa_link_cliente(state),
    }


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

    # Hilos cerrados: una intención explícita y accionable REABRE el hilo.
    if state.get("closed"):
        if _pide_humano_explicito(query):
            pre = _hidratar_contacto(state)
            logger.info("[sesión] hilo cerrado reabierto: humano explícito")
            return {
                "route": ROUTE_HANDOFF_CHOICE,
                "closed": False,
                "intake_exit": None,
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
        # El parent NO intercepta aborts: esa decisión vive en el subgrafo
        # (booking/planner.py). Excepción: pedido explícito de humano.
        if _expirado(state.get("agenda_started_en"), AGENDA_CAPTURA_TTL_HORAS):
            logger.info("booking expirado por TTL → reclasificar")
            reset = {**reset, **RESET_BOOKING}
        elif _pide_humano_explicito(query):
            logger.info("booking: cliente pide humano → salir de booking")
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

    # v22.4 (Plan A): sin ruta de crisis. El riesgo se deriva por el flujo
    # estándar (el clasificador ya lo etiqueta hablar_humano + alta).
    if result.riesgo:
        logger.warning(
            "[riesgo] clasificador marcó riesgo=true; derivación por flujo "
            "estándar (sin contención) | %s", result.reason
        )

    # Guarda anti-ambigüedad: aceptación suelta sin oferta previa → FAQ.
    if route == ROUTE_AGENDAR and _aceptacion_ambigua_agendar(query) \
            and not _ultima_ai_ofrecio_agendar(state):
        route = ROUTE_FAQ
        result = result.model_copy(update={"intent": "respuestas_faq"})
        reason = f"aceptación ambigua sin oferta de agenda previa → FAQ | {result.reason}"

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
    elif forzar_intake:
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

    logger.info("clf | %s/%s/%s/%s → %s | riesgo=%s | %s",
                result.sentiment, result.urgency, result.intent,
                result.category, route, result.riesgo, reason)

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

    Matching exhaustivo contra IntakeExitLabel:

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
        return ROUTE_HANDOFF
    if exit_ == "end":
        return END

    # Fallback defensivo: cubre None (turno sin exit) y futuras extensiones.
    if state.get("intake_completado"):
        return ROUTE_HANDOFF
    return END


def route_post_faq(state: AgentState) -> str:
    """Router de salida del subgrafo FAQ."""
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

    # v22.4: fallback alineado a la voz vigente ("tú"), sin emojis.
    fallback_default = (
        "Para derivarte con nuestro equipo, ¿cómo prefieres continuar?\n\n"
        "1) *Ficha breve* (2 min): nuestro equipo legal revisa tu caso y te\n"
        "   contacta con todo el contexto.\n"
        "2) *Contacto directo*: te entrego ahora el WhatsApp del equipo.\n\n"
        "Responde *1* o *2*."
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
    wa_link = _wa_link_cliente(state)

    # Hilo cerrado: mensaje de seguimiento + botón CTA (v22.4). La URL ya
    # no viaja en el texto; el kwarg whatsapp_ejecutiva queda por si la
    # plantilla aún lo usa (str.format ignora argumentos extra).
    if state.get("closed"):
        response = atn["hilo_cerrado_message"].format(
            whatsapp_ejecutiva=wa_link, thread_id=tid,
        )
        _guardar_ai(state, response)
        return {
            "response": response, "response_bubbles": [response],
            "response_interactive": {
                "kind": "cta", "text": CTA_TEXTO_EQUIPO, "url": wa_link,
            },
            "messages": [AIMessage(content=response)],
        }

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
    <p><b>WhatsApp del cliente:</b> <a href="{wa_link}">Abrir chat con el cliente</a></p>
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

    # Matching contra IntakeExitLabel.
    intake_ya_despidio = state.get("intake_exit") in ("handoff", "cta_cierre") \
        or bool(state.get("intake_completado"))

    if intake_ya_despidio:
        return {
            "summary": summary, "closed": True,
            "notificacion_pendiente": not (notificacion_ok or email_ok),
            "intake_exit": None,
            **({} if pre is None else pre),
        }

    # v22.4: la URL no viaja en el texto; el botón CTA la reemplaza.
    response = atn["handoff_message"].format(
        apertura=_apertura_pres(state),
        whatsapp_ejecutiva=wa_link,
        thread_id=tid,
    )
    _guardar_ai(state, response)
    return {
        "response": response, "response_bubbles": [response],
        "response_interactive": {
            "kind": "cta", "text": CTA_TEXTO_EQUIPO, "url": wa_link,
        },
        "summary": summary, "closed": True,
        "notificacion_pendiente": not (notificacion_ok or email_ok),
        "messages": [AIMessage(content=response)],
        "asistente_presentado": True,
        **({} if pre is None else pre),
    }
