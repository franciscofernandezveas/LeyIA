"""graph/faq/supervisor.py — Routing interno del sub-agente FAQ.

v2 — Política de ofertas one-shot (fix CTA duplicado, producción 20/10):
  El planner PROPONE y el supervisor DISPONE. Si el planner elige
  sugerir_agendar pero la oferta standalone ya se emitió en esta sesión
  (cta_agenda_ofrecida=True, la levanta el propio nodo sugerir_agendar),
  se degrada a responder_pregunta: la invitación a agendar queda embebida
  en la respuesta RAG y el cliente nunca recibe el mismo CTA dos veces.

  Mismo idiom que intake_oferta_qid (intake) y booking_privacidad_dicha
  (booking): el ledger vive en AgentState; ningún nodo compara strings de
  templates. La política de flujo vive aquí, no en el planner.
"""
import logging

from core.contracts import AgentState

logger = logging.getLogger(__name__)


def route_faq(state: AgentState) -> str:
    d = state.get("faq_decision") or {}
    accion = d.get("accion", "responder_pregunta")

    # v2 — Anti-repetición del CTA de agenda (one-shot por sesión).
    # La única línea de política de flujo del subgrafo: el CTA standalone
    # es idempotente. Aplica tanto a fast-path como a LLM — la degradación a
    # responder_pregunta es segura porque la oferta queda embebida en la
    # respuesta del FAQ.
    if accion == "sugerir_agendar" and state.get("cta_agenda_ofrecida"):
        logger.info("[faq] CTA de agenda ya ofrecida esta sesión → "
                    "responder_pregunta (oferta embebida, sin standalone)")
        return "responder_pregunta"

    if accion == "responder_pregunta":
        return "responder_pregunta"
    if accion == "pedir_clarificacion":
        return "pedir_clarificacion"
    if accion == "sugerir_agendar":
        return "sugerir_agendar"
    if accion == "hablar_humano":
        return "derivar_a_ejecutiva"
    if accion == "fuera_de_dominio":
        return "derivar_a_fuera_dominio"

    return "responder_pregunta"
