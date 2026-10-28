"""graph/intake/nodes.py — Acciones del sub-agente INTAKE.

v18.2 — Fix saludos reiterados:
  - iniciar_ficha gatea la burbuja de apertura con `_apertura_pres` para no
    volver a presentarse si el asistente ya lo hizo en el hilo.
  - DEFAULT_AP_IA pasa a voz LeyIA (usted, sin emojis) como fallback seguro.
  - Todos los nodos emisores de burbujas levantan `asistente_presentado=True`.

v18.1 — FIX lead_* en estado:
  - intake ahora copia nombre/email capturados a lead_nombre/lead_email
    (memoria larga del hilo) en todas las salidas del subgrafo. Booking
    lee lead_* directamente y así no repregunta datos ya dados en la ficha.

v18 — Fix "saltar" en campos opcionales:
  - Detección robusta de intención de salto (no solo coincidencia exacta).
  - Campos opcionales saltados se registran como None y el intake avanza.
  - Nuevo campo de estado intake_skipped; la extracción posterior no
    sobreescribe campos que el usuario eligió omitir.
  - Propagación de intake_skipped en todas las ramas de salida.

v17 — Ficha por ramas temáticas (familia / penal / general):
  - QUESTIONS = tronco común + módulos condicionados por area_caso (vía
    skip_if, mecanismo que ya existía pero nunca se usaba).
  - La category del padre NO distingue penal (va a "otro"): la rama penal
    se detecta por extracción (area_caso/detenido del planner) o por la
    pregunta area_caso cuando la rama está "pendiente".
  - Nueva acción derivar_urgente: detención vigente u orden de arresto →
    handoff inmediato con ficha parcial. Emite intake_exit="handoff" (el
    IntakeExitLabel del padre no se toca) y marca urgencia_legal=True en
    las respuestas para que handoff_humano lo vea en el lead.
  - FIX: _texto_ficha_wa recibe la categoría como parámetro — antes leía
    respuestas["categoria"], clave que jamás existió, por lo que el "Área"
    del wa.me siempre decía "consulta_general".
  - La ficha wa.me ahora incluye los campos de rama (subtipo, hijos,
    ingresos, medidas, rol, audiencia, urgencia).

v16 — Resets centralizados (RESET_INTAKE desde graph.utils).
v15 — Cierre dual con botón Agendar + CTA wa.me con ficha prellenada.
"""
import os
import re
import logging
import unicodedata
from urllib.parse import quote

from core.contracts import EMAIL_RE, AgentState
from core.db_client import upsert_conversation, upsert_lead
from graph.intake.utils import _evento_intake, _pack


from graph.utils import (
    RESET_INTAKE,
    _ahora_iso, _apertura_pres, _cfg, _ejecutiva_number, _guardar_ai,
    _primer_nombre, _recent_messages, _telefono_cliente, _wa_link,
    _wa_link_diagnostico, _wa_link_display,
)


logger = logging.getLogger(__name__)

MAX_INTENTOS_CAMPO = 2

SALTAR = ("saltar", "salta", "skip", "omitir", "paso",
          "prefiero no", "prefiero no decir", "prefiero no responder",
          "no quiero responder", "no aplica", "n/a", "-")

RESPUESTAS_REINTENTAR = ("lo intento de nuevo", "intento de nuevo",
                         "intentar de nuevo", "intentarlo de nuevo",
                         "reintentar", "otra vez")
RESPUESTAS_HUMANO = ("hablar con humano", "hablar con una persona",
                     "hablar con un humano", "hablar con una ejecutiva",
                     "hablar con la ejecutiva")


def _detecta_salto(raw: str) -> bool:
    """Detecta intención de saltar una pregunta opcional.

    Acepta frases exactas del catálogo SALTAR y variaciones naturales
    ('quiero saltar esta pregunta', 'paso de esta', etc.), pero descarta
    negaciones como 'no quiero saltar'.
    """
    t = _norm(raw)
    if not t:
        return False

    # 1) Frases exactas del catálogo
    if t in SALTAR:
        return True

    # 2) Negación explícita → no es salto
    if t.startswith("no ") or " no " in t:
        return False

    # 3) Mensaje corto con palabra clave de salto
    palabras = t.split()
    if len(palabras) > 6:
        return False

    claves_salto = {
        "saltar", "salta", "salto", "saltarme", "saltarlo",
        "omitir", "omito", "omítir", "paso", "pasar", "páse",
        "skip", "skipear", "salte",
    }
    if any(p in claves_salto for p in palabras):
        return True

    # 4) Frases típicas de omisión explícita
    omitir_frases = (
        "no quiero responder", "prefiero no responder",
        "prefiero no decir", "no aplica", "n/a", "me salto",
    )
    if any(p in t for p in omitir_frases):
        return True

    return False


# Detección determinista de urgencia (refuerzo 0-LLM al campo `detenido`).
# Solo formulaciones de HECHO VIGENTE en primera persona; las hipotéticas
# ("¿me pueden detener?") las filtra el planner como duda_o_consulta y este
# regex no llega a evaluarse como urgencia.
_URGENTE_RE = re.compile(
    r"(me|lo|la|nos)\s+detuvieron"
    r"|est[aá]\s+detenid[oa]"
    r"|control\s+de\s+detenci[oó]n\s+(es\s+)?(hoy|ma[ñn]ana)"
    r"|me\s+formalizaron"
    r"|tengo\s+orden\s+de\s+arresto"
    r"|orden\s+de\s+arresto\s+en\s+mi\s+contra",
    re.IGNORECASE,
)

DEFAULT_AP_IA = "Soy LeyIA, la asistente virtual de Manzzo y Cía. "
DEFAULT_AP_EXP = ("Te haré unas preguntas rápidas (unos 2 minutos) para "
                "preparar tu caso.\nEn cualquier momento puedes hacer una "
                "pregunta, pedir *hablar con una persona* o escribir "
                "*pausa* para retomar después.")
DEFAULT_AVISO_SALTAR = "Si prefieres no responder, escribe *saltar*."
DEFAULT_ERROR_NO_SALTAR = ("Este dato sí lo necesito para poder derivar tu "
                           "caso 🙏 Si te complica, también puedes pedir "
                           "*hablar con una persona*.")
DEFAULT_OFERTA = ("{nombre}Este dato se nos está resistiendo 😅 ¿Qué "
                  "prefieres?\n• Te paso con una ejecutiva que lo toma "
                  "directamente contigo, o\n• lo intentamos una vez más.")
DEFAULT_REANUDA = "¡Seguimos donde quedamos 💪!"
DEFAULT_REANUDA_DUDA = ("Sobre lo que preguntaste {duda}: espero haberte "
                        "orientado 🙌 Seguimos donde quedamos:")
DEFAULT_CIERRE_LISTO = "🎉 ¡Listo{nombre}! Tu ficha quedó registrada."
DEFAULT_CIERRE_SLA = ("Una ejecutiva revisará tu caso y te contactará *hoy "
                      "dentro del horario hábil* (09:00–18:30). Quedó todo "
                      "anotado a tu nombre.")
DEFAULT_CIERRE_CTAS = (
    "¿Quieres asegurarte más pronto? Agenda la asesoría ahora mismo 👇"
)
DEFAULT_CIERRE_CTA = (
    "Si prefieres que el equipo te contacte directamente, pincha aquí:"
)
DEFAULT_DERIVA = ("Ningún problema{nombre} — te paso directamente con una "
                  "ejecutiva para que lo vean contigo 🤝\n{wa_link}")
DEFAULT_URGENTE = (
    "{nombre}Entiendo — una detención o una orden vigente *no puede "
    "esperar* ⏱️\n"
    "Dejé tu caso marcado como *urgente* para que el equipo te contacte "
    "a la brevedad. Si necesitas hablar ahora mismo, escríbele directo "
    "a la abogada aquí:\n{wa_link}"
)

# Número de la ejecutiva para el CTA wa.me (override vía env var).


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm(s) -> str:
    return _sin_tildes((s or "").strip().lower().rstrip("."))


def _recortar(s, n: int) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def _v_text(minlen: int):
    def check(s):
        s = (s or "").strip()
        if len(s) < minlen:
            return None, f"menos de {minlen} caracteres"
        return s, None
    return check


def _v_email(s):
    s = (s or "").strip().lower().replace(" ", "")
    if not EMAIL_RE.fullmatch(s):
        return None, f"no calza EMAIL_RE: {s!r}"
    return s, None


def _v_si_no(s):
    t = _norm(s)
    if t in ("si", "sipo", "sip", "yes", "claro", "por supuesto",
             "afirmativo", "de acuerdo", "autorizo", "lo autorizo",
             "acepto", "si autorizo", "si, autorizo", "si por supuesto"):
        return True, None
    if t in ("no", "nop", "nope", "negativo", "para nada",
             "no autorizo", "no acepto", "no gracias", "no, gracias"):
        return False, None
    return None, f"no binario: {s!r}"


def _v_opciones(*opciones: str, aliases: dict | None = None):
    alias_map = {_norm(k): v for k, v in (aliases or {}).items()}

    def check(s):
        t = _norm(s)
        if t in alias_map:
            return alias_map[t], None
        if t.isdigit() and 1 <= int(t) <= len(opciones):
            return opciones[int(t) - 1], None
        exact = [o for o in opciones if _norm(o) == t]
        if exact:
            return exact[0], None
        subs = [o for o in opciones if t and t in _norm(o)]
        if len(subs) == 1:
            return subs[0], None
        return None, f"ambigua o fuera de lista: {s!r}"
    return check


def _v_nombre(s):
    s = (s or "").strip()
    palabras = [p for p in s.split() if p]
    if len(s) < 3 or not palabras:
        return None, "nombre no identificable"
    if len(palabras) > 6 or any(len(p) > 20 for p in palabras):
        return None, "parece frase, no nombre"
    return " ".join(p.capitalize() for p in palabras), None


def _botones(*id_title) -> dict:
    return {"kind": "buttons",
            "buttons": [{"id": i, "title": t} for i, t in id_title]}


# ===========================================================================
# RAMAS TEMÁTICAS
# ===========================================================================
# La category del padre es fina para familia pero mete penal/laboral/civil
# en "otro". La rama se resuelve así:
#   1) señales duras ya capturadas (detenido / rol_penal) → penal
#   2) category del padre ∈ familia → familia
#   3) respuesta/extracción de area_caso → texto canónico → rama
#   4) "pendiente" → se pregunta area_caso
# ===========================================================================
_CATS_FAMILIA = frozenset({
    "pension_alimentos", "rebaja_pension", "regimen_visitas",
    "medidas_apremio", "terminacion_pension", "divorcio",
    "compensacion_economica",
})

# Texto canónico almacenado en intake_respuestas["area_caso"] → rama.
_AREA_TXT_A_RAMA = {
    "pensión de alimentos": "familia",
    "régimen de visitas": "familia",
    "divorcio / compensación económica": "familia",
    "medidas de apremio": "familia",
    "derecho penal": "penal",
    "laboral / civil": "general_otra",
    "otro tema": "general_otra",
}

_AREA_LABEL_MAP = {
    "pension_alimentos": "pensión de alimentos",
    "regimen_visitas": "régimen de visitas",
    "divorcio": "divorcio / compensación económica",
    "medidas_apremio": "medidas de apremio",
    "penal": "derecho penal",
    "laboral_civil": "laboral / civil",
    "otro_tema": "otro tema",
}

_ROL_LABEL_MAP = {
    "imputado": "imputado / querellado",
    "victima": "víctima",
    "familiar": "familiar o cercano",
}

_INGRESOS_LABEL_MAP = {
    "dependiente_estable": "trabajo dependiente, sueldo estable",
    "independiente_variable": "independiente / ingresos variables",
    "cesante_reducidos": "cesante o ingresos muy reducidos",
    "jubilado": "jubilado / pensionado",
}


def _rama(state, respuestas=None) -> str:
    """'familia' | 'penal' | 'general_otra' | 'pendiente'."""
    resp = respuestas if respuestas is not None \
        else (state.get("intake_respuestas") or {})

    # (1) señales duras ya capturadas
    if resp.get("detenido") is True or resp.get("rol_penal"):
        return "penal"

    # (2) category del clasificador del padre
    cat = state.get("case_category") or state.get("category") or ""
    if cat in _CATS_FAMILIA:
        return "familia"

    # (3) área declarada o extraída del relato
    area = resp.get("area_caso")
    if area:
        return _AREA_TXT_A_RAMA.get(str(area), "pendiente")

    return "pendiente"


def _es_familia(s, r=None): return _rama(s, r) == "familia"
def _es_penal(s, r=None):   return _rama(s, r) == "penal"
def _area_pendiente(s, r=None): return _rama(s, r) == "pendiente"


def _es_urgencia(state, respuestas: dict) -> bool:
    """Hecho de urgencia vigente: detención u orden de arresto."""
    if respuestas.get("detenido") is True:
        return True
    med = _norm(str(respuestas.get("medidas_apremio") or ""))
    if ("arresto" in med and "ningun" not in med
            and not med.startswith("no ")):
        return True
    return bool(_URGENTE_RE.search(state.get("query") or ""))


# ===========================================================================
# INTERACTIVOS
# ===========================================================================
_INTERACTIVE_CONSENTIMIENTO = _botones(("si", "Sí, autorizo"),
                                       ("no", "No autorizo"))
_BOTONES_OFERTA = _botones(("humano", "Hablar con humano"),
                           ("retry", "Lo intento de nuevo"))
_BOTON_AGENDAR = _botones(("agendar", "Agendar una hora"))
_INTERACTIVE_SI_NO = _botones(("si", "Sí"), ("no", "No"))

_ETAPA_TITULOS = ["Aún no inicio nada", "Me demandaron / avisaron",
                  "Causa en curso", "Hay sentencia previa"]
_INTERACTIVE_ETAPA = {
    "kind": "list", "cta": "Ver opciones", "title": "Estado del caso",
    "options": [{"id": f"etapa_{i + 1}", "title": t}
                for i, t in enumerate(_ETAPA_TITULOS)],
}

_AREA_TITULOS = ["Pensión de alimentos", "Ver a mis hijos",
                 "Divorcio / compensación", "Medidas de apremio",
                 "Tema penal", "Laboral / civil", "Otro tema"]
_INTERACTIVE_AREA = {
    "kind": "list", "cta": "Ver opciones", "title": "Tu caso",
    "options": [{"id": f"area_{i + 1}", "title": t}
                for i, t in enumerate(_AREA_TITULOS)],
}

_ROL_TITULOS = ["Soy el imputado", "Soy la víctima", "Es un familiar"]
_INTERACTIVE_ROL = {
    "kind": "list", "cta": "Ver opciones", "title": "Tu rol en el caso",
    "options": [{"id": f"rol_{i + 1}", "title": t}
                for i, t in enumerate(_ROL_TITULOS)],
}

_INGRESOS_TITULOS = ["Dependiente, sueldo fijo", "Independiente",
                     "Cesante / ingresos bajos", "Jubilado(a)"]
_INTERACTIVE_INGRESOS = {
    "kind": "list", "cta": "Ver opciones", "title": "Tus ingresos",
    "options": [{"id": f"ing_{i + 1}", "title": t}
                for i, t in enumerate(_INGRESOS_TITULOS)],
}


# ===========================================================================
# QUESTIONS — tronco común + módulos por rama
# Orden: la posición define la secuencia; skip_if(state, respuestas) la
# aplica solo si la rama corresponde (se recalcula cada turno).
# ===========================================================================
QUESTIONS: list[dict] = [
    # ================= TRONCO COMÚN =================
    {"id": "consentimiento_datos", "requerida": True, "extractable": False,
     "label": "tu autorización",
     "prompt": ("Para poder revisar tu caso y que un abogado te contacte, "
                "necesito tu *autorización para usar tus datos* (Ley N° "
                "21.719). Los usaremos solo para gestionar tu consulta. "
                "¿Me la das?"),
     "retry_prompt": ("Perdona, necesito un *sí* o un *no* claros: "
                      "¿autorizas el uso de tus datos solo para gestionar "
                      "tu consulta? Puedes usar los botones 👇"),
     "ack": "Gracias ✓",
     "interactive": _INTERACTIVE_CONSENTIMIENTO,
     "validate": _v_si_no},

    {"id": "nombre", "requerida": True,
     "label": "tu nombre",
     "prompt": ("Para partir, ¿me dices tu *nombre completo*, tal como "
                "aparece en tu cédula?"),
     "retry_prompt": ("Perdona, no logré distinguir tu nombre en el mensaje "
                      "🙈 Escríbemelo solo con nombre y apellido — por "
                      "ejemplo: *María Pérez Soto*"),
     "ack": None,
     "validate": _v_nombre},

    {"id": "email", "requerida": True,
     "label": "tu correo",
     "prompt": ("¿A qué *correo* te enviamos el seguimiento de tu caso? "
                "(lo usaremos solo para eso)"),
     "retry_prompt": ("Ese correo no me cuadra 🤔 A veces se pega un "
                      "espacio o falta la '@'. ¿Me lo repites? Algo como "
                      "*maria@gmail.com*"),
     "ack": "Listo, anoté tu correo ✓",
     "validate": _v_email},

    {"id": "situacion_actual", "requerida": True,
     "label": "tu relato",
     "prompt": ("Cuéntame brevemente qué está pasando con tu caso, "
                "con tus palabras:"),
     "retry_prompt": ("Me falta un poco más para derivarte bien: ¿*qué "
                      "ocurrió*, *quiénes* están involucrados y *desde "
                      "cuándo*? (si es un tema penal, solo lo general — "
                      "sin detalles sensibles del hecho)"),
     "ack": None,
     "validate": _v_text(8)},

    # Pregunta clasificadora: solo si la rama no quedó clara por category
    # del padre ni por extracción del relato.
    {"id": "area_caso", "requerida": True,
     "label": "el tipo de caso",
     "skip_if": lambda s, r=None: not _area_pendiente(s, r),
     "prompt": ("¿Cuál de estos se acerca más a tu situación?\n"
                "  1) Pensión de alimentos (me demandan, quiero rebajarla "
                "o terminarla)\n"
                "  2) Ver a mis hijos (régimen de visitas)\n"
                "  3) Divorcio o compensación económica\n"
                "  4) Medidas de apremio (embargo, licencia retenida, "
                "registro de deudores)\n"
                "  5) Tema penal\n"
                "  6) Laboral / civil\n"
                "  7) Otro tema\n"
                "(elige de la lista o respóndeme con el número)"),
     "retry_prompt": ("No logré ubicarlo entre las opciones 😅 Elige de la "
                      "lista o respóndeme solo con el número (1 al 7):"),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_AREA,
     "label_map": _AREA_LABEL_MAP,
     "validate": _v_opciones(
         "pensión de alimentos", "régimen de visitas",
         "divorcio / compensación económica", "medidas de apremio",
         "derecho penal", "laboral / civil", "otro tema",
         aliases={"ver a mis hijos": "régimen de visitas",
                  "divorcio / compensación": "divorcio / compensación económica",
                  "tema penal": "derecho penal"})},

    {"id": "etapa_proceso", "requerida": False,
     "label": "la etapa del caso",
     "skip_if": lambda s, r=None: _es_penal(s, r),  # penal tiene etapa propia
     "prompt": ("¿En qué estado está tu caso hoy?\n"
                "  1) Aún no he iniciado acciones\n"
                "  2) Fui demandado(a) o notificado(a)\n"
                "  3) Hay una causa judicial en curso\n"
                "  4) Ya existe sentencia o acuerdo previo\n"
                "(elige de la lista o respóndeme con el número)"),
     "retry_prompt": ("No logré ubicarlo entre las opciones 😅 Elige de la "
                      "lista o respóndeme solo con el número (1 al 4):"),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_ETAPA,
     "validate": _v_opciones(
         "aún no inicio nada", "me demandaron o me notificaron",
         "causa en curso", "sentencia o acuerdo previa",
         aliases={"aún no inicio nada": "aún no inicio nada",
                  "me demandaron / avisaron": "me demandaron o me notificaron",
                  "causa en curso": "causa en curso",
                  "hay sentencia previa": "sentencia o acuerdo previa"}),
     "label_map": {"sin_inicio": "aún no inicio nada",
                   "demandado": "me demandaron o me notificaron",
                   "causa_en_curso": "causa en curso",
                   "sentencia_previa": "sentencia o acuerdo previa"}},

    # ================= MÓDULO FAMILIA =================
    {"id": "hijos_menores", "requerida": False,
     "label": "los hijos menores",
     "skip_if": lambda s, r=None: not _es_familia(s, r),
     "prompt": ("¿Hay hijos menores de edad involucrados? Si es así, "
                "¿cuántos y de qué edades? (ej: *2, de 5 y 8 años*; si no "
                "hay, responde *no*)"),
     "retry_prompt": ("Solo necesito el número y las edades aproximadas — "
                      "por ejemplo: *una hija de 12 años* — o *no* si no "
                      "hay hijos menores."),
     "ack": "Anotado ✓",
     "validate": _v_text(1)},

    {"id": "ingresos_rango", "requerida": False,
     "label": "tu situación de ingresos",
     "skip_if": lambda s, r=None: not _es_familia(s, r),
     "prompt": ("Para evaluar montos justos (pensión, acuerdos), ¿cuál es "
                "tu situación laboral hoy?\n"
                "  1) Trabajo dependiente, sueldo estable\n"
                "  2) Independiente / ingresos variables\n"
                "  3) Cesante o ingresos muy reducidos\n"
                "  4) Jubilado(a) / pensionado(a)"),
     "retry_prompt": ("Elige de la lista o respóndeme con el número "
                      "(1 al 4):"),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_INGRESOS,
     "label_map": _INGRESOS_LABEL_MAP,
     "validate": _v_opciones(
         "trabajo dependiente, sueldo estable",
         "independiente / ingresos variables",
         "cesante o ingresos muy reducidos",
         "jubilado / pensionado",
         aliases={"dependiente, sueldo fijo": "trabajo dependiente, sueldo estable",
                  "independiente": "independiente / ingresos variables",
                  "cesante / ingresos bajos": "cesante o ingresos muy reducidos",
                  "jubilado(a)": "jubilado / pensionado"})},

    {"id": "medidas_apremio", "requerida": False,
     "label": "las medidas de apremio",
     "skip_if": lambda s, r=None: not _es_familia(s, r),
     "prompt": ("¿Tienes alguna medida vigente en tu contra? Puedes indicar "
                "varias:\n"
                "  1) Retención de licencia de conducir\n"
                "  2) Prohibición de salir del país\n"
                "  3) Embargo o retención de sueldo / AFP\n"
                "  4) Orden de arresto\n"
                "  5) Ninguna\n"
                "(o cuéntamelo con tus palabras)"),
     "retry_prompt": ("Léeme las que apliquen con su número (ej: *1 y 3*), "
                      "o responde *ninguna*."),
     "ack": "Anotado ✓",
     "validate": _v_text(2)},

    # ================= MÓDULO PENAL =================
    # Nota de diseño: en penal la ficha es deliberadamente CORTA y no pide
    # detalles del hecho — un imputado no debe autoincriminarse por
    # WhatsApp. Lo relevante es rol + urgencia + audiencia, y derivar.
    {"id": "rol_penal", "requerida": True,
     "label": "tu rol en el caso",
     "skip_if": lambda s, r=None: not _es_penal(s, r),
     "prompt": ("¿Cuál es tu relación con el caso?\n"
                "  1) Soy el imputado (me acusan / querellaron)\n"
                "  2) Soy la víctima\n"
                "  3) Es un familiar o cercano"),
     "retry_prompt": ("Solo necesito ubicar tu rol: responde con el número "
                      "(1 al 3) o elige de la lista:"),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_ROL,
     "label_map": _ROL_LABEL_MAP,
     "validate": _v_opciones(
         "imputado / querellado", "víctima", "familiar o cercano",
         aliases={"soy el imputado": "imputado / querellado",
                  "soy la víctima": "víctima",
                  "es un familiar": "familiar o cercano",
                  "me acusan": "imputado / querellado",
                  "me querellaron": "imputado / querellado"})},

    {"id": "detenido", "requerida": True,
     "label": "si hay alguien detenido",
     "skip_if": lambda s, r=None: not _es_penal(s, r),
     "prompt": ("Pregunta clave: ¿hay alguien *detenido ahora* o con "
                "audiencia de control de detención en las próximas horas?"),
     "retry_prompt": ("Necesito un *sí* o un *no*: ¿hay alguien detenido "
                      "en este momento?"),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_SI_NO,
     "validate": _v_si_no},

    {"id": "audiencia_proxima", "requerida": False,
     "label": "la audiencia próxima",
     "skip_if": lambda s, r=None: not _es_penal(s, r),
     "prompt": ("¿Tienes fecha de alguna audiencia, citación o "
                "formalización? ¿Para cuándo es?"),
     "retry_prompt": ("Con la fecha aproximada basta — por ejemplo: "
                      "*el martes* o *aún no me citan*."),
     "ack": "Anotado ✓",
     "validate": _v_text(2)},

    # ================= CIERRE (todas las ramas) =================
    {"id": "comuna", "requerida": False,
     "label": "tu comuna",
     "prompt": ("Última pregunta: ¿en qué *comuna o ciudad* estás? (para "
                "ubicar el tribunal que corresponde — atendemos en todo "
                "Chile)"),
     "retry_prompt": ("Solo el nombre de la comuna o ciudad — por ejemplo: "
                      "*Puente Alto* o *Concepción*."),
     "ack": "Anotado ✓",
     "validate": _v_text(3)},
]

_BY_ID = {q["id"]: q for q in QUESTIONS}


def _aplicables(state: AgentState, respuestas: dict | None = None) -> list[dict]:
    resp = respuestas if respuestas is not None else (state.get("intake_respuestas") or {})
    return [q for q in QUESTIONS
            if not q.get("skip_if", lambda s, r=None: False)(state, resp)]


def _pendientes(state: AgentState, respuestas: dict) -> list[dict]:
    hechas = {q["id"] for q in _aplicables(state, respuestas)
              if q["id"] in respuestas}
    return [q for q in _aplicables(state, respuestas) if q["id"] not in hechas]


def _form_pregunta(state: AgentState, q: dict, respuestas: dict,
                   nombre: str | None = None) -> str:
    sec = _cfg()["intake"]
    total = len(_aplicables(state, respuestas))
    restantes = len(_pendientes(state, respuestas))
    if restantes <= 1:
        head = f"Ya casi{', ' + nombre if nombre else ''} 🙌 Solo me falta esto:\n"
    else:
        head = f"_{total - restantes + 1} de {total}_\n"
    body = q["prompt"]
    if not q["requerida"]:
        body += "\n" + sec.get("aviso_saltar", DEFAULT_AVISO_SALTAR)
    return head + body


def _idx_of(qid: str) -> int:
    return next(i for i, q in enumerate(QUESTIONS) if q["id"] == qid)


def _interactive_de(q: dict) -> dict | None:
    return q.get("interactive")


def _ack_campo(qid: str, respuestas: dict) -> str:
    nombre = _primer_nombre(respuestas.get("nombre"))
    if qid == "nombre":
        return f"Perfecto, {nombre} ✓" if nombre else "Perfecto ✓"
    if qid == "situacion_actual":
        resumen = str(respuestas.get(qid) or "").strip()
        if len(resumen) > 180:
            resumen = resumen[:177].rstrip() + "…"
        return (f"Gracias por contármelo{', ' + nombre if nombre else ''}; "
                "sé que no siempre es fácil 🙏 "
                f"Si te entendí bien: “{resumen}”\n"
                "Si se me escapó algo, dímelo; si no, seguimos 👇")
    tpl = _BY_ID[qid].get("ack") or "Anotado ✓"
    return tpl.format(nombre=nombre or "")


def _extras_turno(q_actual: dict, antes: dict, despues: dict) -> list[str]:
    return [qid for qid in _BY_ID
            if qid != q_actual["id"]
            and despues.get(qid) is not None
            and despues.get(qid) != antes.get(qid)]


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " y " + labels[-1]


def _acks_turno(q_actual: dict, antes: dict, respuestas: dict) -> list[str]:
    acks: list[str] = []
    qid = q_actual["id"]
    if respuestas.get(qid) is not None and respuestas.get(qid) != antes.get(qid):
        acks.append(_ack_campo(qid, respuestas))
    elif qid in respuestas and respuestas[qid] is None and qid not in antes:
        acks.append("Ningún problema, lo dejamos así ✓")
    extras = _extras_turno(q_actual, antes, respuestas)
    if extras:
        acks.append("De paso ya tengo "
                    + _join_labels([_BY_ID[q]["label"] for q in extras])
                    + " 👌")
    return acks


def _aplicar_extraccion(respuestas: dict, decision: dict,
                        skipped: set | None = None) -> None:
    for qid, qdef in _BY_ID.items():
        if qid in (skipped or set()):
            continue
        if not qdef.get("extractable", True):
            continue
        raw = decision.get(qid)
        if raw is None:
            continue
        if isinstance(raw, bool):
            respuestas[qid] = raw
            continue
        if "label_map" in qdef:
            raw = qdef["label_map"].get(str(raw))
            if raw is None:
                continue
        val, err = qdef["validate"](raw)
        if err is None and val is not None:
            respuestas[qid] = val
        else:
            logger.info("[intake] extracción rechazada por validador: %s=%r (%s)",
                        qid, raw, err)


def _lead_updates(respuestas: dict) -> dict:
    """Copia nombre/email capturados en intake a lead_nombre/lead_email.

    Booking y handoff_humano leen estos canales como memoria larga del
    lead, por lo que es crítico sincronizarlos al salir del intake.
    No sobrescribe si ya existen.
    """
    updates = {}
    if respuestas.get("nombre"):
        updates["lead_nombre"] = respuestas["nombre"]
    if respuestas.get("email"):
        updates["lead_email"] = respuestas["email"]
    return updates


def _persistir_parcial(state: AgentState, respuestas: dict) -> None:
    if respuestas.get("consentimiento_datos") is not True:
        return
    try:
        upsert_lead(
            thread_id=state.get("thread_id", ""),
            intake_respuestas=respuestas,
            category=state.get("category"),
            completed=False,
        )
    except Exception as e:
        logger.warning("upsert_lead parcial falló (best-effort): %s", e)


def _texto_ficha_wa(respuestas: dict, thread_id: str,
                    categoria: str) -> str:
    """Ficha comprimida para el wa.me. v17: la categoría llega por parámetro
    (antes se leía de respuestas, donde nunca existió → "consulta_general")."""
    nombre = respuestas.get("nombre") or ""
    email = respuestas.get("email") or ""
    telefono = respuestas.get("telefono") or ""

    lineas = [
        f"Hola, soy {nombre}. Acabo de completar mi registro. "
        "Resumen de mi caso:",
        f"• Área: {categoria or 'sin clasificar'}",
    ]
    if respuestas.get("area_caso"):
        lineas.append(f"• Tipo de caso: {_recortar(respuestas['area_caso'], 40)}")
    if respuestas.get("urgencia_legal"):
        lineas.append("• ⚠️ URGENTE: detención u orden vigente")
    if respuestas.get("rol_penal"):
        lineas.append(f"• Rol (penal): {respuestas['rol_penal']}")
    if respuestas.get("audiencia_proxima"):
        lineas.append(
            f"• Audiencia: {_recortar(respuestas['audiencia_proxima'], 60)}")
    if respuestas.get("etapa_proceso"):
        lineas.append(f"• Etapa: {_recortar(respuestas['etapa_proceso'], 40)}")
    if respuestas.get("hijos_menores"):
        lineas.append(
            f"• Hijos menores: {_recortar(respuestas['hijos_menores'], 40)}")
    if respuestas.get("ingresos_rango"):
        lineas.append(
            f"• Ingresos: {_recortar(respuestas['ingresos_rango'], 40)}")
    if respuestas.get("medidas_apremio"):
        lineas.append(
            f"• Medidas: {_recortar(respuestas['medidas_apremio'], 80)}")
    if respuestas.get("comuna"):
        lineas.append(f"• Comuna: {_recortar(respuestas['comuna'], 40)}")
    lineas += [
        f"• Correo: {email}",
        f"• Teléfono: {telefono}",
        f"• Mi situación: {_recortar(respuestas.get('situacion_actual'), 240)}",
        f"(ID de mi atención: {thread_id})",
    ]
    return "\n".join(lineas)


def _cta_humano_con_ficha(respuestas: dict, thread_id: str,
                          categoria: str) -> dict:
    """CTA URL 'Pincha aquí' que abre wa.me con la ficha prellenada."""
    texto = _texto_ficha_wa(respuestas, thread_id, categoria)
    url = f"https://wa.me/{_ejecutiva_number()}?text={quote(texto)}"
    return {
        "text": DEFAULT_CIERRE_CTA,
        "interactive": {
            "kind": "cta",
            "text": "Pincha aquí",
            "url": url,
        },
    }



def iniciar_ficha(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas: dict = {}
    skipped = set(state.get("intake_skipped") or ())
    tel = _telefono_cliente(state)
    if tel:
        respuestas["telefono"] = tel

    _aplicar_extraccion(respuestas, state.get("intake_decision") or {},
                        skipped=skipped)

    # Urgencia ya presente al iniciar (caso borde: decisión pre-poblada).
    if _es_urgencia(state, respuestas):
        return _derivar_urgente(state, respuestas, skipped=skipped)

    pend = _pendientes(state, respuestas)
    q0 = pend[0]

    # Solo presentarnos si el asistente aún no lo hizo en este hilo
    burbujas = []
    apertura_txt = _apertura_pres(state)
    if apertura_txt:
        burbujas.append(sec.get("apertura_ia") or sec.get("apertura") or DEFAULT_AP_IA)

    burbujas += [
        sec.get("apertura_expectativa") or DEFAULT_AP_EXP,
        _form_pregunta(state, q0, respuestas),
    ]
    pack = _pack(burbujas, interactive=_interactive_de(q0))
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_iniciado", state, pregunta=q0["id"],
                   rama=_rama(state, respuestas))
    return {
        **pack,
        "intake_activo": True,
        "intake_idx": _idx_of(q0["id"]),
        "intake_respuestas": respuestas,
        "intake_attempts": 0,
        "intake_exit": None,
        "intake_resume": False,
        "intake_decision": None,
        "intake_stage": "preguntando",
        "intake_oferta_qid": None,
        "intake_skipped": skipped,
        "asistente_presentado": True,
    }


def procesar_respuesta(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas = dict(state.get("intake_respuestas") or {})
    antes = dict(respuestas)
    decision = state.get("intake_decision") or {}
    attempts = int(state.get("intake_attempts") or 0)
    raw = (state.get("query") or "").strip()
    skipped = set(state.get("intake_skipped") or ())

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas, skipped=skipped)
    q_actual = pend[0]
    nombre = _primer_nombre(respuestas.get("nombre"))
    ya_ofrecido = state.get("intake_oferta_qid") == q_actual["id"]

    if ya_ofrecido and _norm(raw) in RESPUESTAS_REINTENTAR:
        pack = _pack([f"Dale 💪 {q_actual['retry_prompt']}"],
                     interactive=_interactive_de(q_actual))
        _guardar_ai(state, pack["response"])
        return {
            **pack,
            "intake_respuestas": respuestas,
            "intake_attempts": attempts,
            "intake_decision": None,
            "intake_skipped": skipped,
            "asistente_presentado": True,
        }

    error_key = None
    if _detecta_salto(raw):
        if q_actual["requerida"]:
            error_key = "no_skip"
        else:
            respuestas[q_actual["id"]] = None
            skipped.add(q_actual["id"])
            logger.info("[intake] campo opcional saltado: %s",
                        q_actual["id"])
    else:
        _aplicar_extraccion(respuestas, decision, skipped=skipped)
        if q_actual["id"] not in respuestas:
            val, err = q_actual["validate"](raw)
            if err:
                logger.info("[intake] validador rechazó %s: %s",
                            q_actual["id"], err)
                error_key = "retry"
            else:
                respuestas[q_actual["id"]] = val

    if respuestas.get("consentimiento_datos") is False:
        return sin_consentimiento(state, respuestas)

    # v17: la urgencia corta la ficha ANTES de seguir preguntando
    # (un detenido no puede esperar el resto del cuestionario).
    if _es_urgencia(state, respuestas):
        return _derivar_urgente(state, respuestas, skipped=skipped)

    # GUARDA: asegurar que un salto opcional nunca deje el intake atascado.
    if (q_actual["id"] in respuestas
            and respuestas[q_actual["id"]] is None
            and q_actual["id"] not in antes):
        logger.info("[intake] avanzando tras salto opcional de %s",
                    q_actual["id"])

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas, skipped=skipped)

    if error_key:
        extras = _extras_turno(q_actual, antes, respuestas)
        if not extras:
            attempts += 1

        if (not extras and q_actual["requerida"]
                and attempts >= MAX_INTENTOS_CAMPO):
            if ya_ofrecido:
                logger.info("[intake] reintento agotado en '%s' → "
                            "derivación parcial", q_actual["id"])
                return _derivar_parcial(state, respuestas, skipped=skipped)
            msg_oferta = sec.get("oferta_salida", DEFAULT_OFERTA).format(
                nombre=f"{nombre}, " if nombre else "")
            pack = _pack([msg_oferta], interactive=_BOTONES_OFERTA)
            _guardar_ai(state, pack["response"])
            return {
                **pack,
                "intake_respuestas": respuestas,
                "intake_attempts": attempts,
                "intake_decision": None,
                "intake_oferta_qid": q_actual["id"],
                "intake_skipped": skipped,
                "asistente_presentado": True,
            }

        burbujas = []
        if extras:
            burbujas.append("De paso ya tengo " + _join_labels(
                [_BY_ID[q]["label"] for q in extras]) + " 👌")
        if error_key == "no_skip":
            burbujas.append(sec.get("error_no_saltar",
                                    DEFAULT_ERROR_NO_SALTAR))
        else:
            burbujas.append(q_actual.get("retry_prompt")
                            or q_actual["prompt"])
        pack = _pack(burbujas, interactive=_interactive_de(q_actual))
        _guardar_ai(state, pack["response"])
        return {
            **pack,
            "intake_respuestas": respuestas,
            "intake_attempts": attempts,
            "intake_decision": None,
            "intake_skipped": skipped,
            "asistente_presentado": True,
        }

    _persistir_parcial(state, respuestas)

    siguiente = pend[0]
    pack = _pack(
        _acks_turno(q_actual, antes, respuestas)
        + [_form_pregunta(state, siguiente, respuestas, nombre)],
        interactive=_interactive_de(siguiente),
    )
    _guardar_ai(state, pack["response"])
    return {
        **pack,
        "intake_idx": _idx_of(siguiente["id"]),
        "intake_respuestas": respuestas,
        "intake_attempts": 0,
        "intake_oferta_qid": None,
        "intake_decision": None,
        "intake_skipped": skipped,
        "asistente_presentado": True,
    }


def _completar_ficha(state: AgentState, respuestas: dict,
                     skipped: set | None = None) -> AgentState:
    sec = _cfg()["intake"]
    nombre = _primer_nombre(respuestas.get("nombre"))
    tid = state.get("thread_id", "")
    categoria = state.get("case_category") or state.get("category") or ""

    burbujas = [
        sec.get("cierre_listo", DEFAULT_CIERRE_LISTO).format(
            nombre=f", {nombre}" if nombre else ""),
        sec.get("cierre_sla", DEFAULT_CIERRE_SLA),
        sec.get("cierre_ctas", DEFAULT_CIERRE_CTAS),
    ]

    pack = _pack(burbujas, interactive=_BOTON_AGENDAR)

    # Segundo mensaje: CTA "Pincha aquí" con la ficha comprimida en URL.
    # v17 FIX: se pasa la categoría explícitamente (antes leía una clave
    # inexistente en respuestas → "consulta_general" en el 100% de casos).
    pack["response_cta"] = _cta_humano_con_ficha(respuestas, tid, categoria)

    try:
        # FIX #10 (se mantiene): notificar_crm=False porque route_post_intake
        # SIEMPRE envía intake_completado=True a ROUTE_HANDOFF en el mismo
        # turno, y handoff_humano es el emisor único. Solo se persiste.
        upsert_lead(
            thread_id=tid,
            intake_respuestas=respuestas,
            category=state.get("category"),
            completed=True,
            notificar_crm=False,
        )
    except Exception as e:
        logger.warning("upsert_lead (completado) falló: %s", e)

    _guardar_ai(state, pack["response"])
    _evento_intake("intake_completado", state,
                   respuestas_keys=list(respuestas.keys()),
                   rama=_rama(state, respuestas),
                   tiene_email=bool(respuestas.get("email")))
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": True,
        "intake_respuestas": respuestas,
        "intake_exit": "cta_cierre",
        "intake_decision": None,
        "intake_stage": "completado",
        "intake_oferta_qid": None,
        "intake_skipped": skipped or set(),
        "asistente_presentado": True,
        **_lead_updates(respuestas),
    }


def pausar_ficha(state: AgentState) -> AgentState:
    """El cliente pide pausa: detener la ficha y esperar al próximo mensaje."""
    skipped = set(state.get("intake_skipped") or ())
    pack = _pack(["Perfecto, quedo aquí esperando. Cuando quieras retomar, "
                  "solo escríbeme 🤙"])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_pausado", state)
    return {
        **pack,
        "intake_activo": True,
        "intake_resume": False,
        "intake_exit": "pausa",
        "intake_stage": "pausado",
        "intake_decision": None,
        "intake_oferta_qid": None,
        "intake_skipped": skipped,
        "asistente_presentado": True,
    }


def pausar_para_faq(state: AgentState) -> AgentState:
    logger.info("[intake] duda lateral → pausa a FAQ")
    _evento_intake("intake_duda_lateral", state)
    return {
        "intake_exit": "faq",
        "intake_resume": True,
        "intake_stage": "pausado",
        "intake_decision": None,
        "response_interactive": None,
    }


def _ultima_duda(state: AgentState) -> str | None:
    actual = (state.get("query") or "").strip()
    for m in reversed(_recent_messages(state, k=8)):
        if getattr(m, "type", "") != "human":
            continue
        txt = (m.content or "").strip()
        if not txt or txt == actual:
            continue
        if "?" not in txt and "¿" not in txt:
            return None
        frag = txt if len(txt) <= 90 else txt[:87].rstrip() + "…"
        return f"«{frag}»"
    return None


def reanudar(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    pend = _pendientes(state, respuestas)
    base = {
        "intake_exit": None,
        "intake_resume": False,
        "intake_decision": None,
        "intake_stage": "preguntando",
        "intake_oferta_qid": None,
        "response_interactive": None,
        "intake_skipped": skipped,
    }
    if not pend or not state.get("intake_activo"):
        return base

    q = pend[0]
    nombre = _primer_nombre(respuestas.get("nombre"))
    duda = _ultima_duda(state)
    puente = (sec.get("reanudar_duda", DEFAULT_REANUDA_DUDA).format(duda=duda)
              if duda else sec.get("reanudar", DEFAULT_REANUDA))

    # FIX #3 (se mantiene): preservar la respuesta del FAQ del MISMO turno
    # en round-trips laterales (intake → FAQ → intake).
    msgs = state.get("messages") or []
    viene_de_faq = bool(msgs and getattr(msgs[-1], "type", "") == "ai")
    previas = list(state.get("response_bubbles") or []) if viene_de_faq else []

    pregunta = _form_pregunta(state, q, respuestas, nombre)
    nuevo = _pack([puente, pregunta], interactive=_interactive_de(q))

    if previas:
        pack = _pack([*previas, puente, pregunta],
                     interactive=_interactive_de(q))
    else:
        pack = nuevo

    _guardar_ai(state, nuevo["response"])

    _evento_intake("intake_reanudado", state, pregunta=q["id"])
    return {**base, **pack, "asistente_presentado": True}


def derivar_parcial(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    return _derivar_parcial(state, respuestas, skipped=skipped)


def _derivar_parcial(state: AgentState, respuestas: dict,
                     skipped: set | None = None) -> AgentState:
    nombre = _primer_nombre(respuestas.get("nombre"))
    msg = _cfg()["intake"].get("derivacion_parcial", DEFAULT_DERIVA).format(
        nombre=f", {nombre}" if nombre else "",
        wa_link=_wa_link_display(
            f"Hola, soy {respuestas.get('nombre', '') or 'un cliente del asistente'}, "
            "prefiero continuar directamente con la ejecutiva.",
            display="Hablar con un humano",
        )
    )
    _persistir_parcial(state, respuestas)
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_derivacion_parcial", state,
                   respuestas_keys=list(respuestas.keys()))
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": respuestas,
        "intake_exit": "handoff",
        "intake_stage": "completado",
        "intake_decision": None,
        "intake_oferta_qid": None,
        "intake_skipped": skipped or set(),
        "asistente_presentado": True,
        **_lead_updates(respuestas),
    }


def derivar_urgente(state: AgentState) -> AgentState:
    """Acción invocable por el planner (detenido=True detectado por LLM)."""
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    _aplicar_extraccion(respuestas, state.get("intake_decision") or {},
                        skipped=skipped)
    return _derivar_urgente(state, respuestas, skipped=skipped)


def _derivar_urgente(state: AgentState, respuestas: dict,
                     skipped: set | None = None) -> AgentState:
    """v17 — Detención vigente u orden de arresto → handoff inmediato.

    Emite intake_exit="handoff" (mismo valor que derivar_parcial: el
    IntakeExitLabel del padre no admite valores nuevos, y el handoff es
    además el emisor único de notificaciones — FIX #10). La marca
    urgencia_legal=True viaja en intake_respuestas: el equipo la ve en el
    lead y handoff_humano la incluye en el resumen que notifica.
    """
    respuestas = {**respuestas, "urgencia_legal": True}
    nombre = _primer_nombre(respuestas.get("nombre"))
    msg = _cfg()["intake"].get("derivacion_urgente", DEFAULT_URGENTE).format(
        nombre=f"{nombre}, " if nombre else "",
        wa_link=_wa_link_display(
            f"URGENTE — soy {respuestas.get('nombre', '') or 'un cliente del asistente'}: "
            "hay una detención u orden vigente. Mi ficha quedó registrada, "
            "necesito hablar con la abogada ahora.",
            display="Hablar con la abogada ahora",
        )
    )
    _persistir_parcial(state, respuestas)
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_derivacion_urgente", state,
                   respuestas_keys=list(respuestas.keys()),
                   gatillo=("detenido" if respuestas.get("detenido") is True
                            else "apremio_o_regex"))
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": respuestas,
        "intake_exit": "handoff",
        "intake_stage": "completado",
        "intake_decision": None,
        "intake_oferta_qid": None,
        "intake_skipped": skipped or set(),
        "asistente_presentado": True,
        **_lead_updates(respuestas),
    }


def abandonar_ficha(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    _persistir_parcial(state, respuestas)
    msg = _cfg()["intake"]["cierre_abandono"].format(
        wa_link=_wa_link_display(display="Hablar con un humano")
    )
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])

    # FIX #4 (se mantiene): cerrar la conversación en BD ANTES de retornar
    # closed=True, para que el gatekeeper de sesión resetee el flujo en el
    # próximo mensaje (status="abandonado" para analítica del embudo).
    upsert_conversation(
        thread_id=state.get("thread_id", ""),
        status="abandonado",
        category=state.get("case_category") or state.get("category"),
        closed=True,
    )

    _evento_intake("intake_abandonado", state,
                   respuestas_keys=list(respuestas.keys()))
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": False,
        "intake_exit": "end",
        "intake_stage": "abandonado",
        "intake_decision": None,
        "intake_oferta_qid": None,
        "closed": True,
        "intake_skipped": skipped,
        "asistente_presentado": True,
        **_lead_updates(respuestas),
    }


def sin_consentimiento(state: AgentState, respuestas: dict | None = None) -> AgentState:
    skipped = set(state.get("intake_skipped") or ())
    msg = _cfg()["intake"]["sin_consentimiento"].format(
        wa_link=_wa_link_display(display="Hablar con un humano")
    )
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])

    # FIX #4 (se mantiene): la negativa de consentimiento (Ley 21.719)
    # también cierra la sesión en BD (status="sin_consentimiento").
    upsert_conversation(
        thread_id=state.get("thread_id", ""),
        status="sin_consentimiento",
        category=state.get("case_category") or state.get("category"),
        closed=True,
    )

    _evento_intake("intake_sin_consentimiento", state)
    return {
        **pack,
        "intake_activo": False,
        "intake_completado": False,
        "intake_respuestas": (respuestas if respuestas is not None
                              else state.get("intake_respuestas")),
        "intake_exit": "end",
        "intake_decision": None,
        "intake_stage": "sin_consentimiento",
        "intake_oferta_qid": None,
        "closed": True,
        "intake_skipped": skipped,
        "asistente_presentado": True,
        **_lead_updates(respuestas or {}),
    }
