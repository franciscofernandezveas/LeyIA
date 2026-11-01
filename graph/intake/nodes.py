"""graph/intake/nodes.py — Acciones del sub-agente INTAKE.

v21 — Fixes de producción (sin tocar contracts; sigue compat core v6.8 /
intake v17). Evidencia en log del 2026-11-01:

  BUG 1 — "Saltar" pausaba la ficha:
    El planner (IntakeDecision.tipo) leyó "Saltar" como 'pausar' y el router
    ejecutó pausar_ficha: _detecta_salto jamás corrió.
    → pausar_ficha ahora tiene GUARDA: si el input es un salto válido para
      la pregunta activa, registra el campo como None (skipped) y AVANZA.
      Si la pregunta es requerida, explica y repregunta. Jamás pausa por
      un salto. La pausa genuina queda intacta.

  BUG 2 — Retry fantasma tras tocar la lista:
    Log: "validador rechazó etapa_proceso: 'Pensión de alimentos'".
    El usuario tocó la lista de ÁREA mientras el sistema creía activa la
    pregunta ETAPA (el planner había contaminado area_caso re-extrayendo
    del historial — incluido el EJEMPLO del propio prompt del bot).
    → RESCATE CRUZADO (_rescue_pendiente): si el input no valida para la
      pregunta activa pero calza con EXACTAMENTE UNA otra pendiente de
      opciones, se adjudica ahí y el turno cuenta como éxito (ack propio,
      cero attempts, siguiente pregunta real).
    → EL USUARIO MANDA SOBRE LA EXTRACCIÓN: _aplicar_extraccion solo llena
      campos AUSENTES y marca lo que llenó en la clave interna "_extraidos".
      Si el usuario responde explícitamente un campo extraído con otro
      valor, su input lo sobrescribe (_fijar_por_usuario).
    → ANTI-CONTAMINACIÓN: valores de texto libre que contienen frases del
      propio bot ("si te entendí bien", el ejemplo «me demandaron por la
      pensión», etc.) o un email en situacion_actual se descartan.
      (El transcript mostraba al usuario reenviando/citando burbujas del
      bot; sin este filtro el planner se autoalimenta.)

  Además:
    - La traducción de inputs interactivos ahora incluye los TÍTULOS de las
      opciones (el dispatcher en producción entrega el título: "Pensión de
      alimentos", "Aún no he iniciado nada"), no solo los ids.
    - _avance_normal: un solo camino para "éxito / salto opcional /
      rescate" (lo usan procesar_respuesta y pausar_ficha).

  De v20 (se mantiene): numeración IVR fuera, confirmación de área 1-toque
  con "_area_sugerida" como clave INTERNA de intake_respuestas (nada de
  claves top-level nuevas en AgentState), mapa numérico alineado al orden
  legado del contrato (1-7, apremio→pensión), bandas de confianza solo si
  el planner la calibra (>0), _salida_intake, _cerrar_con_status,
  _pack_pregunta, _detecta_salto endurecido ("eso ya pasó" no es salto),
  _es_urgencia solo afirmativo/vigente, notificar_crm=False en parcial.
"""
import re
import logging
import unicodedata
from urllib.parse import quote

from core.contracts import EMAIL_RE, AgentState
from core.db_client import upsert_conversation, upsert_lead
from graph.intake.utils import _evento_intake, _pack

from graph.utils import (
    RESET_INTAKE,  # noqa: F401  (re-export para el subgrafo)
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
                         "reintentar", "otra vez", "dale", "ya, dale")
RESPUESTAS_HUMANO = ("hablar con humano", "hablar con una persona",
                     "hablar con un humano", "hablar con una ejecutiva",
                     "hablar con la ejecutiva", "hablar con un abogado",
                     "hablar con una abogada")

_CONFIRMA_SI = frozenset({
    "si", "sipo", "sip", "yes", "asi es", "exacto", "exactamente",
    "correcto", "claro", "eso", "eso mismo", "afirmativo", "de acuerdo",
    "por supuesto", "ya", "ya po",
})
_CONFIRMA_NO = frozenset({
    "no", "nop", "nope", "no es eso", "es otra cosa", "otra cosa",
    "no es eso no", "no exactamente", "en realidad no", "cambia", "cambiar",
})

# Bandas de IntakeDecision.confianza. SOLO se aplican si el planner manda
# confianza > 0; con el default del contrato (0.0) la extracción fija el
# dato directo, como siempre (cero regresión).
_UMBRAL_AREA_DIRECTA = 0.75
_UMBRAL_AREA_SUGERIDA = 0.40


# ===========================================================================
# TEXTO / NORMALIZACIÓN
# ===========================================================================

def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _norm(s) -> str:
    return _sin_tildes((s or "").strip().lower().rstrip("."))


def _recortar(s, n: int) -> str:
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


# ===========================================================================
# FUENTES ÚNICAS DE OPCIONES
# De AREAS/ETAPAS/ROLES/INGRESOS se derivan: títulos UI, interactivos,
# label_maps (extracción del planner), keywords (validador) y ramas.
# Keywords SIN tildes (se comparan contra _norm).
# ===========================================================================

AREAS: list[dict] = [
    {"canon": "pensión de alimentos",
     "ui": "Pensión de alimentos",
     "frase": "la *pensión de alimentos*",
     "rama": "familia",
     "ids_ext": {"pension_alimentos", "rebaja_pension", "terminacion_pension",
                 "medidas_apremio", "alimentos"},
     "kw": {"pension", "alimento", "alimentos", "alimentaria",
            "cuota alimentaria", "cuota", "rebajar", "rebaja", "rebajarla",
            "aumento de pension", "aumentar la pension", "dejar de pagar",
            "no me pasa la pension", "no me paga", "debe la pension",
            "embargo", "embargaron", "embargada", "embargado",
            "licencia retenida", "retuvieron la licencia", "apremio",
            "apremios", "arresto", "registro de deudores", "arraigo",
            "prohibicion de salir", "retencion de sueldo", "retencion"}},
    {"canon": "régimen de visitas",
     "ui": "Ver a mis hijos (visitas)",
     "frase": "*ver a tus hijos* (el régimen de visitas)",
     "rama": "familia",
     "ids_ext": {"regimen_visitas", "visitas", "relacion_directa"},
     "kw": {"visita", "visitas", "ver a mi hijo", "ver a mis hijos",
            "ver a mi hija", "ver a mis hijas", "verlo", "verla", "verlos",
            "relacion directa", "convivencia", "cuidado personal",
            "no me deja ver", "no me dejan ver"}},
    {"canon": "divorcio / compensación económica",
     "ui": "Divorcio o compensación",
     "frase": "el *divorcio* o la *compensación económica*",
     "rama": "familia",
     "ids_ext": {"divorcio", "compensacion_economica", "compensacion"},
     "kw": {"divorcio", "divorciarme", "divorciada", "divorciado",
            "divorciar", "separarme", "separacion", "compensacion",
            "culpabilidad", "separacion de bienes"}},
    {"canon": "derecho penal",
     "ui": "Tema penal",
     "frase": "un *tema penal*",
     "rama": "penal",
     "ids_ext": {"penal", "derecho_penal", "tema_penal"},
     "kw": {"detenido", "detuvieron", "detenida", "carabineros", "fiscalia",
            "querella", "querellaron", "denuncia", "formalizacion",
            "formalizaron", "imputado", "imputada", "control de detencion",
            "delito", "violencia intrafamiliar", "vif", "lesiones", "robo",
            "hurto", "estafa", "agresion", "amenazas de", "causa penal"}},
    {"canon": "laboral / civil",
     "ui": "Laboral o civil",
     "frase": "algo *laboral o civil*",
     "rama": "general_otra",
     "ids_ext": {"laboral_civil", "laboral", "civil"},
     "kw": {"despido", "despidieron", "finiquito", "laboral",
            "contrato de trabajo", "fuero maternal", "embarazo",
            "autodespido", "arriendo", "arrendatario", "arrendador",
            "desalojo", "civil", "cobro ejecutivo", "cheque", "pagare",
            "cobranza judicial", "precario"}},
    {"canon": "otro tema",
     "ui": "Otro tema",
     "frase": "otro tema",
     "rama": "general_otra",
     "ids_ext": {"otro_tema", "otro"},
     "kw": {"herencia", "posesion efectiva", "accidente", "choque",
            "transito", "municipal", "permuta"}},
]

ETAPAS: list[dict] = [
    {"canon": "aún no inicio nada",
     "ui": "Aún no he iniciado nada",
     "ids_ext": {"sin_inicio", "sin_inicio_acciones"},
     "kw": {"aun no", "todavia no", "no he hecho", "quiero demandar",
            "quiero iniciar", "quiero pedir", "ninguna accion",
            "no he iniciado", "recien estoy", "quiero ver si"}},
    {"canon": "me demandaron o me notificaron",
     "ui": "Me demandaron / notificaron",
     "ids_ext": {"demandado", "notificado", "demandada"},
     "kw": {"notificacion", "notificaron", "me demandaron", "me demando",
            "demanda en mi contra", "citacion", "me cito", "receptor",
            "me notifico"}},
    {"canon": "causa en curso",
     "ui": "Causa judicial en curso",
     "ids_ext": {"causa_en_curso", "en_curso"},
     "kw": {"causa", "juicio", "tribunal", "en tramite", "tramitacion",
            "abogado actual", "ya tengo abogado", "mediacion", "rit",
            "rol de la causa"}},
    {"canon": "sentencia o acuerdo previa",
     "ui": "Hay sentencia o acuerdo",
     "ids_ext": {"sentencia_previa", "acuerdo_previo"},
     "kw": {"sentencia", "acuerdo", "transaccion", "resolucion",
            "fijaron la pension", "ya salio", "me condenaron",
            "ya esta la sentencia", "hubo sentencia"}},
]

ROLES: list[dict] = [
    {"canon": "imputado / querellado",
     "ui": "Soy el imputado",
     "ids_ext": {"imputado", "querellado", "acusado"},
     "kw": {"me acusan", "me querellaron", "me imputan", "soy el imputado",
            "soy imputado", "me formalizaron", "acusado", "querellado",
            "me denunciaron", "me denuncio"}},
    {"canon": "víctima",
     "ui": "Soy la víctima",
     "ids_ext": {"victima", "denunciante"},
     "kw": {"soy la victima", "soy victima", "me robaron", "me agredio",
            "me golpeo", "fui victima", "quiero denunciar", "me estafaron",
            "me amenazaron", "quiero querellarme"}},
    {"canon": "familiar o cercano",
     "ui": "Es un familiar o cercano",
     "ids_ext": {"familiar", "cercano", "tercero"},
     "kw": {"mi hijo", "mi hija", "mi hermano", "mi hermana", "mi pareja",
            "mi papa", "mi mama", "mi esposo", "mi esposa", "un familiar",
            "un amigo", "una amiga", "mi sobrino", "mi primo"}},
]

INGRESOS: list[dict] = [
    {"canon": "trabajo dependiente, sueldo estable",
     "ui": "Dependiente, sueldo fijo",
     "ids_ext": {"dependiente_estable", "dependiente"},
     "kw": {"dependiente", "contrato", "sueldo fijo", "indefinido",
            "trabajo en empresa", "empleado", "empleada", "plazo fijo"}},
    {"canon": "independiente / ingresos variables",
     "ui": "Independiente",
     "ids_ext": {"independiente_variable", "independiente"},
     "kw": {"independiente", "boleta", "boletas", "honorarios",
            "emprendimiento", "variable", "variables", "comerciante",
            "a honorarios", "por mi cuenta"}},
    {"canon": "cesante o ingresos muy reducidos",
     "ui": "Cesante / ingresos bajos",
     "ids_ext": {"cesante_reducidos", "cesante"},
     "kw": {"cesante", "sin trabajo", "sin empleo", "desempleado",
            "desempleada", "no estoy trabajando", "sin pega",
            "sueldo minimo", "ingresos bajos"}},
    {"canon": "jubilado / pensionado",
     "ui": "Jubilado(a)",
     "ids_ext": {"jubilado", "pensionado"},
     "kw": {"jubilado", "jubilada", "pensionado", "pensionada", "retiro",
            "jubilacion", "pgu"}},
]

# ---- Derivados (editar solo las fuentes de arriba) ----
_AREA_KEYWORDS = {a["canon"]: a["kw"] for a in AREAS}
_AREA_FRASE = {a["canon"]: a["frase"] for a in AREAS}
_AREA_LABEL_MAP = {ext: a["canon"] for a in AREAS for ext in a["ids_ext"]}
_AREA_TXT_A_RAMA = {a["canon"]: a["rama"] for a in AREAS}
_AREA_TXT_A_RAMA["medidas de apremio"] = "familia"  # compat leads antiguos

_ETAPA_KEYWORDS = {e["canon"]: e["kw"] for e in ETAPAS}
_ETAPA_LABEL_MAP = {ext: e["canon"] for e in ETAPAS for ext in e["ids_ext"]}

_ROL_KEYWORDS = {r["canon"]: r["kw"] for r in ROLES}
_ROL_LABEL_MAP = {ext: r["canon"] for r in ROLES for ext in r["ids_ext"]}

_INGRESOS_KEYWORDS = {g["canon"]: g["kw"] for g in INGRESOS}
_INGRESOS_LABEL_MAP = {ext: g["canon"] for g in INGRESOS
                       for ext in g["ids_ext"]}

_CATS_FAMILIA = frozenset(
    ext for a in AREAS if a["rama"] == "familia" for ext in a["ids_ext"])

# Mapa numérico alineado al orden LEGADO del contrato del planner
# (_AREA_DESDE_NUMERO, 1-7) — no al orden de la UI (6, sin apremio).
_AREA_NUMEROS_LEGACY = [
    "pensión de alimentos",              # 1
    "régimen de visitas",                # 2
    "divorcio / compensación económica", # 3
    "pensión de alimentos",              # 4: apremio → subordinado
    "derecho penal",                     # 5
    "laboral / civil",                   # 6
    "otro tema",                         # 7
]

# Traducción de inputs interactivos: ids nativos Y títulos de opción.
# (En producción el dispatcher entrega el TÍTULO: "Pensión de alimentos".)
_TITULO_A_CANON = {
    _norm(o["ui"]): o["canon"]
    for fuente in (AREAS, ETAPAS, ROLES, INGRESOS) for o in fuente
}
_BUTTON_ID_MAP = {
    **{f"area_{i + 1}": a["canon"] for i, a in enumerate(AREAS)},
    **{f"etapa_{i + 1}": e["canon"] for i, e in enumerate(ETAPAS)},
    **{f"rol_{i + 1}": r["canon"] for i, r in enumerate(ROLES)},
    **{f"ing_{i + 1}": g["canon"] for i, g in enumerate(INGRESOS)},
    **_TITULO_A_CANON,
    "retry": "lo intento de nuevo",
    "humano": "hablar con humano",
    "area_ok": "asi es",
    "area_no": "no, es otra cosa",
}


def _traducir_id(raw: str) -> str:
    """id o título de opción/botón nativo → texto canónico; si no, raw."""
    return _BUTTON_ID_MAP.get(_norm(raw), raw)


def traducir_input_interactivo(raw: str) -> str:
    """API pública para el PLANNER del intake (mismo paquete): traducir ids/
    títulos nativos → texto ANTES de llamar al LLM."""
    return _traducir_id(raw)


# ===========================================================================
# DETECTORES DE INTENCIÓN (0-LLM)
# ===========================================================================

def _detecta_salto(raw: str) -> bool:
    """"paso/pasar/pasa" solo cuenta si abre el mensaje o es ≤2 palabras:
    "eso ya pasó hace tiempo" es RESPUESTA, no salto."""
    t = _norm(_traducir_id(raw))
    if not t:
        return False
    if t in SALTAR:
        return True
    if t.startswith("no ") or " no " in t:
        return False

    palabras = t.split()
    if len(palabras) > 6:
        return False

    claves_fuertes = {
        "saltar", "salta", "salto", "saltarme", "saltarlo", "saltarla",
        "omitir", "omito", "skip", "skipear", "salte",
    }
    if any(p in claves_fuertes for p in palabras):
        return True

    claves_debiles = {"paso", "pasar", "pasa"}
    if any(p in claves_debiles for p in palabras):
        return palabras[0] in claves_debiles or len(palabras) <= 2

    omitir_frases = (
        "no quiero responder", "prefiero no responder",
        "prefiero no decir", "no aplica", "n/a", "me salto",
    )
    return any(p in t for p in omitir_frases)


def _detecta_humano(raw: str) -> bool:
    t = _norm(_traducir_id(raw))
    if not t:
        return False
    if t in RESPUESTAS_HUMANO:
        return True
    if "ejecutiva" in t or "ejecutivo" in t:
        return True
    if "hablar con" in t and any(x in t for x in
                                 ("persona", "humano", "abogad", "ejecutiv")):
        return True
    return False


# ===========================================================================
# URGENCIA — detección determinista (refuerzo 0-LLM al campo `detenido`)
# ===========================================================================
_URGENTE_RE = re.compile(
    r"(me|lo|la|nos)\s+detuvieron"
    r"|est[aá]\s+detenid[oa]"
    r"|control\s+de\s+detenci[oó]n\s+(es\s+)?(hoy|ma[ñn]ana)"
    r"|me\s+formalizaron"
    r"|tengo\s+orden\s+de\s+arresto"
    r"|orden\s+de\s+arresto\s+en\s+mi\s+contra",
    re.IGNORECASE,
)

_ORDEN_ARRESTO_AFIRMA_RE = re.compile(
    r"(tengo|me\s+(dictaron|pusieron|llego)|hay)\s+[^,.]{0,25}orden\s+de\s+arresto"
    r"|^orden\s+de\s+arresto$",
    re.IGNORECASE,
)
_NEGACION_MED_RE = re.compile(
    r"(no\s+tengo|ningun|nunca|sin\s+|todavia\s+no|aun\s+no|podria|"
    r"podrian|amenaza|amenazaron|quizas|puede\s+que)",
    re.IGNORECASE,
)

# ===========================================================================
# TEXTOS POR DEFECTO (override por _cfg()["intake"])
# ===========================================================================
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
DEFAULT_AREA_CONFIRMA = ("Por lo que me cuentas, tu caso iría por el lado "
                         "de {area} 🤔 ¿Voy bien?")
DEFAULT_AREA_CONFIRMA_RETRY = ("Perdona, no me quedó claro 🤔 ¿Tu caso va "
                               "por {area}? Responde *sí* o *no*, o "
                               "cuéntamelo en una frase.")
DEFAULT_AREA_FALLBACK = ("Tranquilo — no necesito que lo clasifiques tú: la "
                         "abogada que revise tu ficha lo ubica 🙌 Seguimos.")
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
DEFAULT_CIERRE_ABANDONO = (
    "Entendido, lo dejamos hasta aquí. Cuando quieras retomar tu caso, "
    "escríbeme o habla directo con el equipo:\n{wa_link}"
)
DEFAULT_SIN_CONSENTIMIENTO = (
    "Sin problema — sin tu autorización no puedo registrar tus datos ni "
    "derivar el caso. Si cambias de opinión, escríbeme o habla directo "
    "con el equipo:\n{wa_link}"
)


# ===========================================================================
# VALIDADORES
# ===========================================================================

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
    t = _norm(_traducir_id(s))
    if t in ("si", "sipo", "sip", "yes", "claro", "por supuesto",
             "afirmativo", "de acuerdo", "autorizo", "lo autorizo",
             "acepto", "si autorizo", "si, autorizo", "si por supuesto",
             "ya", "ya po"):
        return True, None
    if t in ("no", "nop", "nope", "negativo", "para nada",
             "no autorizo", "no acepto", "no gracias", "no, gracias"):
        return False, None
    return None, f"no binario: {s!r}"


def _score_kw(t: str, catalogo: dict) -> list[tuple[str, int]]:
    """keywords contenidas EN el mensaje (frases valen doble)."""
    scores: dict[str, int] = {}
    for canon, kws in catalogo.items():
        for kw in kws:
            if kw in t:
                scores[canon] = scores.get(canon, 0) + (2 if " " in kw else 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _v_opciones_nl(catalogo_kw: dict, aliases: dict | None = None,
                   numeros: list[str] | None = None):
    """Validador por keywords: tolera frases naturales, aliases, títulos e
    ids de opciones, y números. `numeros` fija el orden numérico (área usa
    el legado del contrato)."""
    alias_map = {_norm(k): v for k, v in (aliases or {}).items()}
    orden = numeros or list(catalogo_kw)

    def check(s):
        s = _traducir_id(s or "")
        t = _norm(s)
        if t in alias_map:
            return alias_map[t], None
        if t.isdigit() and 1 <= int(t) <= len(orden):
            return orden[int(t) - 1], None
        exacto = [o for o in catalogo_kw if _norm(o) == t]
        if exacto:
            return exacto[0], None
        ranked = _score_kw(t, catalogo_kw)
        if not ranked:
            return None, f"sin señal reconocible: {s!r}"
        if (len(ranked) > 1 and ranked[0][1] == ranked[1][1]
                and ranked[0][1] <= 1):
            return None, f"empate débil: {[r[0] for r in ranked[:2]]}"
        return ranked[0][0], None
    return check


def _v_nombre(s):
    s = (s or "").strip()
    palabras = [p for p in s.split() if p]
    if len(s) < 3 or not palabras:
        return None, "nombre no identificable"
    if len(palabras) > 6 or any(len(p) > 20 for p in palabras):
        return None, "parece frase, no nombre"
    return " ".join(p.capitalize() for p in palabras), None


def _match_confirmacion(texto_raw: str, sugerida: str) -> str | None:
    """'si' | 'no' | None. Keywords de OTRA área pesan más que el sí/no:
    "no, en realidad es por las visitas" clasifica a visitas."""
    t = _norm(texto_raw)
    alfa = " ".join(
        "".join(c for c in t if c.isalpha() or c.isspace()).split())
    if not alfa:
        return None
    ranked = _score_kw(alfa, _AREA_KEYWORDS)
    otra_gana = bool(ranked and ranked[0][0] != sugerida
                     and ranked[0][1] >= 2)
    if not otra_gana and (
            alfa in _CONFIRMA_SI or alfa == "si" or alfa.startswith("si ")
            or any(alfa == p or alfa.startswith(p + " ")
                   for p in ("asi es", "exacto", "correcto", "claro",
                             "eso", "ya"))):
        return "si"
    if not ranked and (
            alfa in _CONFIRMA_NO or alfa == "no" or alfa.startswith("no ")
            or alfa.startswith(("otra cosa", "es otra"))):
        return "no"
    return None


# ===========================================================================
# INTERACTIVOS
# ===========================================================================

def _botones(*id_title) -> dict:
    return {"kind": "buttons",
            "buttons": [{"id": i, "title": t} for i, t in id_title]}


def _lista(prefix: str, fuente: list[dict], cta: str, title: str) -> dict:
    return {"kind": "list", "cta": cta, "title": title,
            "options": [{"id": f"{prefix}_{i + 1}", "title": o["ui"]}
                        for i, o in enumerate(fuente)]}


_INTERACTIVE_CONSENTIMIENTO = _botones(("si", "Sí, autorizo"),
                                       ("no", "No autorizo"))
_BOTONES_OFERTA = _botones(("humano", "Hablar con humano"),
                           ("retry", "Lo intento de nuevo"))
_BOTON_AGENDAR = _botones(("agendar", "Agendar una hora"))
_INTERACTIVE_SI_NO = _botones(("si", "Sí"), ("no", "No"))
_BOTONES_CONFIRMA_AREA = _botones(("area_ok", "Así es 🙌"),
                                  ("area_no", "No, es otra cosa"))

_INTERACTIVE_AREA = _lista("area", AREAS, "Ver opciones", "Tu caso")
_INTERACTIVE_ETAPA = _lista("etapa", ETAPAS, "Ver opciones", "Estado del caso")
_INTERACTIVE_ROL = _lista("rol", ROLES, "Ver opciones", "Tu rol en el caso")
_INTERACTIVE_INGRESOS = _lista("ing", INGRESOS, "Ver opciones", "Tus ingresos")


# ===========================================================================
# RAMAS TEMÁTICAS
# ===========================================================================

def _rama(state, respuestas=None) -> str:
    resp = respuestas if respuestas is not None \
        else (state.get("intake_respuestas") or {})

    if resp.get("detenido") is True or resp.get("rol_penal"):
        return "penal"

    cat = state.get("case_category") or state.get("category") or ""
    if cat in _CATS_FAMILIA:
        return "familia"

    area = resp.get("area_caso")
    if area:
        return _AREA_TXT_A_RAMA.get(str(area), "pendiente")

    return "pendiente"


def _es_familia(s, r=None): return _rama(s, r) == "familia"
def _es_penal(s, r=None):   return _rama(s, r) == "penal"
def _area_pendiente(s, r=None): return _rama(s, r) == "pendiente"


def _es_urgencia(state, respuestas: dict) -> bool:
    if respuestas.get("detenido") is True:
        return True
    if _URGENTE_RE.search(state.get("query") or ""):
        return True
    med = _norm(respuestas.get("medidas_apremio") or "")
    return bool(med
                and _ORDEN_ARRESTO_AFIRMA_RE.search(med)
                and not _NEGACION_MED_RE.search(med))


# ===========================================================================
# QUESTIONS — tronco común + módulos por rama
# ===========================================================================
QUESTIONS: list[dict] = [
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

    {"id": "area_caso", "requerida": True,
     "label": "el tipo de caso",
     "skip_if": lambda s, r=None: not _area_pendiente(s, r),
     "prompt": ("Para llevarte con la abogada correcta: ¿cuál de estas se "
                "parece más a tu caso? 👇\nSi ninguna calza, cuéntamelo en "
                "una frase corta y yo lo ubico — por ejemplo: *«me "
                "demandaron por la pensión»*"),
     "retry_prompt": ("No lo ubiqué 😅 Toca la lista aquí abajo y elige la "
                      "que más se acerque, o escríbemelo en una frase corta."),
     "ack": "Perfecto, con eso ya sé a quién derivarte ✓",
     "interactive": _INTERACTIVE_AREA,
     "label_map": _AREA_LABEL_MAP,
     "validate": _v_opciones_nl(
         _AREA_KEYWORDS,
         numeros=_AREA_NUMEROS_LEGACY,
         aliases={"ver a mis hijos": "régimen de visitas",
                  "tema penal": "derecho penal",
                  "penal": "derecho penal",
                  "pension": "pensión de alimentos",
                  "alimentos": "pensión de alimentos",
                  "divorcio": "divorcio / compensación económica",
                  "apremio": "pensión de alimentos",
                  "medidas de apremio": "pensión de alimentos",
                  "laboral": "laboral / civil",
                  "civil": "laboral / civil",
                  "otro": "otro tema"})},

    {"id": "etapa_proceso", "requerida": False,
     "label": "la etapa del caso",
     "skip_if": lambda s, r=None: _es_penal(s, r),
     "prompt": ("¿Y cómo va tu caso hoy? 👇 Aún no partes, ya te demandaron "
                "o notificaron, hay una causa en curso, o ya existe una "
                "sentencia o acuerdo."),
     "retry_prompt": ("No lo ubiqué 😅 Toca la lista o respóndeme corto — "
                      "por ejemplo: *«aún no parto»* o *«ya me "
                      "notificaron»*."),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_ETAPA,
     "validate": _v_opciones_nl(_ETAPA_KEYWORDS),
     "label_map": _ETAPA_LABEL_MAP},

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
     "prompt": ("Para evaluar montos justos (pensión, acuerdos): ¿cuál es "
                "tu situación laboral hoy? 👇"),
     "retry_prompt": ("Toca la lista aquí abajo, o cuéntamelo corto — por "
                      "ejemplo: *«trabajo con contrato»* o *«soy "
                      "independiente»*."),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_INGRESOS,
     "label_map": _INGRESOS_LABEL_MAP,
     "validate": _v_opciones_nl(_INGRESOS_KEYWORDS)},

    {"id": "medidas_apremio", "requerida": False,
     "label": "las medidas de apremio",
     "skip_if": lambda s, r=None: not _es_familia(s, r),
     "prompt": ("¿Tienes alguna medida vigente en tu contra? Por ejemplo: "
                "licencia de conducir retenida, prohibición de salir del "
                "país, embargo o retención de sueldo o AFP, orden de "
                "arresto. Pueden ser varias — cuéntamelas con tus palabras; "
                "si no hay, responde *ninguna*."),
     "retry_prompt": ("Cuéntame solo las que apliquen — por ejemplo: "
                      "*«me retuvieron la licencia»* — o responde "
                      "*ninguna*."),
     "ack": "Anotado ✓",
     "validate": _v_text(2)},

    {"id": "rol_penal", "requerida": True,
     "label": "tu rol en el caso",
     "skip_if": lambda s, r=None: not _es_penal(s, r),
     "prompt": ("Para orientar bien la derivación: ¿el imputado eres tú, "
                "eres la víctima, o consultas por un familiar o cercano? 👇"),
     "retry_prompt": ("Solo necesito ubicar tu rol: toca la lista aquí "
                      "abajo o respóndeme corto — *«soy yo»*, *«soy la "
                      "víctima»* o *«es un familiar»*."),
     "ack": "Anotado ✓",
     "interactive": _INTERACTIVE_ROL,
     "label_map": _ROL_LABEL_MAP,
     "validate": _v_opciones_nl(_ROL_KEYWORDS)},

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

    {"id": "comuna", "requerida": False,
     "label": "tu comuna",
     "prompt": ("¿En qué *comuna o ciudad* estás? (para ubicar el tribunal "
                "que corresponde — atendemos en todo Chile)"),
     "retry_prompt": ("Solo el nombre de la comuna o ciudad — por ejemplo: "
                      "*Puente Alto* o *Concepción*."),
     "ack": "Anotado ✓",
     "validate": _v_text(3)},
]

_BY_ID = {q["id"]: q for q in QUESTIONS}


def _aplicables(state: AgentState, respuestas: dict | None = None) -> list[dict]:
    resp = respuestas if respuestas is not None \
        else (state.get("intake_respuestas") or {})
    return [q for q in QUESTIONS
            if not q.get("skip_if", lambda s, r=None: False)(state, resp)]


def _pendientes(state: AgentState, respuestas: dict) -> list[dict]:
    aplicables = _aplicables(state, respuestas)
    return [q for q in aplicables if q["id"] not in respuestas]


# ===========================================================================
# CLAVES INTERNAS (sin tocar AgentState: viajan en intake_respuestas con
# prefijo "_" y se excluyen de upsert_lead / ficha wa.me vía _publicas)
# ===========================================================================

_K_SUGERIDA = "_area_sugerida"    # hipótesis de área pendiente de confirmar
_K_RECHAZADA = "_area_rechazada"  # hipótesis que el cliente descartó
_K_EXTRAIDOS = "_extraidos"       # campos llenados por el planner (lista)


def _publicas(respuestas: dict) -> dict:
    return {k: v for k, v in respuestas.items() if not k.startswith("_")}


def _fijar_por_usuario(respuestas: dict, qid: str, val) -> None:
    """El input explícito del usuario SIEMPRE manda: fija el valor y saca
    el campo de la marca de extraídos (ya no es sobrescribible por nadie)."""
    extraidos = [x for x in (respuestas.get(_K_EXTRAIDOS) or []) if x != qid]
    if extraidos:
        respuestas[_K_EXTRAIDOS] = extraidos
    else:
        respuestas.pop(_K_EXTRAIDOS, None)
    respuestas[qid] = val


def _rescue_pendiente(raw_eff: str, pend: list[dict]):
    """v21 — Si el input calza con EXACTAMENTE UNA otra pendiente DE
    OPCIONES, devolver (question, valor). Absorbe el descuadre entre la
    pregunta mostrada y la que el sistema creía activa (ej: el usuario toca
    la lista de área cuando el estado apuntaba a etapa)."""
    hits = []
    for q in pend[1:]:
        if "label_map" not in q:  # solo preguntas de opciones cerradas
            continue
        val, err = q["validate"](raw_eff)
        if err is None and val is not None:
            hits.append((q, val))
            if len(hits) > 1:
                return None
    return hits[0] if hits else None


def _inferir_area_sugerida(respuestas: dict) -> str | None:
    relato = _norm(respuestas.get("situacion_actual") or "")
    if len(relato) < 6:
        return None
    ranked = _score_kw(relato, _AREA_KEYWORDS)
    if not ranked or ranked[0][1] < 2:
        return None
    if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
        return None
    if ranked[0][0] == respuestas.get(_K_RECHAZADA):
        return None  # jamás re-sugerir la que el cliente descartó
    return ranked[0][0]


def _interactive_para(q: dict, area_sugerida: str | None = None) -> dict | None:
    if q["id"] == "area_caso" and area_sugerida:
        return _BOTONES_CONFIRMA_AREA
    return q.get("interactive")


def _form_pregunta(state: AgentState, q: dict, respuestas: dict,
                   nombre: str | None = None,
                   area_sugerida: str | None = None) -> str:
    sec = _cfg()["intake"]
    restantes = len(_pendientes(state, respuestas))

    if restantes == 2:
        head = (f"Ya casi{', ' + nombre if nombre else ''} 🙌 "
                "quedan solo un par de datos:\n")
    elif restantes <= 1:
        head = (f"Ya casi{', ' + nombre if nombre else ''} 🙌 "
                "solo me falta esto:\n")
    else:
        head = ""

    if q["id"] == "area_caso" and area_sugerida:
        frase = _AREA_FRASE.get(area_sugerida, area_sugerida)
        body = sec.get("area_confirmacion", DEFAULT_AREA_CONFIRMA).format(
            area=frase)
    else:
        body = q["prompt"]
        if not q["requerida"]:
            body += "\n" + sec.get("aviso_saltar", DEFAULT_AVISO_SALTAR)
    return head + body


def _pack_pregunta(state: AgentState, respuestas: dict, skipped: set,
                   *, prefijo: list[str] | None = None,
                   nombre: str | None = None):
    """(pack, pregunta, area_sugerida) para la siguiente pendiente; siembra/
    limpia la hipótesis interna. None si no quedan pendientes."""
    pend = _pendientes(state, respuestas)
    if not pend:
        return None
    q = pend[0]
    sugerida = None
    if q["id"] == "area_caso":
        sugerida = (respuestas.get(_K_SUGERIDA)
                    or _inferir_area_sugerida(respuestas))
        if sugerida:
            respuestas[_K_SUGERIDA] = sugerida
    else:
        respuestas.pop(_K_SUGERIDA, None)
        respuestas.pop(_K_RECHAZADA, None)
    body = _form_pregunta(state, q, respuestas, nombre=nombre,
                          area_sugerida=sugerida)
    pack = _pack([*(prefijo or []), body],
                 interactive=_interactive_para(q, sugerida))
    return pack, q, sugerida


# ===========================================================================
# ACKS / EXTRAS / EXTRACCIÓN
# ===========================================================================

def _idx_of(qid: str) -> int:
    return next(i for i, q in enumerate(QUESTIONS) if q["id"] == qid)


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


def _extras_turno(q_actual: dict, antes: dict, despues: dict,
                  excluir: frozenset = frozenset()) -> list[str]:
    return [qid for qid in _BY_ID
            if qid != q_actual["id"] and qid not in excluir
            and despues.get(qid) is not None
            and despues.get(qid) != antes.get(qid)]


def _join_labels(labels: list[str]) -> str:
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " y " + labels[-1]


def _acks_turno(q_respondida: dict, antes: dict, respuestas: dict) -> list[str]:
    """Acks del turno. q_respondida es la pregunta efectivamente respondida
    (la activa, o la rescatada si el input calzó con otra pendiente)."""
    acks: list[str] = []
    qid = q_respondida["id"]
    if respuestas.get(qid) is not None and respuestas.get(qid) != antes.get(qid):
        acks.append(_ack_campo(qid, respuestas))
    elif qid in respuestas and respuestas[qid] is None and qid not in antes:
        acks.append("Ningún problema, lo dejamos así ✓")
    extras = _extras_turno(q_respondida, antes, respuestas,
                           excluir=frozenset({qid}))
    if extras:
        acks.append("De paso ya tengo "
                    + _join_labels([_BY_ID[q]["label"] for q in extras])
                    + " 👌")
    return acks


# Frases que SOLO puede decir el bot: si aparecen en un valor extraído, el
# planner se autocontaminó con el historial (ej: el usuario cita/reenvía la
# burbuja del asistente) y el valor se descarta.
_FRASES_BOT = (
    "si te entendi bien", "gracias por contarmelo",
    "para llevarte con la abogada", "toca la lista", "elige la opcion",
    "se me escapo algo", "me demandaron por la pension", "no lo ubique",
    "aun no parto", "ya me notificaron", "se que no siempre es facil",
    "cuentame brevemente", "a que correo te enviamos",
)


def _valor_contaminado(qid: str, raw_txt, respuestas: dict) -> bool:
    t = _norm(str(raw_txt or ""))
    if not t:
        return True
    if any(f in t for f in _FRASES_BOT):
        return True
    if qid == "situacion_actual":
        compacto = t.replace(" ", "")
        if EMAIL_RE.fullmatch(compacto):
            return True
        if t == _norm(respuestas.get("email") or ""):
            return True
    return False


def _aplicar_extraccion(respuestas: dict, decision: dict,
                        skipped: set | None = None) -> None:
    """v21 — Tres guardas sobre la extracción del planner:
      (1) SOLO llena campos ausentes: jamás sobrescribe lo que ya está
          (el planner re-lee el historial cada turno y puede degradar).
      (2) Lo que llena queda marcado en "_extraidos": el input explícito
          del usuario puede sobrescribirlo después (_fijar_por_usuario).
      (3) Texto libre contaminado con frases del propio bot se descarta.
      Bandas de confianza (v20): solo si confianza > 0.
    """
    conf = decision.get("confianza") or 0.0
    extraidos = list(respuestas.get(_K_EXTRAIDOS) or [])
    for qid, qdef in _BY_ID.items():
        if qid in (skipped or set()):
            continue
        if not qdef.get("extractable", True):
            continue
        if qid in respuestas:                       # guarda (1)
            continue
        raw = decision.get(qid)
        if raw is None:
            continue
        if isinstance(raw, bool):
            respuestas[qid] = raw
            if qid not in extraidos:
                extraidos.append(qid)
            continue
        if "label_map" in qdef:
            canon = qdef["label_map"].get(str(raw))
            if canon is None:
                logger.info("[intake] extracción sin label_map: %s=%r "
                            "(agregar el id a la fuente única)", qid, raw)
                continue
            raw = canon
        elif _valor_contaminado(qid, raw, respuestas):   # guarda (3)
            logger.info("[intake] extracción descartada (autocontaminación "
                        "del historial): %s=%r", qid, str(raw)[:60])
            continue
        if qid == "area_caso" and conf > 0:
            if conf < _UMBRAL_AREA_SUGERIDA:
                logger.info("[intake] área extraída con confianza %.2f "
                            "< %.2f → se ignora", conf, _UMBRAL_AREA_SUGERIDA)
                continue
            if conf < _UMBRAL_AREA_DIRECTA:
                respuestas.setdefault(_K_SUGERIDA, str(raw))
                continue
        val, err = qdef["validate"](raw)
        if err is None and val is not None:
            respuestas[qid] = val
            if qid not in extraidos:                # guarda (2)
                extraidos.append(qid)
        else:
            logger.info("[intake] extracción rechazada por validador: "
                        "%s=%r (%s)", qid, raw, err)
    if extraidos:
        respuestas[_K_EXTRAIDOS] = extraidos


def _lead_updates(respuestas: dict) -> dict:
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
        # notificar_crm=False: persistir por turno no spamea el CRM;
        # handoff_humano es el emisor único (FIX #10).
        upsert_lead(
            thread_id=state.get("thread_id", ""),
            intake_respuestas=_publicas(respuestas),
            category=state.get("category"),
            completed=False,
            notificar_crm=False,
        )
    except Exception as e:
        logger.warning("upsert_lead parcial falló (best-effort): %s", e)


def _texto_ficha_wa(respuestas: dict, thread_id: str,
                    categoria: str) -> str:
    nombre = respuestas.get("nombre") or ""
    email = respuestas.get("email") or ""
    telefono = respuestas.get("telefono") or ""

    lineas = [
        f"Hola, soy {nombre}. Acabo de completar mi registro. "
        "Resumen de mi caso:",
        f"• Área: {categoria or 'sin clasificar'}",
    ]
    if respuestas.get("area_caso"):
        lineas.append(
            f"• Tipo de caso: {_recortar(respuestas['area_caso'], 40)}")
    if respuestas.get("area_revision_manual"):
        lineas.append("• ⚠️ Clasificar el área a mano "
                      "(el cliente no logró ubicarla)")
    if respuestas.get("urgencia_legal"):
        lineas.append("• URGENTE: detención u orden vigente")
    if respuestas.get("rol_penal"):
        lineas.append(f"• Rol (penal): {respuestas['rol_penal']}")
    if respuestas.get("audiencia_proxima"):
        lineas.append(
            f"• Audiencia: {_recortar(respuestas['audiencia_proxima'], 60)}")
    if respuestas.get("etapa_proceso"):
        lineas.append(
            f"• Etapa: {_recortar(respuestas['etapa_proceso'], 40)}")
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
        f"• Mi situación: "
        f"{_recortar(respuestas.get('situacion_actual'), 240)}",
        f"(ID de mi atención: {thread_id})",
    ]
    return "\n".join(lineas)


def _cta_humano_con_ficha(respuestas: dict, thread_id: str,
                          categoria: str) -> dict:
    texto = _texto_ficha_wa(_publicas(respuestas), thread_id, categoria)
    url = f"https://wa.me/{_ejecutiva_number()}?text={quote(texto)}"
    return {
        "text": DEFAULT_CIERRE_CTA,
        "interactive": {"kind": "cta", "text": "Pincha aquí", "url": url},
    }


# ===========================================================================
# SALIDA ÚNICA
# ===========================================================================

def _salida_intake(pack: dict, *, respuestas: dict, skipped: set,
                   exit_: str | None, stage: str, activo: bool = False,
                   completado: bool = False, closed: bool = False) -> dict:
    out = {
        **pack,
        "intake_activo": activo,
        "intake_completado": completado,
        "intake_respuestas": respuestas,
        "intake_exit": exit_,
        "intake_stage": stage,
        "intake_decision": None,
        "intake_oferta_qid": None,
        "intake_skipped": skipped,
        "asistente_presentado": True,
        **_lead_updates(respuestas),
    }
    if closed:
        out["closed"] = True
    return out


def _avance_normal(state: AgentState, respuestas: dict, skipped: set,
                   q_respondida: dict | None, antes: dict) -> AgentState:
    """Camino único de avance: éxito de validación, rescate cruzado o salto
    opcional (lo usan procesar_respuesta y pausar_ficha v21)."""
    if respuestas.get("consentimiento_datos") is False:
        return sin_consentimiento(state, respuestas)
    if _es_urgencia(state, respuestas):
        return _derivar(state, respuestas, skipped=skipped, urgente=True)

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas, skipped=skipped)

    _persistir_parcial(state, respuestas)
    nombre = _primer_nombre(respuestas.get("nombre"))
    acks = _acks_turno(q_respondida, antes, respuestas) if q_respondida else []
    res = _pack_pregunta(state, respuestas, skipped, prefijo=acks,
                         nombre=nombre)
    if res is None:
        return _completar_ficha(state, respuestas, skipped=skipped)
    pack, siguiente, _sug = res
    _guardar_ai(state, pack["response"])
    return {
        **pack,
        "intake_idx": _idx_of(siguiente["id"]),  # DEPRECADO (compat)
        "intake_respuestas": respuestas,
        "intake_attempts": 0,
        "intake_oferta_qid": None,
        "intake_decision": None,
        "intake_skipped": skipped,
        "asistente_presentado": True,
    }


# ===========================================================================
# NODOS DEL SUBGRAFO
# ===========================================================================

def iniciar_ficha(state: AgentState) -> AgentState:
    sec = _cfg()["intake"]
    respuestas: dict = {}
    skipped = set(state.get("intake_skipped") or ())
    tel = _telefono_cliente(state)
    if tel:
        respuestas["telefono"] = tel

    _aplicar_extraccion(respuestas, state.get("intake_decision") or {},
                        skipped=skipped)

    if _es_urgencia(state, respuestas):
        return _derivar(state, respuestas, skipped=skipped, urgente=True)

    prefijo = []
    if _apertura_pres(state):
        prefijo.append(sec.get("apertura_ia") or sec.get("apertura")
                       or DEFAULT_AP_IA)
    prefijo.append(sec.get("apertura_expectativa") or DEFAULT_AP_EXP)

    res = _pack_pregunta(state, respuestas, skipped, prefijo=prefijo)
    if res is None:
        return _completar_ficha(state, respuestas, skipped=skipped)
    pack, q0, _sug = res

    _guardar_ai(state, pack["response"])
    _evento_intake("intake_iniciado", state, pregunta=q0["id"],
                   rama=_rama(state, respuestas))
    return {
        **pack,
        "intake_activo": True,
        "intake_idx": _idx_of(q0["id"]),  # DEPRECADO (compat)
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
    raw_eff = _traducir_id(raw)  # ids y títulos de opciones → texto
    skipped = set(state.get("intake_skipped") or ())
    nombre = _primer_nombre(respuestas.get("nombre"))

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas, skipped=skipped)
    q_actual = pend[0]
    ya_ofrecido = state.get("intake_oferta_qid") == q_actual["id"]

    # (a) Tras la oferta, "lo intento de nuevo" (texto o botón retry).
    if ya_ofrecido and _norm(raw_eff) in RESPUESTAS_REINTENTAR:
        pack = _pack(
            [f"Dale 💪 {q_actual['retry_prompt']}"],
            interactive=_interactive_para(q_actual,
                                          respuestas.get(_K_SUGERIDA)))
        _guardar_ai(state, pack["response"])
        return {
            **pack,
            "intake_respuestas": respuestas,
            "intake_attempts": attempts,
            "intake_decision": None,
            "intake_skipped": skipped,
            "asistente_presentado": True,
        }

    # (b) Pedido de humano (texto o botón): deriva de inmediato.
    if _detecta_humano(raw):
        logger.info("[intake] pedido de humano en '%s' → derivación",
                    q_actual["id"])
        return _derivar(state, respuestas, skipped=skipped, urgente=False)

    # (c) Interpretación del input.
    campo_respondido: dict | None = None
    error_key = None

    if _detecta_salto(raw):
        if q_actual["requerida"]:
            error_key = "no_skip"
        else:
            _fijar_por_usuario(respuestas, q_actual["id"], None)
            skipped.add(q_actual["id"])
            campo_respondido = q_actual
            logger.info("[intake] campo opcional saltado: %s",
                        q_actual["id"])
    else:
        _aplicar_extraccion(respuestas, decision, skipped=skipped)

        if q_actual["id"] in respuestas:
            campo_respondido = q_actual
            # v21: si el planner la llenó ESTE turno y el usuario respondió
            # algo distinto y válido, el usuario corrige la extracción.
            if q_actual["id"] in (respuestas.get(_K_EXTRAIDOS) or []):
                val2, err2 = q_actual["validate"](raw_eff)
                if (err2 is None and val2 is not None
                        and val2 != respuestas[q_actual["id"]]):
                    logger.info("[intake] el usuario corrige lo extraído en "
                                "%s: %r → %r", q_actual["id"],
                                respuestas[q_actual["id"]], val2)
                    _fijar_por_usuario(respuestas, q_actual["id"], val2)
        else:
            # (c.1) Confirmación de área: sí/no sobre la hipótesis activa.
            sugerida = respuestas.get(_K_SUGERIDA)
            if q_actual["id"] == "area_caso" and sugerida:
                veredicto = _match_confirmacion(raw_eff, sugerida)
                if veredicto == "si":
                    _fijar_por_usuario(respuestas, "area_caso", sugerida)
                    respuestas.pop(_K_SUGERIDA, None)
                    respuestas.pop(_K_RECHAZADA, None)
                    campo_respondido = q_actual
                elif veredicto == "no":
                    respuestas[_K_RECHAZADA] = sugerida
                    respuestas.pop(_K_SUGERIDA, None)
                    pack = _pack(
                        ["Ningún problema 👍 Elige la opción que más se "
                         "acerque, o cuéntamelo en una frase 👇"],
                        interactive=_INTERACTIVE_AREA)
                    _guardar_ai(state, pack["response"])
                    return {
                        **pack,
                        "intake_respuestas": respuestas,
                        "intake_attempts": attempts,  # no penaliza
                        "intake_decision": None,
                        "intake_skipped": skipped,
                        "asistente_presentado": True,
                    }
                # Otro texto → validate / rescate (c.2).

            # (c.2) Validación directa; si falla, RESCATE cruzado (v21):
            # el input se adjudica a la pendiente con la que SÍ calza.
            if campo_respondido is None and q_actual["id"] not in respuestas:
                val, err = q_actual["validate"](raw_eff)
                if err is None and val is not None:
                    _fijar_por_usuario(respuestas, q_actual["id"], val)
                    if q_actual["id"] == "area_caso":
                        respuestas.pop(_K_SUGERIDA, None)
                        respuestas.pop(_K_RECHAZADA, None)
                    campo_respondido = q_actual
                else:
                    hit = _rescue_pendiente(raw_eff, pend)
                    if hit:
                        q_resc, val_r = hit
                        _fijar_por_usuario(respuestas, q_resc["id"], val_r)
                        if q_resc["id"] == "area_caso":
                            respuestas.pop(_K_SUGERIDA, None)
                            respuestas.pop(_K_RECHAZADA, None)
                        campo_respondido = q_resc
                        logger.info(
                            "[intake] rescate: input %r adjudicado a '%s' "
                            "(la activa era '%s')", raw_eff[:50],
                            q_resc["id"], q_actual["id"])
                    else:
                        logger.info("[intake] validador rechazó %s: %s",
                                    q_actual["id"], err)
                        error_key = "retry"

    # ---------------- Éxito / salto / rescate → avance ----------------
    if error_key is None:
        return _avance_normal(state, respuestas, skipped,
                              campo_respondido or q_actual, antes)

    pend = _pendientes(state, respuestas)
    if not pend:
        return _completar_ficha(state, respuestas, skipped=skipped)

    # ---------------- Rama de error ----------------
    extras = _extras_turno(q_actual, antes, respuestas)
    if not extras:
        attempts += 1

    if not extras and attempts >= MAX_INTENTOS_CAMPO:
        # Anti-bucle (P1): clasificar mal el área es barato; atrapar al
        # usuario es carísimo. Fallback + revisión manual, y se AVANZA.
        if q_actual["id"] == "area_caso":
            _fijar_por_usuario(respuestas, "area_caso", "otro tema")
            respuestas["area_revision_manual"] = True
            respuestas.pop(_K_SUGERIDA, None)
            respuestas.pop(_K_RECHAZADA, None)
            logger.info("[intake] área sin clasificar tras %d intentos "
                        "→ 'otro tema' + revisión manual", attempts)
            _persistir_parcial(state, respuestas)
            aviso = sec.get("area_fallback", DEFAULT_AREA_FALLBACK)
            res = _pack_pregunta(state, respuestas, skipped,
                                 prefijo=[aviso], nombre=nombre)
            if res is None:
                return _completar_ficha(state, respuestas, skipped=skipped)
            pack, siguiente, _sug = res
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

        if q_actual["requerida"]:
            if ya_ofrecido:
                logger.info("[intake] reintento agotado en '%s' → "
                            "derivación parcial", q_actual["id"])
                return _derivar(state, respuestas, skipped=skipped,
                                urgente=False)
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
    sugerida = respuestas.get(_K_SUGERIDA)
    if error_key == "no_skip":
        burbujas.append(sec.get("error_no_saltar", DEFAULT_ERROR_NO_SALTAR))
    elif q_actual["id"] == "area_caso" and sugerida:
        burbujas.append(
            sec.get("area_confirmacion_retry",
                    DEFAULT_AREA_CONFIRMA_RETRY).format(
                area=_AREA_FRASE.get(sugerida, sugerida)))
    else:
        burbujas.append(q_actual.get("retry_prompt") or q_actual["prompt"])
    pack = _pack(burbujas, interactive=_interactive_para(q_actual, sugerida))
    _guardar_ai(state, pack["response"])
    return {
        **pack,
        "intake_respuestas": respuestas,
        "intake_attempts": attempts,
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
    pack["response_cta"] = _cta_humano_con_ficha(respuestas, tid, categoria)

    try:
        # notificar_crm=False (FIX #10): handoff_humano es el emisor único.
        upsert_lead(
            thread_id=tid,
            intake_respuestas=_publicas(respuestas),
            category=state.get("category"),
            completed=True,
            notificar_crm=False,
        )
    except Exception as e:
        logger.warning("upsert_lead (completado) falló: %s", e)

    _guardar_ai(state, pack["response"])
    _evento_intake("intake_completado", state,
                   respuestas_keys=list(_publicas(respuestas).keys()),
                   rama=_rama(state, respuestas),
                   tiene_email=bool(respuestas.get("email")))
    return _salida_intake(pack, respuestas=_publicas(respuestas),
                          skipped=skipped or set(),
                          exit_="cta_cierre", stage="completado",
                          completado=True)


def pausar_ficha(state: AgentState) -> AgentState:
    """v21 — GUARDA anti-bug: el planner a veces clasifica "saltar" como
    'pausar'. Si el input es un salto válido para la pregunta activa, NO se
    pausa: la opcional queda en None (skipped) y se avanza; la requerida se
    explica y se repregunta. La pausa genuina queda intacta.
    """
    sec = _cfg()["intake"]
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    raw = (state.get("query") or "").strip()

    pend = _pendientes(state, respuestas)
    q = pend[0] if pend else None

    if q and _detecta_salto(raw):
        if q["requerida"]:
            nombre = _primer_nombre(respuestas.get("nombre"))
            pack = _pack(
                [sec.get("error_no_saltar", DEFAULT_ERROR_NO_SALTAR),
                 _form_pregunta(state, q, respuestas, nombre=nombre,
                                area_sugerida=respuestas.get(_K_SUGERIDA))],
                interactive=_interactive_para(
                    q, respuestas.get(_K_SUGERIDA)))
            _guardar_ai(state, pack["response"])
            logger.info("[intake] 'pausar' del planner era salto en "
                        "requerida '%s' → repregunta", q["id"])
            return {
                **pack,
                "intake_activo": True,
                "intake_respuestas": respuestas,
                "intake_attempts": int(state.get("intake_attempts") or 0),
                "intake_exit": None,
                "intake_stage": "preguntando",
                "intake_decision": None,
                "intake_oferta_qid": None,
                "intake_skipped": skipped,
                "asistente_presentado": True,
            }

        # Salto legítimo de opcional: registrar como None y AVANZAR.
        antes = dict(respuestas)
        _fijar_por_usuario(respuestas, q["id"], None)
        skipped.add(q["id"])
        logger.info("[intake] 'pausar' del planner era un salto de campo "
                    "opcional: %s → avanza", q["id"])
        return _avance_normal(state, respuestas, skipped, q, antes)

    # Pausa genuina.
    pack = _pack(["Perfecto, quedo aquí esperando. Cuando quieras retomar, "
                  "solo escríbeme 🤙"])
    _guardar_ai(state, pack["response"])
    _evento_intake("intake_pausado", state)
    out = _salida_intake(pack, respuestas=respuestas, skipped=skipped,
                         exit_="pausa", stage="pausado", activo=True)
    out["intake_resume"] = False
    return out


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

    # FIX #3 (se mantiene): preservar la respuesta del FAQ del MISMO turno.
    msgs = state.get("messages") or []
    viene_de_faq = bool(msgs and getattr(msgs[-1], "type", "") == "ai")
    previas = (list(state.get("response_bubbles") or [])
               if viene_de_faq else [])

    sugerida = None
    if q["id"] == "area_caso":
        sugerida = (respuestas.get(_K_SUGERIDA)
                    or _inferir_area_sugerida(respuestas))
        if sugerida:
            respuestas[_K_SUGERIDA] = sugerida
    body = _form_pregunta(state, q, respuestas, nombre=nombre,
                          area_sugerida=sugerida)
    inter = _interactive_para(q, sugerida)
    pack = _pack([*previas, puente, body], interactive=inter)
    _guardar_ai(state, _pack([puente, body])["response"])

    _evento_intake("intake_reanudado", state, pregunta=q["id"])
    return {**base, **pack,
            "intake_idx": _idx_of(q["id"]),  # DEPRECADO (compat)
            "intake_respuestas": respuestas,
            "asistente_presentado": True}


# ===========================================================================
# DERIVACIONES
# ===========================================================================

def derivar_parcial(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    return _derivar(state, respuestas, skipped=skipped, urgente=False)


def derivar_urgente(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    _aplicar_extraccion(respuestas, state.get("intake_decision") or {},
                        skipped=skipped)
    return _derivar(state, respuestas, skipped=skipped, urgente=True)


def _derivar(state: AgentState, respuestas: dict, *, skipped: set | None,
             urgente: bool) -> AgentState:
    sec = _cfg()["intake"]
    if urgente:
        respuestas = {**respuestas, "urgencia_legal": True}
    nombre = _primer_nombre(respuestas.get("nombre"))
    display_nombre = respuestas.get("nombre") or "un cliente del asistente"

    if urgente:
        tpl = sec.get("derivacion_urgente", DEFAULT_URGENTE)
        wa = _wa_link_display(
            f"URGENTE — soy {display_nombre}: hay una detención u orden "
            "vigente. Mi ficha quedó registrada, necesito hablar con la "
            "abogada ahora.",
            display="Hablar con la abogada ahora")
        msg = tpl.format(nombre=f"{nombre}, " if nombre else "", wa_link=wa)
        evento, ev_kw = "intake_derivacion_urgente", {
            "gatillo": ("detenido" if respuestas.get("detenido") is True
                        else "apremio_o_regex")}
    else:
        tpl = sec.get("derivacion_parcial", DEFAULT_DERIVA)
        wa = _wa_link_display(
            f"Hola, soy {display_nombre}, prefiero continuar directamente "
            "con la ejecutiva.",
            display="Hablar con un humano")
        msg = tpl.format(nombre=f", {nombre}" if nombre else "", wa_link=wa)
        evento, ev_kw = "intake_derivacion_parcial", {}

    _persistir_parcial(state, respuestas)
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])
    _evento_intake(evento, state,
                   respuestas_keys=list(_publicas(respuestas).keys()),
                   **ev_kw)
    return _salida_intake(pack, respuestas=_publicas(respuestas),
                          skipped=skipped or set(),
                          exit_="handoff", stage="completado")


# ===========================================================================
# CIERRES CON STATUS
# ===========================================================================

def abandonar_ficha(state: AgentState) -> AgentState:
    respuestas = dict(state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    _persistir_parcial(state, respuestas)
    return _cerrar_con_status(
        state, respuestas, skipped,
        status="abandonado",
        cfg_key="cierre_abandono", default_msg=DEFAULT_CIERRE_ABANDONO,
        evento="intake_abandonado",
        ev_kw={"respuestas_keys": list(_publicas(respuestas).keys())})


def sin_consentimiento(state: AgentState,
                       respuestas: dict | None = None) -> AgentState:
    respuestas = dict(respuestas if respuestas is not None
                      else state.get("intake_respuestas") or {})
    skipped = set(state.get("intake_skipped") or ())
    return _cerrar_con_status(
        state, respuestas, skipped,
        status="sin_consentimiento",
        cfg_key="sin_consentimiento", default_msg=DEFAULT_SIN_CONSENTIMIENTO,
        evento="intake_sin_consentimiento")


def _cerrar_con_status(state: AgentState, respuestas: dict, skipped: set,
                       *, status: str, cfg_key: str, default_msg: str,
                       evento: str, ev_kw: dict | None = None) -> AgentState:
    msg = _cfg()["intake"].get(cfg_key, default_msg).format(
        wa_link=_wa_link_display(display="Hablar con un humano"))
    pack = _pack([msg])
    _guardar_ai(state, pack["response"])

    upsert_conversation(
        thread_id=state.get("thread_id", ""),
        status=status,
        category=state.get("case_category") or state.get("category"),
        closed=True,
    )

    _evento_intake(evento, state, **(ev_kw or {}))
    return _salida_intake(pack, respuestas=_publicas(respuestas),
                          skipped=skipped,
                          exit_="end", stage=status, closed=True)
