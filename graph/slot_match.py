"""graph/slot_match.py — Matcher robusto de horarios propuestos.

slots_propuestos (datetimes) son la fuente de verdad. Este matcher entiende
lo que una recepcionista chilena entiende por WhatsApp:
  "2", "2)", "opción 1", "la 2", "la primera", "la última",
  "el lunes a las 17:30", "04/09 - 17:30", "hoy a las 1730",
  "a las 5", "las 15", "15" pelado, "5:30 pm", "3pm".

match_slot() retorna:
  int        → índice único dentro de la lista de slots
  list[int]  → la mención calza con VARIOS slots (ambigüedad real)
  None       → no se interpretó nada útil

Semántica de registro WhatsApp: "la 2" = OPCIÓN 2 (artículo femenino =
"la opción"); "las 2" = hora (las 14:00 candidata).

extraer_dias_offset() reutiliza el mismo parser para el turno de ENTRADA
(aún sin slots): "quiero hora el jueves" → offset determinista para
pedir_otra_fecha, sin LLM ni fecha en el prompt del planner.

Changelog:
  v2 — `ref` OBLIGATORIO (date.today() con TZ del servidor — UTC en
       Railway — desfasaba "mañana"/"hoy" de noche).
       Rama hm con heurística +12 ("5:30" → 17:30 candidato) y soporte
       am/pm ("5:30 pm", "3 p.m."); antes el match con minutos era EXACTO.
       "por la mañana" ya no se traga como fecha: lookbehind (?<!la ).
       Nuevos formatos: "las 15", "la de las 3", "15" (hora), "la N" y
       ordinales (índice). extraer_dias_offset() público para planner.py.
       Semántica "este/esta": "ESTE miércoles" dicho un miércoles = HOY
       (offset 0); "el miércoles" a secas sigue siendo el próximo (+7).
  v1 — Matcher tri-estado (int | list[int] | None).
"""
import re
import unicodedata
from datetime import date, datetime, timedelta

_DIAS = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3,
         "viernes": 4, "sabado": 5, "domingo": 6}

_ORDINALES = {"primer": 1, "primero": 1, "primera": 1,
              "segundo": 2, "segunda": 2,
              "tercer": 3, "tercero": 3, "tercera": 3,
              "cuarto": 4, "cuarta": 4, "quinto": 5, "quinta": 5,
              "ultimo": -1, "ultima": -1}

_MERIDIANO = r"(a\.?\s?m\.?|p\.?\s?m\.?)"        # am / pm / a.m. / p. m.


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def _horas(h: int, marca: str | None) -> set[int]:
    """Candidatas de hora. Sin marca: "5" → {5, 17} (en Chile "a las 5"
    suele ser 17:00). Con marca explícita se respeta y normaliza a 24h."""
    if marca:
        if "p" in marca:
            return {h + 12} if h < 12 else {h}
        return {h % 12}                            # am: 12 am → 0
    return {h, h + 12} if h < 12 else {h}


def _hms(h: int, mi: int, marca: str | None) -> set[tuple[int, int]]:
    return {(x, mi) for x in _horas(h, marca)}


def _parse_criterios(t: str, ref: date) -> dict:
    """Extrae criterios de fecha/hora de un texto YA normalizado."""
    c: dict = {"dm": None, "weekday": None, "hm": None, "horas": None}

    # Fecha explícita: 04/09, 4-9, 04/09/2026
    m = re.search(r"\b(\d{1,2})\s*[/-]\s*(\d{1,2})(?:\s*[/-]\s*\d{2,4})?\b", t)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        if 1 <= dia <= 31 and 1 <= mes <= 12:
            c["dm"] = (mes, dia)
            t = t[:m.start()] + " " + t[m.end():]   # no confundir con hora

    # Día relativo o día de semana.
    # OJO: "mañana" como FECHA excluye "la mañana" (lookbehind). Franja
    # ("por/en/de la mañana") la resuelve _fast_path ANTES de llamar aquí;
    # esto es defensa en profundidad, no la línea principal.
    if c["dm"] is None:
        if "pasado manana" in t:
            d = ref + timedelta(days=2); c["dm"] = (d.month, d.day)
        elif re.search(r"(?<!la )\bmanana\b", t):
            d = ref + timedelta(days=1); c["dm"] = (d.month, d.day)
        elif re.search(r"\bhoy\b", t):
            c["dm"] = (ref.month, ref.day)
        else:
            for nombre, wd in _DIAS.items():
                if re.search(rf"\b{nombre}\b", t):
                    c["weekday"] = wd
                    break

    # Hora con minutos: "17:30" / "17h30" / "17.30" / "5:30 pm" > "1730"
    m = re.search(r"\b([01]?\d|2[0-3])\s*[:.h]\s*([0-5]\d)\s*"
                  + _MERIDIANO + r"?\b", t)
    if m:
        c["hm"] = _hms(int(m.group(1)), int(m.group(2)), m.group(3))
        return c
    m = re.search(r"\b([01]\d|2[0-3])([0-5]\d)\b", t)
    if m:
        c["hm"] = _hms(int(m.group(1)), int(m.group(2)), None)
        return c

    # Hora suelta: "a las 17", "a la 1", "las 15", "la de las 3".
    # "la 2" NO entra aquí: es la OPCIÓN 2 (lo resuelve el match de índice).
    m = re.search(r"\ba\s+las?\s*([01]?\d|2[0-3])\s*" + _MERIDIANO + r"?\b", t)
    if not m:
        m = re.search(r"\blas\s*([01]?\d|2[0-3])\s*" + _MERIDIANO + r"?\b", t)
    if m:
        c["horas"] = _horas(int(m.group(1)), m.group(2))
        return c

    # "5pm", "3 p.m." — marca obligatoria; sin ella "5" es índice/hora_suelta
    m = re.search(r"\b([1-9]|1[0-2])\s*" + _MERIDIANO, t)
    if m:
        c["horas"] = _horas(int(m.group(1)), m.group(2))
    return c


def match_slot(query: str, slots: list[datetime], ref: date):
    """int | list[int] | None — ver docstring del módulo.
    `ref` es la fecha "de hoy" en la TZ del ESTUDIO (la pasa el llamador;
    default de servidor = bug nocturno en UTC)."""
    t = _norm(query)
    hora_suelta = None

    # 1) Índice: "2", "2)", "opción 1", "la 2"
    m = re.fullmatch(r"(?:(?:opcion|la)\s*)?(\d{1,2})[).,]?", t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= len(slots):
            return n - 1
        if n <= 23:
            hora_suelta = n              # "15" pelado: fuera de índice → hora

    # 2) Ordinal: "la primera", "el segundo", "la última"
    m = re.fullmatch(r"(?:(?:la|el)\s+)?([a-z]+)", t)
    if m and m.group(1) in _ORDINALES:
        v = _ORDINALES[m.group(1)]
        idx = v - 1 if v > 0 else len(slots) + v        # -1 → último
        if 0 <= idx < len(slots):
            return idx

    # 3) Criterios de fecha/hora
    c = _parse_criterios(t, ref)
    if hora_suelta is not None and c["horas"] is None and c["hm"] is None:
        c["horas"] = _horas(hora_suelta, None)

    if all(v is None for v in c.values()):
        return None

    def ok(dt: datetime) -> bool:
        if c["dm"] and (dt.month, dt.day) != c["dm"]:
            return False
        if c["weekday"] is not None and dt.weekday() != c["weekday"]:
            return False
        if c["hm"] and (dt.hour, dt.minute) not in c["hm"]:
            return False
        if c["horas"] and dt.hour not in c["horas"]:
            return False
        return True

    hits = [i for i, dt in enumerate(slots) if ok(dt)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        return hits                        # ambigüedad real (ej: "17:30" ×2 días)
    return None


def extraer_dias_offset(query: str, ref: date) -> int | None:
    """Día ESPECÍFICO mencionado (turno de entrada o navegación sin match):
    "quiero hora el jueves" → 4; "mañana" → 1; "este miércoles" → 1 (o 0
    si HOY es miércoles); "18/11" → diff con rollover de año.
    Solo resuelve DÍAS: una hora sin fecha retorna None.

    Convenciones:
      - "ESTE miércoles" = el miércoles de esta semana (incluye HOY);
        "el miércoles" a secas = el PRÓXIMO (si hoy es miércoles → +7).
      - Fin de semana degrada natural: consultar_disponibilidad salta
        sábado/domingo ([]) y aterriza el lunes siguiente.
    """
    t = _norm(query)
    c = _parse_criterios(t, ref)
    if c["dm"]:
        mes, dia = c["dm"]
        anio = ref.year + (1 if (mes, dia) < (ref.month, ref.day) else 0)
        try:
            return (date(anio, mes, dia) - ref).days
        except ValueError:                       # 30/02, 29/02 no bisiesto…
            return None
    if c["weekday"] is not None:
        mismo = bool(re.search(r"\best[ae]\b", t))       # "este/esta" semana
        delta = (c["weekday"] - ref.weekday()) % 7
        return delta if (delta or mismo) else 7
    return None
