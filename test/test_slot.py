# test_slot_match_taps.py
from datetime import date, datetime
from graph.slot_match import match_slot

REF = date(2025, 9, 13)  # sábado

SLOTS = [
    datetime(2025, 9, 15, 10, 0),   # 0 — Lun
    datetime(2025, 9, 15, 16, 30),  # 1 — Lun
    datetime(2025, 9, 17, 16, 0),   # 2 — Mie
    datetime(2025, 9, 26, 10, 0),   # 3 — Vie (título máx largo: "4) Vie 26/09 10:00")
]

# taps reales del tap de botón/lista WhatsApp (títulos post-integración)
assert match_slot("1) Lun 15/09 10:00", SLOTS, REF) == 0
assert match_slot("2) Lun 15/09 16:30", SLOTS, REF) == 1
assert match_slot("3) Mie 17/09 16:00", SLOTS, REF) == 2   # ← el día del bug viejo
assert match_slot("4) Vie 26/09 10:00", SLOTS, REF) == 3

# texto libre tradicional — sin regresión:
assert match_slot("2", SLOTS, REF) == 1
assert match_slot("opción 3", SLOTS, REF) == 2
assert match_slot("la primera", SLOTS, REF) == 0
assert match_slot("a las 16:30", SLOTS, REF) == 1
assert match_slot("mañana a las 5:30 pm", SLOTS, REF) == 1   # 15/09 = lunes
