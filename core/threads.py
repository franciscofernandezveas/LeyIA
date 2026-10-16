import re

def wa_thread_id(phone: str) -> str:
    """
    thread_id determinista para WhatsApp.
    Convención: 'wa-<teléfono>' (guión medio, NO underscore).
    """
    return "wa-" + re.sub(r"\D", "", phone or "")


def normalize_thread_id(thread_id: str) -> str:
    """
    Normaliza variantes históricas de thread_id a la convención actual.
    Sirve de red de seguridad si algún script/backfill introduce el
    formato antiguo con underscore.
    """
    if not thread_id:
        return thread_id

    # wa_12345678 → wa-12345678
    if thread_id.startswith("wa_"):
        return "wa-" + thread_id[3:]

    return thread_id


def channel_from_thread(thread_id: str) -> str:
    """
    Deriva el canal directamente del thread_id.
    Cualquier punto del sistema puede usarlo sin consultar la BD.
    """
    if not thread_id or "-" not in thread_id:
        return "desconocido"
    return thread_id.split("-", 1)[0]
