import re

def wa_thread_id(phone: str) -> str:
    """
    thread_id determinista para WhatsApp.
    Respeta la convención ya existente: 'wa-<teléfono>' (guión medio).
    """
    return "wa-" + re.sub(r"\D", "", phone or "")


def channel_from_thread(thread_id: str) -> str:
    """
    Deriva el canal directamente del thread_id.
    Cualquier punto del sistema puede usarlo sin consultar la BD.
    """
    if not thread_id or "-" not in thread_id:
        return "desconocido"
    return thread_id.split("-", 1)[0]
