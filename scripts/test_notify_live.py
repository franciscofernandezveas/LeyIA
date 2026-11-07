"""Envía una notificación real de prueba al abogado de turno."""

import logging
import os
import sys

# Añade la raíz del proyecto al PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configura logging para ver los mensajes de leyia-notify
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from integrations.whatsapp.notify import notify_escalation, notify_new_lead

LAWYER = os.getenv("LAWYER_WHATSAPP_TO")
print(f"Destinatario configurado: {LAWYER or 'NO CONFIGURADO'}")


def test_new_lead():
    print("\n--- Test: notify_new_lead ---")
    ok = notify_new_lead({
        "nombre": "Prueba Notificación",
        "telefono": "+56965688788",
        "email": "prueba@leyia.cl",
        "categoria": "Test de notificación",
        "situacion": "Este es un mensaje de prueba para validar que el abogado de turno reciba alertas.",
        "etapa_proceso": "Prueba",
    })
    print(f"Resultado: {'✅ ENVIADO' if ok else '❌ FALLÓ'}")


def test_escalation():
    print("\n--- Test: notify_escalation ---")
    ok = notify_escalation(
        thread_id="thread_prueba_001",
        summary="El usuario solicita hablar con un abogado. Esta es una prueba del sistema de notificaciones.",
        telefono="+56965688788",
        motivo="hablar_abogada",
    )
    print(f"Resultado: {'✅ ENVIADO' if ok else '❌ FALLÓ'}")


if __name__ == "__main__":
    test_new_lead()
    test_escalation()
