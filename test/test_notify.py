"""Tests para notificaciones internas de LeyIA al abogado de turno."""

from unittest.mock import MagicMock

import httpx
import pytest

# Importamos el módulo completo para poder parchear sus constantes
import integrations.whatsapp.notify as notify_module


@pytest.fixture(autouse=True)
def configure_notify(monkeypatch):
    """
    Configura manualmente las constantes que notify.py usa.
    Así no dependemos de cuándo se importó el módulo.
    """
    monkeypatch.setattr(notify_module, "ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(notify_module, "PHONE_NUMBER_ID", "123456789")
    # API_VERSION también puede venir de config; la forzamos por si acaso
    monkeypatch.setattr(notify_module, "API_VERSION", "v21.0")
    monkeypatch.setenv("LAWYER_WHATSAPP_TO", "56999999999")
    monkeypatch.setenv("WHATSAPP_OPERATOR_NUMBER", "")
    monkeypatch.setenv("NOTIFY_TEMPLATE", "alerta_abogado_v1")


@pytest.fixture
def mock_post_200(monkeypatch):
    """Simula que Meta siempre acepta el mensaje."""
    def fake_post(self, *args, **kwargs):
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "messages": [{"id": "wamid.test123"}]
        }
        response.text = '{"messages":[{"id":"wamid.test123"}]}'
        return response

    monkeypatch.setattr(httpx.Client, "post", fake_post)


@pytest.fixture
def mock_post_closed_window(monkeypatch):
    """
    Simula la primera llamada rechazada por ventana de 24h cerrada,
    luego acepta el template, y finalmente acepta el texto reintentado.
    """
    call_count = {"n": 0}
    window_open = {"ready": False}  # Se pone True tras enviar template exitoso

    def fake_post(self, *args, **kwargs):
        response = MagicMock(spec=httpx.Response)
        call_count["n"] += 1

        payload = kwargs.get("json") or {}
        msg_type = payload.get("type")

        if msg_type == "text" and not window_open["ready"]:
            # Primera llamada: texto rechazado por ventana cerrada
            response.status_code = 400
            response.json.return_value = {
                "error": {"code": 131047, "message": "24h window closed"}
            }
            response.text = '{"error":{"code":131047,"message":"24h window closed"}}'
        else:
            # Template que abre ventana, o segundo texto con ventana ya abierta
            if msg_type == "template":
                window_open["ready"] = True
            response.status_code = 200
            response.json.return_value = {
                "messages": [{"id": f"wamid.retry{call_count['n']}"}]
            }
            response.text = '{"messages":[{"id":"wamid.retry"}]}'

        return response

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    return fake_post, call_count



def test_notify_new_lead(mock_post_200):
    ok = notify_module.notify_new_lead({
        "nombre": "Juan Pérez",
        "telefono": "+56987654321",
        "email": "juan@example.com",
        "categoria": "Pensión de alimentos",
        "situacion": "Debo pensión hace 3 meses",
        "etapa_proceso": "Sin juicio",
    })

    assert ok is True


def test_notify_escalation(mock_post_200):
    ok = notify_module.notify_escalation(
        thread_id="thread_abc123",
        summary="El cliente quiere hablar con un abogado sobre pensión de alimentos.",
        telefono="+56987654321",
        motivo="hablar_abogada",
    )

    assert ok is True


def test_notify_booking(mock_post_200):
    ok = notify_module.notify_booking({
        "cliente_nombre": "María González",
        "fecha": "2025-11-10",
        "hora_inicio": "10:00",
        "hora_fin": "10:30",
        "modalidad_label": "Videollamada Google Meet",
        "meet_link": "https://meet.google.com/abc-defg-hij",
    })

    assert ok is True


def test_escalation_uses_fallback_template_when_window_closed(
    mock_post_closed_window
):
    """Verifica que se intenta texto, luego template, luego texto de nuevo."""
    fake_post, calls = mock_post_closed_window

    ok = notify_module.notify_escalation(
        thread_id="thread_fallback",
        summary="Resumen de prueba",
        telefono="+56987654321",
        motivo="hablar_abogada",
    )

    assert ok is True
    assert calls["n"] == 3  # 1 texto fallido + 1 template + 1 texto exitoso


def test_notify_skips_when_no_lawyer_number(monkeypatch, mock_post_200):
    monkeypatch.setenv("LAWYER_WHATSAPP_TO", "")
    monkeypatch.setenv("WHATSAPP_OPERATOR_NUMBER", "")

    ok = notify_module.notify_escalation(
        thread_id="thread_no_number",
        summary="Sin destinatario",
    )

    assert ok is False
