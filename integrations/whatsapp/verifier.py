import hashlib
import hmac

from integrations.whatsapp.config import APP_SECRET


def verify_signature(body: bytes, signature_header: str | None) -> bool:
    """
    Verifica la firma HMAC-SHA256 que Meta envía en X-Hub-Signature-256.
    Si APP_SECRET está vacío (modo desarrollo local), pasa sin validar.
    """
    if not APP_SECRET:
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = signature_header.removeprefix("sha256=")

    computed = hmac.new(
        APP_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, expected)
