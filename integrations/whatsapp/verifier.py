import hashlib
import hmac

from .config import APP_SECRET


def verify_signature(body: bytes, signature_header: str | None) -> bool:
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
