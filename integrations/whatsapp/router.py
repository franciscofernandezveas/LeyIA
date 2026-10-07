import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from .config import APP_SECRET, VERIFY_TOKEN
from .handler import handle_messages
from .parser import parse_webhook_payload
from . import verifier

logger = logging.getLogger("leyia-whatsapp")

router = APIRouter(prefix="/webhook/whatsapp", tags=["whatsapp"])


@router.get("")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_verify_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
):
    if hub_mode == "subscribe" and VERIFY_TOKEN and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403, detail="Token de verificación inválido")


@router.post("")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()

    if APP_SECRET and not verifier.verify_signature(
        body, request.headers.get("X-Hub-Signature-256")
    ):
        raise HTTPException(status_code=403, detail="Firma inválida")

    payload = json.loads(body)
    if payload.get("object") != "whatsapp_business_account":
        return {"status": "ignored"}

    messages = parse_webhook_payload(payload)
    if messages:
        background_tasks.add_task(handle_messages, messages)

    return {"status": "ok"}
