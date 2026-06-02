import httpx
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


async def send_callback(callback_url: str, payload: Dict[str, Any]):
    ai_webhook_secret = os.getenv("AI_WEBHOOK_SECRET", "")

    if not ai_webhook_secret:
        logger.error("AI_WEBHOOK_SECRET is not set. Callback will likely be rejected.")

    headers = {
        "Content-Type": "application/json",
        "X-AI-WEBHOOK-SECRET": ai_webhook_secret,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload, headers=headers)

        if response.status_code in (200, 201, 202):
            logger.info(
                f"[{payload.get('video_upload_id')}] Callback accepted: HTTP {response.status_code}"
            )
        else:
            logger.error(
                f"[{payload.get('video_upload_id')}] Callback rejected: "
                f"HTTP {response.status_code} — {response.text[:500]}"
            )

    except httpx.TimeoutException:
        logger.error(f"[{payload.get('video_upload_id')}] Callback timed out")
    except Exception as e:
        logger.error(f"[{payload.get('video_upload_id')}] Callback error: {e}")
