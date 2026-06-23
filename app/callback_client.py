import os
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

CALLBACK_TIMEOUT_SECONDS = 30.0


def _safe_payload_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Log useful callback info without exposing signed URLs or private paths."""
    return {
        "video_upload_id": payload.get("video_upload_id"),
        "job_id": payload.get("job_id"),
        "status": payload.get("status"),
        "analysis_mode": payload.get("analysis_mode"),
        "real_pose_detected": payload.get("real_pose_detected"),
        "pose_reliability": payload.get("pose_reliability"),
        "finding_count": len(payload.get("findings") or []),
        "quality_flags": payload.get("quality_flags") or [],
        "recommended_next_action": payload.get("recommended_next_action"),
    }


async def send_callback(
    callback_url: str,
    payload: Dict[str, Any],
    request_id: Optional[str] = None,
) -> bool:
    """
    Send AI result callback to Vercel.

    Required by Vercel callback route:
    - x-ai-webhook-secret: <AI_WEBHOOK_SECRET>
    """
    if not callback_url:
        logger.error("Callback skipped: missing callback_url")
        return False

    secret = os.getenv("AI_WEBHOOK_SECRET")
    if not secret:
        logger.error("Callback skipped: AI_WEBHOOK_SECRET is not set on Render")
        return False

    headers = {
        "Content-Type": "application/json",
        "x-ai-webhook-secret": secret,
    }

    if request_id:
        headers["x-request-id"] = request_id

    safe_summary = _safe_payload_summary(payload)
    logger.info(f"Sending AI callback: {safe_summary}")

    try:
        async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT_SECONDS) as client:
            response = await client.post(
                callback_url,
                json=payload,
                headers=headers,
            )

        if 200 <= response.status_code < 300:
            logger.info(
                f"Callback accepted: HTTP {response.status_code}, "
                f"job_id={payload.get('job_id')}, video_upload_id={payload.get('video_upload_id')}"
            )
            return True

        logger.error(
            f"Callback rejected: HTTP {response.status_code}, "
            f"body={response.text[:500]}, summary={safe_summary}"
        )
        return False

    except httpx.TimeoutException:
        logger.error(f"Callback timed out after {CALLBACK_TIMEOUT_SECONDS}s: {safe_summary}")
        return False

    except Exception as error:
        logger.error(
            "Callback failed safely: error_type=%s, summary=%s",
            type(error).__name__,
            safe_summary,
        )
        return False
