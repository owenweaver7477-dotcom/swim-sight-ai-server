import asyncio
import os
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

CALLBACK_TIMEOUT_SECONDS = 30.0
# One transient Vercel blip must not lose a finished analysis: retry the
# callback a small number of times with short backoff. 5xx, timeouts, and
# network errors are retried; 4xx responses are not (they will not heal).
CALLBACK_MAX_ATTEMPTS = 3
CALLBACK_RETRY_BACKOFF_SECONDS = (1.0, 3.0)


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

    for attempt in range(1, CALLBACK_MAX_ATTEMPTS + 1):
        retryable = False
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
                    f"job_id={payload.get('job_id')}, video_upload_id={payload.get('video_upload_id')}, "
                    f"attempt={attempt}"
                )
                return True

            retryable = response.status_code >= 500
            logger.error(
                f"Callback rejected: HTTP {response.status_code}, attempt={attempt}, "
                f"body={response.text[:500]}, summary={safe_summary}"
            )

        except httpx.TimeoutException:
            retryable = True
            logger.error(
                f"Callback timed out after {CALLBACK_TIMEOUT_SECONDS}s "
                f"(attempt {attempt}): {safe_summary}"
            )

        except Exception as error:
            retryable = True
            logger.error(
                "Callback failed safely: error_type=%s, attempt=%s, summary=%s",
                type(error).__name__,
                attempt,
                safe_summary,
            )

        if not retryable or attempt >= CALLBACK_MAX_ATTEMPTS:
            return False

        backoff = CALLBACK_RETRY_BACKOFF_SECONDS[
            min(attempt - 1, len(CALLBACK_RETRY_BACKOFF_SECONDS) - 1)
        ]
        logger.info(
            "Retrying AI callback in %ss (attempt %s/%s): job_id=%s",
            backoff,
            attempt + 1,
            CALLBACK_MAX_ATTEMPTS,
            payload.get("job_id"),
        )
        await asyncio.sleep(backoff)

    return False
