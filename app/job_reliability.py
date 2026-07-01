"""Pure reliability helpers for worker timeouts and safe failure callbacks."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_AI_JOB_TIMEOUT_SECONDS = 600
MIN_AI_JOB_TIMEOUT_SECONDS = 30
MAX_AI_JOB_TIMEOUT_SECONDS = 3600

FAILURE_MESSAGES = {
    "worker_timeout": "AI processing took too long and was stopped. Continue with manual coach review.",
    "pose_backend_unavailable": "The AI pose service was unavailable. Continue with manual coach review.",
    "video_download_failed": "The private video could not be downloaded for AI review. Continue with manual coach review or retry later.",
    "video_decode_failed": "The video could not be decoded reliably. Continue with manual coach review or upload a clearer clip.",
    "insufficient_pose_confidence": "The system did not find enough reliable evidence to produce a coach-ready AI draft.",
    "cancelled_by_user": "AI processing was cancelled. Continue with manual coach review.",
    "unknown_worker_failure": "AI processing could not be completed. Continue with manual coach review.",
}

ALLOWED_FAILURE_STATUSES = {"failed", "timed_out", "cancelled"}
UNSAFE_KEYS = {
    "signed_video_url",
    "video_url",
    "file_path",
    "private_path",
    "callback_url",
    "raw_landmarks",
    "landmarks",
    "pose_results",
    "frames",
    "raw_frames",
    "height",
    "height_cm",
    "mass",
    "mass_kg",
    "guardian_email",
    "guardian_name",
    "swimmer_name",
    "stack_trace",
    "traceback",
}
UNSAFE_VALUE = re.compile(r"(?:token=|/tmp/|/var/|/Users/|supabase[^\s]*?/storage/)", re.IGNORECASE)


def job_timeout_seconds(raw: Optional[str] = None) -> int:
    value = os.getenv("AI_JOB_TIMEOUT_SECONDS") if raw is None else raw
    try:
        parsed = int(value) if value is not None else DEFAULT_AI_JOB_TIMEOUT_SECONDS
    except (TypeError, ValueError):
        return DEFAULT_AI_JOB_TIMEOUT_SECONDS
    if parsed < MIN_AI_JOB_TIMEOUT_SECONDS or parsed > MAX_AI_JOB_TIMEOUT_SECONDS:
        return DEFAULT_AI_JOB_TIMEOUT_SECONDS
    return parsed


def safe_reason_code(reason_code: str) -> str:
    return reason_code if reason_code in FAILURE_MESSAGES else "unknown_worker_failure"


def _safe_stage_history(entries: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    safe_entries: List[Dict[str, Any]] = []
    for entry in list(entries or [])[-50:]:
        safe_entries.append({
            key: entry.get(key)
            for key in ("stage", "status", "progress_percent", "timestamp", "at")
            if entry.get(key) is not None
        })
    return safe_entries


def build_failure_callback(
    *,
    video_upload_id: str,
    job_id: str,
    status: str,
    reason_code: str,
    engine: str,
    processing_duration_seconds: float,
    stage_history: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    safe_status = status if status in ALLOWED_FAILURE_STATUSES else "failed"
    safe_reason = safe_reason_code(reason_code)
    coach_message = FAILURE_MESSAGES[safe_reason]
    return {
        "job_id": job_id,
        "server_job_id": job_id,
        "video_upload_id": video_upload_id,
        "engine": engine,
        "status": safe_status,
        "reason_code": safe_reason,
        "error_code": safe_reason,
        "coach_message": coach_message,
        "error_message": coach_message,
        "manual_review_available": True,
        "analysis_mode": "manual_review",
        "real_pose_detected": False,
        "findings": [],
        "overall_score": None,
        "phase_breakdown": {},
        "drag_analysis": [],
        "key_frames": [],
        "technical_summary": coach_message,
        "stage_history": _safe_stage_history(stage_history),
        "processing_duration_seconds": round(max(0.0, processing_duration_seconds), 2),
        "pose_reliability": "failed",
        "quality_flags": [safe_reason],
        "recommended_next_action": "manual_review_recommended",
        "detection_ratio": 0,
        "frame_count_processed": 0,
        "detected_pose_frames": 0,
        "detected_keypoints_count": 0,
    }


def payload_is_safe(payload: Dict[str, Any]) -> bool:
    """True if the payload contains no unsafe keys or values.

    Payload-agnostic: used as the final redaction net for success, manual-review,
    and failure callbacks alike. Rejects unsafe keys (signed URLs, secrets,
    private paths, raw landmarks/frames, athlete profile, stack traces) and unsafe
    string values (URL tokens, /tmp//var//Users/ paths, supabase storage URLs).
    """
    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in UNSAFE_KEYS:
                    return False
                if not walk(child):
                    return False
            return True
        if isinstance(value, list):
            return all(walk(item) for item in value)
        if isinstance(value, str):
            return not UNSAFE_VALUE.search(value)
        return True

    return walk(payload)


# Backwards-compatible alias: the failure path (and existing tests) call this name.
failure_payload_is_safe = payload_is_safe


def unsafe_keys_in(payload: Any) -> List[str]:
    """Return offending KEY NAMES (plus a generic marker for value-pattern hits)
    for safe logging. NEVER returns the offending values themselves, so this can
    be logged without leaking signed URLs, secrets, or private paths.
    """
    found: List[str] = []

    def walk(value: Any, key_hint: Optional[str] = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in UNSAFE_KEYS:
                    found.append(str(key))
                walk(child, str(key))
        elif isinstance(value, list):
            for item in value:
                walk(item, key_hint)
        elif isinstance(value, str):
            if UNSAFE_VALUE.search(value):
                found.append(f"{key_hint or '?'}:<unsafe-value-pattern>")

    walk(payload)
    return sorted(set(found))
