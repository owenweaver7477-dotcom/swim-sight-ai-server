"""Failure callbacks must be useful to the app without leaking private inputs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.job_reliability import (  # noqa: E402
    build_failure_callback,
    failure_payload_is_safe,
    payload_is_safe,
    unsafe_keys_in,
)


for status, reason in [
    ("failed", "unknown_worker_failure"),
    ("timed_out", "worker_timeout"),
    ("cancelled", "cancelled_by_user"),
    ("failed", "video_download_failed"),
]:
    payload = build_failure_callback(
        video_upload_id="synthetic-video-id",
        job_id="synthetic-job-id",
        status=status,
        reason_code=reason,
        engine="pose-mvp-test",
        processing_duration_seconds=12.5,
        stage_history=[{
            "stage": "extracting_frames",
            "status": "running",
            "progress_percent": 35,
            "message": "safe internal message is deliberately omitted",
        }],
    )
    assert payload["status"] == status
    assert payload["reason_code"] == reason
    assert payload["manual_review_available"] is True
    assert payload["analysis_mode"] == "manual_review"
    assert payload["findings"] == []
    assert payload["overall_score"] is None
    assert "message" not in payload["stage_history"][0]
    assert failure_payload_is_safe(payload)

unsafe_payload = build_failure_callback(
    video_upload_id="synthetic-video-id",
    job_id="synthetic-job-id",
    status="failed",
    reason_code="unknown_worker_failure",
    engine="pose-mvp-test",
    processing_duration_seconds=0,
)
unsafe_payload["signed_video_url"] = "https://storage.example/video?token=secret"
assert failure_payload_is_safe(unsafe_payload) is False

# ── Generalized payload_is_safe now guards success + manual-review payloads too ──

# failure_payload_is_safe is a backwards-compatible alias of payload_is_safe.
assert failure_payload_is_safe is payload_is_safe

# Representative success + manual-review payloads are safe.
success_like = {
    "status": "completed",
    "analysis_mode": "real_pose",
    "findings": [{"fault_tag": "body_line_loss", "observation": "hips drop", "coach_review_required": True}],
    "processing_telemetry": {"quality_flags": [], "pose_detection_rate": 0.74},
    "phase_breakdown": {"body_line": {"status": "review_required", "label": "Body Line"}},
}
assert payload_is_safe(success_like) is True

manual_like = {
    "status": "manual_review_recommended",
    "analysis_mode": "manual_review",
    "findings": [],
    "quality_flags": ["manual_review_recommended"],
    "recommended_next_action": "manual_review_recommended",
}
assert payload_is_safe(manual_like) is True

# An unsafe KEY is detected and reported by name only (never the value).
unsafe_key_payload = {"status": "completed", "signed_video_url": "https://x/v?token=abc123", "findings": []}
assert payload_is_safe(unsafe_key_payload) is False
reported = unsafe_keys_in(unsafe_key_payload)
assert "signed_video_url" in reported
assert not any("abc123" in item or "token=" in item for item in reported), "must not leak the secret value"

# An unsafe VALUE (private path) is detected and reported generically, no value leak.
unsafe_value_payload = {"status": "completed", "note": "/Users/owen/private/clip.mp4", "findings": []}
assert payload_is_safe(unsafe_value_payload) is False
reported2 = unsafe_keys_in(unsafe_value_payload)
assert any("unsafe-value-pattern" in item for item in reported2)
assert not any("/Users/" in item for item in reported2), "must not leak the private path"

print("failure callback safety test passed")
