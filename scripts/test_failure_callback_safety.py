"""Failure callbacks must be useful to the app without leaking private inputs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.job_reliability import build_failure_callback, failure_payload_is_safe  # noqa: E402


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

print("failure callback safety test passed")
