"""Worker storage adapter tests.

These tests do not perform network access and do not require Supabase/S3/GCS
credentials.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models import VideoProcessingRequest  # noqa: E402
from app.video_storage import (  # noqa: E402
    build_provider_object_request,
    has_video_access_method,
    payload_contains_private_video_access,
    redact_signed_url,
    redact_video_key,
    safe_video_access_summary,
)


signed_request = VideoProcessingRequest(
    job_id="job-signed",
    video_upload_id="video-signed",
    signed_video_url="https://example.supabase.co/storage/v1/object/sign/private-videos/video.mp4?token=secret",
    callback_url="https://callback.example/api/ai/callback",
    stroke_type="Freestyle",
)
assert has_video_access_method(signed_request) is True
signed_summary = safe_video_access_summary(signed_request)
assert signed_summary["access_mode"] == "signed_url"
assert signed_summary["signed_video_url_present"] is True
assert "token=secret" not in str(signed_summary)
assert redact_signed_url(signed_request.signed_video_url) == "[redacted-signed-url]"

provider_request = VideoProcessingRequest(
    job_id="job-provider",
    video_upload_id="video-provider",
    storage_provider="supabase_private",
    video_key="club-a/swimmer-b/video-c/private-video.mp4",
    callback_url="https://callback.example/api/ai/callback",
    stroke_type="Breaststroke",
)
assert has_video_access_method(provider_request) is True
provider_summary = safe_video_access_summary(provider_request)
assert provider_summary["access_mode"] == "provider_key"
assert provider_summary["storage_provider"] == "supabase_private"
assert provider_summary["video_key_present"] is True
assert "club-a/swimmer-b/video-c" not in str(provider_summary)
assert redact_video_key(provider_request.video_key).startswith("[redacted-key:")

missing_request = VideoProcessingRequest(
    job_id="job-missing",
    video_upload_id="video-missing",
    callback_url="https://callback.example/api/ai/callback",
)
assert has_video_access_method(missing_request) is False
assert safe_video_access_summary(missing_request)["access_mode"] == "missing"

old_url = os.environ.pop("SUPABASE_URL", None)
old_key = os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
old_alt_key = os.environ.pop("SUPABASE_SERVICE_KEY", None)
try:
    unconfigured = build_provider_object_request("supabase_private", "club/video.mp4")
    assert unconfigured.configured is False
    assert unconfigured.reason == "supabase_storage_env_missing"

    os.environ["SUPABASE_URL"] = "https://example.supabase.co"
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "service-secret"
    configured = build_provider_object_request("supabase_private", "club/video.mp4")
    assert configured.configured is True
    assert configured.url == "https://example.supabase.co/storage/v1/object/private-videos/club/video.mp4"
    assert configured.headers == {"Authorization": "Bearer service-secret"}
finally:
    if old_url is not None:
        os.environ["SUPABASE_URL"] = old_url
    else:
        os.environ.pop("SUPABASE_URL", None)
    if old_key is not None:
        os.environ["SUPABASE_SERVICE_ROLE_KEY"] = old_key
    else:
        os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
    if old_alt_key is not None:
        os.environ["SUPABASE_SERVICE_KEY"] = old_alt_key
    else:
        os.environ.pop("SUPABASE_SERVICE_KEY", None)

assert payload_contains_private_video_access({
    "signed_video_url": signed_request.signed_video_url,
}) is True
assert payload_contains_private_video_access({
    "video_key": provider_request.video_key,
}) is True
assert payload_contains_private_video_access({
    "status": "failed",
    "coach_message": "Manual review remains available.",
}) is False

print("video storage adapter tests passed")
