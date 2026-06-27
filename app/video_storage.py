"""Private video storage access adapter for worker jobs.

The current Render worker can still download a short-lived signed URL. The
provider/key shape exists so future workers can use scoped Supabase, S3, or GCS
object access without changing the /process-video request contract again.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import quote


SUPPORTED_STORAGE_PROVIDERS = {
    "supabase_private",
    "s3_private",
    "gcs_private",
}
DEFAULT_SUPABASE_BUCKET = "private-videos"


@dataclass(frozen=True)
class ProviderObjectRequest:
    provider: str
    video_key: str
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    configured: bool = False
    reason: Optional[str] = None


def _clean(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def normalise_storage_provider(value: Optional[str]) -> Optional[str]:
    provider = _clean(value)
    if not provider:
        return None
    provider = provider.lower()
    return provider if provider in SUPPORTED_STORAGE_PROVIDERS else provider


def redact_signed_url(url: Optional[str]) -> str:
    return "[redacted-signed-url]" if _clean(url) else ""


def redact_video_key(video_key: Optional[str]) -> str:
    key = _clean(video_key)
    if not key:
        return ""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    parts = [part for part in key.split("/") if part]
    suffix = parts[-1] if parts else "object"
    safe_suffix = suffix[-24:]
    return f"[redacted-key:{digest}:{safe_suffix}]"


def get_request_field(request: Any, field: str) -> Any:
    if isinstance(request, dict):
        return request.get(field)
    return getattr(request, field, None)


def has_video_access_method(request: Any) -> bool:
    if _clean(get_request_field(request, "signed_video_url")):
        return True
    provider = normalise_storage_provider(get_request_field(request, "storage_provider"))
    video_key = _clean(get_request_field(request, "video_key"))
    return bool(provider and video_key)


def safe_video_access_summary(request: Any) -> Dict[str, Any]:
    provider = normalise_storage_provider(get_request_field(request, "storage_provider"))
    video_key = _clean(get_request_field(request, "video_key"))
    signed_url = _clean(get_request_field(request, "signed_video_url"))
    if signed_url:
        access_mode = "signed_url"
    elif provider and video_key:
        access_mode = "provider_key"
    else:
        access_mode = "missing"
    return {
        "access_mode": access_mode,
        "storage_provider": provider,
        "video_key_present": bool(video_key),
        "video_key_redacted": redact_video_key(video_key),
        "signed_video_url_present": bool(signed_url),
    }


def build_supabase_object_request(video_key: str) -> ProviderObjectRequest:
    supabase_url = _clean(os.getenv("SUPABASE_URL"))
    service_key = _clean(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SERVICE_KEY")
    )
    bucket = _clean(os.getenv("SUPABASE_VIDEO_BUCKET")) or DEFAULT_SUPABASE_BUCKET
    if not supabase_url or not service_key:
        return ProviderObjectRequest(
            provider="supabase_private",
            video_key=video_key,
            configured=False,
            reason="supabase_storage_env_missing",
        )

    safe_base = supabase_url.rstrip("/")
    safe_path = quote(video_key.lstrip("/"), safe="/")
    return ProviderObjectRequest(
        provider="supabase_private",
        video_key=video_key,
        url=f"{safe_base}/storage/v1/object/{bucket}/{safe_path}",
        headers={"Authorization": f"Bearer {service_key}"},
        configured=True,
    )


def build_provider_object_request(provider: str, video_key: str) -> ProviderObjectRequest:
    provider = normalise_storage_provider(provider) or ""
    video_key = _clean(video_key) or ""
    if provider == "supabase_private":
        return build_supabase_object_request(video_key)
    if provider in {"s3_private", "gcs_private"}:
        return ProviderObjectRequest(
            provider=provider,
            video_key=video_key,
            configured=False,
            reason=f"{provider}_adapter_not_configured",
        )
    return ProviderObjectRequest(
        provider=provider or "unknown",
        video_key=video_key,
        configured=False,
        reason="unsupported_storage_provider",
    )


async def download_video_for_request(request: Any):
    """Download the video described by a worker request.

    Returns app.video_processor.DownloadResult. No signed URL or private key is
    logged or returned by this adapter.
    """
    from app.video_processor import download_video

    video_upload_id = _clean(get_request_field(request, "video_upload_id")) or "unknown-video"
    signed_url = _clean(get_request_field(request, "signed_video_url"))
    if signed_url:
        return await download_video(
            video_upload_id=video_upload_id,
            signed_url=signed_url,
            source_label="signed_url",
        )

    provider = normalise_storage_provider(get_request_field(request, "storage_provider"))
    video_key = _clean(get_request_field(request, "video_key"))
    if not provider or not video_key:
        from app.video_processor import DownloadResult

        return DownloadResult(
            failed_reason="missing_video_access_method",
            quality_flags=["missing_video_access_method", "manual_review_recommended"],
        )

    provider_request = build_provider_object_request(provider, video_key)
    if not provider_request.configured or not provider_request.url:
        from app.video_processor import DownloadResult

        return DownloadResult(
            failed_reason=provider_request.reason or "storage_provider_unavailable",
            quality_flags=[
                provider_request.reason or "storage_provider_unavailable",
                "manual_review_recommended",
            ],
        )

    return await download_video(
        video_upload_id=video_upload_id,
        signed_url=provider_request.url,
        headers=provider_request.headers,
        source_label=provider_request.provider,
    )


def payload_contains_private_video_access(payload: Dict[str, Any]) -> bool:
    text = str(payload)
    return (
        "signed_video_url" in text
        or "token=" in text
        or "x-amz-signature" in text.lower()
        or "x-goog-signature" in text.lower()
        or "video_key" in text
        or "private-videos/" in text
    )
