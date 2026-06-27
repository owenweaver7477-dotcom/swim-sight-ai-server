"""Safe video metadata probing and timestamp sampling for AI progress callbacks.

This module is intentionally metadata-only. It never extracts image bytes,
never returns local paths, and never emits pose, 3D, biomechanics, force, or
drag values. Heavy optional dependencies are imported lazily.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_SAMPLING_RATE_FPS = 5.0
DEFAULT_MAX_SAMPLED_FRAMES = 300
FFPROBE_TIMEOUT_SECONDS = 10

UNSAFE_CALLBACK_KEYS = {
    "signed_video_url",
    "video_url",
    "file_url",
    "file_path",
    "path",
    "local_path",
    "private_path",
    "storage_path",
    "callback_url",
    "raw_frame",
    "raw_frames",
    "frame_bytes",
    "image_data",
    "pose_results",
    "landmarks",
    "joints_2d",
    "joints_3d",
    "estimated_drag",
    "drag_force",
    "force_frames",
    "biomechanics_frames",
}

UNSAFE_VALUE_MARKERS = (
    "token=",
    "access_token",
    "supabase.co/storage",
    "/users/",
    "/home/",
    "/var/folders/",
    "/tmp/",
    "\\users\\",
)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        parsed = int(float(value))
        if parsed >= 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def parse_rate(value: Any) -> Optional[float]:
    """Parse ffprobe frame-rate strings such as ``60000/1001``."""
    if value in (None, "", "0/0"):
        return None
    if isinstance(value, (int, float)):
        parsed = _safe_float(value)
        return parsed if parsed and parsed > 0 else None
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        top = _safe_float(numerator)
        bottom = _safe_float(denominator)
        if top and bottom:
            rate = top / bottom
            return rate if rate > 0 else None
        return None
    parsed = _safe_float(text)
    return parsed if parsed and parsed > 0 else None


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _orientation(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if not width or not height:
        return None
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _container_from_format(format_name: Optional[str]) -> Optional[str]:
    if not format_name:
        return None
    return str(format_name).split(",")[0].strip() or None


def _video_stream(streams: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for stream in streams:
        if stream.get("codec_type") == "video":
            return stream
    return None


def metadata_from_ffprobe_json(
    ffprobe_payload: Dict[str, Any],
    file_size_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Normalise an ffprobe JSON object to the worker metadata contract."""
    warnings: List[str] = []
    errors: List[str] = []
    streams = ffprobe_payload.get("streams") or []
    stream = _video_stream(streams)
    format_info = ffprobe_payload.get("format") or {}

    if not stream:
        errors.append("video_stream_not_found")
        stream = {}

    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate"))
    if not fps:
        warnings.append("fps_unavailable")

    duration_seconds = (
        _safe_float(stream.get("duration"))
        or _safe_float(format_info.get("duration"))
    )
    if duration_seconds is None:
        warnings.append("duration_unavailable")

    frame_count = _safe_int(stream.get("nb_frames"))
    if frame_count is None and fps and duration_seconds:
        frame_count = max(0, int(round(fps * duration_seconds)))
        warnings.append("frame_count_estimated")
    elif frame_count is None:
        warnings.append("frame_count_unavailable")

    width = _safe_int(stream.get("width"))
    height = _safe_int(stream.get("height"))
    if not width or not height:
        warnings.append("resolution_unavailable")

    probed_size_mb = None
    size_bytes = _safe_float(format_info.get("size"))
    if size_bytes is not None:
        probed_size_mb = size_bytes / (1024 * 1024)

    metadata = {
        "duration_seconds": _round(duration_seconds),
        "fps": _round(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "aspect_ratio": _round(width / height, 4) if width and height else None,
        "codec": stream.get("codec_name") or None,
        "container": _container_from_format(format_info.get("format_name")),
        "file_size_mb": _round(file_size_mb if file_size_mb is not None else probed_size_mb),
        "orientation": _orientation(width, height),
        "metadata_complete": bool(duration_seconds and fps and frame_count and width and height),
        "probe_source": "ffprobe",
        "warnings": sorted(set(warnings)),
        "errors": errors,
    }
    return metadata


def metadata_from_cv2_capture(
    video_path: str,
    file_size_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Fallback metadata probe using OpenCV, imported lazily."""
    warnings: List[str] = ["ffprobe_unavailable"]
    errors: List[str] = []
    try:
        import cv2  # noqa: WPS433 - intentionally lazy
    except Exception:
        return {
            "duration_seconds": None,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "codec": None,
            "container": None,
            "file_size_mb": _round(file_size_mb),
            "orientation": None,
            "metadata_complete": False,
            "probe_source": "unavailable",
            "warnings": warnings,
            "errors": ["opencv_unavailable"],
        }

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            errors.append("video_open_failed")
            return {
                "duration_seconds": None,
                "fps": None,
                "frame_count": None,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "codec": None,
                "container": None,
                "file_size_mb": _round(file_size_mb),
                "orientation": None,
                "metadata_complete": False,
                "probe_source": "opencv",
                "warnings": warnings,
                "errors": errors,
            }

        fps = _safe_float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = _safe_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = _safe_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _safe_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = frame_count / fps if fps and frame_count else None

        if not fps:
            warnings.append("fps_unavailable")
        if frame_count is None:
            warnings.append("frame_count_unavailable")
        if duration_seconds is None:
            warnings.append("duration_unavailable")
        if not width or not height:
            warnings.append("resolution_unavailable")

        return {
            "duration_seconds": _round(duration_seconds),
            "fps": _round(fps),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "aspect_ratio": _round(width / height, 4) if width and height else None,
            "codec": None,
            "container": None,
            "file_size_mb": _round(file_size_mb),
            "orientation": _orientation(width, height),
            "metadata_complete": bool(duration_seconds and fps and frame_count and width and height),
            "probe_source": "opencv",
            "warnings": sorted(set(warnings)),
            "errors": errors,
        }
    finally:
        capture.release()


def probe_video_file(
    video_path: str,
    file_size_mb: Optional[float] = None,
) -> Dict[str, Any]:
    """Probe a local temporary video file and return safe metadata only."""
    warnings: List[str] = []
    errors: List[str] = []
    if not video_path or not os.path.exists(video_path):
        return {
            "ok": False,
            "video_metadata": {
                "duration_seconds": None,
                "fps": None,
                "frame_count": None,
                "width": None,
                "height": None,
                "aspect_ratio": None,
                "codec": None,
                "container": None,
                "file_size_mb": _round(file_size_mb),
                "orientation": None,
                "metadata_complete": False,
                "probe_source": "unavailable",
                "warnings": [],
                "errors": ["video_file_missing"],
            },
            "warnings": [],
            "errors": ["video_file_missing"],
        }

    if file_size_mb is None:
        try:
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        except OSError:
            file_size_mb = None
            warnings.append("file_size_unavailable")

    ffprobe = shutil.which("ffprobe")
    metadata: Dict[str, Any]
    if ffprobe:
        try:
            completed = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    video_path,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
            if completed.returncode == 0 and completed.stdout:
                metadata = metadata_from_ffprobe_json(
                    json.loads(completed.stdout),
                    file_size_mb=file_size_mb,
                )
            else:
                warnings.append("ffprobe_failed")
                metadata = metadata_from_cv2_capture(video_path, file_size_mb=file_size_mb)
        except subprocess.TimeoutExpired:
            warnings.append("ffprobe_timeout")
            metadata = metadata_from_cv2_capture(video_path, file_size_mb=file_size_mb)
        except Exception:
            warnings.append("ffprobe_error")
            metadata = metadata_from_cv2_capture(video_path, file_size_mb=file_size_mb)
    else:
        metadata = metadata_from_cv2_capture(video_path, file_size_mb=file_size_mb)

    combined_warnings = sorted(set([*warnings, *(metadata.get("warnings") or [])]))
    combined_errors = sorted(set([*errors, *(metadata.get("errors") or [])]))
    metadata["warnings"] = combined_warnings
    metadata["errors"] = combined_errors

    return {
        "ok": bool(metadata.get("metadata_complete")) and not combined_errors,
        "video_metadata": metadata,
        "warnings": combined_warnings,
        "errors": combined_errors,
    }


def build_frame_sampling_plan(
    video_metadata: Dict[str, Any],
    sampling_rate_fps: float = DEFAULT_SAMPLING_RATE_FPS,
    max_sampled_frames: int = DEFAULT_MAX_SAMPLED_FRAMES,
) -> Dict[str, Any]:
    """Build a timestamp-only frame sampling plan. No images are read or saved."""
    warnings: List[str] = []
    errors: List[str] = []
    rate = _safe_float(sampling_rate_fps, DEFAULT_SAMPLING_RATE_FPS) or DEFAULT_SAMPLING_RATE_FPS
    max_frames = _safe_int(max_sampled_frames, DEFAULT_MAX_SAMPLED_FRAMES) or DEFAULT_MAX_SAMPLED_FRAMES
    fps = _safe_float(video_metadata.get("fps"))
    duration = _safe_float(video_metadata.get("duration_seconds"))
    frame_count = _safe_int(video_metadata.get("frame_count"))

    if rate <= 0:
        errors.append("invalid_sampling_rate")
        rate = DEFAULT_SAMPLING_RATE_FPS
    if max_frames <= 0:
        errors.append("invalid_max_sampled_frames")
        max_frames = DEFAULT_MAX_SAMPLED_FRAMES

    if duration is None and fps and frame_count:
        duration = frame_count / fps
        warnings.append("duration_estimated_from_frame_count")
    if frame_count is None and fps and duration:
        frame_count = max(0, int(round(fps * duration)))
        warnings.append("frame_count_estimated_for_sampling")
    if duration is None:
        errors.append("duration_unavailable")
    if fps is None:
        warnings.append("source_fps_unavailable")

    planned_count = 0
    if duration is not None:
        planned_count = max(1, int(math.ceil(max(duration, 0.001) * rate)))
    if planned_count > max_frames:
        warnings.append("sample_count_capped")
        planned_count = max_frames

    interval_ms = 1000.0 / rate if rate > 0 else 200.0
    samples: List[Dict[str, Any]] = []
    for sample_index in range(planned_count):
        timestamp_ms = int(round(sample_index * interval_ms))
        if duration is not None:
            timestamp_ms = min(timestamp_ms, max(0, int(round(duration * 1000))))
        source_frame_index = int(round((timestamp_ms / 1000.0) * fps)) if fps else None
        if frame_count is not None and source_frame_index is not None:
            source_frame_index = min(source_frame_index, max(0, frame_count - 1))
        samples.append({
            "sampleIndex": sample_index,
            "sourceFrameIndex": source_frame_index,
            "timestampMs": timestamp_ms,
            "status": "scheduled",
        })

    return {
        "ok": not errors,
        "sampling_rate_fps": _round(rate),
        "source_fps": _round(fps),
        "max_sampled_frames": max_frames,
        "total_sampled_frames": len(samples),
        "duration_seconds": _round(duration),
        "samples": samples,
        "warnings": sorted(set(warnings)),
        "errors": errors,
    }


def _request_value(request: Any, field: str) -> Any:
    if isinstance(request, dict):
        return request.get(field)
    return getattr(request, field, None)


def build_video_probe_callback_payload(
    request: Any,
    job_id: str,
    status_value: str,
    video_probe: Dict[str, Any],
    frame_sampling: Optional[Dict[str, Any]] = None,
    engine: str = "pose-mvp-0.5",
) -> Dict[str, Any]:
    """Build a safe additive progress callback for the Vercel app."""
    video_metadata = dict(video_probe.get("video_metadata") or {})
    video_metadata.pop("filename", None)
    video_metadata.pop("path", None)
    warnings = sorted(set([
        *(video_probe.get("warnings") or []),
        *(video_metadata.get("warnings") or []),
        *((frame_sampling or {}).get("warnings") or []),
    ]))
    errors = sorted(set([
        *(video_probe.get("errors") or []),
        *(video_metadata.get("errors") or []),
        *((frame_sampling or {}).get("errors") or []),
    ]))

    payload: Dict[str, Any] = {
        "job_id": job_id,
        "app_job_id": _request_value(request, "app_job_id") or job_id,
        "server_job_id": job_id,
        "video_upload_id": _request_value(request, "video_upload_id"),
        "engine": engine,
        "status": status_value,
        "video_metadata": video_metadata,
        "warnings": warnings,
        "errors": errors,
    }

    if frame_sampling is not None:
        payload["frame_sampling"] = frame_sampling

    return payload


def iter_payload_keys_and_values(value: Any) -> Iterable[Tuple[Optional[str], Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key), nested
            yield from iter_payload_keys_and_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_payload_keys_and_values(nested)
    else:
        yield None, value


def callback_payload_is_safe(payload: Dict[str, Any]) -> bool:
    """Return false when a probe callback contains private paths or raw data."""
    for key, value in iter_payload_keys_and_values(payload):
        if key and key.lower() in UNSAFE_CALLBACK_KEYS:
            return False
        if isinstance(value, str):
            lowered = value.lower()
            if any(marker in lowered for marker in UNSAFE_VALUE_MARKERS):
                return False
    return True
