import cv2
import httpx
import tempfile
import os
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


STANDARD_MAX_WIDTH = _env_int("STANDARD_MAX_WIDTH", 640)
REDUCED_MAX_WIDTH = _env_int("REDUCED_MAX_WIDTH", 480)
MINIMAL_MAX_WIDTH = _env_int("MINIMAL_MAX_WIDTH", 360)
STANDARD_MAX_FRAMES = _env_int("STANDARD_MAX_FRAMES", 60)
REDUCED_MAX_FRAMES = _env_int("REDUCED_MAX_FRAMES", 30)
MINIMAL_MAX_FRAMES = _env_int("MINIMAL_MAX_FRAMES", 12)
STANDARD_MAX_DURATION_SECONDS = _env_int("STANDARD_MAX_DURATION_SECONDS", 20)
REDUCED_MAX_DURATION_SECONDS = _env_int("REDUCED_MAX_DURATION_SECONDS", 15)
MINIMAL_MAX_DURATION_SECONDS = _env_int("MINIMAL_MAX_DURATION_SECONDS", 8)
MAX_DOWNLOAD_MB = _env_int("MAX_DOWNLOAD_MB", 150)
HIGH_RES_PIXEL_THRESHOLD = 1920 * 1080
EXTREME_PIXEL_THRESHOLD = 2560 * 1440
VERY_EXTREME_PIXEL_THRESHOLD = 3000 * 2000
MAX_REASONABLE_FPS = _env_int("MAX_REASONABLE_FPS", 120)


@dataclass
class DownloadResult:
    path: Optional[str] = None
    file_size_mb: float = 0.0
    failed_reason: Optional[str] = None
    manual_review_reason: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)


@dataclass
class FrameExtractionResult:
    frames: List[Tuple[int, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


def sanitize_url_for_logs(url: Optional[str]) -> str:
    if not url:
        return ""
    return "[redacted-url]"


async def download_video(video_upload_id: str, signed_url: str) -> DownloadResult:
    """
    Download private signed video URL to a temporary file.

    Does not log the signed URL or token.
    """
    try:
        logger.info(f"[{video_upload_id}] Downloading private video for AI processing")

        max_bytes = MAX_DOWNLOAD_MB * 1024 * 1024
        downloaded = 0

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            path = tmp.name

            async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
                async with client.stream("GET", signed_url) as response:
                    if response.status_code != 200:
                        logger.error(
                            f"[{video_upload_id}] Download failed: HTTP {response.status_code}"
                        )
                        return DownloadResult(
                            failed_reason=f"download_http_{response.status_code}",
                            quality_flags=["signed_url_download_failed"],
                        )

                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue

                        downloaded += len(chunk)

                        if downloaded > max_bytes:
                            logger.warning(
                                f"[{video_upload_id}] Download stopped safely: file exceeds "
                                f"{MAX_DOWNLOAD_MB} MB worker safety limit"
                            )
                            try:
                                os.remove(path)
                            except Exception:
                                pass
                            return DownloadResult(
                                file_size_mb=round(downloaded / (1024 * 1024), 2),
                                manual_review_reason="video_too_heavy_for_ai_processing",
                                quality_flags=["video_too_large_for_worker", "manual_review_recommended"],
                            )

                        tmp.write(chunk)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"[{video_upload_id}] Downloaded video: {size_mb:.1f} MB")
        return DownloadResult(path=path, file_size_mb=round(size_mb, 2))

    except Exception as e:
        logger.exception(f"[{video_upload_id}] Download error: {e}")
        return DownloadResult(
            failed_reason="download_error",
            quality_flags=["signed_url_download_failed"],
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if not math.isfinite(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
        return number if number >= 0 else default
    except (TypeError, ValueError):
        return default


def _looks_like_screen_recording(filename: Optional[str], capture_source: Optional[str]) -> bool:
    joined = f"{filename or ''} {capture_source or ''}".lower()
    indicators = [
        "screen recording",
        "screen_recording",
        "screen-recording",
        "screenrecording",
        "screenshot",
        "screen",
    ]
    return any(indicator in joined for indicator in indicators)


def classify_video_workload(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify workload before expensive frame extraction.

    File size alone is not decisive; decoded pixel workload, duration, FPS,
    and screen-recording hints carry more weight.
    """
    flags = set(metadata.get("quality_flags") or [])

    file_size_mb = _safe_float(metadata.get("file_size_mb"))
    duration_seconds = _safe_float(metadata.get("duration_seconds"))
    fps = _safe_float(metadata.get("fps"))
    frame_count_total = _safe_int(metadata.get("frame_count_total"))
    source_width = _safe_int(metadata.get("source_width"))
    source_height = _safe_int(metadata.get("source_height"))
    filename = metadata.get("filename")
    capture_source = metadata.get("capture_source")
    source_pixel_count = source_width * source_height
    estimated_workload = source_pixel_count * max(frame_count_total, 1)
    looks_screen = bool(metadata.get("looks_like_screen_recording")) or _looks_like_screen_recording(
        filename,
        capture_source,
    )

    manual_reason = None

    if not source_width or not source_height or not frame_count_total:
        flags.add("metadata_unreadable")
        manual_reason = "metadata_unreadable"

    if file_size_mb and file_size_mb > MAX_DOWNLOAD_MB:
        flags.add("video_too_large_for_worker")
        manual_reason = "video_too_heavy_for_ai_processing"

    if duration_seconds and duration_seconds > 180:
        flags.add("video_too_long_for_ai_window")
        manual_reason = "video_too_heavy_for_ai_processing"

    if fps and fps > 60:
        flags.add("high_fps_video")
    if fps and fps > MAX_REASONABLE_FPS:
        if fps > MAX_REASONABLE_FPS * 2:
            manual_reason = "video_too_heavy_for_ai_processing"

    if source_pixel_count >= VERY_EXTREME_PIXEL_THRESHOLD and duration_seconds > 45:
        flags.add("video_too_high_resolution")
        manual_reason = "video_too_heavy_for_ai_processing"

    if manual_reason:
        tier = "manual_review_required"
        max_width = 0
        max_frames = 0
        window = 0
    elif (
        source_pixel_count >= EXTREME_PIXEL_THRESHOLD
        or file_size_mb >= 120
        or duration_seconds > 45
        or fps > 90
        or (looks_screen and source_pixel_count >= HIGH_RES_PIXEL_THRESHOLD)
    ):
        tier = "minimal_ai"
        max_width = MINIMAL_MAX_WIDTH
        max_frames = MINIMAL_MAX_FRAMES
        window = MINIMAL_MAX_DURATION_SECONDS
        flags.add("minimal_ai_sampling")
        flags.add("video_processing_risk")
        if source_pixel_count >= EXTREME_PIXEL_THRESHOLD:
            flags.add("video_too_high_resolution")
        if looks_screen:
            flags.add("screen_recording_possible")
    elif (
        source_pixel_count > HIGH_RES_PIXEL_THRESHOLD
        or file_size_mb >= 80
        or duration_seconds > STANDARD_MAX_DURATION_SECONDS
        or fps > 60
        or looks_screen
    ):
        tier = "reduced_ai"
        max_width = REDUCED_MAX_WIDTH
        max_frames = REDUCED_MAX_FRAMES
        window = REDUCED_MAX_DURATION_SECONDS
        flags.add("heavy_video_downsampled")
        if duration_seconds > STANDARD_MAX_DURATION_SECONDS:
            flags.add("video_too_long_for_ai_window")
        if looks_screen:
            flags.add("screen_recording_possible")
    else:
        tier = "standard_ai"
        max_width = STANDARD_MAX_WIDTH
        max_frames = STANDARD_MAX_FRAMES
        window = STANDARD_MAX_DURATION_SECONDS

    if duration_seconds and window and duration_seconds > window:
        flags.add("sampled_processing_window")

    return {
        **metadata,
        "processing_tier": tier,
        "max_processing_width": max_width,
        "max_sampled_frames": max_frames,
        "processing_window_seconds": min(duration_seconds, window) if duration_seconds and window else window,
        "source_pixel_count": source_pixel_count,
        "estimated_workload": estimated_workload,
        "looks_like_screen_recording": looks_screen,
        "quality_flags": sorted(flags),
        "manual_review_reason": manual_reason,
    }


def _metadata_from_capture(
    cap,
    video_path: str,
    filename: Optional[str] = None,
    capture_source: Optional[str] = None,
) -> Dict[str, Any]:
    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), 0.0)
    if fps <= 0:
        fps = 30.0

    frame_count_total = _safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)
    source_width = _safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0)
    source_height = _safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)
    duration_seconds = frame_count_total / fps if fps > 0 and frame_count_total > 0 else 0.0
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0.0

    return {
        "file_size_mb": round(file_size_mb, 2),
        "duration_seconds": round(duration_seconds, 2),
        "fps": round(fps, 2),
        "frame_count_total": frame_count_total,
        "source_width": source_width,
        "source_height": source_height,
        "filename": filename or None,
        "capture_source": capture_source or None,
        "quality_flags": [],
        "looks_like_screen_recording": _looks_like_screen_recording(filename, capture_source),
    }


def _even_frame_indices(frame_count_window: int, max_frames: int) -> List[int]:
    if frame_count_window <= 0 or max_frames <= 0:
        return []

    sample_count = min(frame_count_window, max_frames)
    if sample_count <= 1:
        return [0]

    last_index = max(0, frame_count_window - 1)
    return sorted({
        int(round(i * last_index / (sample_count - 1)))
        for i in range(sample_count)
    })


def extract_frames(
    video_path: str,
    video_upload_id: str = "unknown-video",
    filename: Optional[str] = None,
    capture_source: Optional[str] = None,
) -> FrameExtractionResult:
    cap = None
    metadata: Dict[str, Any] = {
        "processing_tier": "manual_review_required",
        "quality_flags": ["metadata_unreadable"],
        "manual_review_reason": "metadata_unreadable",
    }

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"[{video_upload_id}] Cannot open downloaded video")
            metadata = classify_video_workload({
                "filename": filename,
                "capture_source": capture_source,
                "quality_flags": ["metadata_unreadable"],
            })
            return FrameExtractionResult(frames=[], metadata=metadata)

        metadata = classify_video_workload(
            _metadata_from_capture(
                cap,
                video_path,
                filename=filename,
                capture_source=capture_source,
            )
        )

        logger.info(
            f"[{video_upload_id}] Video metadata: "
            f"size={metadata.get('file_size_mb')}MB, "
            f"duration={metadata.get('duration_seconds')}s, "
            f"fps={metadata.get('fps')}, "
            f"resolution={metadata.get('source_width')}x{metadata.get('source_height')}, "
            f"tier={metadata.get('processing_tier')}, "
            f"max_frames={metadata.get('max_sampled_frames')}, "
            f"max_width={metadata.get('max_processing_width')}"
        )

        if metadata["processing_tier"] == "manual_review_required":
            logger.warning(
                f"[{video_upload_id}] Manual review selected before frame extraction: "
                f"{metadata.get('manual_review_reason')}"
            )
            return FrameExtractionResult(frames=[], metadata=metadata)

        fps = _safe_float(metadata.get("fps"), 30.0)
        frame_count_total = _safe_int(metadata.get("frame_count_total"), 0)
        processing_window = _safe_float(metadata.get("processing_window_seconds"), 0.0)
        max_frames = _safe_int(metadata.get("max_sampled_frames"), MINIMAL_MAX_FRAMES)
        max_width = _safe_int(metadata.get("max_processing_width"), MINIMAL_MAX_WIDTH)

        frame_count_window = min(
            frame_count_total,
            max(1, int(round(processing_window * fps))) if processing_window and fps else frame_count_total,
        )
        frame_indices = _even_frame_indices(frame_count_window, max_frames)

        frames: List[Tuple[int, Any]] = []
        processed_width = 0
        processed_height = 0
        failed_reads = 0

        for idx in frame_indices:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret or frame is None:
                    failed_reads += 1
                    continue

                original_height, original_width = frame.shape[:2]

                if original_width > max_width:
                    scale = max_width / original_width
                    new_width = max_width
                    new_height = max(1, int(original_height * scale))
                    frame = cv2.resize(
                        frame,
                        (new_width, new_height),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    new_height, new_width = original_height, original_width

                processed_width = processed_width or new_width
                processed_height = processed_height or new_height
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((idx, frame_rgb))

                del frame
            except Exception as frame_error:
                failed_reads += 1
                logger.warning(
                    f"[{video_upload_id}] Frame extraction skipped frame {idx}: {frame_error}"
                )

        flags = set(metadata.get("quality_flags") or [])
        if failed_reads:
            flags.add("partial_frame_read_failures")
        if not frames:
            flags.add("frame_extraction_failed")

        metadata.update({
            "sampled_frame_count": len(frames),
            "requested_frame_count": len(frame_indices),
            "processed_width": processed_width,
            "processed_height": processed_height,
            "failed_frame_reads": failed_reads,
            "quality_flags": sorted(flags),
        })

        if frames:
            logger.info(
                f"[{video_upload_id}] Extracted {len(frames)}/{len(frame_indices)} resized frames "
                f"tier={metadata.get('processing_tier')}, "
                f"processed={processed_width}x{processed_height}, "
                f"window={metadata.get('processing_window_seconds')}s"
            )
        else:
            logger.warning(f"[{video_upload_id}] No frames extracted; manual review required")

        return FrameExtractionResult(frames=frames, metadata=metadata)

    except Exception as error:
        logger.exception(f"[{video_upload_id}] Controlled frame extraction failure: {error}")
        flags = set(metadata.get("quality_flags") or [])
        flags.add("frame_extraction_failed")
        metadata.update({
            "processing_tier": "manual_review_required",
            "manual_review_reason": "frame_extraction_failed",
            "quality_flags": sorted(flags),
            "sampled_frame_count": 0,
        })
        return FrameExtractionResult(frames=[], metadata=metadata)

    finally:
        if cap is not None:
            cap.release()


def cleanup_temp_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info("Temporary video deleted")
    except Exception as e:
        logger.warning(f"Could not remove temp file: {e}")
