import cv2
import httpx
import tempfile
import os
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

MAX_FRAMES = 20
SAMPLE_INTERVAL_SECONDS = 0.5
MAX_FRAME_WIDTH = 640
MAX_DOWNLOAD_MB = 250


async def download_video(video_upload_id: str, signed_url: str) -> Optional[str]:
    """
    Download private signed video URL to a temporary file.

    Does not log the signed URL.
    """
    try:
        logger.info(f"[{video_upload_id}] Downloading video from signed URL...")

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
                        return None

                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        if not chunk:
                            continue

                        downloaded += len(chunk)

                        if downloaded > max_bytes:
                            logger.error(
                                f"[{video_upload_id}] Download stopped: file exceeds "
                                f"{MAX_DOWNLOAD_MB} MB safety limit"
                            )
                            try:
                                os.remove(path)
                            except Exception:
                                pass
                            return None

                        tmp.write(chunk)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"[{video_upload_id}] Downloaded video: {size_mb:.1f} MB")
        return path

    except Exception as e:
        logger.exception(f"[{video_upload_id}] Download error: {e}")
        return None


def extract_frames(video_path: str) -> Tuple[List[tuple], float, float]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return [], 0.0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0

    if total_frames <= 0:
        logger.error(f"Cannot read frame count from video: {video_path}")
        cap.release()
        return [], fps, total_duration

    sample_every_n = max(1, int(fps * SAMPLE_INTERVAL_SECONDS))
    frame_indices = list(range(0, total_frames, sample_every_n))[:MAX_FRAMES]

    frames = []
    logged_resize = False

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        original_height, original_width = frame.shape[:2]

        if original_width > MAX_FRAME_WIDTH:
            scale = MAX_FRAME_WIDTH / original_width
            new_width = MAX_FRAME_WIDTH
            new_height = int(original_height * scale)
            frame = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )
        else:
            new_height, new_width = original_height, original_width

        if not logged_resize:
            logger.info(
                f"Frame resize: original={original_width}x{original_height}, "
                f"processed={new_width}x{new_height}, max_width={MAX_FRAME_WIDTH}"
            )
            logged_resize = True

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((idx, frame_rgb))

    cap.release()

    logger.info(
        f"Extracted {len(frames)}/{len(frame_indices)} frames "
        f"(fps={fps:.1f}, duration={total_duration:.1f}s, "
        f"sample_interval={SAMPLE_INTERVAL_SECONDS}s, max_frames={MAX_FRAMES})"
    )

    return frames, fps, total_duration


def cleanup_temp_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info("Temporary video deleted")
    except Exception as e:
        logger.warning(f"Could not remove temp file: {e}")
