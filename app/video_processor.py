import cv2
import httpx
import tempfile
import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MAX_FRAMES = 120
SAMPLE_INTERVAL_SECONDS = 0.25


async def download_video(video_upload_id: str, signed_url: str) -> Optional[str]:
    try:
        logger.info(f"[{video_upload_id}] Downloading video...")

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            response = await client.get(signed_url)

        if response.status_code != 200:
            logger.error(f"[{video_upload_id}] Download failed: HTTP {response.status_code}")
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(response.content)
            path = tmp.name

        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"[{video_upload_id}] Downloaded video: {size_mb:.1f} MB")
        return path

    except Exception as e:
        logger.error(f"[{video_upload_id}] Download error: {e}")
        return None


def extract_frames(video_path: str) -> Tuple[list, float, float]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return [], 0.0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0.0

    sample_every_n = max(1, int(fps * SAMPLE_INTERVAL_SECONDS))
    frame_indices = list(range(0, total_frames, sample_every_n))[:MAX_FRAMES]

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((idx, frame_rgb))

    cap.release()

    logger.info(
        f"Extracted {len(frames)}/{len(frame_indices)} frames "
        f"(fps={fps:.1f}, duration={total_duration:.1f}s)"
    )

    return frames, fps, total_duration


def cleanup_temp_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info("Temporary video deleted")
    except Exception as e:
        logger.warning(f"Could not remove temp file: {e}")
