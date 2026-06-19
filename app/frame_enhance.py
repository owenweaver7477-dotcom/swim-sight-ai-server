"""
app/frame_enhance.py - optional CLAHE contrast enhancement for pose input.

Underwater / low-contrast / backlit pool footage detects poorly. CLAHE (Contrast
Limited Adaptive Histogram Equalization) on the luminance channel often lifts
keypoint detection on exactly that kind of footage (named as a wanted upgrade in
BASELINE_EVALUATION.md).

OFF unless ENABLE_CLAHE is truthy. cv2 is imported lazily and every call is
wrapped, so this never hard-fails or blocks pose: on any error the original
frame is returned unchanged.

Tunables (env):
  CLAHE_CLIP_LIMIT (default 2.0)   - higher = stronger local contrast
  CLAHE_TILE       (default 8)     - grid size for adaptive equalisation
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_clahe = None


def clahe_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    src = os.environ if env is None else env
    return str(src.get("ENABLE_CLAHE", "false")).strip().lower() in _TRUTHY


def _get_clahe():
    global _clahe
    if _clahe is None:
        import cv2
        try:
            clip = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
        except ValueError:
            clip = 2.0
        try:
            tile = int(os.getenv("CLAHE_TILE", "8"))
        except ValueError:
            tile = 8
        tile = max(2, tile)
        _clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return _clahe


def enhance_frame_rgb(frame_rgb):
    """
    Apply CLAHE to the L channel of an RGB frame and return an RGB frame.
    Returns the input unchanged on any failure (never blocks pose detection).
    """
    try:
        import cv2
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = _get_clahe().apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    except Exception as e:  # pragma: no cover - depends on cv2 at runtime
        logger.warning(f"CLAHE enhancement skipped: {e}")
        return frame_rgb
