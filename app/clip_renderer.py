"""
app/clip_renderer.py - render a short, shareable analysis clip.

Flag-gated behind ENABLE_SHARE_CLIP (default OFF). Two layers:

  * build_render_plan(...)  - PURE, fully unit-tested. Decides what may appear in
    a shareable clip: ONLY coach-approved AND public items, capped, with drag/
    resistance blocks excluded and every field except {frame_idx, text} dropped
    so no PII / internal notes can leak.
  * render_clip(...)        - draws the overlay + approved callouts + title/footer
    onto frames and encodes an MP4. Guarded: if opencv/ffmpeg is unavailable it
    raises ClipRenderUnavailable instead of crashing the worker.

With the flag OFF nothing here runs and behaviour is unchanged. Output is an
AI-assisted, coach-approved draft, never a measurement.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_TRUE_VALUES = {"1", "true", "yes", "on"}
_APPROVED_STATUSES = {"approved", "coach_approved", "coach-approved", "finalised", "finalized"}
_FOOTER = "Made with SwimSight - AI-assisted, coach-approved"
# Only these keys may ever reach the rendered clip. Everything else (swimmer name,
# internal notes, drag numbers, ids, emails, ...) is dropped by construction.
_CALLOUT_ALLOWED_KEYS = ("frame_idx", "text")
_TEXT_FIELDS = ("coach_text", "approved_text", "cue", "cue_text", "label", "text")
_FRAME_FIELDS = ("key_frame", "frame_idx", "frame", "frame_index")
_DRAG_HINTS = ("drag", "resistance", "hydrodynamic")


class ClipRenderUnavailable(RuntimeError):
    """Raised when opencv/ffmpeg isn't available to encode the clip."""


def share_clip_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the shareable-clip feature is enabled (default false)."""

    source = os.environ if env is None else env
    return str(source.get("ENABLE_SHARE_CLIP", "false")).strip().lower() in _TRUE_VALUES


def _is_approved(item: Mapping[str, Any]) -> bool:
    if item.get("coach_approved") is True or item.get("approved") is True:
        return True
    return str(item.get("status", "")).strip().lower() in _APPROVED_STATUSES


def _is_public(item: Mapping[str, Any]) -> bool:
    if item.get("is_public") is True or item.get("public") is True:
        return True
    return str(item.get("visibility", "")).strip().lower() == "public"


def _is_drag(item: Mapping[str, Any]) -> bool:
    if item.get("is_drag") is True or item.get("is_resistance") is True:
        return True
    haystack = " ".join(
        str(item.get(key, "")) for key in ("type", "category", "kind", "metric")
    ).lower()
    return any(hint in haystack for hint in _DRAG_HINTS)


def _callout_text(item: Mapping[str, Any]) -> Optional[str]:
    for field in _TEXT_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _frame_index(item: Mapping[str, Any]) -> Optional[int]:
    for field in _FRAME_FIELDS:
        value = item.get(field)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def build_render_plan(
    items: Sequence[Mapping[str, Any]],
    *,
    max_callouts: int = 3,
    branding: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide what may appear in a shareable clip. Approved + public only; drag
    blocks and any item lacking a usable frame/text are excluded; output callouts
    carry ONLY {frame_idx, text} so no PII or internal notes can leak."""

    excluded = {"unapproved": 0, "private": 0, "drag": 0, "no_frame_or_text": 0, "over_cap": 0}
    eligible: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        if not _is_approved(item):
            excluded["unapproved"] += 1
            continue
        if not _is_public(item):
            excluded["private"] += 1
            continue
        if _is_drag(item):
            excluded["drag"] += 1
            continue
        text = _callout_text(item)
        frame_idx = _frame_index(item)
        if text is None or frame_idx is None:
            excluded["no_frame_or_text"] += 1
            continue
        eligible.append({"frame_idx": frame_idx, "text": text})

    eligible.sort(key=lambda callout: callout["frame_idx"])
    if len(eligible) > max_callouts:
        excluded["over_cap"] = len(eligible) - max_callouts
        eligible = eligible[:max_callouts]

    # Hard guarantee: nothing but the allowed keys survives.
    callouts = [{key: callout[key] for key in _CALLOUT_ALLOWED_KEYS} for callout in eligible]

    safe_branding: Dict[str, Any] = {}
    if branding:
        club = branding.get("club_name")
        if isinstance(club, str) and club.strip():
            safe_branding["club_name"] = club.strip()
        logo = branding.get("logo_path")
        if isinstance(logo, str) and logo.strip():
            safe_branding["logo_path"] = logo.strip()
        link = branding.get("link")
        if isinstance(link, str) and link.strip():
            safe_branding["link"] = link.strip()

    return {
        "callouts": callouts,
        "included_count": len(callouts),
        "excluded": excluded,
        "footer": _FOOTER,
        "branding": safe_branding,
        "labels": "AI-assisted draft / coach-approved",
    }


# --- rendering (guarded; assembly logic above is what tests rely on) -----------

_SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def _require_cv2():
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        return cv2
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ClipRenderUnavailable("opencv/ffmpeg not available to render clip") from exc


def render_clip(
    frames: Sequence[Tuple[int, "Any"]],
    landmarks_by_frame: Mapping[int, Mapping[str, Mapping[str, float]]],
    plan: Mapping[str, Any],
    output_path: str,
    *,
    fps: float = 30.0,
    max_seconds: float = 20.0,
    callout_window: int = 12,
) -> str:
    """Burn skeleton overlay + approved callouts + footer onto frames and encode an
    MP4 at output_path. `frames` is a list of (frame_idx, BGR image). Raises
    ClipRenderUnavailable if encoding isn't possible."""

    cv2 = _require_cv2()
    import numpy as np

    if not frames:
        raise ValueError("no frames to render")
    max_frames = max(1, int(fps * max_seconds))
    frames = list(frames)[:max_frames]
    height, width = frames[0][1].shape[:2]
    callouts = list(plan.get("callouts", []))
    footer = str(plan.get("footer", _FOOTER))
    branding = plan.get("branding", {}) or {}
    link = branding.get("link", "")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():  # pragma: no cover
        raise ClipRenderUnavailable("could not open video writer (no mp4 encoder)")
    try:
        for frame_idx, image in frames:
            canvas = image.copy()
            lm = landmarks_by_frame.get(frame_idx, {})
            for a, b in _SKELETON_EDGES:
                if a in lm and b in lm:
                    pa = (int(lm[a]["x"]), int(lm[a]["y"]))
                    pb = (int(lm[b]["x"]), int(lm[b]["y"]))
                    cv2.line(canvas, pa, pb, (0, 230, 0), 2)
            for name, point in lm.items():
                cv2.circle(canvas, (int(point["x"]), int(point["y"])), 4, (0, 0, 230), -1)
            for callout in callouts:
                if abs(int(callout["frame_idx"]) - int(frame_idx)) <= callout_window:
                    cv2.putText(canvas, str(callout["text"])[:80], (24, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            footer_text = footer + ((" - " + link) if link else "")
            cv2.putText(canvas, footer_text, (16, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    return output_path
