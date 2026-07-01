"""Approximate stroke-cycle and phase analysis from normalized 2D pose tracks.

This module is deliberately detector-agnostic and NumPy-only. Its output is
internal phase-aware context for coach-review drafts, not a validated
biomechanics measurement. Weak or sparse evidence produces empty/low-confidence
results rather than guessed cycles.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


_TRUE_VALUES = {"1", "true", "yes", "on"}
_SUPPORTED_PHASES = {
    "breaststroke": ("extension", "pull", "recovery", "kick"),
    "freestyle": ("entry_catch", "pull", "push", "recovery"),
}


def phase_analysis_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether optional phase analysis is enabled (default false)."""

    source = os.environ if env is None else env
    return str(source.get("PHASE_ANALYSIS", "false")).strip().lower() in _TRUE_VALUES


def _normalize_stroke(stroke_type: str) -> str:
    value = (stroke_type or "").strip().lower().replace("_", " ")
    aliases = {
        "breast": "breaststroke",
        "breaststroke": "breaststroke",
        "free": "freestyle",
        "freestyle": "freestyle",
        "front crawl": "freestyle",
    }
    return aliases.get(value, value or "unknown")


def _midpoint(first: Mapping[str, Any], second: Mapping[str, Any]) -> Tuple[float, float]:
    return (
        (float(first["x"]) + float(second["x"])) / 2.0,
        (float(first["y"]) + float(second["y"])) / 2.0,
    )


def _visibility(point: Mapping[str, Any]) -> float:
    return float(np.clip(point.get("visibility", 1.0), 0.0, 1.0))


def _frame_signal(frame: Mapping[str, Any], stroke: str) -> Optional[Tuple[float, float, bool]]:
    landmarks = frame.get("landmarks") or {}
    required = ("left_hip", "right_hip", "left_wrist", "right_wrist")
    if not all(name in landmarks for name in required):
        return None

    hip_x, hip_y = _midpoint(landmarks["left_hip"], landmarks["right_hip"])
    left_wrist = landmarks["left_wrist"]
    right_wrist = landmarks["right_wrist"]
    points = [landmarks[name] for name in required]
    confidence = float(np.mean([_visibility(point) for point in points]))
    interpolated = bool(frame.get("interpolated")) or any(
        bool(point.get("interpolated")) for point in points
    )

    if stroke == "breaststroke":
        wrist_x, wrist_y = _midpoint(left_wrist, right_wrist)
        # Translation-invariant hand reach. A light vertical term helps when a
        # side-view hand track has little horizontal travel.
        signal = (wrist_x - hip_x) + 0.25 * (hip_y - wrist_y)
    else:
        # Alternating-arm separation is periodic and remains translation-safe.
        signal = (float(left_wrist["x"]) - hip_x) - (
            float(right_wrist["x"]) - hip_x
        )

    if not np.isfinite(signal):
        return None
    return float(signal), confidence, interpolated


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values.copy()
    width = min(int(window), int(values.size))
    if width % 2 == 0:
        width = max(1, width - 1)
    if width <= 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def _select_peaks(
    frame_indices: np.ndarray,
    values: np.ndarray,
    fps: float,
    config: Mapping[str, Any],
) -> List[int]:
    if values.size < 3 or float(np.std(values)) < float(config["min_signal_std"]):
        return []

    threshold = float(np.median(values) + float(config["peak_threshold_std"]) * np.std(values))
    candidates = [
        idx
        for idx in range(1, len(values) - 1)
        if values[idx] >= values[idx - 1]
        and values[idx] > values[idx + 1]
        and values[idx] >= threshold
    ]
    min_gap = max(1, int(round(float(config["min_cycle_seconds"]) * fps)))
    selected: List[int] = []
    for candidate in candidates:
        if not selected:
            selected.append(candidate)
            continue
        frame_gap = int(frame_indices[candidate] - frame_indices[selected[-1]])
        if frame_gap >= min_gap:
            selected.append(candidate)
        elif values[candidate] > values[selected[-1]]:
            selected[-1] = candidate
    return selected


def _phase_ranges(start: int, end: int, phases: Sequence[str], fps: float) -> List[Dict[str, Any]]:
    span = max(1, end - start)
    boundaries = [int(round(start + span * index / len(phases))) for index in range(len(phases) + 1)]
    boundaries[0] = start
    boundaries[-1] = end
    return [
        {
            "phase": phase,
            "start_frame": boundaries[index],
            "end_frame": boundaries[index + 1],
            "duration_seconds": round((boundaries[index + 1] - boundaries[index]) / fps, 3),
        }
        for index, phase in enumerate(phases)
    ]


def _empty_result(
    stroke: str,
    fps: float,
    status: str,
    reason: str,
    quality_flags: Sequence[str],
    confidence: float = 0.0,
) -> Dict[str, Any]:
    return {
        "phase_analysis_enabled": True,
        "stroke_type": stroke,
        "fps": round(float(fps), 3),
        "supported": stroke in _SUPPORTED_PHASES,
        "status": status,
        "reason": reason,
        "cycles": [],
        "summary": {
            "cycle_count": 0,
            "mean_cycle_duration_seconds": None,
            "cycle_regularity": 0.0,
            "confidence": round(float(confidence), 3),
        },
        "quality_flags": list(quality_flags),
    }


def analyze_stroke_cycles(
    pose_results: Sequence[Mapping[str, Any]],
    fps: float,
    stroke_type: str,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect approximate cycles and quarter-cycle phases.

    The phase boundaries are deliberately approximate in this first internal
    implementation. Confidence combines pose coverage, visibility, periodic
    signal strength, and cycle regularity.
    """

    stroke = _normalize_stroke(stroke_type)
    safe_fps = float(fps) if fps and float(fps) > 0 else 30.0
    if stroke not in _SUPPORTED_PHASES:
        return _empty_result(
            stroke,
            safe_fps,
            "unsupported",
            "Unsupported stroke for phase analysis.",
            ["phase_analysis_unsupported_stroke"],
        )

    settings: Dict[str, Any] = {
        "min_samples": 12,
        "min_cycle_seconds": 0.35,
        "max_cycle_seconds": 3.0,
        "min_signal_std": 0.008,
        "peak_threshold_std": 0.15,
        "smoothing_window": max(3, int(round(safe_fps * 0.08))),
    }
    if config:
        settings.update(dict(config))

    detected = [frame for frame in pose_results if frame.get("pose_detected")]
    signal_rows: List[Tuple[int, float, float, bool]] = []
    for frame in detected:
        sample = _frame_signal(frame, stroke)
        if sample is None:
            continue
        signal, visibility, interpolated = sample
        signal_rows.append((int(frame.get("frame_idx", len(signal_rows))), signal, visibility, interpolated))

    sample_count = len(signal_rows)
    coverage = sample_count / max(len(detected), 1)
    if sample_count < int(settings["min_samples"]):
        return _empty_result(
            stroke,
            safe_fps,
            "insufficient_pose",
            "Not enough usable hand and hip landmarks for cycle analysis.",
            ["insufficient_phase_samples"],
            confidence=coverage * 0.25,
        )

    signal_rows.sort(key=lambda row: row[0])
    frame_indices = np.asarray([row[0] for row in signal_rows], dtype=int)
    raw_signal = np.asarray([row[1] for row in signal_rows], dtype=float)
    visibilities = np.asarray([row[2] for row in signal_rows], dtype=float)
    interpolated = np.asarray([row[3] for row in signal_rows], dtype=bool)
    smoothed = _smooth(raw_signal, int(settings["smoothing_window"]))
    peaks = _select_peaks(frame_indices, smoothed, safe_fps, settings)

    quality_flags: List[str] = []
    if coverage < 0.75:
        quality_flags.append("sparse_phase_landmarks")
    interpolated_ratio = float(np.mean(interpolated)) if interpolated.size else 0.0
    if interpolated_ratio > 0.15:
        quality_flags.append("interpolated_phase_landmarks")

    cycles: List[Dict[str, Any]] = []
    durations: List[float] = []
    for start_peak, end_peak in zip(peaks, peaks[1:]):
        start_frame = int(frame_indices[start_peak])
        end_frame = int(frame_indices[end_peak])
        duration = (end_frame - start_frame) / safe_fps
        if not float(settings["min_cycle_seconds"]) <= duration <= float(settings["max_cycle_seconds"]):
            continue

        mask = (frame_indices >= start_frame) & (frame_indices <= end_frame)
        local_visibility = float(np.mean(visibilities[mask])) if np.any(mask) else 0.0
        local_interpolation = float(np.mean(interpolated[mask])) if np.any(mask) else 0.0
        cycle_confidence = np.clip(local_visibility * coverage * (1.0 - 0.45 * local_interpolation), 0.0, 1.0)
        cycles.append({
            "cycle_idx": len(cycles),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_seconds": round(duration, 3),
            "confidence": round(float(cycle_confidence), 3),
            "phases": _phase_ranges(start_frame, end_frame, _SUPPORTED_PHASES[stroke], safe_fps),
        })
        durations.append(duration)

    if not cycles:
        quality_flags.append("no_reliable_cycles")
        signal_strength = float(np.clip(np.std(smoothed) / 0.08, 0.0, 1.0))
        return _empty_result(
            stroke,
            safe_fps,
            "insufficient_periodicity",
            "The usable pose track did not contain repeated cycles with enough confidence.",
            quality_flags,
            confidence=coverage * float(np.mean(visibilities)) * signal_strength * 0.35,
        )

    mean_duration = float(np.mean(durations))
    regularity = float(np.clip(1.0 - np.std(durations) / max(mean_duration, 1e-6), 0.0, 1.0))
    signal_strength = float(np.clip(np.std(smoothed) / 0.08, 0.0, 1.0))
    count_factor = float(np.clip(len(cycles) / 3.0, 0.35, 1.0))
    confidence = (
        0.30 * coverage
        + 0.25 * float(np.mean(visibilities))
        + 0.20 * regularity
        + 0.15 * signal_strength
        + 0.10 * count_factor
    ) * (1.0 - 0.45 * interpolated_ratio)

    return {
        "phase_analysis_enabled": True,
        "stroke_type": stroke,
        "fps": round(safe_fps, 3),
        "supported": True,
        "status": "completed",
        "reason": None,
        "cycles": cycles,
        "summary": {
            "cycle_count": len(cycles),
            "mean_cycle_duration_seconds": round(mean_duration, 3),
            "cycle_regularity": round(regularity, 3),
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
        },
        "quality_flags": quality_flags,
        "signal": {
            "usable_sample_count": sample_count,
            "detected_pose_count": len(detected),
            "coverage": round(coverage, 3),
            "interpolated_ratio": round(interpolated_ratio, 3),
        },
    }


# Keys that are safe to surface in internal telemetry. Everything else from
# analyze_stroke_cycles (per-frame cycle boundaries, phase frame ranges, raw
# signal diagnostics) is intentionally dropped.
_SUMMARY_SCALAR_KEYS = (
    "cycle_count",
    "mean_cycle_duration_seconds",
    "cycle_regularity",
    "confidence",
)

# Strokes per full detected cycle, by stroke. Only the strokes the cycle
# detector supports are mapped; anything else yields null rate fields.
_STROKES_PER_CYCLE = {"freestyle": 2, "breaststroke": 1}

# Gates for emitting an estimated rate (documented in AI_WORKER_CONTRACT.md).
_MIN_CYCLES_FOR_RATE = 3
_MIN_REGULARITY_FOR_RATE = 0.6
_MIN_CONFIDENCE_FOR_RATE = 0.4


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def estimate_stroke_rate(
    *,
    status: Any,
    stroke_type: Any,
    mean_cycle_duration_seconds: Any,
    cycle_count: Any,
    cycle_regularity: Any,
    confidence: Any,
) -> Dict[str, Any]:
    """Estimate stroke rate from real cycle timing. Pure and side-effect free.

    Uses only the cycle summary, which is derived from real frame timing
    (frame indices / fps). Rate fields are null with a reason flag whenever any
    gate fails. `_flags` is an internal list of reason codes for the caller to
    fold into quality_flags. This is a 2D-heuristic ESTIMATE, never a measured
    or public metric.

      estimated_cycle_rate_per_min = 60 / mean_cycle_duration_seconds
      estimated_stroke_rate_spm    = cycle_rate * strokes_per_cycle
    """
    stroke = _normalize_stroke(str(stroke_type or ""))
    strokes_per_cycle = _STROKES_PER_CYCLE.get(stroke)

    def _null(flags: Sequence[str]) -> Dict[str, Any]:
        return {
            "estimated_cycle_rate_per_min": None,
            "estimated_stroke_rate_spm": None,
            "strokes_per_cycle": strokes_per_cycle,
            "stroke_rate_estimated": True,
            "_flags": list(dict.fromkeys(flags)),
        }

    if status != "completed" or strokes_per_cycle is None:
        return _null(["stroke_rate_unavailable"])

    flags: List[str] = []
    duration = mean_cycle_duration_seconds
    if not (_is_number(duration) and math.isfinite(float(duration)) and float(duration) > 0):
        flags.append("stroke_rate_invalid_cycle_duration")
    if not (_is_number(cycle_count) and int(cycle_count) >= _MIN_CYCLES_FOR_RATE):
        flags.append("stroke_rate_insufficient_cycles")
    if not (_is_number(cycle_regularity) and float(cycle_regularity) >= _MIN_REGULARITY_FOR_RATE):
        flags.append("stroke_rate_low_regularity")
    if not (_is_number(confidence) and float(confidence) >= _MIN_CONFIDENCE_FOR_RATE):
        flags.append("stroke_rate_low_confidence")

    if flags:
        return _null(flags)

    cycle_rate = round(60.0 / float(duration), 1)
    stroke_rate = round(cycle_rate * strokes_per_cycle, 1)
    return {
        "estimated_cycle_rate_per_min": cycle_rate,
        "estimated_stroke_rate_spm": stroke_rate,
        "strokes_per_cycle": strokes_per_cycle,
        "stroke_rate_estimated": True,
        "_flags": [],
    }


def sanitized_cycle_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Whitelist-only summary of an analyze_stroke_cycles result.

    Returns a flat dict safe for internal telemetry. It NEVER contains raw
    landmarks, per-frame cycle boundaries, signed URLs, secrets, or private
    video keys -- only the scalar summary, a status, safe quality flags, and an
    internal 2D-heuristic stroke-rate ESTIMATE. Explicitly marked not public.
    """
    summary = result.get("summary") if isinstance(result, Mapping) else None
    summary = summary if isinstance(summary, Mapping) else {}

    sanitized: Dict[str, Any] = {
        "enabled": True,
        "status": str(result.get("status", "unknown")) if isinstance(result, Mapping) else "unknown",
        "quality_flags": [
            str(flag)
            for flag in (result.get("quality_flags") or [] if isinstance(result, Mapping) else [])
        ],
        "basis": "2d_heuristic",
        "public_safe": False,
    }
    for key in _SUMMARY_SCALAR_KEYS:
        sanitized[key] = summary.get(key)

    rate = estimate_stroke_rate(
        status=sanitized["status"],
        stroke_type=result.get("stroke_type") if isinstance(result, Mapping) else None,
        mean_cycle_duration_seconds=summary.get("mean_cycle_duration_seconds"),
        cycle_count=summary.get("cycle_count"),
        cycle_regularity=summary.get("cycle_regularity"),
        confidence=summary.get("confidence"),
    )
    rate_flags = rate.pop("_flags", [])
    sanitized.update(rate)
    if rate_flags:
        sanitized["quality_flags"] = list(dict.fromkeys([*sanitized["quality_flags"], *rate_flags]))
    return sanitized
