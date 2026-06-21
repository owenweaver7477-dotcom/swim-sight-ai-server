"""
app/comparison.py - phase-aligned before/after (side-by-side) swim comparison.

Flag-gated behind ENABLE_COMPARISON (default OFF). Detector-agnostic and
NumPy-only. Consumes two clips' pose_results (the worker contract:
``{frame_idx, pose_detected, landmarks:{name:{x,y,visibility}}}``), aligns them
by stroke phase using ``app.stroke_cycles``, and returns translation/scale-
invariant per-phase and per-joint deltas plus aligned frame pairs for
synchronised playback.

This is AI-assisted draft context for coach review, NOT a validated measurement.
With the flag OFF the worker never calls this and nothing here runs; there is no
change to existing behaviour.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.stroke_cycles import analyze_stroke_cycles

try:  # enrichment only; comparison never depends on this importing
    from app.temporal_metrics import frame_relative_metrics
except Exception:  # pragma: no cover
    frame_relative_metrics = None  # type: ignore

_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_SAMPLES_PER_PHASE = 10
# Joints required to build a translation/scale-invariant frame.
_REQUIRED_FOR_NORM = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")


def comparison_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the comparison feature is enabled (default false)."""

    source = os.environ if env is None else env
    return str(source.get("ENABLE_COMPARISON", "false")).strip().lower() in _TRUE_VALUES


def _midpoint(a: Mapping[str, Any], b: Mapping[str, Any]) -> Tuple[float, float]:
    return ((float(a["x"]) + float(b["x"])) / 2.0, (float(a["y"]) + float(b["y"])) / 2.0)


def _normalized_pose(frame: Mapping[str, Any]) -> Optional[Dict[str, Tuple[float, float]]]:
    """Hip-centred, torso-scaled pose so comparison ignores where in the pool the
    swimmer is and how large they appear. Returns None if the anchor joints are
    missing or degenerate."""

    landmarks = frame.get("landmarks") or {}
    if not all(name in landmarks for name in _REQUIRED_FOR_NORM):
        return None
    hip_x, hip_y = _midpoint(landmarks["left_hip"], landmarks["right_hip"])
    sho_x, sho_y = _midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
    torso = float(np.hypot(sho_x - hip_x, sho_y - hip_y))
    if not np.isfinite(torso) or torso < 1e-6:
        return None
    out: Dict[str, Tuple[float, float]] = {}
    for name, point in landmarks.items():
        try:
            out[name] = ((float(point["x"]) - hip_x) / torso, (float(point["y"]) - hip_y) / torso)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _best_cycle(analysis: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    cycles = analysis.get("cycles") or []
    if not cycles:
        return None
    return max(cycles, key=lambda cycle: float(cycle.get("confidence", 0.0)))


def _detected_frames_by_index(pose_results: Sequence[Mapping[str, Any]]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for frame in pose_results:
        if frame.get("pose_detected"):
            out[int(frame.get("frame_idx", len(out)))] = frame
    return out


def _resample_phase(
    frames_by_idx: Mapping[int, Mapping[str, Any]], start: int, end: int, samples: int
) -> Dict[str, np.ndarray]:
    """Return {joint: (samples, 2)} normalised positions over a 0..1 phase timeline."""

    span = max(1, int(end) - int(start))
    times: List[float] = []
    poses: List[Dict[str, Tuple[float, float]]] = []
    for idx in range(int(start), int(end) + 1):
        frame = frames_by_idx.get(idx)
        if frame is None:
            continue
        norm = _normalized_pose(frame)
        if norm is None:
            continue
        times.append((idx - start) / span)
        poses.append(norm)
    if len(poses) < 2:
        return {}
    times_arr = np.asarray(times, dtype=float)
    common = set(poses[0])
    for pose in poses[1:]:
        common &= set(pose)
    grid = np.linspace(0.0, 1.0, samples)
    out: Dict[str, np.ndarray] = {}
    for joint in common:
        xs = np.asarray([pose[joint][0] for pose in poses], dtype=float)
        ys = np.asarray([pose[joint][1] for pose in poses], dtype=float)
        out[joint] = np.stack([np.interp(grid, times_arr, xs), np.interp(grid, times_arr, ys)], axis=1)
    return out


def _phase_frame_pairs(start_a: int, end_a: int, start_b: int, end_b: int, samples: int) -> List[List[int]]:
    grid = np.linspace(0.0, 1.0, samples)
    return [
        [int(round(start_a + t * (end_a - start_a))), int(round(start_b + t * (end_b - start_b)))]
        for t in grid
    ]


def _flatten_numeric(value: Any, prefix: str = "") -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if isinstance(value, Mapping):
        for key, sub in value.items():
            out.extend(_flatten_numeric(sub, f"{prefix}{key}."))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        out.append((prefix.rstrip("."), float(value)))
    return out


def _phase_metric_deltas(frames_a, phase_a, frames_b, phase_b, fps_a, fps_b) -> Dict[str, float]:
    """Best-effort scalar metric deltas (clip B minus clip A) per phase. Never raises."""

    if frame_relative_metrics is None:
        return {}

    def phase_means(frames_by_idx, phase, fps) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for idx in range(int(phase["start_frame"]), int(phase["end_frame"]) + 1):
            frame = frames_by_idx.get(idx)
            if frame is None:
                continue
            try:
                metrics = frame_relative_metrics(frame, fps) or {}
            except Exception:
                continue
            for key, val in _flatten_numeric(metrics):
                acc.setdefault(key, []).append(val)
        return {key: float(np.mean(vals)) for key, vals in acc.items() if vals}

    try:
        means_a = phase_means(frames_a, phase_a, fps_a)
        means_b = phase_means(frames_b, phase_b, fps_b)
    except Exception:
        return {}
    return {key: round(means_b[key] - means_a[key], 4) for key in (set(means_a) & set(means_b))}


def _insufficient(stroke: str, anchor: str, reason: str) -> Dict[str, Any]:
    return {
        "comparison_enabled": True,
        "stroke_type": stroke,
        "anchor": anchor,
        "status": "insufficient",
        "reason": reason,
        "labels": "AI-assisted draft / estimate (not a measurement)",
        "samples_per_phase": 0,
        "phases": [],
        "aligned_frames": [],
        "per_phase": [],
        "per_joint_delta_series": {},
        "summary": {"overall_mean_delta": None, "largest_change_phase": None, "confidence": 0.0},
    }


def compare_clips(
    pose_results_a: Sequence[Mapping[str, Any]],
    pose_results_b: Sequence[Mapping[str, Any]],
    fps_a: float,
    fps_b: float,
    stroke_type: str,
    *,
    anchor: str = "before_after",
    samples_per_phase: int = _DEFAULT_SAMPLES_PER_PHASE,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase-align two clips and return per-phase / per-joint deltas + frame pairs.

    ``anchor`` is a label only ("before_after" for the same swimmer over time, or
    "reference" for swimmer-vs-reference); the maths is identical. Output is
    draft/estimate context, never a measurement.
    """

    stroke = (stroke_type or "").strip().lower()
    analysis_a = analyze_stroke_cycles(pose_results_a, fps_a, stroke_type, config)
    analysis_b = analyze_stroke_cycles(pose_results_b, fps_b, stroke_type, config)
    cycle_a = _best_cycle(analysis_a)
    cycle_b = _best_cycle(analysis_b)
    if cycle_a is None or cycle_b is None:
        return _insufficient(stroke, anchor, "No reliable stroke cycle in one or both clips.")

    frames_a = _detected_frames_by_index(pose_results_a)
    frames_b = _detected_frames_by_index(pose_results_b)
    phases_a = {phase["phase"]: phase for phase in cycle_a.get("phases", [])}
    phases_b = {phase["phase"]: phase for phase in cycle_b.get("phases", [])}
    order = [phase["phase"] for phase in cycle_a.get("phases", []) if phase["phase"] in phases_b]
    if not order:
        return _insufficient(stroke, anchor, "Clips share no common stroke phases.")

    samples = max(2, int(samples_per_phase))
    aligned: List[List[int]] = []
    per_phase: List[Dict[str, Any]] = []
    series: Dict[str, List[float]] = {}

    for phase in order:
        pa, pb = phases_a[phase], phases_b[phase]
        resampled_a = _resample_phase(frames_a, pa["start_frame"], pa["end_frame"], samples)
        resampled_b = _resample_phase(frames_b, pb["start_frame"], pb["end_frame"], samples)
        common = sorted(set(resampled_a) & set(resampled_b))
        joint_deltas: Dict[str, float] = {}
        phase_samples: List[np.ndarray] = []
        for joint in common:
            distances = np.linalg.norm(resampled_a[joint] - resampled_b[joint], axis=1)
            joint_deltas[joint] = round(float(np.mean(distances)), 4)
            series.setdefault(joint, []).extend(round(float(d), 4) for d in distances)
            phase_samples.append(distances)
        mean_delta = round(float(np.mean(phase_samples)), 4) if phase_samples else None
        per_phase.append({
            "phase": phase,
            "mean_delta": mean_delta,
            "joint_deltas": joint_deltas,
            "metric_deltas": _phase_metric_deltas(frames_a, pa, frames_b, pb, fps_a, fps_b),
            "common_joint_count": len(common),
        })
        aligned.extend(_phase_frame_pairs(pa["start_frame"], pa["end_frame"], pb["start_frame"], pb["end_frame"], samples))

    phase_means = [(item["phase"], item["mean_delta"]) for item in per_phase if item["mean_delta"] is not None]
    overall = round(float(np.mean([value for _, value in phase_means])), 4) if phase_means else None
    largest = max(phase_means, key=lambda item: item[1])[0] if phase_means else None
    confidence = round(min(float(cycle_a.get("confidence", 0.0)), float(cycle_b.get("confidence", 0.0))), 3)

    return {
        "comparison_enabled": True,
        "stroke_type": stroke,
        "anchor": anchor,
        "status": "completed",
        "reason": None,
        "labels": "AI-assisted draft / estimate (not a measurement)",
        "samples_per_phase": samples,
        "phases": order,
        "aligned_frames": aligned,
        "per_phase": per_phase,
        "per_joint_delta_series": series,
        "summary": {
            "overall_mean_delta": overall,
            "largest_change_phase": largest,
            "confidence": confidence,
        },
    }
