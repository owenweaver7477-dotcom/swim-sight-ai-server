"""Conservative relative-2D metrics and heuristic stroke-phase segments.

These values are normalized image-space cues. They are not calibrated angles in
3D, distances, velocities, or hydrodynamic measurements. They support internal
quality review and coach-draft evidence only.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


Point = Dict[str, float]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _midpoint(a: Point, b: Point) -> Point:
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2}


def joint_angle_degrees(a: Point, vertex: Point, c: Point) -> Optional[float]:
    first = np.array([a["x"] - vertex["x"], a["y"] - vertex["y"]], dtype=float)
    second = np.array([c["x"] - vertex["x"], c["y"] - vertex["y"]], dtype=float)
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < 1e-8:
        return None
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _metric_summary(values: Iterable[float]) -> Optional[Dict[str, float]]:
    clean = np.array([float(value) for value in values if math.isfinite(float(value))])
    if clean.size == 0:
        return None
    return {
        "count": int(clean.size),
        "median": round(float(np.median(clean)), 3),
        "p10": round(float(np.percentile(clean, 10)), 3),
        "p90": round(float(np.percentile(clean, 90)), 3),
        "min": round(float(clean.min()), 3),
        "max": round(float(clean.max()), 3),
    }


def frame_relative_metrics(frame: Dict[str, Any], fps: float) -> Dict[str, Any]:
    landmarks = frame.get("landmarks") or {}
    metrics: Dict[str, Any] = {
        "frame_idx": int(frame.get("frame_idx", 0)),
        "timestamp_seconds": round(int(frame.get("frame_idx", 0)) / max(fps, 1.0), 3),
    }
    required_torso = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
    if not required_torso.issubset(landmarks):
        return metrics

    shoulder_mid = _midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
    hip_mid = _midpoint(landmarks["left_hip"], landmarks["right_hip"])
    shoulder_width = _distance(landmarks["left_shoulder"], landmarks["right_shoulder"])
    hip_width = _distance(landmarks["left_hip"], landmarks["right_hip"])
    torso_length = max(_distance(shoulder_mid, hip_mid), shoulder_width, 0.02)

    metrics["body_line_offset"] = abs(hip_mid["y"] - shoulder_mid["y"]) / torso_length
    metrics["shoulder_tilt"] = abs(
        landmarks["left_shoulder"]["y"] - landmarks["right_shoulder"]["y"]
    ) / max(shoulder_width, 0.02)
    metrics["hip_tilt"] = abs(
        landmarks["left_hip"]["y"] - landmarks["right_hip"]["y"]
    ) / max(hip_width, 0.02)

    if "nose" in landmarks:
        metrics["head_alignment"] = abs(landmarks["nose"]["y"] - shoulder_mid["y"]) / torso_length

    if {"left_knee", "right_knee"}.issubset(landmarks):
        metrics["knee_width_ratio"] = _distance(
            landmarks["left_knee"], landmarks["right_knee"]
        ) / max(hip_width, 0.02)

    if {"left_ankle", "right_ankle"}.issubset(landmarks):
        metrics["ankle_width_ratio"] = _distance(
            landmarks["left_ankle"], landmarks["right_ankle"]
        ) / max(hip_width, 0.02)

    for side in ("left", "right"):
        shoulder = landmarks.get(f"{side}_shoulder")
        elbow = landmarks.get(f"{side}_elbow")
        wrist = landmarks.get(f"{side}_wrist")
        hip = landmarks.get(f"{side}_hip")
        knee = landmarks.get(f"{side}_knee")
        ankle = landmarks.get(f"{side}_ankle")
        if shoulder and elbow and wrist:
            angle = joint_angle_degrees(shoulder, elbow, wrist)
            if angle is not None:
                metrics[f"{side}_elbow_angle"] = angle
            metrics[f"{side}_arm_extension"] = _distance(shoulder, wrist) / torso_length
        if hip and knee and ankle:
            angle = joint_angle_degrees(hip, knee, ankle)
            if angle is not None:
                metrics[f"{side}_knee_angle"] = angle
            metrics[f"{side}_leg_extension"] = _distance(hip, ankle) / torso_length

    return metrics


def _mean_available(metrics: Dict[str, Any], names: Tuple[str, ...]) -> Optional[float]:
    values = [float(metrics[name]) for name in names if name in metrics]
    return sum(values) / len(values) if values else None


def infer_phase(metrics: Dict[str, Any], stroke: str) -> str:
    arm_extension = _mean_available(metrics, ("left_arm_extension", "right_arm_extension"))
    leg_extension = _mean_available(metrics, ("left_leg_extension", "right_leg_extension"))
    elbow_angle = _mean_available(metrics, ("left_elbow_angle", "right_elbow_angle"))
    knee_angle = _mean_available(metrics, ("left_knee_angle", "right_knee_angle"))
    head_alignment = metrics.get("head_alignment")

    if stroke == "Breaststroke":
        if arm_extension and leg_extension and arm_extension >= 1.8 and leg_extension >= 1.8:
            return "streamline"
        if knee_angle is not None and knee_angle < 105:
            return "kick_setup"
        if knee_angle is not None and knee_angle >= 145:
            return "kick_drive"
        if head_alignment is not None and head_alignment > 0.9:
            return "breath"
        if elbow_angle is not None and elbow_angle < 115:
            return "pull"
        if arm_extension and arm_extension >= 1.45:
            return "recovery"
        return "line_reset"

    if stroke == "Freestyle":
        if head_alignment is not None and head_alignment > 0.95:
            return "breathing"
        if arm_extension and arm_extension >= 1.6:
            return "entry_extension"
        if elbow_angle is not None and elbow_angle < 115:
            return "pull"
        if elbow_angle is not None and elbow_angle < 145:
            return "catch_setup"
        return "body_line"

    if stroke == "Backstroke":
        if arm_extension and arm_extension >= 1.6:
            return "entry_extension"
        if elbow_angle is not None and elbow_angle < 115:
            return "pull"
        if elbow_angle is not None and elbow_angle < 145:
            return "catch_setup"
        if metrics.get("shoulder_tilt", 0) > 0.18:
            return "recovery"
        return "body_line"

    if stroke == "Butterfly":
        if head_alignment is not None and head_alignment > 0.95:
            return "breathing"
        if arm_extension and arm_extension >= 1.55:
            return "entry_extension"
        if elbow_angle is not None and elbow_angle < 120:
            return "pull"
        if metrics.get("body_line_offset", 0) > 0.75:
            return "body_wave"
        return "recovery"

    return "unclassified"


def _phase_segments(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for frame in frames:
        phase = frame["phase"]
        timestamp = frame["timestamp_seconds"]
        if not segments or segments[-1]["phase"] != phase:
            segments.append({
                "phase": phase,
                "start_seconds": timestamp,
                "end_seconds": timestamp,
                "sample_count": 1,
            })
        else:
            segments[-1]["end_seconds"] = timestamp
            segments[-1]["sample_count"] += 1
    return segments


def build_temporal_analysis(
    pose_results: List[Dict[str, Any]],
    fps: float,
    stroke: str,
) -> Dict[str, Any]:
    relative_frames = [
        frame_relative_metrics(frame, fps)
        for frame in pose_results
        if frame.get("pose_detected")
    ]
    usable = [frame for frame in relative_frames if len(frame) > 2]
    for frame in usable:
        frame["phase"] = infer_phase(frame, stroke)

    metric_names = sorted({
        name
        for frame in usable
        for name in frame
        if name not in {"frame_idx", "timestamp_seconds", "phase"}
    })
    summaries = {
        name: summary
        for name in metric_names
        if (summary := _metric_summary(frame[name] for frame in usable if name in frame))
    }
    phase_counts = Counter(frame["phase"] for frame in usable)
    quality_flags = []
    if len(usable) < 8:
        quality_flags.append("insufficient_temporal_samples")
    if len(phase_counts) < 2:
        quality_flags.append("limited_phase_variation")

    return {
        "metric_basis": "relative_2d_image_space",
        "sample_count": len(relative_frames),
        "usable_sample_count": len(usable),
        "phase_counts": dict(phase_counts),
        "phase_segments": _phase_segments(usable),
        "relative_metrics": summaries,
        "quality_flags": quality_flags,
    }
