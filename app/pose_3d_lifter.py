"""Monocular estimated 3D pose lifting from private 2D pose frames.

This is a first-pass heuristic lifter for internal pipeline readiness. It uses
hip/torso anchoring, relative body-unit scaling, simple anatomical depth
priors, and light temporal smoothing. It is always labelled as an estimate:
source=`monocular_estimate`, measurementType=`estimated`.

It does not calculate biomechanics, force, drag, or public report output.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


POSE_3D_SCHEMA_VERSION = "swim_sight_pose_3d_v1"
POSE_3D_MODEL = "anatomical_heuristic_lift_v1"
METHOD = "anatomical_heuristic_lift"
SOURCE = "monocular_estimate"
MEASUREMENT_TYPE = "estimated"
COORDINATE_SYSTEM = "hip_centered_relative"
SCALE = "relative_body_units"

BODY_JOINTS = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}

UNSAFE_CALLBACK_KEYS = {
    "pose_3d_frames",
    "raw_pose_3d_frames",
    "joints_3d",
    "pose_2d_frames",
    "joints_2d",
    "raw_frames",
    "image_data",
    "frame_bytes",
    "signed_video_url",
    "video_url",
    "file_path",
    "storage_path",
    "local_path",
    "callback_url",
    "force_frames",
    "biomechanics_frames",
    "estimated_drag",
    "drag_force",
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

ASSUMPTIONS = [
    "single-view depth estimated from 2D pose sequence",
    "coordinates are relative body units, not measured metres",
    "z-depth is inferred from anatomical constraints and temporal smoothing",
]

DEPTH_PRIORS = {
    "nose": 0.0,
    "left_eye": -0.02,
    "right_eye": 0.02,
    "left_ear": -0.04,
    "right_ear": 0.04,
    "left_shoulder": -0.06,
    "right_shoulder": 0.06,
    "left_elbow": -0.08,
    "right_elbow": 0.08,
    "left_wrist": -0.1,
    "right_wrist": 0.1,
    "left_hip": -0.04,
    "right_hip": 0.04,
    "left_knee": -0.06,
    "right_knee": 0.06,
    "left_ankle": -0.08,
    "right_ankle": 0.08,
    "left_heel": -0.08,
    "right_heel": 0.08,
    "left_foot_index": -0.1,
    "right_foot_index": 0.1,
    "left_pinky": -0.11,
    "right_pinky": 0.11,
    "left_index": -0.11,
    "right_index": 0.11,
    "left_thumb": -0.11,
    "right_thumb": 0.11,
    "mouth_left": -0.01,
    "mouth_right": 0.01,
}


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(value, digits)


def _joint_ok(joint: Optional[Dict[str, Any]]) -> bool:
    if not joint:
        return False
    if joint.get("status") == "missing":
        return False
    return _finite_float(joint.get("x")) is not None and _finite_float(joint.get("y")) is not None


def _midpoint(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if not (_joint_ok(a) and _joint_ok(b)):
        return None
    return (
        ((_finite_float(a.get("x")) or 0.0) + (_finite_float(b.get("x")) or 0.0)) / 2,
        ((_finite_float(a.get("y")) or 0.0) + (_finite_float(b.get("y")) or 0.0)) / 2,
    )


def _distance(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    if not (_joint_ok(a) and _joint_ok(b)):
        return None
    ax = _finite_float(a.get("x")) or 0.0
    ay = _finite_float(a.get("y")) or 0.0
    bx = _finite_float(b.get("x")) or 0.0
    by = _finite_float(b.get("y")) or 0.0
    return math.hypot(ax - bx, ay - by)


def _anchor_and_scale(joints_2d: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[float, float], float]:
    hips = _midpoint(joints_2d.get("left_hip", {}), joints_2d.get("right_hip", {}))
    shoulders = _midpoint(joints_2d.get("left_shoulder", {}), joints_2d.get("right_shoulder", {}))
    anchor = hips or shoulders or (0.5, 0.5)

    width_candidates = [
        _distance(joints_2d.get("left_shoulder", {}), joints_2d.get("right_shoulder", {})),
        _distance(joints_2d.get("left_hip", {}), joints_2d.get("right_hip", {})),
    ]
    if hips and shoulders:
        width_candidates.append(math.hypot(hips[0] - shoulders[0], hips[1] - shoulders[1]))
    scale = max([value for value in width_candidates if value] or [0.15])
    return anchor, max(scale, 0.08)


def _lift_joint(name: str, joint_2d: Dict[str, Any], anchor: Tuple[float, float], scale: float, frame_confidence: float) -> Dict[str, Any]:
    if not _joint_ok(joint_2d):
        return {
            "x": None,
            "y": None,
            "z": None,
            "confidence": 0.0,
            "source_2d_confidence": _round(_finite_float(joint_2d.get("confidence"), 0.0) or 0.0),
            "status": "missing",
        }

    source_confidence = max(0.0, min(1.0, _finite_float(joint_2d.get("confidence"), 0.0) or 0.0))
    x2 = _finite_float(joint_2d.get("x"), anchor[0]) or anchor[0]
    y2 = _finite_float(joint_2d.get("y"), anchor[1]) or anchor[1]
    rel_x = (x2 - anchor[0]) / scale
    rel_y = (anchor[1] - y2) / scale
    depth_prior = DEPTH_PRIORS.get(name, 0.0)
    z = depth_prior + (rel_x * 0.08)
    confidence = source_confidence * max(0.2, min(1.0, frame_confidence))
    if joint_2d.get("status") == "low_confidence":
        confidence *= 0.75

    return {
        "x": _round(rel_x),
        "y": _round(rel_y),
        "z": _round(z),
        "confidence": _round(confidence),
        "source_2d_confidence": _round(source_confidence),
        "status": "estimated",
    }


def _smooth_estimated_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(frames) < 3:
        return frames
    smoothed = json.loads(json.dumps(frames))
    for index in range(1, len(frames) - 1):
        previous = frames[index - 1].get("joints_3d") or {}
        current = frames[index].get("joints_3d") or {}
        following = frames[index + 1].get("joints_3d") or {}
        target = smoothed[index].get("joints_3d") or {}
        for name, joint in current.items():
            if joint.get("status") != "estimated":
                continue
            neighbors = [
                candidate
                for candidate in (previous.get(name), joint, following.get(name))
                if candidate and candidate.get("status") == "estimated"
            ]
            if len(neighbors) < 2:
                continue
            for axis in ("x", "y", "z"):
                values = [_finite_float(candidate.get(axis)) for candidate in neighbors]
                values = [value for value in values if value is not None]
                if values:
                    target[name][axis] = _round(sum(values) / len(values))
    return smoothed


def lift_pose_2d_frames_to_3d(
    pose_2d_frames: List[Dict[str, Any]],
    input_pose_model: str = "mediapipe_pose",
) -> List[Dict[str, Any]]:
    """Lift private 2D pose frames into estimated relative 3D frames."""
    lifted: List[Dict[str, Any]] = []
    for frame in pose_2d_frames or []:
        joints_2d = frame.get("joints_2d") or {}
        anchor, scale = _anchor_and_scale(joints_2d)
        frame_confidence = _finite_float(frame.get("frame_confidence"), 0.0) or 0.0
        joints_3d = {
            name: _lift_joint(name, joint, anchor, scale, frame_confidence)
            for name, joint in joints_2d.items()
        }
        estimated_count = sum(1 for joint in joints_3d.values() if joint.get("status") == "estimated")
        body_estimated = sum(
            1
            for name in BODY_JOINTS
            if joints_3d.get(name, {}).get("status") == "estimated"
        )
        if estimated_count == 0 or frame.get("tracking_status") in {"failed", "no_person_detected"}:
            tracking_status = frame.get("tracking_status") or "failed"
        elif body_estimated >= 8:
            tracking_status = "estimated"
        else:
            tracking_status = "partial"

        confidence_values = [
            _finite_float(joint.get("confidence"), 0.0) or 0.0
            for joint in joints_3d.values()
            if joint.get("status") == "estimated"
        ]
        lifted.append({
            "timestamp_ms": frame.get("timestamp_ms", 0),
            "source_frame_index": frame.get("source_frame_index", 0),
            "sample_index": frame.get("sample_index", 0),
            "view_type": frame.get("view_type") or "unknown",
            "source": SOURCE,
            "method": METHOD,
            "measurementType": MEASUREMENT_TYPE,
            "pose_model": input_pose_model,
            "pose_3d_model": POSE_3D_MODEL,
            "joints_3d": joints_3d,
            "frame_confidence": _round(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0,
            "tracking_status": tracking_status,
        })
    return _smooth_estimated_frames(lifted)


def build_pose_3d_summary(
    pose_3d_frames: List[Dict[str, Any]],
    input_pose_model: str = "mediapipe_pose",
) -> Dict[str, Any]:
    total = len(pose_3d_frames)
    estimated = sum(1 for frame in pose_3d_frames if frame.get("tracking_status") == "estimated")
    partial = sum(1 for frame in pose_3d_frames if frame.get("tracking_status") == "partial")
    failed = total - estimated - partial
    confidences = [
        _finite_float(frame.get("frame_confidence"), 0.0) or 0.0
        for frame in pose_3d_frames
    ]
    warnings: List[str] = []
    if partial:
        warnings.append("some_frames_have_partial_3d_estimates")
    if failed:
        warnings.append("some_frames_failed_3d_estimation")
    if total and estimated == 0:
        warnings.append("no_complete_3d_estimate_frames")

    return {
        "availabilityState": "pose_3d_estimated",
        "ok": total > 0 and (estimated + partial) > 0,
        "source": SOURCE,
        "method": METHOD,
        "measurementType": MEASUREMENT_TYPE,
        "pose3dModel": POSE_3D_MODEL,
        "inputPoseModel": input_pose_model,
        "inputFrames": total,
        "estimatedFrames": estimated,
        "partialFrames": partial,
        "failedFrames": failed,
        "averageFrameConfidence": _round(sum(confidences) / len(confidences)) if confidences else 0.0,
        "coordinateSystem": COORDINATE_SYSTEM,
        "scale": SCALE,
        "calibration": {
            "cameraCalibrated": False,
            "worldScaleKnown": False,
            "multiView": False,
        },
        "assumptions": ASSUMPTIONS,
        "warnings": warnings,
    }


def write_pose_3d_artifact(
    pose_3d_frames: List[Dict[str, Any]],
    job_id: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    artifact_id = f"{job_id or uuid.uuid4()}-pose-3d"
    root = Path(output_dir or os.getenv("POSE_ARTIFACT_DIR", "pose_artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{artifact_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": POSE_3D_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "source": SOURCE,
            "measurementType": MEASUREMENT_TYPE,
            "frames": pose_3d_frames,
        }, handle, ensure_ascii=True)
    return {
        "artifact_type": "pose_3d_timeseries",
        "artifact_id": artifact_id,
        "storage_visibility": "private",
        "format": "json",
        "frame_count": len(pose_3d_frames),
        "contains_raw_pose": True,
        "contains_video_pixels": False,
        "public_safe": False,
        "source": SOURCE,
        "measurementType": MEASUREMENT_TYPE,
    }


def _request_value(request: Any, field: str) -> Any:
    if isinstance(request, dict):
        return request.get(field)
    return getattr(request, field, None)


def build_pose_3d_callback_payload(
    request: Any,
    job_id: str,
    pose_summary: Dict[str, Any],
    pose_artifact: Dict[str, Any],
    engine: str = "pose-mvp-0.5",
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "app_job_id": _request_value(request, "app_job_id") or job_id,
        "server_job_id": job_id,
        "video_upload_id": _request_value(request, "video_upload_id"),
        "engine": engine,
        "status": "pose_3d_estimated",
        "stage": "pose_3d_estimated",
        "progress_percent": 66,
        "pose_3d_summary": pose_summary,
        "pose_3d_artifact": pose_artifact,
        "warnings": pose_summary.get("warnings") or [],
    }


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


def pose_3d_callback_payload_is_safe(payload: Dict[str, Any]) -> bool:
    for key, value in iter_payload_keys_and_values(payload):
        if key and key.lower() in UNSAFE_CALLBACK_KEYS:
            return False
        if isinstance(value, str):
            lowered = value.lower()
            if any(marker in lowered for marker in UNSAFE_VALUE_MARKERS):
                return False
    return True
