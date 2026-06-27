"""2D pose contract normalisation for Swim Sight AI progress callbacks.

The detector itself remains the existing MediaPipe/POSE_BACKEND pipeline. This
module turns real pose results into a stable 2D schema, writes raw timeseries to
a private local artifact when requested, and builds a public-safe summary
callback. It never emits frame pixels, signed URLs, 3D data, biomechanics, force,
or drag values.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


POSE_2D_SCHEMA_VERSION = "swim_sight_pose_2d_v1"
POSE_MODEL_NAME = "mediapipe_pose"
POSE_MODEL_VERSION = "blazepose_33"
TRACKED_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.40

REQUIRED_JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

OPTIONAL_JOINT_NAMES = (
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "mouth_left",
    "mouth_right",
)

JOINT_NAMES = (*REQUIRED_JOINT_NAMES, *OPTIONAL_JOINT_NAMES)

BODY_TRACKING_JOINTS = {
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
    "pose_2d_frames",
    "raw_pose_frames",
    "pose_results",
    "landmarks",
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
    "joints_3d",
    "pose_3d",
    "force_frames",
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


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def _clamp01(value: Any) -> float:
    parsed = _finite_float(value, 0.0) or 0.0
    return max(0.0, min(1.0, parsed))


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _frame_sampling_lookup(frame_sampling: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    if not frame_sampling:
        return lookup
    for sample in frame_sampling.get("samples") or []:
        source_index = sample.get("sourceFrameIndex")
        if source_index is None:
            continue
        try:
            lookup[int(source_index)] = sample
        except (TypeError, ValueError):
            continue
    return lookup


def joint_status(confidence: Any, present: bool = True) -> str:
    if not present:
        return "missing"
    score = _clamp01(confidence)
    if score >= TRACKED_CONFIDENCE_THRESHOLD:
        return "tracked"
    if score >= LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return "missing"


def normalise_joint(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not value:
        return {
            "x": None,
            "y": None,
            "confidence": 0.0,
            "visibility": None,
            "status": "missing",
        }

    confidence = _clamp01(value.get("visibility", value.get("confidence")))
    status = joint_status(confidence, present=True)
    x = _finite_float(value.get("x")) if status != "missing" else None
    y = _finite_float(value.get("y")) if status != "missing" else None
    if x is not None:
        x = max(0.0, min(1.0, x))
    if y is not None:
        y = max(0.0, min(1.0, y))
    return {
        "x": _round(x),
        "y": _round(y),
        "confidence": _round(confidence),
        "visibility": _round(confidence),
        "status": status,
    }


def _tracking_status(joints: Dict[str, Dict[str, Any]], pose_result: Dict[str, Any]) -> str:
    if pose_result.get("error"):
        return "failed"
    if not pose_result.get("landmarks"):
        return "no_person_detected"

    body_tracked = sum(
        1
        for name in BODY_TRACKING_JOINTS
        if joints.get(name, {}).get("status") == "tracked"
    )
    body_visible = sum(
        1
        for name in BODY_TRACKING_JOINTS
        if joints.get(name, {}).get("status") in {"tracked", "low_confidence"}
    )
    if body_visible < 4:
        return "failed"
    if body_tracked >= 8:
        return "tracked"
    return "partial"


def _frame_confidence(joints: Dict[str, Dict[str, Any]]) -> float:
    observed = [
        joint["confidence"]
        for joint in joints.values()
        if joint.get("status") in {"tracked", "low_confidence"}
    ]
    if not observed:
        return 0.0
    return round(sum(observed) / len(observed), 4)


def pose_results_to_pose_2d_frames(
    pose_results: List[Dict[str, Any]],
    frame_sampling: Optional[Dict[str, Any]] = None,
    fps: Optional[float] = None,
    view_type: str = "unknown",
    pose_model: str = POSE_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """Convert backend pose results into timestamped 2D pose frames."""
    lookup = _frame_sampling_lookup(frame_sampling)
    frames: List[Dict[str, Any]] = []
    source_fps = _finite_float(fps)

    for fallback_sample_index, result in enumerate(pose_results or []):
        source_frame_index = int(result.get("frame_idx") or 0)
        sample = lookup.get(source_frame_index) or {}
        timestamp_ms = sample.get("timestampMs")
        if timestamp_ms is None and source_fps:
            timestamp_ms = int(round((source_frame_index / source_fps) * 1000))
        if timestamp_ms is None:
            timestamp_ms = 0

        landmarks = result.get("landmarks") or {}
        joints = {
            name: normalise_joint(landmarks.get(name))
            for name in JOINT_NAMES
        }
        tracking_status = _tracking_status(joints, result)
        frame_confidence = _frame_confidence(joints)
        frames.append({
            "timestamp_ms": int(timestamp_ms),
            "source_frame_index": source_frame_index,
            "sample_index": sample.get("sampleIndex", fallback_sample_index),
            "view_type": view_type or "unknown",
            "pose_model": pose_model,
            "joints_2d": joints,
            "frame_confidence": frame_confidence,
            "tracking_status": tracking_status,
        })

    return frames


def build_pose_2d_summary(
    pose_frames: List[Dict[str, Any]],
    view_type: str = "unknown",
    model: str = POSE_MODEL_NAME,
    model_version: str = POSE_MODEL_VERSION,
    sampled_frames: Optional[int] = None,
) -> Dict[str, Any]:
    processed = len(pose_frames)
    tracked = sum(1 for frame in pose_frames if frame.get("tracking_status") == "tracked")
    partial = sum(1 for frame in pose_frames if frame.get("tracking_status") == "partial")
    failed = processed - tracked - partial
    confidence_values = [
        _finite_float(frame.get("frame_confidence"), 0.0) or 0.0
        for frame in pose_frames
    ]
    average_confidence = (
        round(sum(confidence_values) / len(confidence_values), 4)
        if confidence_values
        else 0.0
    )

    total_joints = 0
    low_confidence = 0
    for frame in pose_frames:
        for joint in (frame.get("joints_2d") or {}).values():
            total_joints += 1
            if joint.get("status") == "low_confidence":
                low_confidence += 1
    low_conf_rate = round(low_confidence / total_joints, 4) if total_joints else 0.0

    warnings: List[str] = []
    if failed:
        warnings.append("some_frames_failed_pose_tracking")
    if partial:
        warnings.append("some_frames_have_partial_pose_tracking")
    if processed and tracked == 0:
        warnings.append("no_fully_tracked_pose_frames")

    return {
        "availabilityState": "pose_2d_ready",
        "ok": processed > 0 and (tracked + partial) > 0,
        "model": model,
        "modelVersion": model_version,
        "sampledFrames": sampled_frames if sampled_frames is not None else processed,
        "processedFrames": processed,
        "trackedFrames": tracked,
        "partialFrames": partial,
        "failedFrames": failed,
        "averageFrameConfidence": average_confidence,
        "lowConfidenceJointRate": low_conf_rate,
        "viewType": view_type or "unknown",
        "warnings": warnings,
    }


def write_pose_2d_artifact(
    pose_frames: List[Dict[str, Any]],
    job_id: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Write private pose timeseries locally and return path-free metadata."""
    artifact_id = f"{job_id or uuid.uuid4()}-pose-2d"
    root = Path(output_dir or os.getenv("POSE_ARTIFACT_DIR", "pose_artifacts"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{artifact_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": POSE_2D_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "frames": pose_frames,
        }, handle, ensure_ascii=True)

    return {
        "artifact_type": "pose_2d_timeseries",
        "artifact_id": artifact_id,
        "storage_visibility": "private",
        "format": "json",
        "frame_count": len(pose_frames),
        "contains_raw_pose": True,
        "contains_video_pixels": False,
        "public_safe": False,
    }


def _request_value(request: Any, field: str) -> Any:
    if isinstance(request, dict):
        return request.get(field)
    return getattr(request, field, None)


def build_pose_2d_callback_payload(
    request: Any,
    job_id: str,
    pose_summary: Dict[str, Any],
    pose_artifact: Dict[str, Any],
    engine: str = "pose-mvp-0.5",
) -> Dict[str, Any]:
    """Build the safe `pose_2d_ready` progress callback."""
    return {
        "job_id": job_id,
        "app_job_id": _request_value(request, "app_job_id") or job_id,
        "server_job_id": job_id,
        "video_upload_id": _request_value(request, "video_upload_id"),
        "engine": engine,
        "status": "pose_2d_ready",
        "stage": "pose_2d_ready",
        "progress_percent": 62,
        "pose_2d_summary": pose_summary,
        "pose_artifact": pose_artifact,
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


def pose_2d_callback_payload_is_safe(payload: Dict[str, Any]) -> bool:
    for key, value in iter_payload_keys_and_values(payload):
        if key and key.lower() in UNSAFE_CALLBACK_KEYS:
            return False
        if isinstance(value, str):
            lowered = value.lower()
            if any(marker in lowered for marker in UNSAFE_VALUE_MARKERS):
                return False
    return True
