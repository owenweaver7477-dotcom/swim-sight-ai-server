"""
app/pose_backends.py - pluggable pose engine behind the POSE_BACKEND flag.

POSE_BACKEND selects which detector turns frames into pose_results:

  * "mediapipe" (default): the current run_pose_estimation. NO behaviour change.
  * "onnx": a ViTPose / RTMPose model fine-tuned on SwimXYZ and exported to ONNX.
            app.pose_onnx.run_onnx_pose(frames) loads the exported model from
            POSE_ONNX_PATH (see FINE_TUNE_POSE_PLAN.md). It MUST return the same pose_results
            shape as run_pose_estimation: a list of
            {frame_idx, pose_detected, keypoint_count, landmarks:{name:{x,y,visibility}}}
            with the worker's landmark names (reuse app.swimxyz_adapter's mapping).

Lazy imports keep this module importable without mediapipe or onnxruntime
installed. With POSE_BACKEND unset it behaves exactly as the current MediaPipe
worker.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def pose_backend_name(env: Optional[Dict[str, str]] = None) -> str:
    src = os.environ if env is None else env
    return str(src.get("POSE_BACKEND", "mediapipe")).strip().lower() or "mediapipe"


def run_pose_estimation_backend(frames: list,
                                env: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Dispatch frame -> pose_results to the configured backend."""
    name = pose_backend_name(env)

    if name == "mediapipe":
        from app.pose_estimator import run_pose_estimation
        return run_pose_estimation(frames)

    if name == "onnx":
        from app.pose_onnx import run_onnx_pose
        return run_onnx_pose(frames)

    raise ValueError(f"Unknown POSE_BACKEND={name!r}; expected 'mediapipe' or 'onnx'.")
