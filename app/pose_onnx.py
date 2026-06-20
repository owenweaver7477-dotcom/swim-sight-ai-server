"""Lazy ONNX pose backend for an MMDeploy RTMPose/ViTPose export.

The public ``run_onnx_pose`` function preserves the worker pose-result contract.
Neither onnxruntime nor OpenCV is imported until real inference is requested.
"""
from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from app.swimxyz_adapter import (
    COCO17_TO_WORKER,
    DEFAULT_MIN_VIS,
    DETECT_MIN_KEYPOINTS,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT_WIDTH = 192
DEFAULT_INPUT_HEIGHT = 256
SIMCC_SPLIT_RATIO = 2.0
_IMAGE_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGE_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


class PoseOnnxError(RuntimeError):
    """Actionable configuration or inference failure for the optional backend."""


def _require_model_path() -> str:
    raw = os.getenv("POSE_ONNX_PATH", "").strip()
    if not raw:
        raise PoseOnnxError(
            "POSE_BACKEND=onnx requires POSE_ONNX_PATH to reference an exported "
            "MMDeploy end-to-end pose model."
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise PoseOnnxError(
            "POSE_ONNX_PATH does not point to a readable model file. Export the "
            "fine-tuned model first or restore POSE_BACKEND=mediapipe."
        )
    return str(path.resolve())


@lru_cache(maxsize=2)
def _load_session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise PoseOnnxError(
            "POSE_BACKEND=onnx requires the optional 'onnxruntime' package. "
            "Install it in the worker environment before enabling this backend."
        ) from exc

    try:
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise PoseOnnxError(
            "The configured ONNX pose model could not be loaded. Confirm it is "
            "a valid MMDeploy end-to-end pose export."
        ) from exc


def _positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _model_input_size(session) -> Tuple[int, int]:
    try:
        shape = session.get_inputs()[0].shape
        height = _positive_int(shape[-2], DEFAULT_INPUT_HEIGHT)
        width = _positive_int(shape[-1], DEFAULT_INPUT_WIDTH)
        return width, height
    except Exception:
        return DEFAULT_INPUT_WIDTH, DEFAULT_INPUT_HEIGHT


def _prepare_input(frame_rgb: np.ndarray, target_width: int, target_height: int):
    try:
        import cv2
    except ImportError as exc:
        raise PoseOnnxError(
            "ONNX pose preprocessing requires opencv-python-headless."
        ) from exc

    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        raise PoseOnnxError("ONNX pose inference received an invalid RGB frame.")

    source_height, source_width = frame_rgb.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise PoseOnnxError("ONNX pose inference received an empty RGB frame.")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame_rgb[:, :, :3], (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    canvas = np.zeros((target_height, target_width, 3), dtype=np.float32)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized.astype(np.float32)
    canvas = (canvas - _IMAGE_MEAN) / _IMAGE_STD
    tensor = np.transpose(canvas, (2, 0, 1))[None, ...].astype(np.float32)
    transform = {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "source_width": float(source_width),
        "source_height": float(source_height),
    }
    return tensor, transform


def _parse_end2end_outputs(outputs: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(output) for output in outputs]
    squeezed = [np.squeeze(array, axis=0) if array.ndim >= 1 and array.shape[0] == 1 else array
                for array in arrays]

    for array in squeezed:
        if array.ndim == 2 and array.shape == (17, 3):
            return array[:, :2].astype(float), array[:, 2].astype(float)

    coordinates = next(
        (array for array in squeezed if array.ndim == 2 and array.shape == (17, 2)),
        None,
    )
    scores = next(
        (array for array in squeezed if array.ndim == 1 and array.shape[0] == 17),
        None,
    )
    if coordinates is not None and scores is not None:
        return coordinates.astype(float), scores.astype(float)

    simcc_outputs = [
        array for array in squeezed
        if array.ndim == 2 and array.shape[0] == 17 and array.shape[1] > 3
    ]
    if len(simcc_outputs) == 2:
        simcc_x, simcc_y = sorted(simcc_outputs, key=lambda array: array.shape[1])
        x_indices = np.argmax(simcc_x, axis=1).astype(float) / SIMCC_SPLIT_RATIO
        y_indices = np.argmax(simcc_y, axis=1).astype(float) / SIMCC_SPLIT_RATIO
        keypoints = np.column_stack((x_indices, y_indices))
        x_scores = np.max(simcc_x, axis=1)
        y_scores = np.max(simcc_y, axis=1)
        scores = np.clip((x_scores + y_scores) / 2.0, 0.0, 1.0)
        return keypoints, scores

    raise PoseOnnxError(
        "Unsupported ONNX pose output. Expected decoded MMDeploy keypoints and "
        "scores, a combined (1,17,3) output, or RTMPose SimCC x/y outputs."
    )


def _restore_normalised_coordinates(keypoints: np.ndarray, transform: Dict[str, float]) -> np.ndarray:
    scale = transform["scale"]
    if not math.isfinite(scale) or scale <= 0:
        raise PoseOnnxError("ONNX pose preprocessing produced an invalid scale.")
    restored = keypoints.astype(float).copy()
    restored[:, 0] = (restored[:, 0] - transform["pad_x"]) / scale
    restored[:, 1] = (restored[:, 1] - transform["pad_y"]) / scale
    restored[:, 0] = np.clip(restored[:, 0] / transform["source_width"], 0.0, 1.0)
    restored[:, 1] = np.clip(restored[:, 1] / transform["source_height"], 0.0, 1.0)
    return restored


def _infer_frame(frame_rgb: np.ndarray, model_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run one frame and return normalised COCO-17 coordinates and scores."""
    session = _load_session(model_path)
    target_width, target_height = _model_input_size(session)
    tensor, transform = _prepare_input(frame_rgb, target_width, target_height)
    try:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
        keypoints, scores = _parse_end2end_outputs(outputs)
        return _restore_normalised_coordinates(keypoints, transform), scores
    except PoseOnnxError:
        raise
    except Exception as exc:
        raise PoseOnnxError(
            "ONNX pose inference failed. Confirm the model matches the documented "
            "MMDeploy end-to-end output contract."
        ) from exc


def _pose_result(frame_idx: int, keypoints: Any, scores: Any) -> Dict[str, Any]:
    points = np.asarray(keypoints, dtype=float)
    confidence = np.asarray(scores, dtype=float).reshape(-1)
    if points.shape != (17, 2) or confidence.shape != (17,):
        raise PoseOnnxError("ONNX pose inference must return 17 coordinates and 17 scores.")

    landmarks: Dict[str, Dict[str, float]] = {}
    visible_total = 0
    for index in range(17):
        x, y = points[index]
        score = confidence[index]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(score)):
            continue
        score = float(np.clip(score, 0.0, 1.0))
        if score < DEFAULT_MIN_VIS:
            continue
        visible_total += 1
        name = COCO17_TO_WORKER.get(index)
        if name:
            landmarks[name] = {
                "x": float(np.clip(x, 0.0, 1.0)),
                "y": float(np.clip(y, 0.0, 1.0)),
                "visibility": score,
            }

    keypoint_count = len(landmarks)
    return {
        "frame_idx": int(frame_idx),
        "pose_detected": keypoint_count >= DETECT_MIN_KEYPOINTS,
        "keypoint_count": keypoint_count,
        "landmark_count_total": visible_total,
        "landmarks": landmarks,
    }


def run_onnx_pose(frames: list) -> List[Dict[str, Any]]:
    """Convert sampled RGB frames into the worker's stable pose contract."""
    model_path = _require_model_path()
    results: List[Dict[str, Any]] = []
    for fallback_idx, item in enumerate(frames):
        if isinstance(item, tuple) and len(item) == 2:
            frame_idx, frame_rgb = item
        else:
            frame_idx, frame_rgb = fallback_idx, item
        try:
            keypoints, scores = _infer_frame(frame_rgb, model_path)
            results.append(_pose_result(int(frame_idx), keypoints, scores))
        except PoseOnnxError:
            raise
        except Exception as exc:
            raise PoseOnnxError(
                "ONNX pose processing failed for a sampled frame. Restore "
                "POSE_BACKEND=mediapipe while checking the exported model."
            ) from exc
    logger.info("ONNX pose backend processed %s sampled frames", len(results))
    return results
