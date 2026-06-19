import mediapipe as mp
import numpy as np
import logging
import os
from typing import List, Dict, Any

from app.frame_enhance import clahe_enabled, enhance_frame_rgb

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose

LANDMARK = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Preserve the existing quality-gate semantics. Richer face/hand/foot landmarks
# are available internally, but they do not inflate the core keypoint count.
CORE_LANDMARK_NAMES = {
    "nose",
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
}

MIN_VISIBILITY = 0.45


def _model_complexity() -> int:
    """
    MediaPipe BlazePose complexity from POSE_MODEL_COMPLEXITY (0/1/2).
    Default 0 preserves current production behaviour; set 1 (or 2) for markedly
    better keypoint accuracy/stability at higher per-frame cost.
    """
    try:
        c = int(os.getenv("POSE_MODEL_COMPLEXITY", "0"))
    except ValueError:
        c = 0
    return c if c in (0, 1, 2) else 0


def run_pose_estimation(frames: list) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if not frames:
        logger.warning("Pose estimation skipped: no frames provided")
        return results

    logger.info(f"Starting pose estimation on {len(frames)} sampled frames")

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=_model_complexity(),
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            for original_idx, frame_rgb in frames:
                try:
                    if not frame_rgb.flags["C_CONTIGUOUS"]:
                        frame_rgb = np.ascontiguousarray(frame_rgb)
                    frame_rgb.flags.writeable = False
                    frame_result = _process_frame(pose, frame_rgb, original_idx)
                    results.append(frame_result)
                except Exception as frame_error:
                    logger.warning(
                        f"Pose failed on frame {original_idx}: {frame_error}"
                    )
                    results.append({
                        "frame_idx": original_idx,
                        "pose_detected": False,
                        "keypoint_count": 0,
                        "landmarks": {},
                        "error": "frame_pose_failed",
                    })
                finally:
                    try:
                        frame_rgb.flags.writeable = True
                    except Exception:
                        pass

    except Exception as error:
        logger.exception(f"Pose estimation failed completely: {error}")

        return [
            {
                "frame_idx": original_idx,
                "pose_detected": False,
                "keypoint_count": 0,
                "landmarks": {},
                "error": "pose_estimation_failed",
            }
            for original_idx, _frame_rgb in frames
        ]

    detected = sum(1 for r in results if r.get("pose_detected"))
    avg_kps = (
        np.mean([r.get("keypoint_count", 0) for r in results])
        if results
        else 0
    )

    logger.info(
        f"Pose estimation complete: {detected}/{len(results)} frames detected, "
        f"avg keypoints={avg_kps:.1f}"
    )

    if results and detected / len(results) < 0.10:
        logger.warning("Pose estimation produced very weak evidence; manual review likely")

    return results


def _process_frame(pose, frame_rgb, original_frame_idx: int) -> Dict[str, Any]:
    # Optional CLAHE contrast lift before detection (ENABLE_CLAHE). Safe no-op
    # otherwise; enhance_frame_rgb returns the frame unchanged on any error.
    if clahe_enabled():
        frame_rgb = enhance_frame_rgb(frame_rgb)

    result = pose.process(frame_rgb)

    if not result.pose_landmarks:
        return {
            "frame_idx": original_frame_idx,
            "pose_detected": False,
            "keypoint_count": 0,
            "landmarks": {},
        }

    landmarks = {}
    core_visible_count = 0

    for name, idx in LANDMARK.items():
        lm = result.pose_landmarks.landmark[idx]

        if lm.visibility >= MIN_VISIBILITY:
            landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
            if name in CORE_LANDMARK_NAMES:
                core_visible_count += 1

    return {
        "frame_idx": original_frame_idx,
        "pose_detected": core_visible_count >= 4,
        "keypoint_count": core_visible_count,
        "landmark_count_total": len(landmarks),
        "landmarks": landmarks,
    }


def get_midpoint(lm_a: Dict, lm_b: Dict) -> Dict:
    return {
        "x": (lm_a["x"] + lm_b["x"]) / 2,
        "y": (lm_a["y"] + lm_b["y"]) / 2,
    }


def vertical_distance(point_a: Dict, point_b: Dict) -> float:
    return point_b["y"] - point_a["y"]


def horizontal_distance(point_a: Dict, point_b: Dict) -> float:
    return abs(point_a["x"] - point_b["x"])
