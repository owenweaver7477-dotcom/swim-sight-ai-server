import mediapipe as mp
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose

LANDMARK = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

MIN_VISIBILITY = 0.45


def run_pose_estimation(frames: list) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    if not frames:
        logger.warning("Pose estimation skipped: no frames provided")
        return results

    logger.info(f"Starting pose estimation on {len(frames)} sampled frames")

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            for original_idx, frame_rgb in frames:
                try:
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

    return results


def _process_frame(pose, frame_rgb, original_frame_idx: int) -> Dict[str, Any]:
    result = pose.process(frame_rgb)

    if not result.pose_landmarks:
        return {
            "frame_idx": original_frame_idx,
            "pose_detected": False,
            "keypoint_count": 0,
            "landmarks": {},
        }

    landmarks = {}
    visible_count = 0

    for name, idx in LANDMARK.items():
        lm = result.pose_landmarks.landmark[idx]

        if lm.visibility >= MIN_VISIBILITY:
            landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "visibility": lm.visibility,
            }
            visible_count += 1

    return {
        "frame_idx": original_frame_idx,
        "pose_detected": visible_count >= 4,
        "keypoint_count": visible_count,
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