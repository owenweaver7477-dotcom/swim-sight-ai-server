"""
app/pose_landmarks.py - pure landmark constants + geometry helpers.

These have NO heavy dependencies (no mediapipe / cv2), so the analysis stack
(pose_postprocess, swim_analyzer findings, the drag estimate) can import them and
run with zero footage and without the detection libraries installed. The
detector (pose_estimator) re-exports these so existing imports keep working.
"""
from __future__ import annotations

from typing import Dict

# Core keypoints that count toward the quality gate. Richer face/hand/foot
# landmarks may be tracked internally but must not inflate this count.
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


def get_midpoint(lm_a: Dict, lm_b: Dict) -> Dict:
    return {
        "x": (lm_a["x"] + lm_b["x"]) / 2,
        "y": (lm_a["y"] + lm_b["y"]) / 2,
    }


def vertical_distance(point_a: Dict, point_b: Dict) -> float:
    return point_b["y"] - point_a["y"]


def horizontal_distance(point_a: Dict, point_b: Dict) -> float:
    return abs(point_a["x"] - point_b["x"])
