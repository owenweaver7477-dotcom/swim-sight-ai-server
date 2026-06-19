"""Focused tests for relative-2D temporal metrics and phase segmentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_estimator import CORE_LANDMARK_NAMES, LANDMARK  # noqa: E402
from app.temporal_metrics import build_temporal_analysis, joint_angle_degrees  # noqa: E402


def point(x, y):
    return {"x": x, "y": y, "z": 0.0, "visibility": 0.9}


assert len(LANDMARK) == 33
assert len(CORE_LANDMARK_NAMES) == 15
assert round(joint_angle_degrees(point(0, 0), point(1, 0), point(2, 0))) == 180

frames = []
for index in range(12):
    frames.append({
        "frame_idx": index * 5,
        "pose_detected": True,
        "keypoint_count": 15,
        "landmark_count_total": 33,
        "landmarks": {
            "nose": point(0.5, 0.25),
            "left_shoulder": point(0.4, 0.4),
            "right_shoulder": point(0.6, 0.4),
            "left_elbow": point(0.3, 0.4),
            "right_elbow": point(0.7, 0.4),
            "left_wrist": point(0.1, 0.4),
            "right_wrist": point(0.9, 0.4),
            "left_hip": point(0.44, 0.6),
            "right_hip": point(0.56, 0.6),
            "left_knee": point(0.45, 0.75),
            "right_knee": point(0.55, 0.75),
            "left_ankle": point(0.46, 0.9),
            "right_ankle": point(0.54, 0.9),
        },
    })

result = build_temporal_analysis(frames, fps=30.0, stroke="Breaststroke")
assert result["metric_basis"] == "relative_2d_image_space"
assert result["usable_sample_count"] == 12
assert result["phase_segments"]
assert "left_elbow_angle" in result["relative_metrics"]
assert "insufficient_temporal_samples" not in result["quality_flags"]
print("temporal metrics tests passed")
