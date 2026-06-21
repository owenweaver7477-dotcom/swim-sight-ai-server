#!/usr/bin/env python3
"""Synthetic tests for optional known-distance calibration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.calibration import calculate_known_distance_calibration  # noqa: E402
from app.pose_worker_integration import analyse_clip, synthetic_pose_results  # noqa: E402


def run() -> None:
    normalised_config = {
        "calibration_type": "known_distance",
        "image_points": [{"x": 0.25, "y": 0.6}, {"x": 0.75, "y": 0.6}],
        "real_distance_m": 2.5,
        "coordinate_space": "normalised",
    }
    normalised = calculate_known_distance_calibration(normalised_config)
    assert normalised["calibration_status"] == "calibrated"
    assert normalised["metres_per_normalised_unit"] == 5.0
    assert normalised["metres_per_pixel"] is None
    assert normalised["notes"]

    pixel_config = {
        "calibration_type": "known_distance",
        "image_points": [{"x": 480, "y": 640}, {"x": 1440, "y": 640}],
        "real_distance_m": 2.5,
        "coordinate_space": "pixels",
        "image_width": 1920,
        "image_height": 1080,
    }
    pixels = calculate_known_distance_calibration(pixel_config)
    assert pixels["calibration_status"] == "calibrated"
    assert pixels["metres_per_normalised_unit"] == 5.0
    assert abs(pixels["metres_per_pixel"] - 0.002604167) < 1e-9

    zero_points = dict(normalised_config)
    zero_points["image_points"] = [{"x": 0.25, "y": 0.6}, {"x": 0.25, "y": 0.6}]
    assert "greater than zero" in calculate_known_distance_calibration(zero_points)["reason"]

    negative_distance = dict(normalised_config)
    negative_distance["real_distance_m"] = -1
    assert "positive" in calculate_known_distance_calibration(negative_distance)["reason"]

    missing_dimensions = dict(pixel_config)
    missing_dimensions.pop("image_width")
    missing = calculate_known_distance_calibration(missing_dimensions)
    assert missing["calibration_status"] == "invalid"
    assert "dimensions" in missing["reason"]

    unknown_space = dict(normalised_config)
    unknown_space["coordinate_space"] = "screen"
    assert "coordinate_space" in calculate_known_distance_calibration(unknown_space)["reason"]

    too_many_points = dict(normalised_config)
    too_many_points["image_points"] = [
        {"x": 0.1, "y": 0.1},
        {"x": 0.5, "y": 0.5},
        {"x": 0.9, "y": 0.9},
    ]
    assert "exactly two" in calculate_known_distance_calibration(too_many_points)["reason"]

    low_confidence = dict(normalised_config)
    low_confidence["confidence"] = 0.3
    low_result = calculate_known_distance_calibration(low_confidence)
    assert low_result["calibration_status"] == "low_confidence"

    unknown_type = dict(normalised_config)
    unknown_type["calibration_type"] = "camera_guess"
    assert "calibration_type" in calculate_known_distance_calibration(unknown_type)["reason"]

    pose_results = synthetic_pose_results()
    existing = analyse_clip(
        pose_results,
        fps=30,
        height_cm=180.0,
        mass_kg=75.0,
    )
    explicit_none = analyse_clip(
        pose_results,
        fps=30,
        height_cm=180.0,
        mass_kg=75.0,
        calibration_config=None,
    )
    assert existing == explicit_none
    assert "calibration" not in existing
    assert "metric_basis" not in existing

    invalid_config = dict(normalised_config)
    invalid_config["real_distance_m"] = 0
    invalid_fallback = analyse_clip(
        pose_results,
        fps=30,
        height_cm=180.0,
        mass_kg=75.0,
        calibration_config=invalid_config,
    )
    assert invalid_fallback is not None
    assert invalid_fallback["metric_basis"] == "estimated"
    assert invalid_fallback["calibration"]["calibration_status"] == "invalid"
    assert invalid_fallback["scale_m_per_unit"] == existing["scale_m_per_unit"]

    low_confidence_fallback = analyse_clip(
        pose_results,
        fps=30,
        height_cm=180.0,
        mass_kg=75.0,
        calibration_config=low_confidence,
    )
    assert low_confidence_fallback["metric_basis"] == "estimated"
    assert low_confidence_fallback["calibration"]["calibration_status"] == "low_confidence"
    assert low_confidence_fallback["scale_m_per_unit"] == existing["scale_m_per_unit"]

    calibrated = analyse_clip(
        pose_results,
        fps=30,
        height_cm=180.0,
        mass_kg=75.0,
        calibration_config=normalised_config,
    )
    assert calibrated["scale_m_per_unit"] == 5.0
    assert calibrated["metric_basis"] == "calibrated"
    assert calibrated["calibration"]["calibration_status"] == "calibrated"
    assert "calibrated from marked image distance" in calibrated["basis"]

    serialized = json.dumps(calibrated)
    for private_field in (
        "image_points",
        "height_cm",
        "mass_kg",
        "height_m",
        "signed_video_url",
        "file_path",
    ):
        assert private_field not in serialized

    canonical_callback = json.loads(
        (ROOT / "fixtures" / "callback_success.example.json").read_text(encoding="utf-8")
    )
    assert "calibration" not in canonical_callback
    assert "calibration_status" not in canonical_callback
    assert "metric_basis" not in canonical_callback

    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.calibration; "
                "assert 'mediapipe' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules; print('pure calibration import ok')"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr

    print("Calibration checks passed.")


if __name__ == "__main__":
    run()
