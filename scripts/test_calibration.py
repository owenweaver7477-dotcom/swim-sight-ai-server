#!/usr/bin/env python3
"""Local synthetic inspection for known-distance calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.calibration import calculate_known_distance_calibration  # noqa: E402
from app.pose_worker_integration import analyse_clip, synthetic_pose_results  # noqa: E402


def main() -> int:
    calibration = {
        "calibration_type": "known_distance",
        "image_points": [
            {"x": 0.25, "y": 0.60},
            {"x": 0.75, "y": 0.60},
        ],
        "real_distance_m": 2.5,
        "coordinate_space": "normalised",
        "image_width": 1920,
        "image_height": 1080,
    }
    result = calculate_known_distance_calibration(calibration)
    pose_results = synthetic_pose_results()
    estimated = analyse_clip(
        pose_results,
        fps=30,
        height_cm=180.0,
        mass_kg=75.0,
    )
    calibrated = analyse_clip(
        pose_results,
        fps=30,
        height_cm=180.0,
        mass_kg=75.0,
        calibration_config=calibration,
    )

    print("Known-distance calibration is internal draft context; coach review remains required.")
    print(f"Calibration status: {result['calibration_status']}")
    print(f"Metres per normalised unit: {result['metres_per_normalised_unit']}")
    print(f"Metres per pixel: {result['metres_per_pixel']}")
    print(f"Estimated basis: {estimated['basis']}")
    print(f"Calibrated basis: {calibrated['basis']}")
    print("Safe calibration summary:")
    print(json.dumps(calibrated["calibration"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
