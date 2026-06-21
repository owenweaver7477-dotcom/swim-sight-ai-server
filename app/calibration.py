"""Known-distance image calibration for optional internal scale metrics.

Calibration improves image-space scale only. It does not correct refraction,
camera perspective, occlusion, pose error, or turn a monocular estimate into a
validated biomechanics measurement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def _invalid(reason: str) -> Dict[str, Any]:
    return {
        "calibration_status": "invalid",
        "reason": reason,
        "confidence": 0.0,
    }


def _number(value: Any, field: str) -> Tuple[Optional[float], Optional[str]]:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None, f"{field} must be numeric"
    result = float(value)
    if not math.isfinite(result):
        return None, f"{field} must be finite"
    return result, None


def _dimensions(config: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    width, width_error = _number(config.get("image_width"), "image_width")
    if width_error:
        return None, None, width_error
    height, height_error = _number(config.get("image_height"), "image_height")
    if height_error:
        return None, None, height_error
    if width is None or width <= 0:
        return None, None, "image_width must be positive"
    if height is None or height <= 0:
        return None, None, "image_height must be positive"
    return width, height, None


def calculate_known_distance_calibration(config: Any) -> Dict[str, Any]:
    """Validate a known-distance config and derive safe scale values.

    Raw image points are intentionally omitted from the returned result so an
    internal summary can be attached without echoing unnecessary input data.
    """

    if not isinstance(config, Mapping):
        return _invalid("calibration config must be an object")
    calibration_type = str(config.get("calibration_type", "")).strip().lower()
    if calibration_type != "known_distance":
        return _invalid("unsupported calibration_type; expected 'known_distance'")

    points = config.get("image_points")
    if not isinstance(points, (list, tuple)) or len(points) != 2:
        return _invalid("image_points must contain exactly two points")

    parsed_points = []
    for index, point in enumerate(points):
        if not isinstance(point, Mapping):
            return _invalid(f"image_points[{index}] must be an object")
        x, x_error = _number(point.get("x"), f"image_points[{index}].x")
        if x_error:
            return _invalid(x_error)
        y, y_error = _number(point.get("y"), f"image_points[{index}].y")
        if y_error:
            return _invalid(y_error)
        parsed_points.append((float(x), float(y)))

    real_distance, distance_error = _number(config.get("real_distance_m"), "real_distance_m")
    if distance_error:
        return _invalid(distance_error)
    if real_distance is None or real_distance <= 0:
        return _invalid("real_distance_m must be positive")

    supplied_confidence = config.get("confidence", 1.0)
    confidence, confidence_error = _number(supplied_confidence, "confidence")
    if confidence_error:
        return _invalid(confidence_error)
    if confidence is None or not 0.0 <= confidence <= 1.0:
        return _invalid("confidence must be between 0 and 1")
    if confidence < 0.5:
        return {
            "calibration_status": "low_confidence",
            "reason": "calibration confidence is below the internal use threshold",
            "confidence": round(confidence, 3),
        }

    coordinate_space = str(config.get("coordinate_space", "")).strip().lower()
    if coordinate_space not in {"normalised", "pixels"}:
        return _invalid("coordinate_space must be 'normalised' or 'pixels'")

    first, second = parsed_points
    notes = []
    metres_per_pixel: Optional[float] = None

    if coordinate_space == "pixels":
        width, height, dimensions_error = _dimensions(config)
        if dimensions_error:
            return _invalid(f"pixel coordinate space requires dimensions: {dimensions_error}")
        if not (0 <= first[0] <= width and 0 <= second[0] <= width):
            return _invalid("pixel x coordinates must fall within image_width")
        if not (0 <= first[1] <= height and 0 <= second[1] <= height):
            return _invalid("pixel y coordinates must fall within image_height")
        pixel_distance = math.hypot(second[0] - first[0], second[1] - first[1])
        normalised_distance = math.hypot(
            (second[0] - first[0]) / width,
            (second[1] - first[1]) / height,
        )
        if pixel_distance <= 1e-12 or normalised_distance <= 1e-12:
            return _invalid("image point distance must be greater than zero")
        metres_per_pixel = real_distance / pixel_distance
    else:
        if not all(0.0 <= value <= 1.0 for point in parsed_points for value in point):
            return _invalid("normalised coordinates must be between 0 and 1")
        normalised_distance = math.hypot(second[0] - first[0], second[1] - first[1])
        if normalised_distance <= 1e-12:
            return _invalid("image point distance must be greater than zero")

        has_width = config.get("image_width") is not None
        has_height = config.get("image_height") is not None
        if has_width or has_height:
            width, height, dimensions_error = _dimensions(config)
            if dimensions_error:
                return _invalid(f"optional image dimensions are invalid: {dimensions_error}")
            pixel_distance = math.hypot(
                (second[0] - first[0]) * width,
                (second[1] - first[1]) * height,
            )
            if pixel_distance > 1e-12:
                metres_per_pixel = real_distance / pixel_distance
        else:
            notes.append("metres_per_pixel unavailable without image dimensions")

    return {
        "calibration_status": "calibrated",
        "calibration_type": "known_distance",
        "coordinate_space": coordinate_space,
        "metres_per_normalised_unit": round(real_distance / normalised_distance, 9),
        "metres_per_pixel": round(metres_per_pixel, 9) if metres_per_pixel is not None else None,
        "known_distance_m": round(real_distance, 6),
        "image_distance_normalised": round(normalised_distance, 9),
        "confidence": round(confidence, 3),
        "notes": notes,
    }


def safe_calibration_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the non-sensitive subset suitable for an internal metric block."""

    allowed = (
        "calibration_status",
        "calibration_type",
        "coordinate_space",
        "metres_per_normalised_unit",
        "metres_per_pixel",
        "known_distance_m",
        "image_distance_normalised",
        "confidence",
        "notes",
        "reason",
    )
    return {key: result[key] for key in allowed if key in result}
