"""Provisional internal phase-aware technique reference comparisons.

Reference bands are replaceable coach-review aids. They are not validated
biomechanics standards and must not be used to publish automatic final advice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from app.temporal_metrics import frame_relative_metrics


REFERENCE_DIR = Path(__file__).resolve().parent / "reference_bands"


def _normalize_stroke(stroke_type: str) -> str:
    value = (stroke_type or "").strip().lower()
    return {"breast": "breaststroke", "free": "freestyle", "front crawl": "freestyle"}.get(
        value, value
    )


def validate_reference_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and return a copy of a provisional reference configuration."""

    if not isinstance(config, Mapping):
        raise ValueError("Reference config must be a JSON object.")
    if config.get("status") != "provisional_internal":
        raise ValueError("Reference config status must be 'provisional_internal'.")
    if config.get("validated") is not False:
        raise ValueError("Reference config must explicitly set validated=false.")
    stroke = _normalize_stroke(str(config.get("stroke_type", "")))
    if stroke not in {"breaststroke", "freestyle"}:
        raise ValueError("Reference config stroke_type must be breaststroke or freestyle.")
    metrics = config.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("Reference config must define at least one metric.")

    for metric_name, metric_config in metrics.items():
        if not isinstance(metric_config, Mapping):
            raise ValueError(f"Metric '{metric_name}' must be an object.")
        phases = metric_config.get("phases")
        if not isinstance(phases, Mapping) or not phases:
            raise ValueError(f"Metric '{metric_name}' must define phase bands.")
        for phase, band in phases.items():
            if not isinstance(band, Mapping):
                raise ValueError(f"Metric '{metric_name}' phase '{phase}' must be an object.")
            if "min" not in band or "max" not in band:
                raise ValueError(f"Metric '{metric_name}' phase '{phase}' needs min and max.")
            minimum = float(band["min"])
            maximum = float(band["max"])
            if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
                raise ValueError(f"Metric '{metric_name}' phase '{phase}' has an invalid range.")
            if not str(band.get("coach_language", "")).strip():
                raise ValueError(f"Metric '{metric_name}' phase '{phase}' needs coach_language.")
    return dict(config)


def load_reference_bands(
    stroke_type: str,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    stroke = _normalize_stroke(stroke_type)
    config_path = Path(path) if path else REFERENCE_DIR / f"{stroke}_basic.json"
    if not config_path.is_file():
        raise ValueError(f"No provisional reference bands found for stroke '{stroke}'.")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read reference config '{config_path.name}': {exc}") from exc
    validated = validate_reference_config(config)
    if _normalize_stroke(str(validated["stroke_type"])) != stroke:
        raise ValueError("Reference config stroke_type does not match the requested stroke.")
    return validated


def _metric_value(frame_metrics: Mapping[str, Any], metric_name: str) -> Optional[float]:
    if metric_name == "hip_drop":
        value = frame_metrics.get("body_line_offset")
    elif metric_name == "head_alignment":
        value = frame_metrics.get("head_alignment")
    elif metric_name == "knee_width_ratio":
        value = frame_metrics.get("knee_width_ratio")
    elif metric_name == "elbow_flexion":
        available = [
            float(frame_metrics[name])
            for name in ("left_elbow_angle", "right_elbow_angle")
            if name in frame_metrics
        ]
        value = float(np.mean(available)) if available else None
    else:
        value = frame_metrics.get(metric_name)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _severity(value: float, minimum: float, maximum: float, band: Mapping[str, Any]) -> str:
    deviation = minimum - value if value < minimum else value - maximum
    span = max(maximum - minimum, 0.05)
    high_threshold = float(band.get("high_deviation", span * 1.5))
    return "high" if deviation >= high_threshold else "moderate"


def compare_phase_technique(
    pose_results: Sequence[Mapping[str, Any]],
    phase_analysis: Mapping[str, Any],
    reference_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare persistent phase metrics with provisional internal bands."""

    config = validate_reference_config(reference_config)
    stroke = _normalize_stroke(str(phase_analysis.get("stroke_type", "")))
    if stroke != _normalize_stroke(str(config.get("stroke_type", ""))):
        raise ValueError("Phase analysis and reference config stroke types do not match.")
    fps = float(phase_analysis.get("fps") or 30.0)
    frames_by_index = {
        int(frame.get("frame_idx", index)): frame
        for index, frame in enumerate(pose_results)
        if frame.get("pose_detected")
    }
    metrics_by_index = {
        frame_idx: frame_relative_metrics(frame, fps)
        for frame_idx, frame in frames_by_index.items()
    }

    contexts: List[Dict[str, Any]] = []
    for cycle in phase_analysis.get("cycles") or []:
        cycle_confidence = float(cycle.get("confidence", 0.0))
        for phase in cycle.get("phases") or []:
            phase_name = str(phase.get("phase", ""))
            start = int(phase.get("start_frame", 0))
            end = int(phase.get("end_frame", start))
            phase_indices = sorted(index for index in metrics_by_index if start <= index <= end)
            for metric_name, metric_config in config["metrics"].items():
                band = (metric_config.get("phases") or {}).get(phase_name)
                if not band:
                    continue
                minimum = float(band["min"])
                maximum = float(band["max"])
                samples = [
                    (index, value)
                    for index in phase_indices
                    if (value := _metric_value(metrics_by_index[index], metric_name)) is not None
                ]
                min_frames = int(band.get("min_evidence_frames", 3))
                if len(samples) < min_frames:
                    continue
                violations = [
                    (index, value)
                    for index, value in samples
                    if value < minimum or value > maximum
                ]
                persistence = len(violations) / len(samples)
                if len(violations) < min_frames or persistence < float(band.get("persistence_ratio", 0.5)):
                    continue

                aggregate = float(np.median([value for _, value in violations]))
                ranked = sorted(
                    violations,
                    key=lambda item: max(minimum - item[1], item[1] - maximum),
                    reverse=True,
                )
                evidence_frames = [index for index, _ in ranked[:3]]
                coverage = len(samples) / max(len(phase_indices), 1)
                confidence = np.clip(
                    cycle_confidence * (0.55 + 0.45 * persistence) * (0.6 + 0.4 * coverage),
                    0.0,
                    1.0,
                )
                contexts.append({
                    "metric": metric_name,
                    "phase": phase_name,
                    "cycle_idx": int(cycle.get("cycle_idx", 0)),
                    "value": round(aggregate, 3),
                    "reference_min": minimum,
                    "reference_max": maximum,
                    "severity": _severity(aggregate, minimum, maximum, band),
                    "confidence": round(float(confidence), 3),
                    "evidence_frames": evidence_frames,
                    "persistence_ratio": round(persistence, 3),
                    "coach_language": str(band["coach_language"]),
                    "reference_status": "provisional_internal",
                    "coach_review_required": True,
                })

    return {
        "stroke_type": stroke,
        "reference_status": "provisional_internal",
        "validated": False,
        "phase_context": contexts,
    }
