#!/usr/bin/env python3
"""Inspect default-off stroke phase analysis with JSON or synthetic pose data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.stroke_cycles import analyze_stroke_cycles  # noqa: E402
from app.technique_reference import compare_phase_technique, load_reference_bands  # noqa: E402


def build_synthetic_pose(
    kind: str,
    *,
    frame_count: int = 240,
    fps: float = 60.0,
) -> tuple[List[Dict[str, Any]], str]:
    """Build periodic normalized 2D pose tracks for logic testing only."""

    stroke = "freestyle" if kind.startswith("freestyle") else "breaststroke"
    sparse = kind.endswith("sparse")
    extension_fault = kind == "breaststroke_fault"
    period = 48
    frames: List[Dict[str, Any]] = []

    for frame_idx in range(frame_count):
        angle = 2.0 * math.pi * frame_idx / period
        shoulder_y = 0.45
        hip_y = 0.48
        if extension_fault and (frame_idx % period) < period // 4:
            hip_y = 0.68

        landmarks: Dict[str, Dict[str, float]] = {
            "nose": {"x": 0.30, "y": 0.44, "visibility": 0.95},
            "left_shoulder": {"x": 0.38, "y": shoulder_y, "visibility": 0.95},
            "right_shoulder": {"x": 0.42, "y": shoulder_y, "visibility": 0.95},
            "left_elbow": {"x": 0.48, "y": 0.44, "visibility": 0.95},
            "right_elbow": {"x": 0.50, "y": 0.46, "visibility": 0.95},
            "left_hip": {"x": 0.58, "y": hip_y, "visibility": 0.95},
            "right_hip": {"x": 0.62, "y": hip_y, "visibility": 0.95},
            "left_knee": {"x": 0.58, "y": 0.49, "visibility": 0.95},
            "right_knee": {"x": 0.62, "y": 0.49, "visibility": 0.95},
            "left_ankle": {"x": 0.76, "y": 0.49, "visibility": 0.95},
            "right_ankle": {"x": 0.80, "y": 0.49, "visibility": 0.95},
        }

        if stroke == "breaststroke":
            wrist_mid = 0.60 + 0.22 * math.cos(angle)
            landmarks["left_wrist"] = {"x": wrist_mid - 0.015, "y": 0.43, "visibility": 0.95}
            landmarks["right_wrist"] = {"x": wrist_mid + 0.015, "y": 0.43, "visibility": 0.95}
        else:
            arm_offset = 0.22 * math.sin(angle)
            landmarks["left_wrist"] = {"x": 0.60 + arm_offset, "y": 0.42, "visibility": 0.95}
            landmarks["right_wrist"] = {"x": 0.60 - arm_offset, "y": 0.48, "visibility": 0.95}

        if sparse and frame_idx % 3 == 0:
            landmarks.pop("left_wrist")

        frames.append({
            "frame_idx": frame_idx,
            "pose_detected": True,
            "keypoint_count": len(landmarks),
            "landmarks": landmarks,
        })
    return frames, stroke


def inspect(pose_results: List[Dict[str, Any]], stroke: str, fps: float) -> Dict[str, Any]:
    phase_analysis = analyze_stroke_cycles(pose_results, fps, stroke)
    context = {
        "stroke_type": phase_analysis.get("stroke_type"),
        "reference_status": "provisional_internal",
        "validated": False,
        "phase_context": [],
    }
    if phase_analysis.get("supported") and phase_analysis.get("cycles"):
        context = compare_phase_technique(
            pose_results,
            phase_analysis,
            load_reference_bands(stroke),
        )
    return {"phase_analysis": phase_analysis, "phase_context": context}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect approximate phase cycles (internal logic test only)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--synthetic",
        choices=[
            "breaststroke_clean",
            "breaststroke_fault",
            "breaststroke_sparse",
            "freestyle_clean",
            "freestyle_sparse",
        ],
    )
    source.add_argument("--pose-json", type=Path)
    parser.add_argument("--stroke", default=None)
    parser.add_argument("--fps", type=float, default=60.0)
    args = parser.parse_args()

    if args.synthetic:
        pose_results, synthetic_stroke = build_synthetic_pose(args.synthetic, fps=args.fps)
        stroke = args.stroke or synthetic_stroke
    else:
        try:
            pose_results = json.loads(args.pose_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Could not read pose JSON: {exc}")
        if not isinstance(pose_results, list):
            parser.error("Pose JSON must contain a list of pose-result frames.")
        stroke = args.stroke
        if not stroke:
            parser.error("--stroke is required with --pose-json.")

    result = inspect(pose_results, stroke, args.fps)
    phase = result["phase_analysis"]
    print("Phase analysis is an internal estimate; coach review remains required.")
    print(f"Stroke: {phase['stroke_type']}")
    print(f"Status: {phase['status']}")
    print(f"Cycle count: {phase['summary']['cycle_count']}")
    print(f"Confidence: {phase['summary']['confidence']:.3f}")
    for cycle in phase.get("cycles") or []:
        phase_names = ", ".join(item["phase"] for item in cycle["phases"])
        print(
            f"  Cycle {cycle['cycle_idx']}: frames {cycle['start_frame']}-{cycle['end_frame']} "
            f"({cycle['duration_seconds']:.3f}s), phases: {phase_names}"
        )
    contexts = result["phase_context"].get("phase_context") or []
    print(f"Phase-aware contexts: {len(contexts)}")
    for item in contexts:
        print(
            f"  {item['metric']} / {item['phase']} / cycle {item['cycle_idx']}: "
            f"value={item['value']}, confidence={item['confidence']}, "
            f"frames={item['evidence_frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
