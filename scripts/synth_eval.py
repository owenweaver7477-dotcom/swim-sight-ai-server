#!/usr/bin/env python3
"""
Footage-free evaluation harness.

You do NOT need any swim video to exercise the ANALYSIS stage. This script
builds synthetic pose_results (the same dict shape run_pose_estimation emits) for
a side-view freestyle swimmer, with an optional injected fault and a known
forward velocity, then runs them straight through:

  * pose post-processing  (app.pose_postprocess)
  * the findings engine   (app.swim_analyzer.analyze_pose_data)

It bypasses video + MediaPipe entirely, so it works with zero footage.

Examples
--------
    python3 scripts/synth_eval.py --fault hip_drop
    python3 scripts/synth_eval.py --fault dropped_elbow
    python3 scripts/synth_eval.py --inject-noise --compare-flag ENABLE_POSE_SMOOTHING
    python3 scripts/synth_eval.py --fault head_lift --compare-flag ROBUST_FINDINGS

What it can / can't validate
----------------------------
* CAN (footage-free): findings logic, robust-findings, and pose smoothing.
* CANNOT: the detection stage itself (CLAHE, POSE_MODEL_COMPLEXITY, sequential
  reads), or real-swimmer finding quality. Those need representative, licensed
  footage and coach labels.

Synthetic results are logic checks only. They are not calibrated swimming
results and must not be treated as coach-validated product evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.pose_postprocess import smooth_pose_results, pose_smoothing_enabled  # noqa: E402
from app.swim_analyzer import analyze_pose_data  # noqa: E402


# Side-view layout: x = lane (forward), y = vertical (depth). A "good" swimmer is
# a flat horizontal line (all y ~ 0.5); faults are vertical/relational deviations.
# Left/right pairs overlap on screen (side view) so swim_analyzer's shoulder-width
# normaliser uses its 0.22 fallback -- which keeps the fault magnitudes predictable.
_BASE_OFFSETS_X = {          # along the body, relative to hip centre
    "nose": 0.25, "shoulder": 0.12, "elbow": 0.22, "wrist": 0.32,
    "hip": -0.02, "knee": -0.14, "ankle": -0.25,
}


def build_pose_results(fault: str = "none", n_frames: int = 40, fps: float = 30.0,
                       frame_step: int = 2, true_v: float = 1.6,
                       height_cm: float = 180.0, jitter: float = 0.004,
                       inject_noise: bool = False, seed: int = 7) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    height_m = height_cm / 100.0
    body_len_units = _BASE_OFFSETS_X["nose"] - _BASE_OFFSETS_X["ankle"]  # 0.5
    scale_true = height_m / body_len_units
    base_y = 0.5

    out: List[Dict[str, Any]] = []
    for k in range(n_frames):
        fi = k * frame_step
        t = fi / fps
        cx = 0.2 + (true_v * t) / scale_true                  # hip centre advances down the lane

        def jit():
            return rng.normal(0, jitter)

        def pt(name_x, y):
            return {"x": float(cx + _BASE_OFFSETS_X[name_x] + jit()),
                    "y": float(y + jit()), "visibility": 0.9}

        nose_y, sh_y, el_y, wr_y, hip_y = base_y, base_y, base_y, base_y, base_y
        if fault == "hip_drop":
            hip_y = base_y + 0.40                              # hips sag well below the line
        elif fault == "head_lift":
            nose_y = base_y - 0.12                             # head lifts above the line
        elif fault == "dropped_elbow":
            el_y, wr_y = base_y + 0.12, base_y                 # elbow drops below the wrist

        lm = {
            "nose": pt("nose", nose_y),
            "left_shoulder": pt("shoulder", sh_y), "right_shoulder": pt("shoulder", sh_y),
            "left_elbow": pt("elbow", el_y), "right_elbow": pt("elbow", el_y),
            "left_wrist": pt("wrist", wr_y), "right_wrist": pt("wrist", wr_y),
            "left_hip": pt("hip", hip_y), "right_hip": pt("hip", hip_y),
            "left_knee": pt("knee", base_y), "right_knee": pt("knee", base_y),
            "left_ankle": pt("ankle", base_y), "right_ankle": pt("ankle", base_y),
        }

        # Optionally drop a couple of landmarks / add an outlier so the smoothing
        # flag has something to fix.
        if inject_noise:
            if k % 9 == 4:
                lm.pop("left_hip", None)                       # short gap
            if k % 13 == 6:
                lm["nose"] = {"x": float(cx + 4.0), "y": 0.5, "visibility": 0.9}  # outlier

        out.append({"frame_idx": fi, "pose_detected": True,
                    "keypoint_count": len(lm), "landmarks": lm})
    return out


def _pose_signal_summary(pose_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Small deterministic signal summary used to verify post-processing."""
    nose_x = [
        frame["landmarks"]["nose"]["x"]
        for frame in pose_results
        if "nose" in (frame.get("landmarks") or {})
    ]
    max_nose_step = max(
        (abs(current - previous) for previous, current in zip(nose_x, nose_x[1:])),
        default=0.0,
    )
    return {
        "frames_with_left_hip": sum(
            "left_hip" in (frame.get("landmarks") or {}) for frame in pose_results
        ),
        "max_nose_x_step": round(float(max_nose_step), 6),
    }


def evaluate(pose_results: List[Dict[str, Any]], *, stroke: str, fps: float) -> Dict[str, Any]:
    """Run synthetic findings and honour smoothing exactly like main.py."""
    pr = pose_results
    if pose_smoothing_enabled():
        try:
            pr = smooth_pose_results(pr)
        except Exception:
            pr = pose_results

    analysis = analyze_pose_data(
        pose_results=pr, frames=list(range(len(pr))), fps=fps,
        total_duration=len(pr) / fps if fps else 0.0,
        stroke_type=stroke, camera_angle="Side", video_upload_id="synth")
    findings = analysis.get("findings") or []
    out: Dict[str, Any] = {
        "frames": len(pr),
        "pose_signal": _pose_signal_summary(pr),
        "finding_count": len(findings),
        "finding_titles": [f.get("finding_title") for f in findings],
        "analysis_mode": analysis.get("analysis_mode"),
    }
    return out


def _print(label: str, res: Dict[str, Any]) -> None:
    print(f"--- {label} ---")
    for k, v in res.items():
        print(f"  {k:22s}: {v}")


def main() -> int:
    p = argparse.ArgumentParser(description="Footage-free synthetic evaluation.")
    p.add_argument("--stroke", default="Freestyle")
    p.add_argument("--fault", default="none",
                   choices=["none", "hip_drop", "head_lift", "dropped_elbow"])
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--true-v", type=float, default=1.6)
    p.add_argument("--height-cm", type=float, default=180.0)
    p.add_argument("--inject-noise", action="store_true",
                   help="add short gaps + an outlier so ENABLE_POSE_SMOOTHING has work to do")
    p.add_argument("--compare-flag", default=None,
                   help="env flag to A/B, e.g. ROBUST_FINDINGS or ENABLE_POSE_SMOOTHING")
    p.add_argument("--emit-json", default=None, help="write the synthetic pose_results to this path")
    args = p.parse_args()

    pr = build_pose_results(fault=args.fault, n_frames=args.frames, fps=args.fps,
                            true_v=args.true_v, height_cm=args.height_cm,
                            inject_noise=args.inject_noise)

    if args.emit_json:
        Path(args.emit_json).write_text(json.dumps(pr, indent=2))
        print(f"Wrote {len(pr)} synthetic frames -> {args.emit_json}")

    print(f"Synthetic clip: stroke={args.stroke} fault={args.fault} "
          f"frames={len(pr)} (logic check only; not real-swimmer evidence)\n")

    if args.compare_flag:
        flag = args.compare_flag
        prev = os.environ.get(flag)
        os.environ[flag] = "false"
        off = evaluate(pr, stroke=args.stroke, fps=args.fps)
        os.environ[flag] = "true"
        on = evaluate(pr, stroke=args.stroke, fps=args.fps)
        if prev is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = prev
        print(f"BEFORE/AFTER for {flag}:\n")
        _print(f"{flag}=false", off)
        _print(f"{flag}=true", on)
    else:
        _print("result", evaluate(pr, stroke=args.stroke, fps=args.fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
