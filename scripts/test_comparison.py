#!/usr/bin/env python3
"""Synthetic tests for app/comparison.py (no real footage, no mediapipe).

Verifies the contract the side-by-side feature relies on:
  * identical clips -> ~zero deltas;
  * a translated + scaled copy -> ~zero deltas (translation/scale invariance);
  * a wrist-only fault -> the wrist deltas rise clearly above the unchanged
    shoulder/hip joints, and the overall delta grows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.comparison import compare_clips, comparison_enabled  # noqa: E402


def make_clip(n=90, period=20, wrist_amp=0.12, wrist_y_offset=0.0, translate=0.0, scale=1.0):
    """Periodic breaststroke-like pose_results that analyze_stroke_cycles can cycle."""
    frames = []
    for i in range(n):
        t = 2.0 * np.pi * (i / period)
        hip_x, hip_y = 0.5, 0.55
        sho_x, sho_y = 0.5, 0.45
        wrist_x = 0.5 + wrist_amp * np.cos(t)
        wrist_y = 0.40 + 0.03 * np.sin(t) + wrist_y_offset

        def P(x, y):
            return {"x": (x * scale) + translate, "y": y * scale, "visibility": 0.9}

        landmarks = {
            "left_hip": P(hip_x - 0.05, hip_y),
            "right_hip": P(hip_x + 0.05, hip_y),
            "left_shoulder": P(sho_x - 0.07, sho_y),
            "right_shoulder": P(sho_x + 0.07, sho_y),
            "left_wrist": P(wrist_x - 0.02, wrist_y),
            "right_wrist": P(wrist_x + 0.02, wrist_y),
            "left_elbow": P(wrist_x - 0.04, (wrist_y + sho_y) / 2.0),
            "right_elbow": P(wrist_x + 0.04, (wrist_y + sho_y) / 2.0),
        }
        frames.append({"frame_idx": i, "pose_detected": True, "keypoint_count": len(landmarks), "landmarks": landmarks})
    return frames


_passed = 0
_failed = 0


def check(name, condition):
    global _passed, _failed
    print(("PASS" if condition else "FAIL") + " - " + name)
    if condition:
        _passed += 1
    else:
        _failed += 1


# Flag defaults.
check("flag default OFF", comparison_enabled({}) is False)
check("flag on when truthy", comparison_enabled({"ENABLE_COMPARISON": "1"}) is True)

base = make_clip()

# 1) Identical clips -> ~zero deltas.
r1 = compare_clips(base, make_clip(), 30, 30, "breaststroke")
check("identical: status completed", r1["status"] == "completed")
check("identical: overall delta ~0", r1["summary"]["overall_mean_delta"] is not None and r1["summary"]["overall_mean_delta"] < 1e-6)
check("identical: aligned frame pairs present", len(r1["aligned_frames"]) > 0)
check("identical: phases present", len(r1["per_phase"]) > 0)

# 2) Translation + scale invariance -> still ~zero.
r2 = compare_clips(base, make_clip(translate=0.25, scale=1.4), 30, 30, "breaststroke")
check("translate+scale: status completed", r2["status"] == "completed")
check("translate+scale: overall delta ~0", r2["summary"]["overall_mean_delta"] is not None and r2["summary"]["overall_mean_delta"] < 1e-6)

# 3) Wrist-only fault -> wrist deltas rise above unchanged joints.
r3 = compare_clips(base, make_clip(wrist_y_offset=0.03), 30, 30, "breaststroke")
check("fault: overall delta grows", r3["summary"]["overall_mean_delta"] is not None and r3["summary"]["overall_mean_delta"] > 0.05)
wrist_beats_shoulder = True
for item in r3["per_phase"]:
    deltas = item["joint_deltas"]
    if "left_wrist" in deltas and "left_shoulder" in deltas:
        if not deltas["left_wrist"] > deltas["left_shoulder"] + 0.05:
            wrist_beats_shoulder = False
check("fault: wrist delta exceeds shoulder delta in every phase", wrist_beats_shoulder)
check("fault: largest_change_phase reported", r3["summary"]["largest_change_phase"] is not None)

# 4) Missing cycles -> clean insufficient result, never a crash.
flat = [{"frame_idx": i, "pose_detected": True, "keypoint_count": 0, "landmarks": {}} for i in range(5)]
r4 = compare_clips(flat, flat, 30, 30, "breaststroke")
check("degenerate input: insufficient, no crash", r4["status"] == "insufficient")

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
