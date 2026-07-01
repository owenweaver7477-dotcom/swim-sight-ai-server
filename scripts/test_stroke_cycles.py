"""Tests for the heuristic stroke-cycle segmentation and its sanitized summary.

Pure/offline: builds synthetic 2D pose tracks, exercises analyze_stroke_cycles
across periodic / sparse / non-periodic / unsupported cases, and asserts the
sanitized telemetry summary leaks no raw landmarks, per-frame data, URLs, or
secrets. No network, no deploy.

Run:  python3 scripts/test_stroke_cycles.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.stroke_cycles import analyze_stroke_cycles, sanitized_cycle_summary  # noqa: E402

FPS = 30.0
PERIOD = 30  # frames per cycle -> 1.0s cycles, peaks well above the min gap

EXPECTED_SUMMARY_KEYS = {
    "enabled",
    "status",
    "cycle_count",
    "mean_cycle_duration_seconds",
    "cycle_regularity",
    "confidence",
    "quality_flags",
    "basis",
    "public_safe",
}
FORBIDDEN_SUBSTRINGS = (
    "landmark", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "start_frame", "end_frame",
    "token=", "http://", "https://", "/users/", "/tmp/", "secret", "signed",
    '"x"', '"y"', "visibility",
)


def _frame(idx, landmarks, detected=True):
    return {"frame_idx": idx, "pose_detected": detected, "keypoint_count": 12, "landmarks": landmarks}


def _lm(x, y):
    return {"x": float(x), "y": float(y), "z": 0.0, "visibility": 0.9}


def periodic_freestyle(n=120):
    """Alternating-arm signal: left/right wrists oscillate out of phase."""
    frames = []
    for i in range(n):
        s = 0.2 * math.sin(2 * math.pi * i / PERIOD)
        frames.append(_frame(i, {
            "left_hip": _lm(0.45, 0.80),
            "right_hip": _lm(0.55, 0.80),
            "left_wrist": _lm(0.50 + s, 0.50),
            "right_wrist": _lm(0.50 - s, 0.50),
        }))
    return frames


def periodic_breaststroke(n=120):
    """Symmetric hand-reach oscillation forward/back of the hips."""
    frames = []
    for i in range(n):
        s = 0.2 * math.sin(2 * math.pi * i / PERIOD)
        frames.append(_frame(i, {
            "left_hip": _lm(0.48, 0.80),
            "right_hip": _lm(0.52, 0.80),
            "left_wrist": _lm(0.50 + s, 0.55 + 0.05 * s),
            "right_wrist": _lm(0.50 + s, 0.55 + 0.05 * s),
        }))
    return frames


def sparse(n=5):
    return periodic_freestyle(n)


def non_periodic(n=60):
    """Constant signal -> no periodicity, must not fabricate cycles."""
    frames = []
    for i in range(n):
        frames.append(_frame(i, {
            "left_hip": _lm(0.45, 0.80),
            "right_hip": _lm(0.55, 0.80),
            "left_wrist": _lm(0.60, 0.50),
            "right_wrist": _lm(0.40, 0.50),
        }))
    return frames


def _check(cond, label):
    if not cond:
        raise AssertionError(label)


def _assert_summary_safe(result, label):
    summary = sanitized_cycle_summary(result)
    _check(set(summary) == EXPECTED_SUMMARY_KEYS,
           f"{label}: summary keys {sorted(summary)} != expected")
    _check(summary["basis"] == "2d_heuristic", f"{label}: basis must be 2d_heuristic")
    _check(summary["public_safe"] is False, f"{label}: public_safe must be False")
    _check(summary["enabled"] is True, f"{label}: enabled must be True")
    _check(isinstance(summary["quality_flags"], list)
           and all(isinstance(f, str) for f in summary["quality_flags"]),
           f"{label}: quality_flags must be a list of strings")
    blob = json.dumps(summary).lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        _check(bad not in blob, f"{label}: sanitized summary leaked '{bad}'")
    return summary


def main() -> int:
    # 1. periodic freestyle -> completed, cycles found
    res = analyze_stroke_cycles(periodic_freestyle(), FPS, "Freestyle")
    _check(res["status"] == "completed", f"freestyle status={res['status']}")
    _check(res["summary"]["cycle_count"] > 0, "freestyle cycle_count must be > 0")
    s = _assert_summary_safe(res, "freestyle")
    _check(s["cycle_count"] > 0 and s["status"] == "completed", "freestyle summary mismatch")

    # 2. periodic breaststroke -> completed, cycles found
    res = analyze_stroke_cycles(periodic_breaststroke(), FPS, "Breaststroke")
    _check(res["status"] == "completed", f"breaststroke status={res['status']}")
    _check(res["summary"]["cycle_count"] > 0, "breaststroke cycle_count must be > 0")
    _assert_summary_safe(res, "breaststroke")

    # 3. sparse -> safe insufficient status, no cycles
    res = analyze_stroke_cycles(sparse(), FPS, "Freestyle")
    _check(res["status"] == "insufficient_pose", f"sparse status={res['status']}")
    _check(res["summary"]["cycle_count"] == 0, "sparse must have 0 cycles")
    _assert_summary_safe(res, "sparse")

    # 4. non-periodic -> no fabricated cycles
    res = analyze_stroke_cycles(non_periodic(), FPS, "Freestyle")
    _check(res["summary"]["cycle_count"] == 0, "non-periodic must not fabricate cycles")
    _check(res["status"] in {"insufficient_periodicity", "insufficient_pose"},
           f"non-periodic status={res['status']}")
    _assert_summary_safe(res, "non_periodic")

    # 5. unsupported stroke -> safe empty result
    res = analyze_stroke_cycles(periodic_freestyle(), FPS, "Backstroke")
    _check(res["status"] == "unsupported", f"backstroke status={res['status']}")
    _check(res["supported"] is False, "backstroke supported must be False")
    _check(res["summary"]["cycle_count"] == 0, "unsupported must have 0 cycles")
    _assert_summary_safe(res, "unsupported")

    # 6. empty input -> safe, no crash
    res = analyze_stroke_cycles([], FPS, "Freestyle")
    _check(res["summary"]["cycle_count"] == 0, "empty input must have 0 cycles")
    _assert_summary_safe(res, "empty")

    print("stroke cycle tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
