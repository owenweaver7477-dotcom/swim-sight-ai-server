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

from app.stroke_cycles import (  # noqa: E402
    analyze_stroke_cycles,
    estimate_stroke_rate,
    sanitized_cycle_summary,
)

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
    "estimated_cycle_rate_per_min",
    "estimated_stroke_rate_spm",
    "strokes_per_cycle",
    "stroke_rate_estimated",
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


def _assert_rate_null(summary, label):
    _check(summary["estimated_cycle_rate_per_min"] is None, f"{label}: cycle rate must be null")
    _check(summary["estimated_stroke_rate_spm"] is None, f"{label}: spm must be null")
    _check(any(f.startswith("stroke_rate_") for f in summary["quality_flags"]),
           f"{label}: must include a stroke_rate_* reason flag")


def test_rate_gates():
    """Direct, deterministic tests of the pure estimate_stroke_rate gates."""
    base = dict(
        status="completed",
        stroke_type="Freestyle",
        mean_cycle_duration_seconds=1.0,
        cycle_count=4,
        cycle_regularity=0.8,
        confidence=0.7,
    )

    ok = estimate_stroke_rate(**base)
    _check(ok["estimated_cycle_rate_per_min"] == 60.0, "gate freestyle cycle rate == 60")
    _check(ok["strokes_per_cycle"] == 2, "gate freestyle strokes_per_cycle == 2")
    _check(ok["estimated_stroke_rate_spm"] == 120.0, "gate freestyle spm == 120")
    _check(ok["stroke_rate_estimated"] is True, "gate stroke_rate_estimated True")
    _check(ok["_flags"] == [], "gate: no flags when all pass")

    br = estimate_stroke_rate(**{**base, "stroke_type": "Breaststroke"})
    _check(br["strokes_per_cycle"] == 1 and br["estimated_stroke_rate_spm"] == 60.0, "gate breaststroke spm == 60")

    def null_with(flag, **over):
        r = estimate_stroke_rate(**{**base, **over})
        _check(r["estimated_cycle_rate_per_min"] is None, f"gate {flag}: cycle rate null")
        _check(r["estimated_stroke_rate_spm"] is None, f"gate {flag}: spm null")
        _check(flag in r["_flags"], f"gate {flag}: reason flag present ({r['_flags']})")

    null_with("stroke_rate_insufficient_cycles", cycle_count=2)
    null_with("stroke_rate_low_regularity", cycle_regularity=0.5)
    null_with("stroke_rate_low_confidence", confidence=0.3)
    null_with("stroke_rate_unavailable", status="insufficient_pose")
    null_with("stroke_rate_invalid_cycle_duration", mean_cycle_duration_seconds=0)
    null_with("stroke_rate_invalid_cycle_duration", mean_cycle_duration_seconds=None)
    null_with("stroke_rate_invalid_cycle_duration", mean_cycle_duration_seconds=float("nan"))

    # unknown / unsupported stroke -> null rate + null multiplier
    un = estimate_stroke_rate(**{**base, "stroke_type": "Backstroke"})
    _check(un["estimated_stroke_rate_spm"] is None and un["strokes_per_cycle"] is None, "gate unsupported stroke null")
    _check("stroke_rate_unavailable" in un["_flags"], "gate unsupported flag present")


def main() -> int:
    # 1. periodic freestyle -> completed, cycles found, cycle rate ~60 / spm ~120
    res = analyze_stroke_cycles(periodic_freestyle(), FPS, "Freestyle")
    _check(res["status"] == "completed", f"freestyle status={res['status']}")
    _check(res["summary"]["cycle_count"] > 0, "freestyle cycle_count must be > 0")
    s = _assert_summary_safe(res, "freestyle")
    _check(s["cycle_count"] > 0 and s["status"] == "completed", "freestyle summary mismatch")
    _check(s["stroke_rate_estimated"] is True, "freestyle stroke_rate_estimated must be True")
    _check(s["strokes_per_cycle"] == 2, "freestyle strokes_per_cycle must be 2")
    _check(s["estimated_cycle_rate_per_min"] is not None and abs(s["estimated_cycle_rate_per_min"] - 60.0) <= 5,
           f"freestyle cycle rate ~60, got {s['estimated_cycle_rate_per_min']}")
    _check(s["estimated_stroke_rate_spm"] is not None and abs(s["estimated_stroke_rate_spm"] - 120.0) <= 10,
           f"freestyle spm ~120, got {s['estimated_stroke_rate_spm']}")
    _check(15 <= s["estimated_stroke_rate_spm"] <= 200, "freestyle spm must be plausible")

    # 2. periodic breaststroke -> completed, cycle rate ~60 / spm ~60 (1 stroke/cycle)
    res = analyze_stroke_cycles(periodic_breaststroke(), FPS, "Breaststroke")
    _check(res["status"] == "completed", f"breaststroke status={res['status']}")
    _check(res["summary"]["cycle_count"] > 0, "breaststroke cycle_count must be > 0")
    s2 = _assert_summary_safe(res, "breaststroke")
    _check(s2["strokes_per_cycle"] == 1, "breaststroke strokes_per_cycle must be 1")
    _check(s2["estimated_cycle_rate_per_min"] is not None and abs(s2["estimated_cycle_rate_per_min"] - 60.0) <= 5,
           f"breaststroke cycle rate ~60, got {s2['estimated_cycle_rate_per_min']}")
    _check(s2["estimated_stroke_rate_spm"] is not None and abs(s2["estimated_stroke_rate_spm"] - 60.0) <= 5,
           f"breaststroke spm ~60, got {s2['estimated_stroke_rate_spm']}")

    # 3. sparse -> safe insufficient status, no cycles, null rate
    res = analyze_stroke_cycles(sparse(), FPS, "Freestyle")
    _check(res["status"] == "insufficient_pose", f"sparse status={res['status']}")
    _check(res["summary"]["cycle_count"] == 0, "sparse must have 0 cycles")
    _assert_rate_null(_assert_summary_safe(res, "sparse"), "sparse")

    # 4. non-periodic -> no fabricated cycles, null rate
    res = analyze_stroke_cycles(non_periodic(), FPS, "Freestyle")
    _check(res["summary"]["cycle_count"] == 0, "non-periodic must not fabricate cycles")
    _check(res["status"] in {"insufficient_periodicity", "insufficient_pose"},
           f"non-periodic status={res['status']}")
    _assert_rate_null(_assert_summary_safe(res, "non_periodic"), "non_periodic")

    # 5. unsupported stroke -> safe empty result, null rate
    res = analyze_stroke_cycles(periodic_freestyle(), FPS, "Backstroke")
    _check(res["status"] == "unsupported", f"backstroke status={res['status']}")
    _check(res["supported"] is False, "backstroke supported must be False")
    _check(res["summary"]["cycle_count"] == 0, "unsupported must have 0 cycles")
    us = _assert_summary_safe(res, "unsupported")
    _assert_rate_null(us, "unsupported")
    _check(us["strokes_per_cycle"] is None, "unsupported strokes_per_cycle must be null")

    # 6. empty input -> safe, no crash, null rate
    res = analyze_stroke_cycles([], FPS, "Freestyle")
    _check(res["summary"]["cycle_count"] == 0, "empty input must have 0 cycles")
    _assert_rate_null(_assert_summary_safe(res, "empty"), "empty")

    # 7. direct gate unit tests
    test_rate_gates()

    print("stroke cycle tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
