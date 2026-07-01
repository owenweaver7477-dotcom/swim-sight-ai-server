"""Tests for the pure validation verdict logic (synthetic rows, no footage).

Run:  python3 scripts/test_validation_report.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validation_report import compare_clip  # noqa: E402
from scripts.build_validation_report import assert_report_safe, build_report  # noqa: E402


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _row(**over):
    row = {
        "clip_name": "clip1.mp4",
        "stroke": "Freestyle",
        "finding_count": 1,
        "finding_fault_tags": ["body_line_loss"],
        "overall_score": 78,
        "fallback_triggered": False,
        "pose_detection_rate": 0.74,
        "processing_seconds": 18.0,
        "analysis_mode": "real_pose",
        "stroke_cycles": None,
    }
    row.update(over)
    return row


def _cycles(**over):
    c = {
        "enabled": True,
        "status": "completed",
        "cycle_count": 3,
        "cycle_regularity": 0.9,
        "confidence": 0.8,
        "estimated_cycle_rate_per_min": 60.0,
        "estimated_stroke_rate_spm": 120.0,
        "basis": "2d_heuristic",
        "public_safe": False,
    }
    c.update(over)
    return c


def test_forbidden_tag_fails():
    v = compare_clip(
        _row(),
        _row(finding_fault_tags=["body_line_loss", "head_lift"], finding_count=2),
        label={"forbidden_fault_tags": ["head_lift"]},
    )
    _check(v["verdict"] == "FAIL", f"forbidden tag must FAIL, got {v['verdict']}")
    _check("forbidden_fault_tag_found" in v["fail_reasons"], "forbidden reason missing")
    _check(v["forbidden_fault_tags_found"] == ["head_lift"], "forbidden tag not reported")


def test_big_score_swing_needs_review():
    v = compare_clip(_row(), _row(overall_score=58))  # delta -20
    _check(v["verdict"] == "NEEDS_REVIEW", f"big swing must review, got {v['verdict']}")
    _check("big_score_swing" in v["review_reasons"], "score swing reason missing")
    _check(v["score_delta"] == -20, f"score delta wrong: {v['score_delta']}")


def test_clean_small_delta_passes():
    v = compare_clip(_row(), _row(overall_score=80))  # delta +2
    _check(v["verdict"] == "PASS", f"clean delta must PASS, got {v['verdict']} {v['review_reasons']}")


def test_false_cycle_fails():
    v = compare_clip(
        _row(),
        _row(stroke_cycles=_cycles(status="completed", cycle_count=3)),
        label={"expected_cycle_status": "insufficient"},
    )
    _check(v["verdict"] == "FAIL", f"false cycle must FAIL, got {v['verdict']}")
    _check("false_cycle_on_no_cycle_clip" in v["fail_reasons"], "false-cycle reason missing")


def test_expected_completed_passes():
    v = compare_clip(
        _row(),
        _row(stroke_cycles=_cycles(status="completed", cycle_count=3)),
        label={"expected_cycle_status": "completed"},
    )
    _check(v["verdict"] == "PASS", f"expected+completed must PASS, got {v['verdict']} {v['review_reasons']}")


def test_rate_within_tolerance_passes():
    v = compare_clip(
        _row(),
        _row(stroke_cycles=_cycles(estimated_cycle_rate_per_min=61.0)),
        label={"coach_counted_cycles": 3, "counted_over_seconds": 3},  # expected 60/min
    )
    _check(v["verdict"] == "PASS", f"rate within tol must PASS, got {v['verdict']} {v['review_reasons']}")
    _check(v["stroke_rate"]["within_tolerance"] is True, "within_tolerance should be True")


def test_rate_outside_tolerance_reviews():
    v = compare_clip(
        _row(),
        _row(stroke_cycles=_cycles(estimated_cycle_rate_per_min=100.0)),
        label={"coach_counted_cycles": 3, "counted_over_seconds": 3},  # expected 60/min
    )
    _check(v["verdict"] == "NEEDS_REVIEW", f"rate out of tol must review, got {v['verdict']}")
    _check("stroke_rate_out_of_tolerance" in v["review_reasons"], "out-of-tolerance reason missing")


def test_rate_missing_when_expected_reviews():
    v = compare_clip(
        _row(),
        _row(stroke_cycles=_cycles(estimated_cycle_rate_per_min=None)),
        label={"coach_counted_cycles": 3, "counted_over_seconds": 3},
    )
    _check(v["verdict"] == "NEEDS_REVIEW", f"missing rate must review, got {v['verdict']}")
    _check("stroke_rate_missing_when_expected" in v["review_reasons"], "missing-rate reason missing")


def test_report_is_footage_safe():
    results = [
        {"variant": "baseline", **_row()},
        {"variant": "phase_analysis", **_row(stroke_cycles=_cycles())},
        {"variant": "extended_stroke_findings",
         **_row(stroke="Butterfly", finding_fault_tags=["butterfly_rhythm_break", "butterfly_body_line_loss"], finding_count=2)},
    ]
    report = build_report(results, labels_by_clip={})
    assert_report_safe(report)  # raises if unsafe
    blob = json.dumps(report).lower()
    for bad in ("landmark", "token=", "/users/", "/tmp/", "signed_video_url", "eyj", "secret"):
        _check(bad not in blob, f"report leaked '{bad}'")
    _check(report["overall_verdict"] in ("PASS", "NEEDS_REVIEW", "FAIL"), "overall verdict invalid")


def main() -> int:
    test_forbidden_tag_fails()
    test_big_score_swing_needs_review()
    test_clean_small_delta_passes()
    test_false_cycle_fails()
    test_expected_completed_passes()
    test_rate_within_tolerance_passes()
    test_rate_outside_tolerance_reviews()
    test_rate_missing_when_expected_reviews()
    test_report_is_footage_safe()
    print("validation report tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
