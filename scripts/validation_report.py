"""Pure diff/verdict logic for AI worker flag validation.

Given a `baseline` per-clip evaluation row and a flag-on `variant` row (from
evaluate_baseline / compare_upgrade_flags), compute a footage-safe verdict:
finding/score/fallback/detection/timing deltas, stroke-cycle quality, and
estimated stroke-rate accuracy vs coach-counted rate.

This module is pure: no file I/O, no network, no env reads. Rows carry only
derived metrics and fault tags (no landmarks / URLs / secrets), so verdicts are
footage-safe by construction. Verdict precedence: FAIL > NEEDS_REVIEW > PASS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

# Cycle statuses that mean "no clear cycles" — a coach labelling a clip with one
# of these expects the worker NOT to report completed cycles.
NO_CYCLE_STATUSES = {
    "insufficient",
    "insufficient_periodicity",
    "insufficient_pose",
    "none",
    "no_cycles",
    "unsupported",
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "score_swing_review": 10.0,        # |overall_score delta| above this -> coach review
    "detection_drop_tolerance": 0.05,  # pose-detection-rate drop beyond this -> regression
    "stroke_rate_tolerance_percent": 15.0,
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(value: Any, default: Optional[float]) -> Optional[float]:
    return float(value) if _is_number(value) else default


def expected_cycle_rate_value(label: Mapping[str, Any]) -> Optional[float]:
    """Coach-provided expected cycle rate (per minute), or None.

    Priority: explicit `expected_cycle_rate_per_min`, else derived from
    `coach_counted_cycles` / `counted_over_seconds`.
    """
    explicit = label.get("expected_cycle_rate_per_min")
    if _is_number(explicit) and explicit > 0:
        return float(explicit)
    cycles = label.get("coach_counted_cycles")
    seconds = label.get("counted_over_seconds")
    if _is_number(cycles) and _is_number(seconds) and seconds > 0:
        return round(60.0 * float(cycles) / float(seconds), 2)
    return None


def compare_clip(
    baseline_row: Mapping[str, Any],
    variant_row: Mapping[str, Any],
    label: Optional[Mapping[str, Any]] = None,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Return a footage-safe verdict dict comparing one clip's baseline vs variant."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    label = dict(label or {})
    stroke = variant_row.get("stroke") or baseline_row.get("stroke")
    clip = variant_row.get("clip_name") or baseline_row.get("clip_name")

    b_tags = set(baseline_row.get("finding_fault_tags") or [])
    v_tags = set(variant_row.get("finding_fault_tags") or [])
    new_tags = sorted(v_tags - b_tags)
    removed_tags = sorted(b_tags - v_tags)
    finding_count_delta = int(variant_row.get("finding_count") or 0) - int(baseline_row.get("finding_count") or 0)

    b_score, v_score = baseline_row.get("overall_score"), variant_row.get("overall_score")
    score_delta = round(v_score - b_score, 2) if _is_number(b_score) and _is_number(v_score) else None

    b_fallback = bool(baseline_row.get("fallback_triggered"))
    v_fallback = bool(variant_row.get("fallback_triggered"))

    b_det, v_det = baseline_row.get("pose_detection_rate"), variant_row.get("pose_detection_rate")
    detection_delta = round(v_det - b_det, 4) if _is_number(b_det) and _is_number(v_det) else None

    b_proc, v_proc = baseline_row.get("processing_seconds"), variant_row.get("processing_seconds")
    processing_delta = round(v_proc - b_proc, 3) if _is_number(b_proc) and _is_number(v_proc) else None

    cycles = variant_row.get("stroke_cycles") or {}
    cycle_status = cycles.get("status")
    cycle_count = cycles.get("cycle_count")
    est_cycle_rate = cycles.get("estimated_cycle_rate_per_min")
    est_spm = cycles.get("estimated_stroke_rate_spm")
    completed_cycle = cycle_status == "completed" and _is_number(cycle_count) and cycle_count > 0

    forbidden = set(label.get("forbidden_fault_tags") or [])
    expected = set(label.get("expected_fault_tags") or [])
    forbidden_found = sorted(forbidden & v_tags)
    unexpected = sorted((v_tags - expected) - forbidden) if expected else []

    fail_reasons: List[str] = []
    review_reasons: List[str] = []

    # ── Hard fails (regressions / safety) ──
    if forbidden_found:
        fail_reasons.append("forbidden_fault_tag_found")
    if v_fallback and not b_fallback:
        fail_reasons.append("fallback_regression")
    if detection_delta is not None and detection_delta < -t["detection_drop_tolerance"]:
        fail_reasons.append("pose_detection_regression")

    expected_cycle_status = label.get("expected_cycle_status")
    if expected_cycle_status in NO_CYCLE_STATUSES and completed_cycle:
        fail_reasons.append("false_cycle_on_no_cycle_clip")

    # ── Needs coach review ──
    if expected_cycle_status == "completed" and cycle_status is not None and not completed_cycle:
        review_reasons.append("cycle_expected_not_detected")
    if score_delta is not None and abs(score_delta) > t["score_swing_review"]:
        review_reasons.append("big_score_swing")
    if new_tags and not expected:
        review_reasons.append("new_fault_tags_unlabelled")
    if unexpected:
        review_reasons.append("unexpected_fault_tags")

    # ── Estimated stroke rate accuracy (cycle rate + optional spm) ──
    rate: Dict[str, Any] = {
        "expected_cycle_rate_per_min": expected_cycle_rate_value(label),
        "estimated_cycle_rate_per_min": est_cycle_rate if _is_number(est_cycle_rate) else None,
        "error_percent": None,
        "within_tolerance": None,
    }
    tol = _number(label.get("stroke_rate_tolerance_percent"), t["stroke_rate_tolerance_percent"])
    expected_rate = rate["expected_cycle_rate_per_min"]
    if expected_rate is not None:
        if not _is_number(est_cycle_rate):
            review_reasons.append("stroke_rate_missing_when_expected")
        else:
            err = abs(est_cycle_rate - expected_rate) / expected_rate * 100 if expected_rate else None
            rate["error_percent"] = round(err, 2) if err is not None else None
            rate["within_tolerance"] = err is not None and err <= tol
            if not rate["within_tolerance"]:
                review_reasons.append("stroke_rate_out_of_tolerance")

    exp_spm = _number(label.get("expected_stroke_rate_spm"), None)
    if exp_spm is not None and exp_spm > 0:
        rate["expected_stroke_rate_spm"] = exp_spm
        rate["estimated_stroke_rate_spm"] = est_spm if _is_number(est_spm) else None
        if not _is_number(est_spm):
            if "stroke_rate_missing_when_expected" not in review_reasons:
                review_reasons.append("stroke_rate_missing_when_expected")
        else:
            err = abs(est_spm - exp_spm) / exp_spm * 100
            rate["spm_error_percent"] = round(err, 2)
            if err > tol and "stroke_rate_out_of_tolerance" not in review_reasons:
                review_reasons.append("stroke_rate_out_of_tolerance")

    verdict = "FAIL" if fail_reasons else ("NEEDS_REVIEW" if review_reasons else "PASS")

    return {
        "clip_name": clip,
        "stroke": stroke,
        "verdict": verdict,
        "fail_reasons": sorted(set(fail_reasons)),
        "review_reasons": sorted(set(review_reasons)),
        "finding_count_delta": finding_count_delta,
        "new_fault_tags": new_tags,
        "removed_fault_tags": removed_tags,
        "forbidden_fault_tags_found": forbidden_found,
        "unexpected_fault_tags": unexpected,
        "score_delta": score_delta,
        "fallback_delta": {"baseline": b_fallback, "variant": v_fallback},
        "pose_detection_delta": detection_delta,
        "processing_time_delta": processing_delta,
        "stroke_cycle": {
            "status": cycle_status,
            "confidence": cycles.get("confidence"),
            "regularity": cycles.get("cycle_regularity"),
            "cycle_count": cycle_count if _is_number(cycle_count) else None,
            "expected_status": expected_cycle_status,
        },
        "stroke_rate": rate,
    }


def summarize(clip_verdicts: List[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {"PASS": 0, "NEEDS_REVIEW": 0, "FAIL": 0}
    per_stroke: Dict[str, Dict[str, int]] = {}
    for cv in clip_verdicts:
        verdict = cv.get("verdict", "PASS")
        counts[verdict] = counts.get(verdict, 0) + 1
        stroke = cv.get("stroke") or "Unknown"
        bucket = per_stroke.setdefault(stroke, {"PASS": 0, "NEEDS_REVIEW": 0, "FAIL": 0})
        bucket[verdict] = bucket.get(verdict, 0) + 1
    overall = "FAIL" if counts["FAIL"] else ("NEEDS_REVIEW" if counts["NEEDS_REVIEW"] else "PASS")
    return {
        "overall_verdict": overall,
        "counts": counts,
        "per_stroke": per_stroke,
        "clip_count": len(clip_verdicts),
    }


def build_flag_verdict(
    variant_name: str,
    baseline_by_clip: Mapping[str, Mapping[str, Any]],
    variant_by_clip: Mapping[str, Mapping[str, Any]],
    labels_by_clip: Optional[Mapping[str, Mapping[str, Any]]] = None,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Verdict for one flag variant across all clips that have a baseline row."""
    labels_by_clip = labels_by_clip or {}
    clip_verdicts: List[Dict[str, Any]] = []
    for clip_name, variant_row in variant_by_clip.items():
        baseline_row = baseline_by_clip.get(clip_name)
        if baseline_row is None:
            continue
        clip_verdicts.append(
            compare_clip(baseline_row, variant_row, labels_by_clip.get(clip_name), thresholds)
        )
    return {
        "variant": variant_name,
        "summary": summarize(clip_verdicts),
        "clips": clip_verdicts,
    }
