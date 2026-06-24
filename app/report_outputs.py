"""Private report-output request planning for the AI worker.

The module is deliberately pure and dependency-free. Missing or legacy output
selection keeps the existing worker behaviour. New structured requests receive
explicit accepted/skipped/completed metadata without exposing athlete inputs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional


CORE_OUTPUTS = {
    "general_fault_scan",
    "stroke_phase_breakdown",
    "body_line_analysis",
    "head_position_analysis",
    "breathing_timing",
    "kick_timing",
    "kick_width",
    "pull_catch_timing",
    "recovery_timing",
    "stroke_rhythm",
    "coach_cue_suggestions",
    "drill_recommendations",
}
ESTIMATE_ONLY_OUTPUTS = {"body_line_drag_risk", "estimated_drag_force"}
KNOWN_UNAVAILABLE_OUTPUTS = {
    "stroke_rate",
    "stroke_count",
    "stroke_cycle_timing",
    "phase_duration",
    "tempo_consistency",
    "breakout_timing",
    "start_turn_timing",
    "distance_per_stroke",
    "estimated_velocity",
    "split_movement_timing",
    "distance_per_cycle",
    "stroke_efficiency_indicators",
    "estimated_propulsive_drag",
    "estimated_frontal_drag_trend",
    "velocity_loss_cycle",
    "intra_cycle_acceleration",
    "kick_propulsion_timing",
    "pull_propulsion_timing",
    "start_setup_review",
    "dive_breakout_sequence",
    "turn_approach",
    "wall_contact_rotation",
    "push_off_line",
    "underwater_phase",
    "first_stroke_timing",
}
ALL_OUTPUTS = CORE_OUTPUTS | ESTIMATE_ONLY_OUTPUTS | KNOWN_UNAVAILABLE_OUTPUTS
TRUTHY = {"1", "true", "yes", "on"}

TECHNICAL_FINDING_TERMS = {
    "body_line_analysis": ("body line", "body_line", "hip", "alignment", "low hips"),
    "head_position_analysis": ("head", "head_line"),
    "breathing_timing": ("breath", "breathing"),
    "kick_timing": ("kick", "leg timing"),
    "kick_width": ("kick width", "wide knee", "knee", "ankle"),
    "pull_catch_timing": ("pull", "catch", "elbow"),
    "recovery_timing": ("recovery",),
    "stroke_rhythm": ("rhythm", "timing"),
}


def _value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _unique_strings(values: Optional[Iterable[Any]]) -> List[str]:
    result: List[str] = []
    for value in values or []:
        if not isinstance(value, str):
            continue
        clean = value.strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def _enabled(env: Mapping[str, str], key: str) -> bool:
    return str(env.get(key, "false")).strip().lower() in TRUTHY


def _skip(output_id: str, reason: str) -> Dict[str, str]:
    return {"id": output_id, "reason": reason}


def build_report_output_plan(request: Any, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    source_env = os.environ if env is None else env
    requested = _unique_strings(_value(request, "selected_report_outputs", []))
    if not requested:
        return {
            "legacy_request": True,
            "requested_outputs": [],
            "accepted_outputs": [],
            "skipped_outputs": [],
            "estimate_only_outputs": [],
        }

    accepted: List[str] = []
    skipped: List[Dict[str, str]] = []
    camera_angle = str(_value(request, "camera_angle", "") or "").lower()
    readiness = _value(request, "athlete_profile_readiness", {}) or {}

    for output_id in requested:
        if output_id not in ALL_OUTPUTS:
            skipped.append(_skip(output_id, "Unsupported report output ID"))
            continue
        if output_id in CORE_OUTPUTS:
            accepted.append(output_id)
            continue
        if output_id == "body_line_drag_risk":
            if "side" not in camera_angle:
                skipped.append(_skip(output_id, "A clear side-view video is required"))
            else:
                accepted.append(output_id)
            continue
        if output_id == "estimated_drag_force":
            missing: List[str] = []
            if not _enabled(source_env, "ENABLE_ESTIMATED_DRAG"):
                missing.append("internal estimated-drag flag is disabled")
            if _value(request, "swimmer_mass_kg") is None or not readiness.get("body_mass_available"):
                missing.append("body mass")
            if _value(request, "swimmer_height_cm") is None or not readiness.get("height_available"):
                missing.append("approximate height")
            if not _value(request, "calibration_available", False) or not readiness.get("calibration_available"):
                missing.append("scale calibration")
            if "side" not in camera_angle:
                missing.append("side-view video")
            if missing:
                skipped.append(_skip(output_id, "Missing requirements: " + ", ".join(missing)))
            else:
                accepted.append(output_id)
            continue

        skipped.append(_skip(output_id, "This report output is not available in the current worker"))

    return {
        "legacy_request": False,
        "requested_outputs": requested,
        "accepted_outputs": accepted,
        "skipped_outputs": skipped,
        "estimate_only_outputs": [item for item in accepted if item in ESTIMATE_ONLY_OUTPUTS],
    }


def _finding_text(finding: Mapping[str, Any]) -> str:
    fields = (
        finding.get("finding_id"),
        finding.get("finding_title"),
        finding.get("finding_name"),
        finding.get("finding_description"),
        finding.get("observation"),
        finding.get("fault_tag"),
        finding.get("stroke_phase"),
    )
    return " ".join(str(value) for value in fields if value).lower()


def filter_findings_for_outputs(payload: Dict[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Limit draft findings only when specific technical output types were selected."""
    if plan.get("legacy_request") or "general_fault_scan" in plan.get("accepted_outputs", []):
        return payload

    accepted = set(plan.get("accepted_outputs", []))
    terms = [
        term
        for output_id, output_terms in TECHNICAL_FINDING_TERMS.items()
        if output_id in accepted
        for term in output_terms
    ]
    if not terms:
        return payload

    filtered = [
        finding for finding in payload.get("findings", [])
        if any(term in _finding_text(finding) for term in terms)
    ]
    payload["findings"] = filtered
    return payload


def attach_report_output_metadata(payload: Dict[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    if plan.get("legacy_request"):
        return payload

    completed: List[str] = []
    skipped = list(plan.get("skipped_outputs", []))
    accepted = list(plan.get("accepted_outputs", []))
    real_pose = payload.get("analysis_mode") == "real_pose" and payload.get("real_pose_detected") is True
    findings = payload.get("findings") or []
    phase_breakdown = payload.get("phase_breakdown") or {}

    for output_id in accepted:
        available = real_pose
        if output_id == "general_fault_scan":
            available = real_pose and bool(findings)
        elif output_id == "stroke_phase_breakdown":
            available = real_pose and bool(phase_breakdown)
        elif output_id in TECHNICAL_FINDING_TERMS:
            available = real_pose and any(
                any(term in _finding_text(finding) for term in TECHNICAL_FINDING_TERMS[output_id])
                for finding in findings
            )
        elif output_id == "coach_cue_suggestions":
            available = real_pose and any(
                finding.get("recommended_correction") or finding.get("correction_cue") or finding.get("cue")
                for finding in findings
            )
        elif output_id == "drill_recommendations":
            available = real_pose and any(
                finding.get("drill") or finding.get("drill_recommendation") or finding.get("recommended_drill")
                for finding in findings
            )
        elif output_id == "estimated_drag_force":
            available = real_pose and bool(payload.get("estimated_drag"))

        if available:
            completed.append(output_id)
        else:
            skipped.append(_skip(output_id, "Reliable evidence was not available for this clip"))

    payload["requested_outputs"] = list(plan.get("requested_outputs", []))
    payload["completed_outputs"] = completed
    payload["skipped_outputs"] = skipped
    payload["estimate_only_outputs"] = [
        output_id for output_id in completed if output_id in ESTIMATE_ONLY_OUTPUTS
    ]
    return payload
