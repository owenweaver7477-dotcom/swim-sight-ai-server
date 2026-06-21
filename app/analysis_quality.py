"""Internal safety QA for worker analysis and coach-feedback payloads.

The summary is a local pilot gate, not a performance claim. It never mutates
the payload and intentionally treats public-report safety more strictly than
internal coach-review safety.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from app.coach_feedback import validate_feedback_privacy


MIN_REAL_POSE_FRAMES = 10
MIN_REAL_POSE_RATIO = 0.30

_PRIVATE_KEYS = {
    "signed_video_url",
    "signed_url",
    "video_url",
    "raw_video_url",
    "file_path",
    "storage_path",
    "private_path",
    "landmarks",
    "raw_landmarks",
    "pose_results",
    "raw_pose",
    "frames",
    "raw_frames",
    "frame_data",
    "height",
    "height_cm",
    "height_m",
    "mass",
    "mass_kg",
    "swimmer_name",
    "swimmer_id",
    "athlete_name",
    "guardian_name",
    "guardian_email",
    "guardian_info",
    "coach_name",
    "coach_note",
    "coach_notes",
}
_PUBLIC_INTERNAL_KEYS = {
    "job_id",
    "server_job_id",
    "video_upload_id",
    "stage_history",
    "processing_telemetry",
    "temporal_metrics",
    "phase_analysis",
    "phase_context",
    "estimated_drag",
    "calibration",
    "detected_keypoints_count",
    "detected_pose_frames",
    "detection_ratio",
    "pose_reliability",
    "processing_tier",
}
_UNSAFE_WORDING = {
    "guaranteed": re.compile(r"\bguaranteed\b", re.IGNORECASE),
    "perfect": re.compile(r"\bperfect(?:ly)?\b", re.IGNORECASE),
    "fully_accurate": re.compile(r"\bfully accurate\b", re.IGNORECASE),
    "measured_drag": re.compile(r"\bmeasured drag\b|\bdrag (?:was |is )?measured\b", re.IGNORECASE),
    "exact_biomechanics": re.compile(r"\bexact biomechanics\b", re.IGNORECASE),
    "true_3d": re.compile(r"\btrue 3d\b", re.IGNORECASE),
    "validated_model": re.compile(r"\bvalidated (?:model|biomechanics|hydrodynamics)\b", re.IGNORECASE),
    "coach_replacement": re.compile(r"\b(?:replaces?|replacement for) (?:the )?coach\b", re.IGNORECASE),
    "automatic_diagnosis": re.compile(r"\bautomatic (?:diagnosis|coaching)\b", re.IGNORECASE),
}
_PRIVATE_VALUE_PATTERNS = (
    re.compile(r"https?://[^\s]*(?:token=|signature=|signed)", re.IGNORECASE),
    re.compile(r"(?:^|\s)/(?:Users|home|tmp|var|private|synthetic)/[^\s]*", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\[^\s]+"),
)


def _walk(value: Any, path: str = "$") -> Iterable[Tuple[str, Any, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield child_path, key, child
            yield from _walk(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _all_text(value: Any) -> Iterable[Tuple[str, str]]:
    if isinstance(value, Mapping):
        for path, _, child in _walk(value):
            if isinstance(child, str):
                yield path, child
    elif isinstance(value, str):
        yield "$", value


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def inspect_analysis_quality(
    payload: Any,
    *,
    target: str = "coach_review",
) -> Dict[str, Any]:
    """Inspect one payload and return a value-safe internal QA summary."""

    if target not in {"coach_review", "public_report"}:
        raise ValueError("target must be 'coach_review' or 'public_report'")

    checks: List[Dict[str, str]] = []
    warnings: List[str] = []
    failures: List[str] = []

    def add(name: str, status: str, message: str) -> None:
        checks.append({"check": name, "status": status, "message": message})
        if status == "warn":
            warnings.append(message)
        elif status == "fail":
            failures.append(message)

    if not isinstance(payload, Mapping):
        add("payload_shape", "fail", "Analysis payload must be a JSON object.")
        return {
            "qa_status": "fail",
            "target": target,
            "checks": checks,
            "warnings": warnings,
            "failures": failures,
            "safe_for_coach_review": False,
            "safe_for_public_report": False,
        }
    add("payload_shape", "pass", "Payload is a JSON object.")

    is_feedback = payload.get("schema_version") == "coach_feedback_v1"
    if is_feedback:
        privacy_valid, privacy_issues = validate_feedback_privacy(payload)
        if privacy_valid:
            add("coach_feedback_privacy", "pass", "Coach feedback record passed privacy validation.")
        else:
            for issue in privacy_issues:
                add("coach_feedback_privacy", "fail", f"Coach feedback privacy issue: {issue}")
    else:
        mode = str(payload.get("analysis_mode") or "").strip().lower()
        real_pose = payload.get("real_pose_detected") is True
        detection_ratio = _number(payload.get("detection_ratio"))
        processed_frames = _number(payload.get("frame_count_processed"))
        findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []

        if mode == "real_pose":
            if not real_pose:
                add("pose_detection", "fail", "real_pose mode is inconsistent with real_pose_detected=false.")
            elif detection_ratio is not None and detection_ratio < MIN_REAL_POSE_RATIO:
                add("pose_detection", "fail", "Pose detection ratio is below the real-pose QA threshold.")
            else:
                add("pose_detection", "pass", "Pose detection is consistent with real-pose review.")
            if processed_frames is None:
                add("processed_frames", "warn", "Processed frame count is missing.")
            elif processed_frames < MIN_REAL_POSE_FRAMES:
                add("processed_frames", "fail", "Too few frames were processed for real-pose review.")
            else:
                add("processed_frames", "pass", "Enough frames were processed for the current quality gate.")
            if findings:
                add("findings", "pass", "AI-assisted draft findings are present.")
            else:
                add("findings", "warn", "Real-pose analysis produced no draft findings.")
        elif mode in {"manual_review", "placeholder"}:
            if findings:
                add("manual_review_findings", "fail", "Manual-review output must not contain AI draft findings.")
            else:
                add("manual_review_findings", "pass", "Manual-review fallback contains no AI draft findings.")
            add("pose_detection", "warn", "Pose evidence was not strong enough; manual coach review is required.")
        else:
            add("analysis_mode", "warn", "Analysis mode is missing or not recognised by the QA tool.")

        if findings:
            payload_text = " ".join(text for _, text in _all_text(payload)).lower()
            explicitly_reviewed = all(
                finding.get("coach_review_required") is True
                for finding in findings
                if isinstance(finding, Mapping)
            )
            framed = explicitly_reviewed or any(
                phrase in payload_text
                for phrase in ("coach review", "coach-reviewed", "coach-approved", "draft finding")
            )
            if framed:
                add("coach_review_framing", "pass", "Findings are framed as coach-review drafts.")
            else:
                add("coach_review_framing", "fail", "Findings are not clearly framed for coach review.")

    private_paths: List[str] = []
    internal_public_paths: List[str] = []
    for path, key, value in _walk(payload):
        normalised_key = str(key).strip().lower()
        if normalised_key in _PRIVATE_KEYS:
            # Empty legacy URL/path placeholders carry no private value, but raw
            # pose/frame/identity fields are unsafe by presence.
            if value not in (None, "", [], {}) or normalised_key in {
                "landmarks", "raw_landmarks", "pose_results", "raw_pose",
                "frames", "raw_frames", "frame_data", "swimmer_name",
                "swimmer_id", "athlete_name", "guardian_name", "guardian_email",
            }:
                private_paths.append(path)
        if normalised_key in _PUBLIC_INTERNAL_KEYS:
            internal_public_paths.append(path)
        if isinstance(value, str) and any(pattern.search(value) for pattern in _PRIVATE_VALUE_PATTERNS):
            private_paths.append(path)

    if private_paths:
        for path in sorted(set(private_paths)):
            add("private_data", "fail", f"Private or raw analysis data detected at {path}.")
    else:
        add("private_data", "pass", "No private URL, path, raw pose, identity, or profile fields detected.")

    unsafe_wording_hits: List[Tuple[str, str]] = []
    for path, text in _all_text(payload):
        for label, pattern in _UNSAFE_WORDING.items():
            if pattern.search(text):
                unsafe_wording_hits.append((path, label))
    if unsafe_wording_hits:
        for path, label in sorted(set(unsafe_wording_hits)):
            add("claim_safety", "fail", f"Unsafe certainty wording '{label}' detected at {path}.")
    else:
        add("claim_safety", "pass", "No unsafe certainty or measurement wording detected.")

    estimated_drag = payload.get("estimated_drag")
    if estimated_drag is not None:
        if not isinstance(estimated_drag, Mapping):
            add("estimated_drag", "fail", "estimated_drag must be an object when present.")
        else:
            drag_label = " ".join(
                str(estimated_drag.get(key, "")) for key in ("label", "basis")
            ).lower()
            if "estimated" not in drag_label:
                add("estimated_drag", "fail", "Experimental drag output is not clearly labelled estimated.")
            elif payload.get("analysis_mode") != "real_pose" or payload.get("real_pose_detected") is not True:
                add("estimated_drag", "fail", "Estimated drag must not appear on manual or weak-pose output.")
            else:
                add("estimated_drag", "warn", "Estimated drag is present as internal experimental context.")

    phase_analysis = payload.get("phase_analysis")
    phase_context = payload.get("phase_context")
    if phase_analysis is not None or phase_context is not None:
        phase_marker = None
        validated = None
        for candidate in (phase_analysis, phase_context):
            if isinstance(candidate, Mapping):
                phase_marker = phase_marker or candidate.get("reference_status") or candidate.get("status_label")
                if "validated" in candidate:
                    validated = candidate.get("validated")
        if phase_marker != "provisional_internal" or validated is True:
            add("phase_analysis", "fail", "Phase analysis is not clearly marked provisional/internal.")
        else:
            add("phase_analysis", "warn", "Provisional internal phase analysis is present for coach review.")

    safe_for_coach_review = not failures

    public_reasons: List[str] = []
    if failures:
        public_reasons.append("payload has QA failures")
    if internal_public_paths:
        public_reasons.append("payload contains internal worker IDs or telemetry")
    if is_feedback:
        public_reasons.append("coach feedback records are internal evaluation data")
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    if findings and not all(
        isinstance(finding, Mapping)
        and (
            finding.get("coach_approved") is True
            or str(finding.get("approval_status", "")).lower() == "approved"
        )
        for finding in findings
    ):
        public_reasons.append("findings are not explicitly coach-approved for public output")
    safe_for_public_report = not public_reasons
    if safe_for_public_report:
        add("public_report_boundary", "pass", "Payload meets the strict public-report boundary.")
    else:
        add(
            "public_report_boundary",
            "warn",
            "Not safe for public report: " + "; ".join(dict.fromkeys(public_reasons)) + ".",
        )

    if target == "public_report" and not safe_for_public_report:
        failures.append("Payload failed the strict public-report QA target.")
        checks.append({
            "check": "target_gate",
            "status": "fail",
            "message": "Payload failed the strict public-report QA target.",
        })

    qa_status = "fail" if failures else "warn" if warnings else "pass"
    return {
        "qa_status": qa_status,
        "target": target,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
        "safe_for_coach_review": safe_for_coach_review,
        "safe_for_public_report": safe_for_public_report,
    }


assess_analysis_quality = inspect_analysis_quality
