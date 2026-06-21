"""Privacy-safe coach feedback labels for offline worker evaluation.

This module does not train or update any model. It converts coach decisions on
AI-assisted draft findings into local evaluation records while removing video,
identity, anthropometric, and raw-pose data.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "coach_feedback_v1"
DEFAULT_EXPORT_PATH = Path("coach_feedback_exports/feedback.local.jsonl")
ALLOWED_DECISIONS = {
    "approved",
    "rejected",
    "edited",
    "needs_more_context",
    "unsafe_to_use",
}

_UNSAFE_KEYS = {
    "video_upload_id",
    "review_id",
    "swimmer_id",
    "club_id",
    "user_id",
    "coach_id",
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
    "guardian_name",
    "guardian_email",
    "guardian_info",
    "coach_name",
}
_PRIVACY_KEYS = {
    "contains_video_url",
    "contains_raw_landmarks",
    "contains_swimmer_identity",
    "stripped_unsafe_field_count",
}
_UNSAFE_VALUE_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"(?:^|[?&])token=", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?:^|\s)/(?:Users|home|tmp|var|private)/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\"),
    re.compile(r"data:(?:video|image)/", re.IGNORECASE),
)


class FeedbackValidationError(ValueError):
    """Raised when feedback cannot become a trustworthy evaluation label."""


def _unsafe_key(key: Any, path: str) -> bool:
    normalised = str(key).strip().lower()
    if path == "$.privacy" and normalised in _PRIVACY_KEYS:
        return False
    return normalised in _UNSAFE_KEYS


def _unsafe_string(value: str) -> bool:
    if len(value) > 2000:
        return True
    return any(pattern.search(value) for pattern in _UNSAFE_VALUE_PATTERNS)


def detect_unsafe_content(value: Any, path: str = "$") -> List[str]:
    """Return paths containing unsafe keys or URL/path/contact-like values."""

    issues: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _unsafe_key(key, path):
                issues.append(f"{child_path}: unsafe key")
            issues.extend(detect_unsafe_content(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            issues.extend(detect_unsafe_content(child, f"{path}[{index}]"))
    elif isinstance(value, str) and _unsafe_string(value):
        issues.append(f"{path}: unsafe-looking value")
    return issues


def validate_feedback_privacy(record: Any) -> Tuple[bool, List[str]]:
    """Validate a sanitised record before local persistence."""

    if not isinstance(record, Mapping):
        return False, ["$: feedback record must be an object"]
    issues = detect_unsafe_content(record)
    if record.get("schema_version") != SCHEMA_VERSION:
        issues.append("$.schema_version: expected coach_feedback_v1")
    privacy = record.get("privacy")
    if not isinstance(privacy, Mapping):
        issues.append("$.privacy: privacy declaration is required")
    else:
        for key in (
            "contains_video_url",
            "contains_raw_landmarks",
            "contains_swimmer_identity",
        ):
            if privacy.get(key) is not False:
                issues.append(f"$.privacy.{key}: must be false")
    return not issues, issues


def _safe_text(value: Any, *, fallback: Optional[str] = None, limit: int = 300) -> Optional[str]:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or _unsafe_string(text):
        return fallback
    return text[:limit]


def _safe_identifier(value: Any, fallback: str) -> str:
    text = _safe_text(value, fallback=fallback, limit=100) or fallback
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", text):
        return text
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_")
    return cleaned[:100] or fallback


def _normalise_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FeedbackValidationError("finding confidence must be numeric between 0 and 1")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise FeedbackValidationError("finding confidence must be numeric between 0 and 1") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise FeedbackValidationError("finding confidence must be numeric between 0 and 1")
    return round(confidence, 4)


def _evidence_frame_count(value: Any) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return 0
    count = 0
    for frame in value:
        if isinstance(frame, bool):
            continue
        try:
            numeric = float(frame)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric >= 0:
            count += 1
    return count


def _decision_type(
    decision: str,
    ai_severity: Optional[str],
    coach_severity: Optional[str],
    corrected_cue: Optional[str],
) -> str:
    if decision in {"rejected", "needs_more_context", "unsafe_to_use"}:
        return decision
    if coach_severity and coach_severity != ai_severity:
        return "severity_edited"
    if corrected_cue:
        return "cue_edited"
    if decision == "edited":
        return "content_edited"
    return "approved"


def normalize_coach_feedback(payload: Any) -> Dict[str, Any]:
    """Create a privacy-safe evaluation record from app coach-review data."""

    if not isinstance(payload, Mapping):
        raise FeedbackValidationError("Coach feedback input must be a JSON object")
    ai_findings = payload.get("ai_findings")
    decisions = payload.get("coach_decisions")
    if not isinstance(ai_findings, list):
        raise FeedbackValidationError("ai_findings must be a list")
    if not isinstance(decisions, list):
        raise FeedbackValidationError("coach_decisions must be a list")

    raw_issues = detect_unsafe_content(payload)
    decision_map: Dict[str, Mapping[str, Any]] = {}
    for index, decision_item in enumerate(decisions):
        if not isinstance(decision_item, Mapping):
            raise FeedbackValidationError(f"coach_decisions[{index}] must be an object")
        finding_id = str(decision_item.get("finding_id", "")).strip()
        if not finding_id:
            raise FeedbackValidationError(f"coach_decisions[{index}] requires finding_id")
        decision = str(decision_item.get("decision", "")).strip().lower()
        if decision not in ALLOWED_DECISIONS:
            allowed = ", ".join(sorted(ALLOWED_DECISIONS))
            raise FeedbackValidationError(
                f"Unknown coach decision '{decision}'. Allowed decisions: {allowed}"
            )
        if finding_id in decision_map:
            raise FeedbackValidationError(f"Duplicate coach decision for finding_id '{finding_id}'")
        decision_map[finding_id] = decision_item

    findings_by_id: Dict[str, Mapping[str, Any]] = {}
    ordered_ids: List[str] = []
    for index, finding in enumerate(ai_findings):
        if not isinstance(finding, Mapping):
            raise FeedbackValidationError(f"ai_findings[{index}] must be an object")
        finding_id = str(finding.get("finding_id", finding.get("id", ""))).strip()
        if not finding_id:
            raise FeedbackValidationError(f"ai_findings[{index}] requires finding_id")
        if finding_id in findings_by_id:
            raise FeedbackValidationError(f"Duplicate AI finding_id '{finding_id}'")
        findings_by_id[finding_id] = finding
        ordered_ids.append(finding_id)

    unmatched = sorted(set(decision_map) - set(findings_by_id))
    if unmatched:
        raise FeedbackValidationError(
            "Coach decisions reference unknown finding_id values: " + ", ".join(unmatched)
        )

    items: List[Dict[str, Any]] = []
    for index, raw_id in enumerate(ordered_ids):
        decision_item = decision_map.get(raw_id)
        if decision_item is None:
            continue
        finding = findings_by_id[raw_id]
        decision = str(decision_item["decision"]).strip().lower()
        ai_severity = _safe_text(finding.get("severity"), limit=40)
        coach_severity = _safe_text(decision_item.get("coach_severity"), limit=40)
        if ai_severity:
            ai_severity = ai_severity.lower()
        if coach_severity:
            coach_severity = coach_severity.lower()
        corrected_cue = _safe_text(decision_item.get("corrected_cue"), limit=300)

        item: Dict[str, Any] = {
            "finding_id": _safe_identifier(raw_id, f"finding_{index + 1:03d}"),
            "ai_title": _safe_text(
                finding.get("title", finding.get("finding_title")),
                fallback=f"Draft finding {index + 1}",
                limit=180,
            ),
            "finding_type": _safe_text(
                finding.get("category", finding.get("fault_tag")),
                fallback=None,
                limit=100,
            ),
            "ai_severity": ai_severity,
            "coach_decision": decision,
            "coach_severity": coach_severity,
            "decision_type": _decision_type(
                decision,
                ai_severity,
                coach_severity,
                corrected_cue,
            ),
            "phase": _safe_text(finding.get("phase"), limit=80),
            "confidence": _normalise_confidence(finding.get("confidence")),
            "evidence_frame_count": _evidence_frame_count(finding.get("evidence_frames")),
        }
        if corrected_cue:
            item["corrected_cue"] = corrected_cue
        if decision_item.get("coach_note_is_safe") is True:
            safe_note = _safe_text(decision_item.get("coach_note"), limit=300)
            if safe_note:
                item["coach_note_summary"] = safe_note
        items.append(item)

    decision_count = len(items)
    finding_count = len(ai_findings)
    if decision_count == 0:
        review_status = "unreviewed"
    elif decision_count < finding_count:
        review_status = "partially_reviewed"
    else:
        review_status = "reviewed"

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stroke_type": _safe_text(payload.get("stroke_type"), fallback="unknown", limit=60),
        "camera_angle": _safe_text(payload.get("camera_angle"), fallback="Unknown", limit=60),
        "worker_version": _safe_text(payload.get("worker_version"), fallback="unknown", limit=80),
        "review_status": review_status,
        "ai_finding_count": finding_count,
        "coach_decision_count": decision_count,
        "unreviewed_count": max(0, finding_count - decision_count),
        "items": items,
        "privacy": {
            "contains_video_url": False,
            "contains_raw_landmarks": False,
            "contains_swimmer_identity": False,
            "stripped_unsafe_field_count": len(raw_issues),
        },
    }
    valid, privacy_issues = validate_feedback_privacy(record)
    if not valid:
        raise FeedbackValidationError(
            "Sanitised feedback failed privacy validation: " + "; ".join(privacy_issues)
        )
    return record


def append_feedback_record(
    record: Mapping[str, Any],
    output_path: Any = DEFAULT_EXPORT_PATH,
) -> Path:
    """Append one sanitised record as JSONL after a second privacy check."""

    valid, issues = validate_feedback_privacy(record)
    if not valid:
        raise FeedbackValidationError(
            "Refusing to write privacy-invalid feedback: " + "; ".join(issues)
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return path


# British spelling retained as a small convenience for internal scripts/docs.
normalise_coach_feedback = normalize_coach_feedback
