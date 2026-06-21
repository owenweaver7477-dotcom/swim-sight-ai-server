"""Internal AI-draft versus coach-decision evaluation metrics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from app.coach_feedback import FeedbackValidationError, validate_feedback_privacy


HIGH_CONFIDENCE_THRESHOLD = 0.75


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _records(value: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise ValueError("Feedback metrics require a record or list of records")


def _is_edited(item: Mapping[str, Any]) -> bool:
    return str(item.get("coach_decision", "")) == "edited" or str(
        item.get("decision_type", "")
    ).endswith("_edited")


def compute_feedback_metrics(records: Any) -> Dict[str, Any]:
    """Compute internal alignment metrics from sanitised feedback records."""

    items: List[Mapping[str, Any]] = []
    for record in _records(records):
        if not isinstance(record, Mapping):
            raise ValueError("Every feedback record must be an object")
        items.extend(item for item in (record.get("items") or []) if isinstance(item, Mapping))

    total = len(items)
    edited_count = sum(_is_edited(item) for item in items)
    approved_count = sum(
        str(item.get("coach_decision", "")) == "approved" and not _is_edited(item)
        for item in items
    )
    rejected_count = sum(str(item.get("coach_decision", "")) == "rejected" for item in items)
    unsafe_count = sum(str(item.get("coach_decision", "")) == "unsafe_to_use" for item in items)
    needs_context_count = sum(
        str(item.get("coach_decision", "")) == "needs_more_context" for item in items
    )

    severity_pairs = [
        item
        for item in items
        if item.get("ai_severity") is not None and item.get("coach_severity") is not None
    ]
    severity_agreements = sum(
        str(item["ai_severity"]).lower() == str(item["coach_severity"]).lower()
        for item in severity_pairs
    )
    phase_available = sum(bool(item.get("phase")) for item in items)
    high_confidence = [
        item
        for item in items
        if isinstance(item.get("confidence"), (int, float))
        and float(item["confidence"]) >= HIGH_CONFIDENCE_THRESHOLD
    ]
    high_confidence_rejected = sum(
        str(item.get("coach_decision", "")) == "rejected" for item in high_confidence
    )

    breakdown: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "total": 0,
        "approved": 0,
        "rejected": 0,
        "edited": 0,
        "unsafe_to_use": 0,
        "needs_more_context": 0,
    })
    for item in items:
        label = str(item.get("finding_type") or item.get("ai_title") or "unknown")
        row = breakdown[label]
        row["total"] += 1
        decision = str(item.get("coach_decision", ""))
        if _is_edited(item):
            row["edited"] += 1
        elif decision == "approved":
            row["approved"] += 1
        elif decision in row:
            row[decision] += 1

    return {
        "total_findings": total,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "edited_count": edited_count,
        "unsafe_count": unsafe_count,
        "needs_more_context_count": needs_context_count,
        "approval_rate": _rate(approved_count, total),
        "rejection_rate": _rate(rejected_count, total),
        "edited_rate": _rate(edited_count, total),
        "severity_agreement_rate": _rate(severity_agreements, len(severity_pairs)),
        "phase_context_available_rate": _rate(phase_available, total),
        "high_confidence_rejection_rate": _rate(
            high_confidence_rejected, len(high_confidence)
        ),
        "high_confidence_finding_count": len(high_confidence),
        "finding_type_breakdown": dict(sorted(breakdown.items())),
    }


def summarise_feedback_records(records: Any) -> Dict[str, Any]:
    record_list = list(_records(records))
    metrics = compute_feedback_metrics(record_list)
    rejected_titles: Counter[str] = Counter()
    high_total: Counter[str] = Counter()
    high_rejected: Counter[str] = Counter()
    edits_by_context: Counter[str] = Counter()

    for record in record_list:
        stroke = str(record.get("stroke_type") or "unknown")
        camera = str(record.get("camera_angle") or "Unknown")
        for item in record.get("items") or []:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("finding_type") or item.get("ai_title") or "unknown")
            decision = str(item.get("coach_decision", ""))
            if decision == "rejected":
                rejected_titles[title] += 1
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)) and confidence >= HIGH_CONFIDENCE_THRESHOLD:
                high_total[title] += 1
                if decision == "rejected":
                    high_rejected[title] += 1
            if _is_edited(item):
                edits_by_context[f"{stroke} | {camera}"] += 1

    high_confidence_rejection = [
        {
            "finding": title,
            "high_confidence_count": high_total[title],
            "rejected_count": count,
            "rejection_rate": _rate(count, high_total[title]),
        }
        for title, count in high_rejected.most_common()
    ]
    return {
        "metrics": metrics,
        "most_rejected_findings": [
            {"finding": title, "rejected_count": count}
            for title, count in rejected_titles.most_common()
        ],
        "high_confidence_rejections": high_confidence_rejection,
        "edits_by_stroke_camera": dict(sorted(edits_by_context.items())),
    }


def load_feedback_jsonl(path: Any) -> List[Dict[str, Any]]:
    """Read privacy-valid sanitised feedback records from JSONL."""

    input_path = Path(path)
    records: List[Dict[str, Any]] = []
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FeedbackValidationError(f"Could not read feedback JSONL: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FeedbackValidationError(
                f"Invalid JSON on feedback line {line_number}: {exc.msg}"
            ) from exc
        valid, issues = validate_feedback_privacy(record)
        if not valid:
            raise FeedbackValidationError(
                f"Privacy-invalid feedback line {line_number}: " + "; ".join(issues)
            )
        records.append(record)
    return records
