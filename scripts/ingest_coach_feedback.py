#!/usr/bin/env python3
"""Sanitise app coach-review feedback into a local ignored JSONL export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.coach_feedback import (  # noqa: E402
    DEFAULT_EXPORT_PATH,
    FeedbackValidationError,
    append_feedback_record,
    normalize_coach_feedback,
    validate_feedback_privacy,
)
from app.feedback_metrics import compute_feedback_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a privacy-safe local coach-feedback evaluation record."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        record = normalize_coach_feedback(payload)
        privacy_valid, privacy_issues = validate_feedback_privacy(record)
        metrics = compute_feedback_metrics(record)
    except (OSError, json.JSONDecodeError, FeedbackValidationError, ValueError) as exc:
        print(f"Feedback ingestion failed safely: {exc}", file=sys.stderr)
        return 2

    print("Internal evaluation label only; no automatic model learning occurs.")
    print(f"AI findings: {record['ai_finding_count']}")
    print(f"Approved unchanged: {metrics['approved_count']}")
    print(f"Rejected: {metrics['rejected_count']}")
    print(f"Edited: {metrics['edited_count']}")
    print(f"Unsafe to use: {metrics['unsafe_count']}")
    print(f"Needs more context: {metrics['needs_more_context_count']}")
    print(f"Privacy validation: {'PASS' if privacy_valid else 'FAIL'}")
    if privacy_issues:
        print("Privacy issues: " + "; ".join(privacy_issues))

    if args.summary:
        print("Evaluation summary:")
        print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.dry_run:
        print(f"Dry run: nothing written. Intended output: {args.output}")
        return 0

    try:
        written = append_feedback_record(record, args.output)
    except FeedbackValidationError as exc:
        print(f"Feedback write refused safely: {exc}", file=sys.stderr)
        return 2
    print(f"Sanitised feedback written to: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
