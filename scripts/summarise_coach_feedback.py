#!/usr/bin/env python3
"""Summarise privacy-safe local coach feedback for internal evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.coach_feedback import FeedbackValidationError  # noqa: E402
from app.feedback_metrics import load_feedback_jsonl, summarise_feedback_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise internal coach feedback JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = load_feedback_jsonl(args.input)
        summary = summarise_feedback_records(records)
    except (FeedbackValidationError, ValueError) as exc:
        print(f"Feedback summary failed safely: {exc}", file=sys.stderr)
        return 2

    print("INTERNAL EVALUATION DATA ONLY - coach judgement remains the source of truth.")
    print(f"Records: {len(records)}")
    print("Summary metrics:")
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    print("Most rejected findings:")
    print(json.dumps(summary["most_rejected_findings"], indent=2))
    print("High-confidence rejections:")
    print(json.dumps(summary["high_confidence_rejections"], indent=2))
    print("Edits by stroke / camera angle:")
    print(json.dumps(summary["edits_by_stroke_camera"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
