#!/usr/bin/env python3
"""Run local internal QA against a worker callback or analysis JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis_quality import inspect_analysis_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a worker payload before pilot review.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--public-report", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"QA input failed safely: {exc}", file=sys.stderr)
        return 2

    target = "public_report" if args.public_report else "coach_review"
    summary = inspect_analysis_quality(payload, target=target)
    print("Internal QA only; coach review remains required.")
    print(f"QA status: {summary['qa_status']}")
    print(f"Warnings: {len(summary['warnings'])}")
    print(f"Failures: {len(summary['failures'])}")
    print(f"Safe for coach review: {summary['safe_for_coach_review']}")
    print(f"Safe for public report: {summary['safe_for_public_report']}")
    print("QA summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["qa_status"] == "fail" and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
