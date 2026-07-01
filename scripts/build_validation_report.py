"""Build a footage-safe validation report from a flag-comparison JSON.

Reads the JSON written by scripts/compare_upgrade_flags.py, groups rows by
variant, optionally attaches coach labels from a local manifest, and writes a
verdict report (per flag, per stroke) plus a human-readable summary.

The output contains only derived metrics, fault tags, clip filenames, and
stroke labels — never raw landmarks, video frames, signed URLs, or secrets. A
final safety scan aborts if any unsafe pattern is detected.

Run:  python3 scripts/build_validation_report.py --comparison-file <report.json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validation_report import build_flag_verdict  # noqa: E402

# Report must never contain these (defence in depth; rows are metrics-only).
_UNSAFE_PATTERNS = [
    re.compile(r"token=", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}"),
    re.compile(r"supabase[^\s\"']*?/storage", re.IGNORECASE),
    re.compile(r"/Users/|/home/|/var/folders/|/tmp/", re.IGNORECASE),
    re.compile(r"\blandmarks?\b", re.IGNORECASE),
    re.compile(r"service_role", re.IGNORECASE),
]


def group_by_variant(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in results:
        variant = row.get("variant")
        clip = row.get("clip_name")
        if not variant or not clip:
            continue
        grouped.setdefault(variant, {})[clip] = row
    return grouped


def load_labels(labels_path: Path) -> Dict[str, Dict[str, Any]]:
    if not labels_path or not labels_path.exists():
        return {}
    with labels_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    clips = payload.get("clips") if isinstance(payload, dict) else None
    labels: Dict[str, Dict[str, Any]] = {}
    for clip in clips or []:
        name = clip.get("file")
        if name:
            labels[name] = clip
    return labels


def assert_report_safe(report: Dict[str, Any]) -> None:
    blob = json.dumps(report)
    for pattern in _UNSAFE_PATTERNS:
        match = pattern.search(blob)
        if match:
            raise AssertionError(
                f"Validation report contains an unsafe pattern ({pattern.pattern}); aborting write."
            )


def build_report(results: List[Dict[str, Any]], labels_by_clip: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    grouped = group_by_variant(results)
    baseline = grouped.get("baseline", {})
    flag_verdicts = []
    for variant_name, variant_rows in grouped.items():
        if variant_name == "baseline":
            continue
        flag_verdicts.append(
            build_flag_verdict(variant_name, baseline, variant_rows, labels_by_clip)
        )

    overall = "PASS"
    for fv in flag_verdicts:
        v = fv["summary"]["overall_verdict"]
        if v == "FAIL":
            overall = "FAIL"
            break
        if v == "NEEDS_REVIEW":
            overall = "NEEDS_REVIEW"

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_clip_count": len(baseline),
        "labels_present": bool(labels_by_clip),
        "overall_verdict": overall,
        "flag_verdicts": flag_verdicts,
    }


def print_summary(report: Dict[str, Any]) -> None:
    print(f"\nValidation report — overall: {report['overall_verdict']} "
          f"(baseline clips: {report['baseline_clip_count']}, labels: {report['labels_present']})")
    for fv in report["flag_verdicts"]:
        s = fv["summary"]
        print(f"  {fv['variant']:<28} {s['overall_verdict']:<13} "
              f"pass={s['counts']['PASS']} review={s['counts']['NEEDS_REVIEW']} fail={s['counts']['FAIL']}")
        for clip in fv["clips"]:
            if clip["verdict"] != "PASS":
                reasons = clip["fail_reasons"] + clip["review_reasons"]
                print(f"      - {clip['clip_name']} [{clip['stroke']}] {clip['verdict']}: {', '.join(reasons)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a footage-safe validation report from a flag comparison.")
    parser.add_argument("--comparison-file", required=True, help="JSON written by compare_upgrade_flags.py")
    parser.add_argument("--labels-file", default=str(ROOT / "samples" / "labels.local.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "baseline_reports"))
    args = parser.parse_args()

    comparison_path = Path(args.comparison_file)
    if not comparison_path.exists():
        print(f"Comparison file not found: {comparison_path}")
        return 1
    with comparison_path.open("r", encoding="utf-8") as handle:
        comparison = json.load(handle)
    results = comparison.get("results") if isinstance(comparison, dict) else None
    if not isinstance(results, list) or not results:
        print("Comparison file has no results.")
        return 1

    labels_by_clip = load_labels(Path(args.labels_file))
    report = build_report(results, labels_by_clip)
    assert_report_safe(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"validation_report_{stamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print_summary(report)
    print(f"\nValidation report written locally: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
