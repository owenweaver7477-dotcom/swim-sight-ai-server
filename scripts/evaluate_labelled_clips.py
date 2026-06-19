"""Evaluate worker findings against a local coach-labelled clip manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_baseline import evaluate_video  # noqa: E402


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    clips = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(clips, list) or not clips:
        raise ValueError("Label manifest must contain a non-empty clips list")
    required = {"file", "stroke", "camera_angle"}
    for index, clip in enumerate(clips):
        missing = required - set(clip)
        if missing:
            raise ValueError(
                f"Clip {index + 1} is missing: {', '.join(sorted(missing))}"
            )
    return clips


def _tags(findings: Iterable[Dict[str, Any]]) -> Set[str]:
    return {
        str(finding.get("fault_tag"))
        for finding in findings
        if finding.get("fault_tag")
    }


def compare_labels(
    label: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    expected = set(label.get("expected_fault_tags") or [])
    forbidden = set(label.get("forbidden_fault_tags") or [])
    found = set(evaluation.get("finding_fault_tags") or [])
    if not found:
        found = _tags(evaluation.get("findings") or [])
    matched = expected & found
    missed = expected - found
    unexpected = found - expected
    forbidden_found = forbidden & found
    expected_mode = label.get("expected_analysis_mode")
    actual_mode = evaluation.get("analysis_mode")
    return {
        "expected_fault_tags": sorted(expected),
        "found_fault_tags": sorted(found),
        "matched_fault_tags": sorted(matched),
        "missed_fault_tags": sorted(missed),
        "unexpected_fault_tags": sorted(unexpected),
        "forbidden_fault_tags_found": sorted(forbidden_found),
        "expected_analysis_mode": expected_mode,
        "analysis_mode_match": expected_mode is None or expected_mode == actual_mode,
        "precision": round(len(matched) / len(found), 4) if found else (1.0 if not expected else 0.0),
        "recall": round(len(matched) / len(expected), 4) if expected else 1.0,
        "passed": not missed
        and not forbidden_found
        and (expected_mode is None or expected_mode == actual_mode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare AI draft findings with local coach-labelled clips."
    )
    parser.add_argument(
        "--labels-file",
        default=str(ROOT / "samples" / "labels.local.json"),
    )
    parser.add_argument("--samples-dir", default=str(ROOT / "samples" / "videos"))
    parser.add_argument("--output-dir", default=str(ROOT / "baseline_reports"))
    args = parser.parse_args()

    labels_path = Path(args.labels_file)
    if not labels_path.exists():
        print(f"No local label manifest found at {labels_path}.")
        print(
            "Copy fixtures/labelled_clip_manifest.example.json to "
            "samples/labels.local.json and reference local sample clips."
        )
        return 0

    clips = load_manifest(labels_path)
    samples_dir = Path(args.samples_dir)
    results = []
    for label in clips:
        video_path = samples_dir / label["file"]
        if not video_path.is_file():
            raise FileNotFoundError(f"Labelled clip not found: {video_path}")
        print(f"Evaluating labelled clip: {video_path.name}", flush=True)
        evaluation = evaluate_video(
            video_path,
            label["stroke"],
            label["camera_angle"],
        )
        comparison = compare_labels(label, evaluation)
        results.append({
            "clip_name": video_path.name,
            "stroke": label["stroke"],
            "camera_angle": label["camera_angle"],
            "coach_notes_present": bool(label.get("coach_notes")),
            "evaluation": evaluation,
            "comparison": comparison,
        })

    created_at = datetime.now(timezone.utc)
    report = {
        "created_at": created_at.isoformat(),
        "clip_count": len(results),
        "passed_clip_count": sum(1 for result in results if result["comparison"]["passed"]),
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"labelled_evaluation_{created_at.strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Labelled evaluation report written locally: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
