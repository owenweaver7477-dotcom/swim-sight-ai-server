#!/usr/bin/env python3
"""Offline pilot-readiness checks for the AI worker repository."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
_TRUTHY = {"1", "true", "yes", "on"}
_REQUIRED_IGNORES = (
    "pilot_reports/",
    "qa_reports/",
    "coach_feedback_exports/",
    "baseline_data/",
    "baseline_reports/",
    "backend_eval_reports/",
    "*.onnx",
    "*.pth",
    "*.mp4",
    "*.webm",
)
_REQUIRED_FIXTURES = (
    "fixtures/callback_success.example.json",
    "fixtures/callback_manual_review.example.json",
    "fixtures/qa/analysis_payload_safe.example.json",
    "fixtures/qa/analysis_payload_unsafe.example.json",
    "fixtures/coach_feedback/approved_rejected_edited.example.json",
    "fixtures/coach_feedback/sample_feedback.jsonl",
)


def _check(name: str, status: str, detail: str) -> Dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


def _tracked_files(root: Path) -> List[str]:
    # A temporary stream avoids occasional nested-process pipe stalls on the
    # pilot Mac while remaining fully local and dependency-free.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as output:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=False,
            stdout=output,
            stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            return []
        output.seek(0)
        return [line.strip() for line in output if line.strip()]


def run_readiness_checks(root: Path = ROOT) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []

    test_runner = root / "scripts" / "test_upgrades.py"
    runner_text = test_runner.read_text(encoding="utf-8") if test_runner.is_file() else ""
    checks.append(_check(
        "safe test suite",
        "pass" if "SAFE_TESTS" in runner_text else "fail",
        "scripts/test_upgrades.py is available." if runner_text else "Safe test runner is missing.",
    ))

    gitignore_path = root / ".gitignore"
    ignore_text = gitignore_path.read_text(encoding="utf-8") if gitignore_path.is_file() else ""
    missing_ignores = [pattern for pattern in _REQUIRED_IGNORES if pattern not in ignore_text]
    checks.append(_check(
        "private/generated ignores",
        "pass" if not missing_ignores else "fail",
        "All required private/generated patterns are ignored."
        if not missing_ignores
        else "Missing ignore patterns: " + ", ".join(missing_ignores),
    ))

    missing_fixtures = [path for path in _REQUIRED_FIXTURES if not (root / path).is_file()]
    checks.append(_check(
        "synthetic fixtures",
        "pass" if not missing_fixtures else "fail",
        "Required synthetic QA fixtures exist."
        if not missing_fixtures
        else "Missing fixtures: " + ", ".join(missing_fixtures),
    ))

    tracked = _tracked_files(root)
    blocked_tracked = []
    for path in tracked:
        lower = path.lower()
        if lower.endswith((".onnx", ".pth", ".mp4", ".webm")):
            blocked_tracked.append(path)
        elif path.startswith((
            "pilot_reports/", "qa_reports/", "coach_feedback_exports/",
            "baseline_data/", "backend_eval_reports/",
        )):
            blocked_tracked.append(path)
        elif path.startswith("baseline_reports/") and not path.endswith(".gitkeep"):
            blocked_tracked.append(path)
    checks.append(_check(
        "tracked private artefacts",
        "pass" if not blocked_tracked else "fail",
        "No videos, models, or generated reports are tracked."
        if not blocked_tracked
        else "Tracked private/generated artefacts: " + ", ".join(blocked_tracked),
    ))

    backend_source = (root / "app" / "pose_backends.py").read_text(encoding="utf-8")
    backend_default_safe = 'src.get("POSE_BACKEND", "mediapipe")' in backend_source
    active_backend = os.getenv("POSE_BACKEND", "mediapipe").strip().lower() or "mediapipe"
    checks.append(_check(
        "pose backend default",
        "pass" if backend_default_safe and active_backend == "mediapipe" else "fail",
        "POSE_BACKEND defaults to MediaPipe and is not locally overridden."
        if backend_default_safe and active_backend == "mediapipe"
        else f"Unsafe pose backend configuration detected: {active_backend or 'blank'}.",
    ))

    drag_source = (root / "app" / "pose_worker_integration.py").read_text(encoding="utf-8")
    drag_default_safe = 'source.get(ENABLE_FLAG, "false")' in drag_source
    drag_active = os.getenv("ENABLE_ESTIMATED_DRAG", "false").strip().lower() in _TRUTHY
    checks.append(_check(
        "estimated drag default",
        "pass" if drag_default_safe and not drag_active else "fail",
        "ENABLE_ESTIMATED_DRAG defaults off and is not locally enabled."
        if drag_default_safe and not drag_active
        else "ENABLE_ESTIMATED_DRAG is enabled or its safe default could not be verified.",
    ))

    phase_source = (root / "app" / "stroke_cycles.py").read_text(encoding="utf-8")
    phase_default_safe = 'source.get("PHASE_ANALYSIS", "false")' in phase_source
    phase_active = os.getenv("PHASE_ANALYSIS", "false").strip().lower() in _TRUTHY
    checks.append(_check(
        "phase analysis default",
        "pass" if phase_default_safe and not phase_active else "fail",
        "PHASE_ANALYSIS defaults off and is not locally enabled."
        if phase_default_safe and not phase_active
        else "PHASE_ANALYSIS is enabled or its safe default could not be verified.",
    ))

    models_source = (root / "app" / "models.py").read_text(encoding="utf-8")
    calibration_optional = "calibration_config: Optional[Dict[str, Any]] = None" in models_source
    checks.append(_check(
        "calibration optionality",
        "pass" if calibration_optional else "fail",
        "Known-distance calibration remains optional."
        if calibration_optional
        else "Calibration no longer appears to default to None.",
    ))

    private_path_hits = []
    for path in tracked:
        if path == "scripts/pilot_readiness_check.py":
            continue
        candidate = root / path
        if not candidate.is_file() or candidate.stat().st_size > 2_000_000:
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "/Users/owen_weaver" in text or "/home/owen" in text:
            private_path_hits.append(path)
    checks.append(_check(
        "committed local paths",
        "pass" if not private_path_hits else "warn",
        "No personal local filesystem paths found in tracked files."
        if not private_path_hits
        else "Review local path references in: " + ", ".join(private_path_hits),
    ))
    return checks


def _print_table(checks: Sequence[Dict[str, str]]) -> None:
    print("Swim Sight AI Worker - Local Pilot Readiness")
    print("Internal checks only; coach review remains required.\n")
    width = max(len(item["check"]) for item in checks)
    print(f"{'STATUS':<8}  {'CHECK':<{width}}  DETAIL")
    print(f"{'-' * 8}  {'-' * width}  {'-' * 48}")
    for item in checks:
        print(f"{item['status'].upper():<8}  {item['check']:<{width}}  {item['detail']}")


def main() -> int:
    checks = run_readiness_checks()
    _print_table(checks)
    failures = [item for item in checks if item["status"] == "fail"]
    warnings = [item for item in checks if item["status"] == "warn"]
    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    print(f"\nPilot readiness: {status} ({len(failures)} failures, {len(warnings)} warnings)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
