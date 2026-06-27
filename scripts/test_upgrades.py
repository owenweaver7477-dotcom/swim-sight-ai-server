"""Run the AI worker's safe local upgrade checks with one summary."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DEPENDENCIES = {
    "cv2": "opencv-python-headless",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "pydantic": "pydantic",
}

SAFE_TESTS: Sequence[Tuple[str, Sequence[str]]] = (
    ("Fixture validation", ("scripts/validate_contract_fixtures.py",)),
    ("Worker contract", ("scripts/test_worker_contract.py",)),
    ("Drag integration", ("scripts/test_drag_integration.py",)),
    ("Pose postprocess", ("scripts/test_pose_postprocess.py",)),
    ("Robust findings", ("scripts/test_findings_robust.py",)),
    ("Temporal metrics", ("scripts/test_temporal_metrics.py",)),
    ("Labelled evaluation", ("scripts/test_labelled_evaluation.py",)),
    ("Durable queue configuration", ("scripts/test_durable_queue.py",)),
    ("Worker health", ("scripts/test_worker_health.py",)),
    ("Job timeout", ("scripts/test_job_timeout.py",)),
    ("Job cancellation", ("scripts/test_job_cancellation.py",)),
    ("Failure callback safety", ("scripts/test_failure_callback_safety.py",)),
    ("Report output selection", ("scripts/test_report_outputs.py",)),
    ("Video probe callbacks", ("scripts/test_video_probe.py",)),
    ("Pose 2D engine", ("scripts/test_pose_2d_engine.py",)),
    ("Pose 3D lifter", ("scripts/test_pose_3d_lifter.py",)),
    ("Video storage adapter", ("scripts/test_video_storage_adapter.py",)),
    ("Worker storage access", ("scripts/test_worker_storage_access.py",)),
)


def missing_dependencies() -> List[str]:
    return [
        package
        for module, package in REQUIRED_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]


def compile_command() -> List[str]:
    files = [ROOT / "main.py"]
    files.extend(sorted((ROOT / "app").glob("*.py")))
    files.extend(sorted((ROOT / "scripts").glob("*.py")))
    return [sys.executable, "-m", "py_compile", *(str(path) for path in files)]


def run_check(label: str, command: Sequence[str]) -> bool:
    print(f"\n{'=' * 72}\nRUNNING: {label}\n{'=' * 72}", flush=True)
    completed = subprocess.run(
        [sys.executable, *command] if command and command[0].endswith(".py") else list(command),
        cwd=ROOT,
        check=False,
    )
    passed = completed.returncode == 0
    print(f"RESULT: {'PASS' if passed else 'FAIL'} - {label}", flush=True)
    return passed


def main() -> int:
    missing = missing_dependencies()
    if missing:
        print("Cannot run upgrade tests because required dependencies are missing:")
        for package in missing:
            print(f"  - {package}")
        print("Install the worker dependencies with: python3 -m pip install -r requirements.txt")
        return 1

    results: List[Tuple[str, bool]] = []
    for label, command in SAFE_TESTS:
        results.append((label, run_check(label, command)))

    results.append(("Python compile", run_check("Python compile", compile_command())))

    print(f"\n{'=' * 72}\nUPGRADE TEST SUMMARY\n{'=' * 72}")
    for label, passed in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")

    failed = [label for label, passed in results if not passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("Failed checks: " + ", ".join(failed))
        return 1

    print("All safe AI worker upgrade checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
