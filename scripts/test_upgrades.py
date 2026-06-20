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
    ("Synthetic pose evaluation", ("scripts/test_synth_eval.py",)),
    ("SwimXYZ adapter", ("scripts/test_swimxyz_adapter.py",)),
    ("Pose backend dispatch", ("scripts/test_pose_backends.py",)),
    ("ONNX pose contract", ("scripts/test_pose_onnx.py",)),
    ("SwimXYZ pipeline tools", ("scripts/test_swimxyz_pipeline_tools.py",)),
    ("MMPose conversion self-test", ("scripts/swimxyz_to_mmpose.py", "--self-test")),
    ("Temporal metrics", ("scripts/test_temporal_metrics.py",)),
    ("Labelled evaluation", ("scripts/test_labelled_evaluation.py",)),
    ("Durable queue configuration", ("scripts/test_durable_queue.py",)),
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
