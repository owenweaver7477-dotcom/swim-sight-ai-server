#!/usr/bin/env python3
"""Synthetic tests for analysis QA and local pilot readiness."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis_quality import inspect_analysis_quality  # noqa: E402
from scripts.pilot_readiness_check import run_readiness_checks  # noqa: E402


def _load(name: str):
    return json.loads((ROOT / "fixtures" / "qa" / name).read_text(encoding="utf-8"))


def _assert_private_failure(payload, expected_path):
    result = inspect_analysis_quality(payload)
    assert result["qa_status"] == "fail", result
    assert any(expected_path in failure for failure in result["failures"]), result


def run() -> None:
    safe_payload = _load("analysis_payload_safe.example.json")
    unsafe_payload = _load("analysis_payload_unsafe.example.json")

    safe = inspect_analysis_quality(safe_payload)
    assert safe["qa_status"] in {"pass", "warn"}
    assert safe["safe_for_coach_review"] is True
    assert safe["safe_for_public_report"] is False

    unsafe = inspect_analysis_quality(unsafe_payload)
    assert unsafe["qa_status"] == "fail"
    assert unsafe["safe_for_coach_review"] is False
    assert unsafe["safe_for_public_report"] is False

    public_target = inspect_analysis_quality(safe_payload, target="public_report")
    assert public_target["qa_status"] == "fail"
    assert public_target["safe_for_coach_review"] is True
    assert public_target["safe_for_public_report"] is False

    private_cases = (
        ("signed_video_url", "https://redacted.invalid/video?token=synthetic", "signed_video_url"),
        ("file_path", "/synthetic/private/video.mp4", "file_path"),
        ("height_cm", 180, "height_cm"),
        ("mass_kg", 75, "mass_kg"),
        ("landmarks", {"synthetic": {"x": 0.1, "y": 0.2}}, "landmarks"),
        ("frames", [[0.1, 0.2]], "frames"),
    )
    for key, value, expected_path in private_cases:
        payload = copy.deepcopy(safe_payload)
        payload[key] = value
        _assert_private_failure(payload, expected_path)

    certainty = copy.deepcopy(safe_payload)
    certainty["technical_summary"] = "Guaranteed fully accurate correction."
    certainty_result = inspect_analysis_quality(certainty)
    assert certainty_result["qa_status"] == "fail"
    assert any("guaranteed" in failure for failure in certainty_result["failures"])

    measured_drag = copy.deepcopy(safe_payload)
    measured_drag["estimated_drag"] = {"label": "drag", "basis": "measured drag"}
    measured_result = inspect_analysis_quality(measured_drag)
    assert measured_result["qa_status"] == "fail"
    assert any("measured_drag" in failure for failure in measured_result["failures"])

    estimated_drag = copy.deepcopy(safe_payload)
    estimated_drag["estimated_drag"] = {
        "label": "estimated_drag",
        "basis": "estimated internal pilot context -- not measured",
    }
    estimated_result = inspect_analysis_quality(estimated_drag)
    assert estimated_result["safe_for_coach_review"] is True
    assert not estimated_result["failures"]

    phase_marked = copy.deepcopy(safe_payload)
    phase_marked["phase_analysis"] = {
        "reference_status": "provisional_internal",
        "validated": False,
        "cycles": [],
    }
    marked_result = inspect_analysis_quality(phase_marked)
    assert marked_result["safe_for_coach_review"] is True
    assert not marked_result["failures"]

    phase_unmarked = copy.deepcopy(safe_payload)
    phase_unmarked["phase_analysis"] = {"cycles": []}
    unmarked_result = inspect_analysis_quality(phase_unmarked)
    assert unmarked_result["qa_status"] == "fail"
    assert any("provisional/internal" in failure for failure in unmarked_result["failures"])

    feedback_record = json.loads(
        (ROOT / "fixtures" / "coach_feedback" / "sample_feedback.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    feedback_result = inspect_analysis_quality(feedback_record)
    assert feedback_result["safe_for_coach_review"] is True
    assert feedback_result["safe_for_public_report"] is False
    assert not feedback_result["failures"]

    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.analysis_quality; "
                "assert 'mediapipe' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules; print('analysis QA import ok')"
            ),
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert import_check.returncode == 0

    readiness_import = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.pilot_readiness_check; "
                "assert 'mediapipe' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules; "
                "assert 'cv2' not in sys.modules; print('readiness import ok')"
            ),
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert readiness_import.returncode == 0

    readiness = run_readiness_checks(ROOT)
    assert not [item for item in readiness if item["status"] == "fail"], readiness

    cli_fail = subprocess.run(
        [
            sys.executable,
            "scripts/qa_analysis_payload.py",
            "--input",
            "fixtures/qa/analysis_payload_unsafe.example.json",
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert cli_fail.returncode != 0
    cli_allowed = subprocess.run(
        [
            sys.executable,
            "scripts/qa_analysis_payload.py",
            "--input",
            "fixtures/qa/analysis_payload_unsafe.example.json",
            "--allow-fail",
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert cli_allowed.returncode == 0

    print("Analysis QA and pilot readiness checks passed.")


if __name__ == "__main__":
    run()
