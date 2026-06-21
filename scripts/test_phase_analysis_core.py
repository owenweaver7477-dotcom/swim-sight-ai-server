#!/usr/bin/env python3
"""Footage-free checks for the default-off phase analysis layer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.stroke_cycles import analyze_stroke_cycles, phase_analysis_enabled  # noqa: E402
from app.swim_analyzer import analyze_pose_data  # noqa: E402
from app.technique_reference import (  # noqa: E402
    compare_phase_technique,
    load_reference_bands,
    validate_reference_config,
)
from scripts.test_phase_analysis import build_synthetic_pose  # noqa: E402


def _analysis(pose_results, stroke):
    return analyze_pose_data(
        pose_results=pose_results,
        frames=list(range(len(pose_results))),
        fps=60.0,
        total_duration=len(pose_results) / 60.0,
        stroke_type=stroke,
        camera_angle="Side",
        video_upload_id="phase-test",
    )


def run() -> None:
    breast, _ = build_synthetic_pose("breaststroke_clean")
    breast_fault, _ = build_synthetic_pose("breaststroke_fault")
    breast_sparse, _ = build_synthetic_pose("breaststroke_sparse")
    freestyle, _ = build_synthetic_pose("freestyle_clean")

    breast_cycles = analyze_stroke_cycles(breast, 60.0, "Breaststroke")
    assert breast_cycles["summary"]["cycle_count"] >= 2, breast_cycles
    assert {phase["phase"] for phase in breast_cycles["cycles"][0]["phases"]} == {
        "extension", "pull", "recovery", "kick"
    }

    freestyle_cycles = analyze_stroke_cycles(freestyle, 60.0, "Freestyle")
    assert freestyle_cycles["summary"]["cycle_count"] >= 2, freestyle_cycles

    sparse_cycles = analyze_stroke_cycles(breast_sparse, 60.0, "Breaststroke")
    assert sparse_cycles["supported"] is True
    assert sparse_cycles["summary"]["confidence"] < breast_cycles["summary"]["confidence"]

    unsupported = analyze_stroke_cycles(breast, 60.0, "Butterfly")
    assert unsupported["status"] == "unsupported"
    assert unsupported["cycles"] == []
    assert unsupported["summary"]["confidence"] == 0.0

    breast_reference = load_reference_bands("Breaststroke")
    freestyle_reference = load_reference_bands("Freestyle")
    assert breast_reference["status"] == "provisional_internal"
    assert freestyle_reference["validated"] is False

    fault_cycles = analyze_stroke_cycles(breast_fault, 60.0, "Breaststroke")
    fault_context = compare_phase_technique(breast_fault, fault_cycles, breast_reference)
    assert any(
        item["metric"] == "hip_drop" and item["phase"] == "extension"
        for item in fault_context["phase_context"]
    ), fault_context

    bad_config = dict(breast_reference)
    bad_config["status"] = "validated_reference"
    try:
        validate_reference_config(bad_config)
    except ValueError as exc:
        assert "provisional_internal" in str(exc)
    else:
        raise AssertionError("Bad reference config should fail clearly")

    assert phase_analysis_enabled({}) is False
    assert phase_analysis_enabled({"PHASE_ANALYSIS": "true"}) is True

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PHASE_ANALYSIS", None)
        default_output = _analysis(breast, "Breaststroke")
    with patch.dict(os.environ, {"PHASE_ANALYSIS": "false"}):
        explicit_off = _analysis(breast, "Breaststroke")
    assert default_output == explicit_off
    assert "phase_analysis" not in explicit_off
    assert "phase_context" not in explicit_off

    with patch.dict(os.environ, {"PHASE_ANALYSIS": "true"}):
        enabled_output = _analysis(breast_fault, "Breaststroke")
    assert enabled_output["phase_analysis"]["summary"]["cycle_count"] >= 2
    assert "phase_context" in enabled_output
    assert enabled_output["findings"] == _analysis_with_flag_off(breast_fault, "Breaststroke")["findings"]
    for finding in enabled_output["findings"]:
        assert {"finding_title", "observation", "coach_review_required"}.issubset(finding)

    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.stroke_cycles, app.technique_reference; "
                "assert 'mediapipe' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules; print('pure phase imports ok')"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "bad.json"
        config_path.write_text(json.dumps({"status": "provisional_internal"}), encoding="utf-8")
        try:
            validate_reference_config(json.loads(config_path.read_text(encoding="utf-8")))
        except ValueError as exc:
            assert "validated=false" in str(exc)
        else:
            raise AssertionError("Incomplete config should fail validation")

    print("Phase analysis checks passed.")


def _analysis_with_flag_off(pose_results, stroke):
    with patch.dict(os.environ, {"PHASE_ANALYSIS": "false"}):
        return _analysis(pose_results, stroke)


if __name__ == "__main__":
    run()
