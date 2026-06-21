"""Footage-free tests for pose baseline metrics, comparison, and reports."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.swimxyz_adapter import keypoint_errors  # noqa: E402
from scripts.measure_pose_baseline import (  # noqa: E402
    comparison_delta,
    render_markdown_report,
    validate_backend_configuration,
    write_reports,
)


results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def frame(frame_idx, landmarks):
    return {
        "frame_idx": frame_idx,
        "pose_detected": bool(landmarks),
        "keypoint_count": len(landmarks),
        "landmarks": {
            name: {"x": x, "y": y, "visibility": 1.0}
            for name, (x, y) in landmarks.items()
        },
    }


truth = [
    frame(0, {"nose": (0.10, 0.10), "left_hip": (0.20, 0.20)}),
    frame(1, {"nose": (0.50, 0.50), "left_hip": (0.60, 0.60)}),
]
prediction = [
    frame(0, {"nose": (0.10, 0.10), "left_hip": (0.30, 0.20)}),
    frame(1, {"nose": (0.54, 0.50)}),
]
metrics = keypoint_errors(prediction, truth)

check("overall matched-keypoint count", metrics["matched_keypoints"] == 3)
check("overall ground-truth count", metrics["truth_keypoints"] == 4)
check("mean-error maths", metrics["mean_error"] == 0.0467, str(metrics["mean_error"]))
check("median-error maths", metrics["median_error"] == 0.04, str(metrics["median_error"]))
check("PCK@0.05 maths", metrics["pck_0.05"] == 0.6667, str(metrics["pck_0.05"]))
check("missing keypoint reduces recall", metrics["recall"] == 0.75, str(metrics["recall"]))
check("per-joint recall", metrics["per_joint"]["left_hip"]["recall"] == 0.5)
check("per-joint mean error", metrics["per_joint"]["nose"]["mean_error"] == 0.02)
check("per-joint PCK", metrics["per_joint"]["left_hip"]["pck_0.05"] == 0.0)

missing_frame_metrics = keypoint_errors([prediction[0]], truth)
check("missing prediction frame counts against recall", missing_frame_metrics["recall"] == 0.5)

no_matches = keypoint_errors([], truth)
check("no matches returns null error", no_matches["mean_error"] is None)
check("no matches returns zero recall", no_matches["recall"] == 0.0)
check("missing joint remains in per-joint output", "left_hip" in no_matches["per_joint"])

first = {"mean_error": 0.5, "median_error": 0.4, "pck_0.05": 0.1, "recall": 0.3}
second = {"mean_error": 0.4, "median_error": 0.4, "pck_0.05": 0.2, "recall": 0.2}
deltas = comparison_delta(first, second)
check("lower mean error is improved", deltas["mean_error"] == {"delta": -0.1, "result": "improved"})
check("equal median error is unchanged", deltas["median_error"]["result"] == "unchanged")
check("higher PCK is improved", deltas["pck_0.05"]["result"] == "improved")
check("lower recall is worsened", deltas["recall"]["result"] == "worsened")

try:
    validate_backend_configuration("onxx")
    typo_error = ""
except ValueError as exc:
    typo_error = str(exc)
check("backend typo fails with ONNX hint", "Did you mean 'onnx'?" in typo_error)

with patch.dict(os.environ, {"POSE_ONNX_PATH": ""}):
    try:
        validate_backend_configuration("onnx")
        onnx_error = ""
    except ValueError as exc:
        onnx_error = str(exc)
check("unconfigured ONNX fails before inference", "POSE_ONNX_PATH" in onnx_error)

backend_result = {
    "backend": "synthetic",
    "frames_compared": 2,
    "processing_seconds": 0.01,
    **metrics,
}
report = {
    "generated_at": "2026-06-21T00:00:00Z",
    "mode": "single",
    "dataset_label": "synthetic-test",
    "fps": 60.0,
    "frames_compared": 2,
    "image_size": {"width": 200, "height": 100},
    "source": {"joints": "synthetic", "frames_dir": "synthetic"},
    "backend_order": ["synthetic"],
    "results": {"synthetic": backend_result},
}

markdown = render_markdown_report(report)
check("markdown contains per-joint table", "### Per-joint metrics" in markdown)
check("markdown contains baseline disclaimer", "evaluation baseline, not a product claim" in markdown)
check("markdown contains coach-approval framing", "Coach approval remains required" in markdown)

with tempfile.TemporaryDirectory() as temp_dir:
    json_path, markdown_path = write_reports(report, Path(temp_dir))
    saved = json.loads(json_path.read_text())
    check("JSON report is written", json_path.is_file() and saved["dataset_label"] == "synthetic-test")
    check("Markdown report is written", markdown_path.is_file() and "left_hip" in markdown_path.read_text())

check("baseline imports do not load MediaPipe", "mediapipe" not in sys.modules)
check("baseline imports do not load ONNX Runtime", "onnxruntime" not in sys.modules)

print("\n" + "=" * 60)
failed = results.count(False)
print(f"{results.count(True)}/{len(results)} passed", "ALL PASS" if failed == 0 else f"{failed} FAILED")
sys.exit(1 if failed else 0)
