"""Synthetic tests for Phase 2 backend comparison and training preparation."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_backends_against_truth import (  # noqa: E402
    generate_overlay_images,
    metric_deltas,
    render_markdown,
    run_backend,
    validate_comparison_configuration,
    write_evaluation_reports,
)
from scripts.swimxyz_to_mmpose import (  # noqa: E402
    COCO17_NAMES,
    build_coco_document,
    convert_joint_layout,
    flip_unity_y_to_image,
    split_indices,
)


results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def joint_metrics(mean_error, median_error, pck, recall):
    return {
        "matched_keypoints": 5,
        "truth_keypoints": 10,
        "mean_error": mean_error,
        "median_error": median_error,
        "pck_0.05": pck,
        "recall": recall,
    }


baseline = {
    "mean_error": 0.5,
    "median_error": 0.4,
    "pck_0.05": 0.2,
    "recall": 0.3,
    "per_joint": {
        "nose": joint_metrics(0.5, 0.4, 0.2, 0.3),
        "left_hip": joint_metrics(0.2, 0.2, 0.6, 0.8),
    },
}
candidate = {
    "mean_error": 0.4,
    "median_error": 0.4,
    "pck_0.05": 0.25,
    "recall": 0.2,
    "per_joint": {
        "nose": joint_metrics(0.4, 0.4, 0.3, 0.4),
        "left_hip": joint_metrics(0.3, 0.2, 0.5, 0.8),
    },
}
deltas = metric_deltas(baseline, candidate)
check("mean-error improvement", deltas["overall"]["mean_error"] == {"delta": -0.1, "result": "improved"})
check("median unchanged", deltas["overall"]["median_error"]["result"] == "unchanged")
check("PCK improvement", deltas["overall"]["pck_0.05"]["result"] == "improved")
check("recall regression", deltas["overall"]["recall"]["result"] == "worsened")
check("per-joint improvement", deltas["per_joint"]["nose"]["mean_error"]["result"] == "improved")
check("per-joint regression", deltas["per_joint"]["left_hip"]["mean_error"]["result"] == "worsened")

with patch.dict(os.environ, {"POSE_ONNX_PATH": ""}):
    try:
        validate_comparison_configuration("mediapipe", "onnx")
        missing_model_error = ""
    except ValueError as exc:
        missing_model_error = str(exc)
check("missing ONNX path fails clearly", "POSE_ONNX_PATH" in missing_model_error)

try:
    validate_comparison_configuration("mediapipe", "onxx")
    typo_error = ""
except ValueError as exc:
    typo_error = str(exc)
check("onxx typo suggests onnx", "Did you mean 'onnx'?" in typo_error)

truth = [{
    "frame_idx": 0,
    "pose_detected": True,
    "keypoint_count": 1,
    "landmarks": {"nose": {"x": 0.2, "y": 0.3, "visibility": 1.0}},
}]
captured_env = {}


def fake_backend(frames, env=None):
    captured_env.update(env or {})
    return truth


with patch("scripts.eval_backends_against_truth.run_pose_estimation_backend", fake_backend):
    with patch.dict(os.environ, {"POSE_BACKEND": "leave-me-alone"}):
        prediction, isolated_metrics = run_backend([(0, "frame")], truth, "candidate-test")
        global_backend_after = os.environ["POSE_BACKEND"]
check("backend selection passed locally", captured_env == {"POSE_BACKEND": "candidate-test"})
check("global POSE_BACKEND is not changed", global_backend_after == "leave-me-alone")
check("mock backend produces contract metrics", prediction == truth and isolated_metrics["recall"] == 1.0)

overlay_skip = generate_overlay_images([], [], {}, Path("unused"), cv2_module=None)
check("overlay skip is non-fatal without OpenCV", overlay_skip["paths"] == [] and "OpenCV" in overlay_skip["warning"])

report = {
    "generated_at": "2026-06-21T00:00:00Z",
    "dataset_label": "synthetic-phase2",
    "frames_compared": 2,
    "fps": 60.0,
    "image_size": {"width": 200, "height": 100},
    "source": {"joints": "synthetic", "frames_dir": "synthetic"},
    "baseline_backend": "mediapipe",
    "candidate_backend": "onnx",
    "results": {
        "mediapipe": {"backend": "mediapipe", "frames_compared": 2, "processing_seconds": 0.1, **baseline},
        "onnx": {"backend": "onnx", "frames_compared": 2, "processing_seconds": 0.1, **candidate},
    },
    "deltas": deltas,
    "overlays": {"paths": ["overlay-one.jpg"], "warning": None},
}
markdown = render_markdown(report)
check("Markdown explains delta direction", "Negative mean/median error deltas" in markdown)
check("Markdown requires overlay inspection", "overlay images confirm" in markdown)
check("Markdown preserves coach approval", "Coach approval remains required" in markdown)

with tempfile.TemporaryDirectory() as temp_dir:
    json_path, markdown_path = write_evaluation_reports(report, Path(temp_dir))
    saved = json.loads(json_path.read_text())
    check("evaluation JSON report written", saved["deltas"]["overall"]["mean_error"]["result"] == "improved")
    check("evaluation Markdown report written", markdown_path.is_file() and "Per-joint deltas" in markdown_path.read_text())

coco = np.zeros((2, 17, 2), dtype=float)
coco[:, :, 0] = np.arange(17)
check("COCO-17 layout preserved", np.array_equal(convert_joint_layout(coco, "coco17")[:, :, 0], coco[:, :, 0]))

openpose18 = np.zeros((2, 18, 2), dtype=float)
openpose18[:, :, 0] = np.arange(18)
converted_openpose = convert_joint_layout(openpose18, "openpose_coco18")
check("OpenPose COCO-18 converts to 17 joints", converted_openpose.shape == (2, 17, 2))
check("OpenPose nose mapping preserved", np.all(converted_openpose[:, 0, 0] == 0))

openpose25 = np.zeros((2, 25, 2), dtype=float)
check("OpenPose BODY-25 converts to 17 joints", convert_joint_layout(openpose25, "openpose_body25").shape == (2, 17, 2))

try:
    convert_joint_layout(np.zeros((2, 19, 2)), "unsupported")
    layout_error = ""
except ValueError as exc:
    layout_error = str(exc)
check("unsupported layout fails clearly", "Unsupported joint layout" in layout_error)

unity = np.zeros((1, 17, 2), dtype=float)
unity[:, :, 1] = 25.0
flipped = flip_unity_y_to_image(unity, 100)
check("Unity y-up converts to image y-down", np.all(flipped[:, :, 1] == 75.0))

first_split = split_indices(20, 0.2, 42)
second_split = split_indices(20, 0.2, 42)
check("train/validation split is deterministic", first_split == second_split)

try:
    build_coco_document(coco, ["only-one-frame.jpg"], width=100, height=100)
    frame_count_error = ""
except ValueError as exc:
    frame_count_error = str(exc)
check("converter rejects frame-count mismatch", "image count" in frame_count_error)
check("converter import does not load MMPose", "mmpose" not in sys.modules)
check("evaluation import does not load ONNX Runtime", "onnxruntime" not in sys.modules)

print("\n" + "=" * 64)
failed = results.count(False)
print(f"{results.count(True)}/{len(results)} passed", "ALL PASS" if failed == 0 else f"{failed} FAILED")
sys.exit(1 if failed else 0)
