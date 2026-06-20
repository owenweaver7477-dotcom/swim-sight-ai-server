"""Pure contract tests for the optional ONNX pose backend."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.pose_onnx as pose_onnx  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


check("module import does not load onnxruntime", "onnxruntime" not in sys.modules)
check("module import does not load cv2", "cv2" not in sys.modules)

simcc_x = np.zeros((1, 17, 384), dtype=float)
simcc_y = np.zeros((1, 17, 512), dtype=float)
simcc_x[:, :, 100] = 0.8
simcc_y[:, :, 200] = 0.6
decoded_points, decoded_scores = pose_onnx._parse_end2end_outputs([simcc_x, simcc_y])
check("raw RTMPose SimCC outputs decode to 17 keypoints", decoded_points.shape == (17, 2))
check("SimCC coordinates use the configured split ratio",
      np.all(decoded_points[:, 0] == 50) and np.all(decoded_points[:, 1] == 100))
check("SimCC x/y confidence becomes a per-keypoint score",
      np.allclose(decoded_scores, 0.7))

previous_path = os.environ.get("POSE_ONNX_PATH")
original_infer = pose_onnx._infer_frame
try:
    os.environ.pop("POSE_ONNX_PATH", None)
    try:
        pose_onnx.run_onnx_pose([])
        missing_path_error = None
    except pose_onnx.PoseOnnxError as exc:
        missing_path_error = str(exc)
    check("missing POSE_ONNX_PATH fails clearly",
          missing_path_error is not None and "POSE_ONNX_PATH" in missing_path_error,
          missing_path_error or "no error")

    with tempfile.NamedTemporaryFile(suffix=".onnx") as model_file:
        os.environ["POSE_ONNX_PATH"] = model_file.name

        def fake_infer(_frame, _model_path):
            keypoints = np.zeros((17, 2), dtype=float)
            keypoints[:, 0] = np.linspace(0.05, 0.95, 17)
            keypoints[:, 1] = 0.5
            scores = np.full(17, 0.9, dtype=float)
            return keypoints, scores

        pose_onnx._infer_frame = fake_infer
        output = pose_onnx.run_onnx_pose([
            (12, np.zeros((20, 40, 3), dtype=np.uint8)),
        ])

    check("fake model returns one frame", len(output) == 1)
    frame = output[0]
    check("frame index is preserved", frame["frame_idx"] == 12)
    check("COCO eyes are omitted through the shared mapping",
          "left_eye" not in frame["landmarks"] and "right_eye" not in frame["landmarks"])
    check("worker core count is 15", frame["keypoint_count"] == 15,
          str(frame["keypoint_count"]))
    check("all 17 visible model points are counted internally",
          frame["landmark_count_total"] == 17)
    check("pose passes the existing detection gate", frame["pose_detected"] is True)
    check("coordinates are normalised",
          all(0.0 <= point["x"] <= 1.0 and 0.0 <= point["y"] <= 1.0
              for point in frame["landmarks"].values()))
    check("visibility is the model score",
          all(point["visibility"] == 0.9 for point in frame["landmarks"].values()))
    check("fake inference path still does not load onnxruntime", "onnxruntime" not in sys.modules)
finally:
    pose_onnx._infer_frame = original_infer
    if previous_path is None:
        os.environ.pop("POSE_ONNX_PATH", None)
    else:
        os.environ["POSE_ONNX_PATH"] = previous_path

print("\n" + "=" * 50)
failed = results.count(False)
print(f"{results.count(True)}/{len(results)} passed",
      "ALL PASS" if failed == 0 else f"{failed} FAILED")
sys.exit(1 if failed else 0)
