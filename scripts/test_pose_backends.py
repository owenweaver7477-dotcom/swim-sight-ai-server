"""Pure dispatch tests for app/pose_backends.py."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_backends import pose_backend_name, run_pose_estimation_backend  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


check("pose_backends imports without MediaPipe", "mediapipe" not in sys.modules)
check("pose_backends imports without onnxruntime", "onnxruntime" not in sys.modules)
check("default backend is mediapipe", pose_backend_name({}) == "mediapipe")
check("blank backend falls back to mediapipe", pose_backend_name({"POSE_BACKEND": ""}) == "mediapipe")

fake_mediapipe = types.ModuleType("app.pose_estimator")
fake_mediapipe.run_pose_estimation = lambda frames: [{"backend": "mediapipe", "frames": frames}]
with patch.dict(sys.modules, {"app.pose_estimator": fake_mediapipe}):
    mediapipe_result = run_pose_estimation_backend(["frame"], env={})
check("unset backend delegates unchanged to current MediaPipe function",
      mediapipe_result == [{"backend": "mediapipe", "frames": ["frame"]}])

fake_onnx = types.ModuleType("app.pose_onnx")
fake_onnx.run_onnx_pose = lambda frames: [{"backend": "onnx", "frames": frames}]
with patch.dict(sys.modules, {"app.pose_onnx": fake_onnx}):
    onnx_result = run_pose_estimation_backend(["frame"], env={"POSE_BACKEND": "ONNX"})
check("onnx backend delegates lazily", onnx_result == [{"backend": "onnx", "frames": ["frame"]}])

try:
    run_pose_estimation_backend([], env={"POSE_BACKEND": "unknown"})
    unknown_error = None
except ValueError as exc:
    unknown_error = str(exc)
check("unknown backend fails clearly", unknown_error is not None and "Unknown POSE_BACKEND" in unknown_error)

print("\n" + "=" * 50)
failed = results.count(False)
print(f"{results.count(True)}/{len(results)} passed",
      "ALL PASS" if failed == 0 else f"{failed} FAILED")
sys.exit(1 if failed else 0)
