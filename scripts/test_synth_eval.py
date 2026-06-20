"""Footage-free logic tests for scripts/synth_eval.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.synth_eval import build_pose_results, evaluate  # noqa: E402


results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def run_fault(fault):
    return evaluate(build_pose_results(fault=fault), stroke="Freestyle", fps=30.0)


previous_smoothing = os.environ.get("ENABLE_POSE_SMOOTHING")
previous_robust = os.environ.get("ROBUST_FINDINGS")
try:
    os.environ["ENABLE_POSE_SMOOTHING"] = "false"
    os.environ["ROBUST_FINDINGS"] = "false"

    none_result = run_fault("none")
    check("none produces zero findings", none_result["finding_count"] == 0,
          str(none_result["finding_titles"]))

    hip_result = run_fault("hip_drop")
    hip_titles = " ".join(hip_result["finding_titles"]).lower()
    check("hip_drop produces a body-line finding",
          hip_result["finding_count"] > 0 and "body line" in hip_titles,
          str(hip_result["finding_titles"]))

    head_result = run_fault("head_lift")
    head_titles = " ".join(head_result["finding_titles"]).lower()
    check("head_lift produces a breathing/head-line finding",
          head_result["finding_count"] > 0
          and ("breath" in head_titles or "head" in head_titles),
          str(head_result["finding_titles"]))

    elbow_result = run_fault("dropped_elbow")
    elbow_titles = " ".join(elbow_result["finding_titles"]).lower()
    check("dropped_elbow produces a catch/elbow finding",
          elbow_result["finding_count"] > 0
          and ("catch" in elbow_titles or "elbow" in elbow_titles),
          str(elbow_result["finding_titles"]))

    noisy_pose = build_pose_results(fault="none", inject_noise=True)
    smoothing_off = evaluate(noisy_pose, stroke="Freestyle", fps=30.0)
    os.environ["ENABLE_POSE_SMOOTHING"] = "true"
    smoothing_on = evaluate(noisy_pose, stroke="Freestyle", fps=30.0)
    off_signal = smoothing_off["pose_signal"]
    on_signal = smoothing_on["pose_signal"]
    check("ENABLE_POSE_SMOOTHING changes the synthetic result", off_signal != on_signal,
          f"off={off_signal}, on={on_signal}")
    check("smoothing fills short landmark gaps",
          on_signal["frames_with_left_hip"] > off_signal["frames_with_left_hip"],
          f"off={off_signal['frames_with_left_hip']}, on={on_signal['frames_with_left_hip']}")
    check("smoothing reduces injected nose outliers",
          on_signal["max_nose_x_step"] < off_signal["max_nose_x_step"],
          f"off={off_signal['max_nose_x_step']}, on={on_signal['max_nose_x_step']}")

    check("synthetic analysis imports without MediaPipe", "mediapipe" not in sys.modules)
finally:
    if previous_smoothing is None:
        os.environ.pop("ENABLE_POSE_SMOOTHING", None)
    else:
        os.environ["ENABLE_POSE_SMOOTHING"] = previous_smoothing
    if previous_robust is None:
        os.environ.pop("ROBUST_FINDINGS", None)
    else:
        os.environ["ROBUST_FINDINGS"] = previous_robust

print("\n" + "=" * 50)
failed = results.count(False)
print(f"{results.count(True)}/{len(results)} passed",
      "ALL PASS" if failed == 0 else f"{failed} FAILED")
sys.exit(1 if failed else 0)
