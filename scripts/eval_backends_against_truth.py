#!/usr/bin/env python3
"""Compare two pose backends against the same labelled SwimXYZ sequence.

This is local evaluation tooling. It does not change POSE_BACKEND, worker
defaults, API contracts, or callback payloads. SwimXYZ is CC-BY-4.0; cite
Fiche et al., "SwimXYZ", ACM MIG 2023.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_backends import run_pose_estimation_backend  # noqa: E402
from app.swimxyz_adapter import joints_to_pose_results, keypoint_errors  # noqa: E402
from scripts.measure_pose_baseline import (  # noqa: E402
    REPORT_METRICS,
    concise_result,
    load_array,
    load_frames,
    resolve_input_paths,
    validate_backend_configuration,
)

DEFAULT_OUTPUT_DIR = ROOT / "backend_eval_reports"
_AUTO_CV2 = object()
WORKER_SKELETON = (
    ("nose", "left_ear"),
    ("nose", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def _metric_delta(baseline_value: Any, candidate_value: Any,
                  preferred: str) -> Dict[str, Any]:
    if baseline_value is None or candidate_value is None:
        return {"delta": None, "result": "unavailable"}
    delta = round(float(candidate_value) - float(baseline_value), 4)
    if abs(delta) < 0.00005:
        result = "unchanged"
    elif (preferred == "lower" and delta < 0) or (preferred == "higher" and delta > 0):
        result = "improved"
    else:
        result = "worsened"
    return {"delta": delta, "result": result}


def metric_deltas(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Build overall and per-joint candidate-minus-baseline metric deltas."""
    overall = {
        key: _metric_delta(baseline.get(key), candidate.get(key), preferred)
        for key, _label, preferred in REPORT_METRICS
    }
    joint_names = sorted(
        set(baseline.get("per_joint", {})) | set(candidate.get("per_joint", {}))
    )
    per_joint: Dict[str, Any] = {}
    for joint in joint_names:
        baseline_joint = baseline.get("per_joint", {}).get(joint, {})
        candidate_joint = candidate.get("per_joint", {}).get(joint, {})
        per_joint[joint] = {
            key: _metric_delta(
                baseline_joint.get(key), candidate_joint.get(key), preferred
            )
            for key, _label, preferred in REPORT_METRICS
        }
    return {"overall": overall, "per_joint": per_joint}


def validate_comparison_configuration(baseline: str, candidate: str) -> Tuple[str, str]:
    baseline_name = str(baseline or "").strip().lower()
    candidate_name = str(candidate or "").strip().lower()
    if baseline_name == candidate_name:
        raise ValueError("Baseline and candidate must be different backends.")
    validate_backend_configuration(baseline_name)
    validate_backend_configuration(candidate_name)
    return baseline_name, candidate_name


def run_backend(frames, truth, backend: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one explicitly selected backend without mutating global POSE_BACKEND."""
    started = time.perf_counter()
    prediction = run_pose_estimation_backend(frames, env={"POSE_BACKEND": backend})
    metrics = {
        "backend": backend,
        "frames_compared": len(truth),
        "processing_seconds": round(time.perf_counter() - started, 3),
        **keypoint_errors(prediction, truth),
    }
    return prediction, metrics


def _point_pixels(point: Dict[str, Any], width: int, height: int) -> Optional[Tuple[int, int]]:
    try:
        x = float(point["x"])
        y = float(point["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return int(round(x * width)), int(round(y * height))


def _draw_pose(cv2, image, landmarks: Dict[str, Any], color: Tuple[int, int, int],
               label: str, label_y: int) -> None:
    height, width = image.shape[:2]
    pixels = {
        name: _point_pixels(point, width, height)
        for name, point in landmarks.items()
    }
    pixels = {name: point for name, point in pixels.items() if point is not None}
    for first, second in WORKER_SKELETON:
        if first in pixels and second in pixels:
            cv2.line(image, pixels[first], pixels[second], color, 2, cv2.LINE_AA)
    for point in pixels.values():
        cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (16, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def generate_overlay_images(
    frames,
    truth,
    predictions: Dict[str, Sequence[Dict[str, Any]]],
    output_dir: Path,
    frame_limit: int = 5,
    cv2_module=_AUTO_CV2,
) -> Dict[str, Any]:
    """Write GT/prediction overlays, or return a non-fatal warning."""
    if frame_limit <= 0:
        return {"paths": [], "warning": "Overlay generation was disabled."}
    if cv2_module is _AUTO_CV2:
        try:
            import cv2 as cv2_module
        except ImportError:
            return {
                "paths": [],
                "warning": "OpenCV is unavailable; metric reports were produced without overlays.",
            }
    if cv2_module is None:
        return {
            "paths": [],
            "warning": "OpenCV is unavailable; metric reports were produced without overlays.",
        }

    truth_by_frame = {item["frame_idx"]: item.get("landmarks", {}) for item in truth}
    predictions_by_backend = {
        backend: {item["frame_idx"]: item.get("landmarks", {}) for item in items}
        for backend, items in predictions.items()
    }
    written: List[str] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame_rgb in frames[:frame_limit]:
            for backend, backend_frames in predictions_by_backend.items():
                image = cv2_module.cvtColor(frame_rgb, cv2_module.COLOR_RGB2BGR)
                _draw_pose(
                    cv2_module, image, truth_by_frame.get(frame_idx, {}),
                    (40, 200, 40), "GT (green)", 56,
                )
                _draw_pose(
                    cv2_module,
                    image,
                    backend_frames.get(frame_idx, {}),
                    (255, 120, 40),
                    f"Prediction: {backend} (blue)",
                    82,
                )
                cv2_module.putText(
                    image,
                    f"Frame {frame_idx}",
                    (16, 28),
                    cv2_module.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2_module.LINE_AA,
                )
                path = output_dir / f"frame_{frame_idx:05d}_{backend}.jpg"
                if not cv2_module.imwrite(str(path), image):
                    raise OSError(f"OpenCV could not write {path.name}")
                written.append(str(path))
    except Exception as exc:
        # Overlay evidence is valuable but must never suppress completed metrics.
        return {
            "paths": written,
            "warning": f"Overlay generation was incomplete: {exc}",
        }
    return {"paths": written, "warning": None}


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: Dict[str, Any]) -> str:
    baseline_name = report["baseline_backend"]
    candidate_name = report["candidate_backend"]
    baseline = report["results"][baseline_name]
    candidate = report["results"][candidate_name]
    lines = [
        "# Swim Sight 3D Backend Evaluation",
        "",
        f"Generated: {report['generated_at']}",
        f"Dataset / clip: `{report['dataset_label']}`",
        f"Baseline backend: `{baseline_name}`",
        f"Candidate backend: `{candidate_name}`",
        f"Frames compared: {report['frames_compared']}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Baseline | Candidate | Delta (candidate − baseline) | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, label, _preferred in REPORT_METRICS:
        delta = report["deltas"]["overall"][key]
        lines.append(
            f"| {label} | {_format(baseline.get(key))} | {_format(candidate.get(key))} | "
            f"{_format(delta['delta'])} | {delta['result']} |"
        )

    lines.extend([
        "",
        "## Per-joint deltas",
        "",
        "| Joint | Mean error Δ | Result | Median error Δ | Result | PCK@0.05 Δ | Result | Recall Δ | Result |",
        "| --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |",
    ])
    for joint, joint_deltas in report["deltas"]["per_joint"].items():
        lines.append(
            f"| {joint} | {_format(joint_deltas['mean_error']['delta'])} | "
            f"{joint_deltas['mean_error']['result']} | "
            f"{_format(joint_deltas['median_error']['delta'])} | "
            f"{joint_deltas['median_error']['result']} | "
            f"{_format(joint_deltas['pck_0.05']['delta'])} | "
            f"{joint_deltas['pck_0.05']['result']} | "
            f"{_format(joint_deltas['recall']['delta'])} | "
            f"{joint_deltas['recall']['result']} |"
        )

    lines.extend(["", "## Overlay sanity images", ""])
    if report["overlays"]["paths"]:
        lines.extend(f"- `{path}`" for path in report["overlays"]["paths"])
    else:
        lines.append(f"- {report['overlays']['warning'] or 'No overlays were requested.'}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Negative mean/median error deltas are improvements; positive PCK/recall deltas are improvements.",
        "- Numeric improvement is not trusted until overlay images confirm coordinate and landmark alignment.",
        "- This is an internal evaluation, not a public product claim.",
        "- Coach approval remains required for product findings.",
        "- The candidate backend remains disabled unless separately validated and enabled.",
        "",
    ])
    return "\n".join(lines)


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return safe.strip("-").lower() or "sequence"


def write_evaluation_reports(report: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
    stem = (
        f"backend_eval_{_safe_name(report['dataset_label'])}_"
        f"{_safe_name(report['baseline_backend'])}-vs-{_safe_name(report['candidate_backend'])}_{timestamp}"
    )
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare pose backends against labelled SwimXYZ truth.")
    parser.add_argument("--sequence-dir", help="Prepared folder containing seq_joints.npy and frames/")
    parser.add_argument("--joints", help="Ground-truth .npy/.npz/.json array")
    parser.add_argument("--joints-key", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--visibility-key", default=None)
    parser.add_argument("--frames-dir", help="Directory of matching image frames")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--baseline", default="mediapipe")
    parser.add_argument("--candidate", default="onnx")
    parser.add_argument("--overlay-frames", type=int, default=5)
    parser.add_argument("--label", help="Dataset/clip label used in reports")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baseline_name, candidate_name = validate_comparison_configuration(
            args.baseline, args.candidate
        )
        joints_path, frames_dir = resolve_input_paths(args)
        if args.fps <= 0:
            raise ValueError("FPS must be greater than zero.")
        if args.overlay_frames < 0:
            raise ValueError("--overlay-frames must be zero or greater.")
    except ValueError as exc:
        print(f"Could not prepare backend evaluation: {exc}", file=sys.stderr)
        return 2

    if not joints_path.is_file():
        print(f"Ground-truth joints file not found: {joints_path}", file=sys.stderr)
        return 2
    if not frames_dir.is_dir():
        print(f"Matching frames directory not found: {frames_dir}", file=sys.stderr)
        return 2

    try:
        joints = load_array(joints_path, args.joints_key)
        visibility = (
            load_array(Path(args.visibility).expanduser(), args.visibility_key)
            if args.visibility else None
        )
        frames = load_frames(frames_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Could not prepare backend evaluation: {exc}", file=sys.stderr)
        return 2

    if joints.ndim != 3 or joints.shape[1] < 17 or joints.shape[2] < 2:
        print("Ground-truth joints must have shape (frames, >=17, >=2).", file=sys.stderr)
        return 2
    if not frames:
        print("No readable matching image frames were found.", file=sys.stderr)
        return 2

    count = min(len(frames), joints.shape[0])
    frames = frames[:count]
    joints = joints[:count]
    if visibility is not None:
        visibility = visibility[:count]
    height, width = frames[0][1].shape[:2]
    truth = joints_to_pose_results(
        joints,
        image_size=(width, height),
        fps=args.fps,
        frame_indices=[frame_idx for frame_idx, _frame in frames],
        visibility=visibility,
    )

    predictions: Dict[str, List[Dict[str, Any]]] = {}
    results: Dict[str, Dict[str, Any]] = {}
    for backend in (baseline_name, candidate_name):
        try:
            predictions[backend], results[backend] = run_backend(frames, truth, backend)
        except Exception as exc:
            print(f"Pose backend {backend!r} could not run: {exc}", file=sys.stderr)
            return 1

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    dataset_label = args.label or (Path(args.sequence_dir).name if args.sequence_dir else joints_path.parent.name)
    output_dir = Path(args.output_dir).expanduser()
    overlays = generate_overlay_images(
        frames,
        truth,
        predictions,
        output_dir / "overlays" / f"{_safe_name(dataset_label)}_{generated_at.replace(':', '')}",
        frame_limit=args.overlay_frames,
    )
    report = {
        "generated_at": generated_at,
        "dataset_label": dataset_label,
        "frames_compared": count,
        "fps": args.fps,
        "image_size": {"width": width, "height": height},
        "source": {
            "joints": str(joints_path.resolve()),
            "frames_dir": str(frames_dir.resolve()),
            "swimxyz_y_convention": "prepared image-space y-down",
        },
        "baseline_backend": baseline_name,
        "candidate_backend": candidate_name,
        "results": results,
        "deltas": metric_deltas(results[baseline_name], results[candidate_name]),
        "overlays": overlays,
    }
    try:
        json_path, markdown_path = write_evaluation_reports(report, output_dir)
    except OSError as exc:
        print(f"Could not write backend evaluation reports: {exc}", file=sys.stderr)
        return 2

    summary = {
        "baseline": concise_result(results[baseline_name]),
        "candidate": concise_result(results[candidate_name]),
        "deltas": report["deltas"]["overall"],
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
        "overlay_paths": overlays["paths"],
        "overlay_warning": overlays["warning"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
