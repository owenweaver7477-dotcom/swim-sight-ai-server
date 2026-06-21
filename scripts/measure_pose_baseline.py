#!/usr/bin/env python3
"""Measure pose backends against one labelled SwimXYZ sequence.

SwimXYZ is CC-BY-4.0. Cite Fiche et al., "SwimXYZ: A large-scale
dataset of synthetic swimming motions and videos", ACM MIG 2023.

This local evaluation tool does not download data or alter worker defaults.
Prepared SwimXYZ labels must use image coordinates (top-left origin, y-down).
Use ``swimxyz_labels_to_npy.py --flip-y`` when converting raw Unity labels.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_backends import pose_backend_name, run_pose_estimation_backend  # noqa: E402
from app.swimxyz_adapter import joints_to_pose_results, keypoint_errors  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_METRICS = (
    ("mean_error", "Mean error", "lower"),
    ("median_error", "Median error", "lower"),
    ("pck_0.05", "PCK@0.05", "higher"),
    ("recall", "Recall", "higher"),
)


def load_array(path: Path, key: Optional[str] = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            selected = key or (archive.files[0] if len(archive.files) == 1 else None)
            if not selected or selected not in archive.files:
                raise ValueError(
                    f"NPZ contains {archive.files}; select one with --joints-key/--visibility-key."
                )
            return np.asarray(archive[selected])
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            selected = key or "joints"
            if selected not in payload:
                raise ValueError(f"JSON does not contain key {selected!r}.")
            payload = payload[selected]
        return np.asarray(payload)
    raise ValueError("Ground-truth arrays must be .npy, .npz, or .json files.")


def load_frames(frames_dir: Path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Reading image frames requires opencv-python-headless.") from exc

    paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    frames = []
    for frame_idx, path in enumerate(paths):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise ValueError(f"Could not read matching frame {path.name}.")
        frames.append((frame_idx, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    return frames


def resolve_input_paths(args) -> Tuple[Path, Path]:
    if args.sequence_dir:
        if args.joints or args.frames_dir:
            raise ValueError("Use either --sequence-dir or --joints with --frames-dir, not both.")
        sequence_dir = Path(args.sequence_dir).expanduser()
        return sequence_dir / "seq_joints.npy", sequence_dir / "frames"
    if not args.joints or not args.frames_dir:
        raise ValueError("Provide --joints and --frames-dir, or use --sequence-dir.")
    return Path(args.joints).expanduser(), Path(args.frames_dir).expanduser()


def parse_compare_backends(value: Optional[str]) -> Optional[Tuple[str, str]]:
    if not value:
        return None
    names = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if len(names) != 2:
        raise ValueError("--compare requires exactly two comma-separated backends, e.g. mediapipe,onnx.")
    if names[0] == names[1]:
        raise ValueError("--compare requires two different backends.")
    return names[0], names[1]


def validate_backend_configuration(backend: str) -> None:
    if backend not in {"mediapipe", "onnx"}:
        hint = " Did you mean 'onnx'?" if backend == "onxx" else ""
        raise ValueError(
            f"Unknown pose backend {backend!r}; expected 'mediapipe' or 'onnx'.{hint}"
        )
    if backend == "onnx":
        model_path = os.getenv("POSE_ONNX_PATH", "").strip()
        if not model_path:
            raise ValueError(
                "The ONNX comparison backend is unavailable: set POSE_ONNX_PATH "
                "to a readable exported model before using --compare mediapipe,onnx."
            )
        if not Path(model_path).expanduser().is_file():
            raise ValueError(
                f"The ONNX comparison backend is unavailable: POSE_ONNX_PATH does not "
                f"reference a readable file ({model_path})."
            )


def comparison_delta(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
    """Return backend-B minus backend-A deltas with metric-aware direction."""
    output: Dict[str, Any] = {}
    for key, _label, preferred in REPORT_METRICS:
        first_value = first.get(key)
        second_value = second.get(key)
        if first_value is None or second_value is None:
            output[key] = {"delta": None, "result": "unavailable"}
            continue
        delta = round(float(second_value) - float(first_value), 4)
        if abs(delta) < 0.00005:
            result = "unchanged"
        elif (preferred == "lower" and delta < 0) or (preferred == "higher" and delta > 0):
            result = "improved"
        else:
            result = "worsened"
        output[key] = {"delta": delta, "result": result}
    return output


def evaluate_backend(frames, truth, backend: str) -> Dict[str, Any]:
    started = time.perf_counter()
    prediction = run_pose_estimation_backend(frames, env={"POSE_BACKEND": backend})
    metrics = keypoint_errors(prediction, truth)
    return {
        "backend": backend,
        "frames_compared": len(truth),
        "processing_seconds": round(time.perf_counter() - started, 3),
        **metrics,
    }


def _safe_report_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return cleaned or "sequence"


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Swim Sight 3D Pose Baseline Evaluation",
        "",
        f"Generated: {report['generated_at']}",
        f"Dataset / clip: `{report['dataset_label']}`",
        f"Frames available for comparison: {report['frames_compared']}",
        f"Image size: {report['image_size']['width']}×{report['image_size']['height']}",
        f"FPS supplied: {report['fps']}",
        "",
    ]

    if report["mode"] == "compare":
        first_name, second_name = report["backend_order"]
        lines.extend([
            "## Backend comparison",
            "",
            f"Backend A: `{first_name}`  ",
            f"Backend B: `{second_name}`",
            "",
            "| Metric | Backend A | Backend B | Delta (B − A) | Result |",
            "| --- | ---: | ---: | ---: | --- |",
        ])
        first = report["results"][first_name]
        second = report["results"][second_name]
        for key, label, _preferred in REPORT_METRICS:
            delta = report["comparison"][key]
            lines.append(
                f"| {label} | {_format_metric(first.get(key))} | "
                f"{_format_metric(second.get(key))} | {_format_metric(delta['delta'])} | "
                f"{delta['result']} |"
            )
        lines.append("")

    for backend in report["backend_order"]:
        result = report["results"][backend]
        lines.extend([
            f"## `{backend}` baseline",
            "",
            f"Frames compared: {result['frames_compared']}  ",
            f"Matched keypoints: {result['matched_keypoints']}  ",
            f"Ground-truth keypoints: {result['truth_keypoints']}  ",
            f"Processing time: {result['processing_seconds']:.3f} seconds",
            "",
            "| Metric | Result |",
            "| --- | ---: |",
        ])
        for key, label, _preferred in REPORT_METRICS:
            lines.append(f"| {label} | {_format_metric(result.get(key))} |")
        lines.extend([
            "",
            "### Per-joint metrics",
            "",
            "| Joint | Truth | Matched | Mean error | Median error | PCK@0.05 | Recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for joint, joint_metrics in result.get("per_joint", {}).items():
            lines.append(
                f"| {joint} | {joint_metrics['truth_keypoints']} | "
                f"{joint_metrics['matched_keypoints']} | "
                f"{_format_metric(joint_metrics['mean_error'])} | "
                f"{_format_metric(joint_metrics['median_error'])} | "
                f"{_format_metric(joint_metrics['pck_0.05'])} | "
                f"{_format_metric(joint_metrics['recall'])} |"
            )
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "- This is an evaluation baseline, not a product claim.",
        "- This number is the floor future swim-specific pose backends must beat.",
        "- Coach approval remains required for product findings.",
        "- SwimXYZ is synthetic labelled footage; representative consented real-world evaluation is still required.",
        "",
        "## Coordinate convention",
        "",
        "SwimXYZ source labels use Unity screen space (bottom-left origin, y-up). "
        "The prepared labels used here must already be converted to image space "
        "with `image_y = image_height - label_y`. Do not flip an already prepared array again.",
        "",
    ])
    return "\n".join(lines)


def write_reports(report: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
    backend_label = "-vs-".join(report["backend_order"])
    stem = f"pose_baseline_{_safe_report_name(report['dataset_label'])}_{backend_label}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def concise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "backend",
            "frames_compared",
            "matched_keypoints",
            "truth_keypoints",
            "mean_error",
            "median_error",
            "pck_0.05",
            "recall",
            "processing_seconds",
        )
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pose backends on a local SwimXYZ sequence.")
    parser.add_argument("--sequence-dir", help="Prepared folder containing seq_joints.npy and frames/")
    parser.add_argument("--joints", help="Ground-truth .npy/.npz/.json array")
    parser.add_argument("--joints-key", default=None, help="Array key for .npz or object JSON")
    parser.add_argument("--visibility", default=None, help="Optional visibility .npy/.npz/.json")
    parser.add_argument("--visibility-key", default=None)
    parser.add_argument("--frames-dir", help="Directory of matching extracted frames")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--label", help="Dataset/clip label used in reports")
    parser.add_argument("--compare", help="Two comma-separated backends, e.g. mediapipe,onnx")
    parser.add_argument("--output-dir", default=str(ROOT / "baseline_reports"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        joints_path, frames_dir = resolve_input_paths(args)
        compare_backends = parse_compare_backends(args.compare)
        backends: Sequence[str] = compare_backends or (pose_backend_name(),)
        for backend in backends:
            validate_backend_configuration(backend)
    except ValueError as exc:
        print(f"Could not prepare baseline inputs: {exc}", file=sys.stderr)
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
        print(f"Could not prepare baseline inputs: {exc}", file=sys.stderr)
        return 2

    if joints.ndim != 3 or joints.shape[1] < 17 or joints.shape[2] < 2:
        print("Ground-truth joints must have shape (frames, >=17, >=2).", file=sys.stderr)
        return 2
    if not frames:
        print("No readable matching image frames were found.", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("FPS must be greater than zero.", file=sys.stderr)
        return 2

    count = min(len(frames), joints.shape[0])
    frames = frames[:count]
    joints = joints[:count]
    if visibility is not None:
        visibility = visibility[:count]
    height, width = frames[0][1].shape[:2]
    frame_indices = [frame_idx for frame_idx, _frame in frames]
    truth = joints_to_pose_results(
        joints,
        image_size=(width, height),
        fps=args.fps,
        frame_indices=frame_indices,
        visibility=visibility,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for backend in backends:
        try:
            results[backend] = evaluate_backend(frames, truth, backend)
        except Exception as exc:
            print(f"Pose backend {backend!r} could not run: {exc}", file=sys.stderr)
            return 1

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    dataset_label = args.label or (Path(args.sequence_dir).name if args.sequence_dir else joints_path.parent.name)
    report: Dict[str, Any] = {
        "generated_at": generated_at,
        "mode": "compare" if compare_backends else "single",
        "dataset_label": dataset_label,
        "fps": args.fps,
        "frames_compared": count,
        "image_size": {"width": width, "height": height},
        "source": {
            "joints": str(joints_path.resolve()),
            "frames_dir": str(frames_dir.resolve()),
            "swimxyz_y_convention": "prepared image-space y-down",
        },
        "backend_order": list(backends),
        "results": results,
    }
    if compare_backends:
        report["comparison"] = comparison_delta(results[backends[0]], results[backends[1]])

    try:
        json_path, markdown_path = write_reports(report, Path(args.output_dir).expanduser())
    except OSError as exc:
        print(f"Could not write baseline reports: {exc}", file=sys.stderr)
        return 2

    if compare_backends:
        summary: Dict[str, Any] = {
            "mode": "compare",
            "dataset_label": dataset_label,
            "backends": {name: concise_result(result) for name, result in results.items()},
            "comparison": report["comparison"],
        }
    else:
        summary = concise_result(results[backends[0]])
    summary["report_json"] = str(json_path)
    summary["report_markdown"] = str(markdown_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
