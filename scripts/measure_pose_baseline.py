#!/usr/bin/env python3
"""Measure a pose backend against one labelled SwimXYZ sequence.

SwimXYZ is CC-BY-4.0. Cite Fiche et al., "SwimXYZ: A large-scale
dataset of synthetic swimming motions and videos", ACM MIG 2023.

This script does not download data. It expects local ground-truth joints and
matching image frames, converts truth through ``joints_to_pose_results``, then
scores the configured backend with ``keypoint_errors``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_backends import run_pose_estimation_backend  # noqa: E402
from app.swimxyz_adapter import joints_to_pose_results, keypoint_errors  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        raise RuntimeError(
            "Reading image frames requires opencv-python-headless."
        ) from exc

    paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    frames = []
    for frame_idx, path in enumerate(paths):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise ValueError(f"Could not read matching frame {path.name}.")
        frames.append((frame_idx, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure pose accuracy on a local SwimXYZ sequence.")
    parser.add_argument("--joints", required=True, help="Ground-truth .npy/.npz/.json array")
    parser.add_argument("--joints-key", default=None, help="Array key for .npz or object JSON")
    parser.add_argument("--visibility", default=None, help="Optional visibility .npy/.npz/.json")
    parser.add_argument("--visibility-key", default=None)
    parser.add_argument("--frames-dir", required=True, help="Directory of matching extracted frames")
    parser.add_argument("--fps", type=float, default=30.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    joints_path = Path(args.joints).expanduser()
    frames_dir = Path(args.frames_dir).expanduser()
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

    count = min(len(frames), joints.shape[0])
    frames = frames[:count]
    joints = joints[:count]
    if visibility is not None:
        visibility = visibility[:count]
    height, width = frames[0][1].shape[:2]
    frame_indices = [frame_idx for frame_idx, _ in frames]
    truth = joints_to_pose_results(
        joints,
        image_size=(width, height),
        fps=args.fps,
        frame_indices=frame_indices,
        visibility=visibility,
    )

    try:
        prediction = run_pose_estimation_backend(frames)
    except Exception as exc:
        print(f"Pose backend could not run: {exc}", file=sys.stderr)
        return 1

    result = {
        "backend": os.getenv("POSE_BACKEND", "mediapipe") or "mediapipe",
        "frames_compared": count,
        **keypoint_errors(prediction, truth),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
