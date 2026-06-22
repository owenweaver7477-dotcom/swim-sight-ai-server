#!/usr/bin/env python3
"""
Inspect a SwimXYZ label/joints file (or a folder, or a video) so you know exactly
how to feed scripts/measure_pose_baseline.py. No data is downloaded.

Point it at:
  - a FOLDER  -> lists files + sizes so you can find the joints file and the video
  - a JOINTS file (.npy/.npz/.json/.pkl) -> shape, PIXEL-vs-NORMALISED verdict, and
    the first frame's joints so you can confirm COCO-17 order
  - a VIDEO   -> resolution + fps + the exact ffmpeg line to extract frames

    python3 scripts/inspect_swimxyz.py /path/to/Freestyle_labels/
    python3 scripts/inspect_swimxyz.py /path/to/seq_joints.npy
    python3 scripts/inspect_swimxyz.py /path/to/seq.mp4
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ARRAY_SUFFIXES = {".npy", ".npz", ".json", ".pkl", ".pickle"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
_JOINT_KEYS = ("joints", "keypoints", "joints2d", "keypoints_2d", "pose", "kpts")


def load_any(path: Path, key=None) -> np.ndarray:
    s = path.suffix.lower()
    if s == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if s == ".npz":
        with np.load(path, allow_pickle=False) as a:
            print("  npz keys:", list(a.files))
            k = key or (a.files[0] if len(a.files) == 1 else None)
            if not k:
                raise ValueError("npz has multiple arrays; pick one with --key")
            return np.asarray(a[k])
    if s == ".json":
        d = json.loads(path.read_text())
        if isinstance(d, dict):
            print("  json keys:", list(d.keys()))
            k = key or next((kk for kk in _JOINT_KEYS if kk in d), None)
            if not k:
                raise ValueError("json is a dict; pick a key with --key")
            d = d[k]
        return np.asarray(d)
    if s in (".pkl", ".pickle"):
        print("  [!] loading a pickle runs code — only for files YOU downloaded and trust.")
        with open(path, "rb") as f:
            d = pickle.load(f)
        if isinstance(d, dict):
            print("  pickle keys:", list(d.keys()))
            k = key or next((kk for kk in _JOINT_KEYS if kk in d), None)
            if not k:
                raise ValueError("pickle is a dict; pick a key with --key")
            d = d[k]
        return np.asarray(d)
    raise ValueError(f"{path.name} is not a recognised array file")


def describe_array(arr: np.ndarray) -> None:
    print(f"  shape: {arr.shape}, dtype: {arr.dtype}")
    a = np.asarray(arr, dtype=float)
    if a.ndim != 3 or a.shape[-1] < 2:
        print("  [!] expected (frames, joints, 2 or 3). If it's (frames, 17*2) or has an"
              " extra batch dim, reshape/select one sequence before measuring.")
        return
    n, j, dim = a.shape
    print(f"  -> {n} frames, {j} joints/frame, {dim} values each")
    xs, ys = a[..., 0], a[..., 1]
    xmax, ymax = float(np.nanmax(xs)), float(np.nanmax(ys))
    print(f"  x range: {float(np.nanmin(xs)):.3f} .. {xmax:.3f}")
    print(f"  y range: {float(np.nanmin(ys)):.3f} .. {ymax:.3f}")
    biggest = max(xmax, ymax)
    if biggest <= 1.5:
        verdict = ("NORMALISED (0..1). measure_pose_baseline assumes PIXELS, so multiply "
                   "x by frame width and y by height before saving, OR scale them in.")
    elif biggest <= 6000:
        verdict = "PIXELS. Feed directly — the frame size does the normalising."
    else:
        verdict = "unusual scale — inspect manually."
    print(f"  COORDINATE VERDICT: {verdict}")
    if j != 17:
        print(f"  [!] {j} joints, not 17 — pick the 17 COCO joints and/or edit COCO17_TO_WORKER.")
    print("  first-frame joints (confirm COCO-17 order: 0 nose, 1/2 eyes, 3/4 ears,")
    print("    5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles):")
    for i in range(min(j, 17)):
        print(f"    [{i:2d}]  x={a[0, i, 0]:8.2f}  y={a[0, i, 1]:8.2f}")


def describe_video(path: Path) -> None:
    try:
        import cv2
    except ImportError:
        print("  (install opencv-python to inspect video)")
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("  could not open video")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  video: {w}x{h}, fps={fps:.2f}, frames={n}")
    print(f"  extract frames:  mkdir frames && ffmpeg -i {path.name} frames/f_%05d.png")
    print(f"  pixel joints should sit within x:0..{w}, y:0..{h}")
    print(f"  then:  python3 scripts/measure_pose_baseline.py --joints JOINTS --frames-dir frames --fps {fps:.0f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Inspect SwimXYZ joints/video before measuring.")
    p.add_argument("path")
    p.add_argument("--key", default=None, help="array key for .npz / dict .json / .pkl")
    args = p.parse_args(argv)
    path = Path(args.path).expanduser()

    if path.is_dir():
        print(f"Folder: {path}")
        for f in sorted(path.iterdir()):
            sz = f.stat().st_size / 1e6 if f.is_file() else 0.0
            kind = ("joints?" if f.suffix.lower() in ARRAY_SUFFIXES
                    else "video" if f.suffix.lower() in VIDEO_SUFFIXES else "")
            print(f"  {f.name:42s} {sz:9.1f} MB  {kind}")
        print("\nNow re-run pointing at the joints file, then at the video.")
        return 0

    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 2

    print(f"Inspecting: {path.name}")
    if path.suffix.lower() in VIDEO_SUFFIXES:
        describe_video(path)
        return 0
    try:
        describe_array(load_any(path, args.key))
    except Exception as exc:
        print(f"  could not load as a joints array: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
