#!/usr/bin/env python3
"""
Convert a SwimXYZ 2D keypoint label file into a clean (frames, 17, 2) pixel-
coordinate .npy in STANDARD COCO-17 order — exactly what scripts/swimxyz_adapter
and scripts/measure_pose_baseline.py expect.

SwimXYZ labels (CC-BY-4.0; cite Fiche et al., ACM MIG 2023) are text files:
  - ';'-separated, COMMA as the decimal point (e.g. "874,80" = 874.80),
  - a header row, then one row per frame,
  - x / y / z per joint (z is dropped),
  - joints in OpenPose order: COCO-18 (18 joints) or BODY_25 (25 joints),
  - 2D coords are in Unity SCREEN space (origin BOTTOM-left, y points UP). To score
    against an image-space detector like MediaPipe (origin top-left, y down) you MUST
    pass --flip-y with the render height (--image-height, default 1080); otherwise the
    ground truth is vertically mirrored and the baseline is meaningless.

Usage:
  python3 scripts/swimxyz_labels_to_npy.py --label-file .../COCO/2D_cam.txt \
      --out joints.npy --flip-y --image-height 1080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# OpenPose name -> column index, for the two layouts SwimXYZ ships.
OPENPOSE_COCO18 = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17,
}
OPENPOSE_BODY25 = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
    "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
    "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19,
    "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24,
}

# Output order = standard COCO-17 (what COCO17_TO_WORKER maps from), using the
# OpenPose joint name for each slot.
COCO17_FROM_OPENPOSE = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
]


def parse_swimxyz_txt(text: str) -> np.ndarray:
    """Return (frames, njoints, 3) from a SwimXYZ 2D/3D label text file."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty label file")
    # Drop the header row if its first field isn't numeric.
    first = lines[0].split(";")[0].replace(",", ".")
    try:
        float(first)
        data_lines = lines
    except ValueError:
        data_lines = lines[1:]

    rows = []
    for ln in data_lines:
        vals = [float(v.replace(",", ".")) for v in ln.split(";") if v != ""]
        rows.append(vals)
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("ragged rows — frames have different column counts")
    if width % 3 != 0:
        raise ValueError(f"row width {width} is not a multiple of 3 (x,y,z per joint)")
    arr = np.asarray(rows, dtype=float)
    return arr.reshape(arr.shape[0], width // 3, 3)


def to_coco17_pixels(joints_xyz: np.ndarray) -> np.ndarray:
    n_joints = joints_xyz.shape[1]
    if n_joints == 18:
        name_to_idx = OPENPOSE_COCO18
    elif n_joints == 25:
        name_to_idx = OPENPOSE_BODY25
    else:
        raise ValueError(
            f"{n_joints} joints — expected 18 (OpenPose COCO) or 25 (BODY_25). "
            "Use the COCO/ or body25/ label file."
        )
    n_frames = joints_xyz.shape[0]
    out = np.zeros((n_frames, 17, 2), dtype=float)
    for slot, name in enumerate(COCO17_FROM_OPENPOSE):
        out[:, slot, :] = joints_xyz[:, name_to_idx[name], :2]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="SwimXYZ 2D label .txt -> COCO-17 .npy (pixels).")
    p.add_argument("--label-file", required=True, help="path to a SwimXYZ COCO/2D_cam.txt")
    p.add_argument("--out", required=True, help="output .npy path")
    p.add_argument("--flip-y", action="store_true",
                   help="flip Unity y-up coords to image y-down (REQUIRED to match MediaPipe)")
    p.add_argument("--image-height", type=int, default=1080,
                   help="render height in px, used by --flip-y (default 1080)")
    args = p.parse_args(argv)

    path = Path(args.label_file).expanduser()
    if not path.is_file():
        print(f"Not found: {path}", file=sys.stderr)
        return 2
    try:
        xyz = parse_swimxyz_txt(path.read_text(encoding="utf-8", errors="replace"))
        coco17 = to_coco17_pixels(xyz)
    except ValueError as exc:
        print(f"Could not convert: {exc}", file=sys.stderr)
        return 2

    if args.flip_y:
        coco17[:, :, 1] = args.image_height - coco17[:, :, 1]

    np.save(args.out, coco17.astype(np.float32))
    space = ("image space (y-down)" if args.flip_y
             else "RAW Unity space (y-UP — pass --flip-y to match MediaPipe)")
    print(f"wrote {args.out}  shape={coco17.shape} (frames, 17 COCO joints, x/y pixels; {space})")
    print(f"  x range {coco17[:, :, 0].min():.1f}..{coco17[:, :, 0].max():.1f}  "
          f"y range {coco17[:, :, 1].min():.1f}..{coco17[:, :, 1].max():.1f}")
    print("  -> feed this to measure_pose_baseline.py --joints, with the matching frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
