#!/usr/bin/env python3
"""Convert one local SwimXYZ stroke sequence to MMPose COCO JSON.

SwimXYZ is CC-BY-4.0. Required attribution: Fiche et al., "SwimXYZ: A
large-scale dataset of synthetic swimming motions and videos", ACM MIG 2023.

No data is downloaded by this script. Use only locally obtained footage and
labels under the SwimXYZ licence terms.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.swimxyz_adapter import COCO17_TO_WORKER  # noqa: E402
from scripts.measure_pose_baseline import IMAGE_SUFFIXES, load_array  # noqa: E402
from scripts.swimxyz_labels_to_npy import (  # noqa: E402
    COCO17_FROM_OPENPOSE,
    OPENPOSE_BODY25,
    OPENPOSE_COCO18,
)

COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_SKELETON = (
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12),
    (7, 13), (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),
    (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
)
SUPPORTED_LAYOUTS = ("coco17", "openpose_coco18", "openpose_body25")


def parse_joint_order(raw: str) -> Tuple[str, ...]:
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(names) != 17 or len(set(names)) != 17:
        raise ValueError("--joint-order must contain 17 unique comma-separated names.")
    missing = sorted(set(COCO17_NAMES) - set(names))
    if missing:
        raise ValueError(f"--joint-order is missing COCO joints: {', '.join(missing)}")
    return names


def reorder_to_coco17(joints: np.ndarray, source_order: Sequence[str]) -> np.ndarray:
    if joints.ndim != 3 or joints.shape[2] < 2:
        raise ValueError("joints must have shape (frames, joints, >=2)")
    if joints.shape[1] != len(source_order):
        raise ValueError(
            f"joint array contains {joints.shape[1]} joints but the declared order contains "
            f"{len(source_order)} names"
        )
    indices = [source_order.index(name) for name in COCO17_NAMES]
    return np.asarray(joints[:, indices, :2], dtype=float)


def reorder_visibility_to_coco17(visibility: np.ndarray,
                                 source_order: Sequence[str]) -> np.ndarray:
    if visibility.ndim != 2 or visibility.shape[1] != len(source_order):
        raise ValueError(
            f"visibility must have shape (frames, {len(source_order)}) for the declared layout"
        )
    indices = [source_order.index(name) for name in COCO17_NAMES]
    return np.asarray(visibility[:, indices], dtype=float)


def _openpose_indices(layout: str) -> Tuple[int, ...]:
    if layout == "openpose_coco18":
        mapping = OPENPOSE_COCO18
    elif layout == "openpose_body25":
        mapping = OPENPOSE_BODY25
    else:
        raise ValueError(
            f"Unsupported joint layout {layout!r}; expected one of {', '.join(SUPPORTED_LAYOUTS)}."
        )
    return tuple(mapping[name] for name in COCO17_FROM_OPENPOSE)


def convert_joint_layout(joints: np.ndarray, layout: str,
                         source_order: Sequence[str] = COCO17_NAMES) -> np.ndarray:
    array = np.asarray(joints)
    if layout == "coco17":
        return reorder_to_coco17(array, source_order)
    indices = _openpose_indices(layout)
    expected = 18 if layout == "openpose_coco18" else 25
    if array.ndim != 3 or array.shape[1] != expected or array.shape[2] < 2:
        raise ValueError(
            f"{layout} joints must have shape (frames, {expected}, >=2); got {array.shape}."
        )
    return np.asarray(array[:, indices, :2], dtype=float)


def convert_visibility_layout(visibility: np.ndarray, layout: str,
                              source_order: Sequence[str] = COCO17_NAMES) -> np.ndarray:
    array = np.asarray(visibility)
    if layout == "coco17":
        return reorder_visibility_to_coco17(array, source_order)
    indices = _openpose_indices(layout)
    expected = 18 if layout == "openpose_coco18" else 25
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError(
            f"{layout} visibility must have shape (frames, {expected}); got {array.shape}."
        )
    return np.asarray(array[:, indices], dtype=float)


def flip_unity_y_to_image(joints: np.ndarray, image_height: int) -> np.ndarray:
    if image_height <= 0:
        raise ValueError("image height must be positive before applying --flip-y")
    flipped = np.asarray(joints, dtype=float).copy()
    flipped[:, :, 1] = float(image_height) - flipped[:, :, 1]
    return flipped


def _bbox_from_keypoints(points: np.ndarray, visible: np.ndarray,
                         width: int, height: int) -> Tuple[List[float], float]:
    usable = points[visible > 0]
    if not len(usable):
        return [0.0, 0.0, 0.0, 0.0], 0.0
    min_x, min_y = np.min(usable, axis=0)
    max_x, max_y = np.max(usable, axis=0)
    pad_x = max(2.0, (max_x - min_x) * 0.05)
    pad_y = max(2.0, (max_y - min_y) * 0.05)
    x1 = float(np.clip(min_x - pad_x, 0, width))
    y1 = float(np.clip(min_y - pad_y, 0, height))
    x2 = float(np.clip(max_x + pad_x, 0, width))
    y2 = float(np.clip(max_y + pad_y, 0, height))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    return [round(x1, 3), round(y1, 3), round(box_width, 3), round(box_height, 3)], round(box_width * box_height, 3)


def build_coco_document(joints: np.ndarray, image_names: Sequence[str],
                        width: int, height: int,
                        visibility: Optional[np.ndarray] = None,
                        coordinate_convention: str = "image-space top-left origin, y-down") -> Dict:
    if joints.ndim != 3 or joints.shape[1] != 17 or joints.shape[2] < 2:
        raise ValueError("training joints must have shape (frames, 17, >=2) in COCO-17 order")
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if len(image_names) != joints.shape[0]:
        raise ValueError("image count must match the number of joint frames")
    if visibility is not None and visibility.shape[:2] != joints.shape[:2]:
        raise ValueError("visibility must have shape (frames, joints)")

    images = []
    annotations = []
    for index, image_name in enumerate(image_names):
        image_id = index + 1
        points = np.asarray(joints[index, :17, :2], dtype=float)
        finite = np.isfinite(points).all(axis=1)
        if visibility is None:
            visible = finite.astype(int) * 2
        else:
            visible = ((np.asarray(visibility[index, :17], dtype=float) > 0) & finite).astype(int) * 2
        safe_points = np.where(finite[:, None], points, 0.0)
        keypoints = [
            value
            for joint_index in range(17)
            for value in (
                round(float(safe_points[joint_index, 0]), 3),
                round(float(safe_points[joint_index, 1]), 3),
                int(visible[joint_index]),
            )
        ]
        bbox, area = _bbox_from_keypoints(safe_points, visible, width, height)
        images.append({"id": image_id, "file_name": image_name, "width": width, "height": height})
        annotations.append({
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": int(np.count_nonzero(visible)),
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })

    return {
        "info": {
            "description": "SwimXYZ conversion for Swim Sight pose training",
            "license": "CC-BY-4.0",
            "citation": "Fiche et al., SwimXYZ, ACM MIG 2023",
            "coordinate_convention": coordinate_convention,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": list(COCO17_NAMES),
            "skeleton": [list(edge) for edge in COCO_SKELETON],
        }],
    }


def validate_coco_document(document: Dict) -> None:
    if not {"images", "annotations", "categories"} <= set(document):
        raise ValueError("COCO document is missing a required top-level field")
    if len(document["images"]) != len(document["annotations"]):
        raise ValueError("COCO image and annotation counts differ")
    if not document["categories"] or len(document["categories"][0].get("keypoints", [])) != 17:
        raise ValueError("COCO category must define 17 keypoints")
    if tuple(document["categories"][0]["keypoints"]) != COCO17_NAMES:
        raise ValueError("COCO category keypoints are not in the required COCO-17 order")
    image_ids = [image.get("id") for image in document["images"]]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("COCO image IDs must be unique")
    for annotation in document["annotations"]:
        if len(annotation.get("keypoints", [])) != 51:
            raise ValueError("Every annotation must contain 17x3 keypoint values")
        if annotation.get("image_id") not in image_ids:
            raise ValueError("COCO annotation references an unknown image")


def split_indices(count: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 (inclusive) and 1 (exclusive)")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(count * val_ratio))) if count > 1 and val_ratio > 0 else 0
    val_count = min(val_count, max(0, count - 1))
    return sorted(indices[val_count:]), sorted(indices[:val_count])


def subset_document(document: Dict, selected: Sequence[int]) -> Dict:
    images_by_id = {image["id"]: image for image in document["images"]}
    annotations_by_image = {annotation["image_id"]: annotation for annotation in document["annotations"]}
    image_ids = [index + 1 for index in selected]
    return {
        "info": document["info"],
        "images": [images_by_id[image_id] for image_id in image_ids],
        "annotations": [annotations_by_image[image_id] for image_id in image_ids],
        "categories": document["categories"],
    }


def run_self_test() -> int:
    assert set(COCO17_TO_WORKER) == {0, *range(3, 17)}
    joints = np.zeros((4, 17, 2), dtype=float)
    for frame_index in range(4):
        joints[frame_index, :, 0] = np.arange(17) * 4 + 20 + frame_index
        joints[frame_index, :, 1] = np.arange(17) * 2 + 30
    document = build_coco_document(
        joints,
        [f"synthetic_{index:03d}.png" for index in range(4)],
        width=160,
        height=120,
    )
    validate_coco_document(document)
    train_indices, val_indices = split_indices(4, 0.25, 7)
    validate_coco_document(subset_document(document, train_indices))
    validate_coco_document(subset_document(document, val_indices))
    openpose = np.zeros((2, 18, 2), dtype=float)
    converted = convert_joint_layout(openpose, "openpose_coco18")
    assert converted.shape == (2, 17, 2)
    with tempfile.TemporaryDirectory() as temp_dir:
        target = Path(temp_dir) / "self_test.json"
        target.write_text(json.dumps(document))
        validate_coco_document(json.loads(target.read_text()))
    print("SwimXYZ to MMPose self-test passed (synthetic data only).")
    return 0


def _infer_image_size(path: Path) -> Tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Inferring image dimensions requires opencv-python-headless.") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image dimensions from {path.name}.")
    height, width = image.shape[:2]
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert local SwimXYZ labels to MMPose COCO JSON.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--joints", help="Ground-truth .npy/.npz/.json array")
    parser.add_argument("--joints-key", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--visibility-key", default=None)
    parser.add_argument("--images-dir", help="Matching image frames")
    parser.add_argument("--output-dir", default="data/swimxyz/annotations")
    parser.add_argument("--stroke", default="freestyle")
    parser.add_argument("--joint-layout", choices=SUPPORTED_LAYOUTS, default="coco17",
                        help="Input array layout before conversion to COCO-17")
    parser.add_argument("--joint-order", default=",".join(COCO17_NAMES))
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flip-y", action="store_true",
        help=("Convert raw SwimXYZ Unity y-up coordinates to image y-down. "
              "Required for raw labels; do not use for an already prepared/flipped array."),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not args.joints or not args.images_dir:
        print("--joints and --images-dir are required unless --self-test is used.", file=sys.stderr)
        return 2

    joints_path = Path(args.joints).expanduser()
    images_dir = Path(args.images_dir).expanduser()
    if not joints_path.is_file() or not images_dir.is_dir():
        print("The joints file or images directory does not exist.", file=sys.stderr)
        return 2

    try:
        if (args.image_width is None) != (args.image_height is None):
            raise ValueError("Provide both --image-width and --image-height, or neither.")
        source_order = parse_joint_order(args.joint_order)
        joints = convert_joint_layout(
            load_array(joints_path, args.joints_key), args.joint_layout, source_order
        )
        visibility = (
            convert_visibility_layout(
                load_array(Path(args.visibility).expanduser(), args.visibility_key),
                args.joint_layout,
                source_order,
            )
            if args.visibility else None
        )
        images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) != joints.shape[0]:
            raise ValueError(
                f"Found {len(images)} images but {joints.shape[0]} labelled frames. "
                "Use a matching extracted sequence."
            )
        if not images:
            raise ValueError("No matching image frames were found.")
        width, height = (
            (args.image_width, args.image_height)
            if args.image_width and args.image_height
            else _infer_image_size(images[0])
        )
        if args.flip_y:
            joints = flip_unity_y_to_image(joints, height)
        convention = (
            "converted from Unity bottom-left origin/y-up to image top-left origin/y-down"
            if args.flip_y else
            "input used unchanged; expected image-space top-left origin/y-down"
        )
        document = build_coco_document(
            joints, [path.name for path in images], width, height, visibility,
            coordinate_convention=convention,
        )
        validate_coco_document(document)
        train_indices, val_indices = split_indices(len(images), args.val_ratio, args.seed)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Could not convert SwimXYZ labels: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{args.stroke}_train.json"
    val_path = output_dir / f"{args.stroke}_val.json"
    train_path.write_text(json.dumps(subset_document(document, train_indices), indent=2))
    val_path.write_text(json.dumps(subset_document(document, val_indices), indent=2))
    print(f"Wrote {len(train_indices)} training frames to {train_path}")
    print(f"Wrote {len(val_indices)} validation frames to {val_path}")
    if not args.flip_y:
        print("Y coordinates were used unchanged. Raw SwimXYZ Unity labels require --flip-y.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
