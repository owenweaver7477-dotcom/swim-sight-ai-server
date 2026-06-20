"""Pure tests for SwimXYZ baseline and MMPose conversion helpers."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.measure_pose_baseline import load_array  # noqa: E402
from scripts.swimxyz_to_mmpose import (  # noqa: E402
    COCO17_NAMES,
    build_coco_document,
    parse_joint_order,
    reorder_to_coco17,
    reorder_visibility_to_coco17,
    split_indices,
    subset_document,
    validate_coco_document,
)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


with tempfile.TemporaryDirectory() as temp_dir:
    temp = Path(temp_dir)
    sample = np.arange(3 * 17 * 2, dtype=float).reshape(3, 17, 2)
    npy_path = temp / "joints.npy"
    npz_path = temp / "joints.npz"
    json_path = temp / "joints.json"
    np.save(npy_path, sample)
    np.savez(npz_path, joints=sample)
    json_path.write_text(json.dumps({"joints": sample.tolist()}))
    check("baseline loader reads NPY", np.array_equal(load_array(npy_path), sample))
    check("baseline loader reads keyed NPZ", np.array_equal(load_array(npz_path, "joints"), sample))
    check("baseline loader reads JSON", np.array_equal(load_array(json_path), sample))

source_order = tuple(reversed(COCO17_NAMES))
source_joints = np.zeros((3, 17, 2), dtype=float)
for source_index, name in enumerate(source_order):
    source_joints[:, source_index, 0] = COCO17_NAMES.index(name) + 10
    source_joints[:, source_index, 1] = COCO17_NAMES.index(name) + 20
reordered = reorder_to_coco17(source_joints, source_order)
check("parameterised source order is converted to COCO-17",
      np.array_equal(reordered[0, :, 0], np.arange(17) + 10))
check("default joint-order parser accepts COCO-17",
      parse_joint_order(",".join(COCO17_NAMES)) == COCO17_NAMES)
source_visibility = np.tile(np.arange(17, dtype=float), (3, 1))
reordered_visibility = reorder_visibility_to_coco17(source_visibility, source_order)
check("parameterised visibility follows the same COCO-17 order",
      np.array_equal(reordered_visibility[0], np.arange(16, -1, -1)))

document = build_coco_document(
    reordered,
    [f"frame_{index:03d}.png" for index in range(3)],
    width=200,
    height=120,
)
validate_coco_document(document)
check("COCO document contains 17x3 keypoints",
      all(len(annotation["keypoints"]) == 51 for annotation in document["annotations"]))
check("SwimXYZ attribution is embedded in output", "Fiche" in document["info"]["citation"])

train_indices, val_indices = split_indices(3, 0.34, 42)
train_document = subset_document(document, train_indices)
val_document = subset_document(document, val_indices)
validate_coco_document(train_document)
validate_coco_document(val_document)
check("train/validation split preserves all frames",
      len(train_document["images"]) + len(val_document["images"]) == 3)
check("tool imports do not load MMPose", "mmpose" not in sys.modules)
check("tool imports do not load onnxruntime", "onnxruntime" not in sys.modules)

print("\n" + "=" * 50)
failed = results.count(False)
print(f"{results.count(True)}/{len(results)} passed",
      "ALL PASS" if failed == 0 else f"{failed} FAILED")
sys.exit(1 if failed else 0)
