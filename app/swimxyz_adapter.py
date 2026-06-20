"""
app/swimxyz_adapter.py - SwimXYZ ground-truth joints -> worker pose_results.

SwimXYZ (CC-BY-4.0; cite Fiche et al., ACM MIG 2023) ships every frame with
ground-truth 2D/3D joints. Converting those joints into the worker's
pose_results landmark-dict format lets you:

  * run labelled synthetic clips straight through findings + drag, and
  * MEASURE the current MediaPipe detector against ground truth (and, later,
    your fine-tuned model against the same truth).

SwimXYZ's 2D joints follow the COCO-17 ordering (the format ViTPose trains on).
The worker's CORE landmark names are a superset of those minus the eyes. If your
copy of SwimXYZ uses a different joint order, edit COCO17_TO_WORKER (one dict) --
that is the only thing tied to the dataset layout.

Pure NumPy; no mediapipe / cv2.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# COCO-17 index -> worker landmark name. COCO indices 1,2 (eyes) are dropped
# because the worker's CORE keypoint set has no eyes.
COCO17_TO_WORKER: Dict[int, str] = {
    0: "nose",
    3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}

DETECT_MIN_KEYPOINTS = 4
DEFAULT_MIN_VIS = 0.45


def joints_to_pose_results(joints: Sequence,
                           image_size: Optional[Tuple[float, float]] = None,
                           fps: float = 30.0,
                           frame_indices: Optional[Sequence[int]] = None,
                           visibility: Optional[Sequence] = None,
                           joint_map: Dict[int, str] = COCO17_TO_WORKER,
                           min_visibility: float = DEFAULT_MIN_VIS) -> List[Dict[str, Any]]:
    """
    joints       : array (frames, J, 2|3). Pixel coords if image_size is given,
                   else assumed already normalised to 0..1.
    image_size   : (width, height) used to normalise pixel coords to 0..1.
    visibility   : optional (frames, J) in [0,1]; default 1.0 (ground truth = visible).
                   Joints below min_visibility are omitted (mimics the detector's gate).
    frame_indices: optional real frame indices (else 0..frames-1).
    Returns the worker's pose_results: list of
        {frame_idx, pose_detected, keypoint_count, landmarks:{name:{x,y[,z],visibility}}}.
    """
    arr = np.asarray(joints, dtype=float)
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError("joints must be (frames, J, 2 or 3)")
    n, num_j = arr.shape[0], arr.shape[1]
    has_z = arr.shape[2] >= 3
    w, h = (image_size if image_size else (1.0, 1.0))
    vis = np.asarray(visibility, dtype=float) if visibility is not None else None

    out: List[Dict[str, Any]] = []
    for i in range(n):
        fi = int(frame_indices[i]) if frame_indices is not None else i
        lm: Dict[str, Any] = {}
        for idx, name in joint_map.items():
            if idx >= num_j:
                continue
            x = arr[i, idx, 0] / w
            y = arr[i, idx, 1] / h
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            v = float(vis[i, idx]) if vis is not None else 1.0
            if v < min_visibility:
                continue
            point: Dict[str, Any] = {"x": float(x), "y": float(y), "visibility": v}
            if has_z and np.isfinite(arr[i, idx, 2]):
                point["z"] = float(arr[i, idx, 2])
            lm[name] = point
        kc = sum(1 for p in lm.values() if p["visibility"] >= min_visibility)
        out.append({
            "frame_idx": fi,
            "pose_detected": kc >= DETECT_MIN_KEYPOINTS,
            "keypoint_count": kc,
            "landmarks": lm,
        })
    return out


# ---------------------------------------------------------------------------
# Accuracy scoring: compare a DETECTOR's pose_results against ground truth.
# ---------------------------------------------------------------------------
def keypoint_errors(pred: List[Dict[str, Any]],
                    truth: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Mean normalised keypoint error (in normalised-coord units) between a
    predicted pose_results and the ground-truth pose_results, matched by
    frame_idx and landmark name. Also reports PCK@0.05 (fraction of keypoints
    within 0.05 of truth). Lower error / higher PCK = better.
    """
    truth_by_frame = {t["frame_idx"]: t.get("landmarks", {}) for t in truth}
    dists: List[float] = []
    matched = 0
    total_truth = 0
    for p in pred:
        gt = truth_by_frame.get(p["frame_idx"])
        if not gt:
            continue
        pl = p.get("landmarks", {})
        for name, tpt in gt.items():
            total_truth += 1
            pp = pl.get(name)
            if not pp:
                continue
            matched += 1
            dists.append(float(np.hypot(pp["x"] - tpt["x"], pp["y"] - tpt["y"])))
    if not dists:
        return {"matched_keypoints": 0, "mean_error": None, "pck_0.05": None,
                "recall": 0.0}
    d = np.asarray(dists)
    return {
        "matched_keypoints": matched,
        "mean_error": round(float(d.mean()), 4),
        "median_error": round(float(np.median(d)), 4),
        "pck_0.05": round(float(np.mean(d <= 0.05)), 4),
        "recall": round(matched / total_truth, 4) if total_truth else 0.0,
    }
