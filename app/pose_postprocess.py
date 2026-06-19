"""
app/pose_postprocess.py - temporal stabilisation of sampled pose tracks.

The /process-video worker SAMPLES frames across the clip, which largely defeats
MediaPipe's in-model temporal smoothing, and any landmark below the visibility
gate is dropped entirely (leaving holes). This module post-processes the
per-frame landmark dicts produced by run_pose_estimation to:

  - linearly interpolate SHORT gaps (a landmark missing for <= max_gap sampled
    frames but present on both sides), spacing-aware via frame_idx,
  - replace single-frame position outliers using a rolling median (MAD test),
  - smooth x/y with a small centred moving average,
  - recompute keypoint_count / pose_detected after filling.

Design choices that keep the quality gate honest:
  - interpolated points are marked {"interpolated": true} and given a visibility
    JUST BELOW the detection floor, so they are usable for geometry (findings &
    drag read landmarks by presence) but do NOT inflate keypoint_count /
    detection_ratio that the app-side gate depends on.

Pure NumPy. OFF unless ENABLE_POSE_SMOOTHING is truthy. smooth_pose_results
returns a NEW list and should be wrapped by the caller so any failure falls back
to the raw results.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from app.pose_estimator import CORE_LANDMARK_NAMES

_TRUTHY = {"1", "true", "yes", "on"}
DETECT_MIN_KEYPOINTS = 4      # matches pose_estimator pose_detected rule
DEFAULT_VIS = 0.45            # matches pose_estimator MIN_VISIBILITY
INTERP_VIS = 0.40            # interpolated points: usable but below the count gate


def pose_smoothing_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    src = os.environ if env is None else env
    return str(src.get("ENABLE_POSE_SMOOTHING", "false")).strip().lower() in _TRUTHY


def _moving_average(a: np.ndarray, w: int) -> np.ndarray:
    if w < 2 or a.size < 2:
        return a.copy()
    if w % 2 == 0:
        w += 1
    half = w // 2
    pad = np.pad(a, half, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def smooth_pose_results(pose_results: List[Dict[str, Any]],
                        window: int = 3,
                        max_gap: int = 2,
                        outlier_mad: float = 6.0,
                        min_visibility: float = DEFAULT_VIS) -> List[Dict[str, Any]]:
    if not pose_results:
        return pose_results
    n = len(pose_results)
    frame_idx = np.array([float(r.get("frame_idx", i)) for i, r in enumerate(pose_results)])

    names = set()
    for r in pose_results:
        names.update((r.get("landmarks") or {}).keys())

    # Per-landmark x/y/visibility arrays (NaN where absent) + an "observed" mask.
    series: Dict[str, List[np.ndarray]] = {}
    observed: Dict[str, np.ndarray] = {}
    for name in names:
        xs = np.full(n, np.nan)
        ys = np.full(n, np.nan)
        zs = np.full(n, np.nan)
        vs = np.full(n, np.nan)
        for i, r in enumerate(pose_results):
            lm = (r.get("landmarks") or {}).get(name)
            if lm and "x" in lm and "y" in lm:
                xs[i], ys[i] = lm["x"], lm["y"]
                zs[i], vs[i] = lm.get("z", np.nan), lm.get("visibility", 1.0)
        series[name] = [xs, ys, zs, vs]
        observed[name] = ~np.isnan(xs)

    def interp_short_gaps(arr: np.ndarray) -> np.ndarray:
        idxs = np.where(~np.isnan(arr))[0]
        if idxs.size < 2:
            return arr
        out = arr.copy()
        for a, b in zip(idxs[:-1], idxs[1:]):
            gap = int(b - a - 1)
            if 0 < gap <= max_gap:
                span = frame_idx[b] - frame_idx[a]
                for k in range(a + 1, b):
                    t = (frame_idx[k] - frame_idx[a]) / span if span else 0.5
                    out[k] = arr[a] + t * (arr[b] - arr[a])
        return out

    def replace_outliers(arr: np.ndarray) -> np.ndarray:
        present = ~np.isnan(arr)
        if present.sum() < 5:
            return arr
        out = arr.copy()
        half = 2
        for i in range(arr.size):
            if np.isnan(arr[i]):
                continue
            wnd = arr[max(0, i - half):min(arr.size, i + half + 1)]
            wnd = wnd[~np.isnan(wnd)]
            if wnd.size < 3:
                continue
            med = float(np.median(wnd))
            mad = float(np.median(np.abs(wnd - med))) + 1e-9
            if abs(arr[i] - med) > outlier_mad * mad:
                out[i] = med
        return out

    for name in names:
        xs, ys, zs, vs = series[name]
        xs = replace_outliers(interp_short_gaps(xs))
        ys = replace_outliers(interp_short_gaps(ys))
        zs = replace_outliers(interp_short_gaps(zs))
        present = ~np.isnan(xs)
        if present.any():
            pidx = np.where(present)[0]
            xs[pidx] = _moving_average(xs[pidx], window)
            ys[pidx] = _moving_average(ys[pidx], window)
        series[name] = [xs, ys, zs, vs]

    out_results: List[Dict[str, Any]] = []
    for i, r in enumerate(pose_results):
        new_lm: Dict[str, Any] = {}
        for name in names:
            xs, ys, zs, vs = series[name]
            if np.isnan(xs[i]) or np.isnan(ys[i]):
                continue
            was_observed = bool(observed[name][i])
            visibility = float(vs[i]) if was_observed else INTERP_VIS
            point = {"x": float(xs[i]), "y": float(ys[i]), "visibility": visibility}
            if not np.isnan(zs[i]):
                point["z"] = float(zs[i])
            if not was_observed:
                point["interpolated"] = True
            new_lm[name] = point
        # Count ONLY genuinely-visible (observed) keypoints toward the gate.
        kc = sum(
            1
            for name, lm in new_lm.items()
            if name in CORE_LANDMARK_NAMES and lm["visibility"] >= min_visibility
        )
        nr = dict(r)
        nr["landmarks"] = new_lm
        nr["keypoint_count"] = kc
        nr["landmark_count_total"] = sum(
            1 for lm in new_lm.values() if lm["visibility"] >= min_visibility
        )
        nr["pose_detected"] = kc >= DETECT_MIN_KEYPOINTS
        nr["smoothed"] = True
        out_results.append(nr)
    return out_results
