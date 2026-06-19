"""
SwimSight AI worker - anthropometric drag integration
=====================================================

Consumes THIS worker's real MediaPipe pose output (app/pose_estimator.py) and
produces an `estimated_drag` block for the /process-video callback.

Real input schema (per frame, from run_pose_estimation):
    {
      "frame_idx": int,                # original video frame index (sampled)
      "pose_detected": bool,
      "keypoint_count": int,
      "landmarks": {                   # ONLY landmarks with visibility >= 0.45
          "nose":       {"x": .., "y": .., "visibility": ..},
          "left_hip":   {...}, "right_hip": {...},
          "left_ankle": {...}, "right_ankle": {...}, ...
      }
    }

Landmarks are MediaPipe-normalised (x, y in 0..1, y increasing downward) and are
OMITTED entirely when the detector's visibility gate drops them, so a landmark
may be absent in any frame.

Mapping from what the drag math needs -> real MediaPipe landmark names:
    head (scale)      -> "nose"
    ankle (scale)     -> midpoint("left_ankle", "right_ankle")
    forward/velocity  -> midpoint("left_hip", "right_hip")   (lane axis)
    (the old neck/lumbar assumptions are dropped: MediaPipe has no equivalent
     and the drag math never used them)

Timestamps come from frame_idx / fps because the worker samples frames, so the
spacing is not 1/fps.

PRIVACY: height_cm / mass_kg are inputs only; they are never written into the
returned payload (asserted in the self-test).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # works when imported as part of the app package (main.py)
    from app.anthropometric_drag import Swimmer, AnthropometricDragModel, POSE_PRESETS
except ImportError:  # works when run directly from inside app/
    from anthropometric_drag import Swimmer, AnthropometricDragModel, POSE_PRESETS

logger = logging.getLogger(__name__)

# Real MediaPipe landmark NAMES this module needs (all emitted by pose_estimator).
HEAD = "nose"
HIP_L, HIP_R = "left_hip", "right_hip"
ANKLE_L, ANKLE_R = "left_ankle", "right_ankle"

MIN_RELIABLE_FRAME_FRAC = 0.5    # below this share of frames with a hip midpoint -> low confidence
MIN_TRACK_FRAMES = 5             # need at least this many hip frames to estimate kinematics

# Map the app's stroke_type ("Freestyle", "free", ...) onto anthropometric_drag pose presets.
STROKE_ALIASES = {
    "free": "freestyle", "freestyle": "freestyle", "front crawl": "freestyle",
    "back": "backstroke", "backstroke": "backstroke",
    "breast": "breaststroke", "breaststroke": "breaststroke",
    "fly": "butterfly", "butterfly": "butterfly",
}


def _pose_key(stroke: Optional[str]) -> str:
    key = (stroke or "").strip().lower()
    key = STROKE_ALIASES.get(key, key)
    return key if key in POSE_PRESETS else "mid_stroke"


# ---------------------------------------------------------------------------
# Pilot feature flag + gate (single source of truth used by /process-video).
# estimated_drag is an INTERNAL PILOT prototype and is OFF unless explicitly
# enabled via the ENABLE_ESTIMATED_DRAG environment variable.
# ---------------------------------------------------------------------------
ENABLE_FLAG = "ENABLE_ESTIMATED_DRAG"
_TRUTHY = {"1", "true", "yes", "on"}


def estimated_drag_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """
    Pilot flag. Returns False unless ENABLE_ESTIMATED_DRAG is explicitly truthy
    ("1"/"true"/"yes"/"on", case-insensitive). Missing, blank, or any other
    value -> False, so the prototype is OFF by default.
    """
    source = os.environ if env is None else env
    return str(source.get(ENABLE_FLAG, "false")).strip().lower() in _TRUTHY


def should_emit_estimated_drag(*, analysis_mode: Optional[str],
                               real_pose_detected: Any,
                               height_cm: Any,
                               mass_kg: Any,
                               env: Optional[Dict[str, str]] = None) -> bool:
    """
    The complete /process-video gate. ALL of the following must hold:
      - the pilot flag is ON,
      - real pose keypoints were detected,
      - analysis_mode is 'real_pose' (i.e. NOT a manual-review fallback),
      - both swimmer anthropometrics are present.
    """
    return bool(
        estimated_drag_enabled(env)
        and real_pose_detected
        and analysis_mode == "real_pose"
        and height_cm
        and mass_kg
    )


def build_swimmer(height_cm: float, mass_kg: float, name: str = "swimmer") -> Swimmer:
    """Coach profile -> validated Swimmer. cm -> m happens HERE (the /100), and
    anthropometric_drag's unit guard fires if a value still looks like cm."""
    return Swimmer(mass_kg=float(mass_kg), height_m=float(height_cm) / 100.0, name=name)


def _landmark(frame: Dict[str, Any], name: str, min_vis: float) -> Optional[Tuple[float, float]]:
    """(x, y) for a landmark if present and visible enough, else None."""
    lm = frame.get("landmarks", {}).get(name)
    if not lm:
        return None
    if min_vis > 0.0 and float(lm.get("visibility", 1.0)) < min_vis:
        return None
    try:
        return float(lm["x"]), float(lm["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def estimate_scale(pose_results: List[Dict[str, Any]], height_m: float,
                   min_vis: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Step 6. metres-per-normalised-unit from the most-extended frame.

    body length (units) = max over frames of |nose - ankle_midpoint|, using only
    frames where nose + both ankles are present. scale = height_m / that length.
    """
    best_len = 0.0
    best_frame: Optional[int] = None
    for i, fr in enumerate(pose_results):
        head = _landmark(fr, HEAD, min_vis)
        al = _landmark(fr, ANKLE_L, min_vis)
        ar = _landmark(fr, ANKLE_R, min_vis)
        if head is None or al is None or ar is None:
            continue
        amid = _midpoint(al, ar)
        d = float(np.hypot(head[0] - amid[0], head[1] - amid[1]))
        if d > best_len:
            best_len = d
            best_frame = int(fr.get("frame_idx", i))
    if best_len <= 1e-6:
        return None
    return {
        "scale_m_per_unit": height_m / best_len,
        "body_length_units": best_len,
        "extended_frame": best_frame,
    }


def analyse_clip(pose_results: List[Dict[str, Any]],
                 fps: float,
                 *,
                 height_cm: Optional[float],
                 mass_kg: Optional[float],
                 stroke: str = "freestyle",
                 lane_axis: str = "x",
                 min_visibility: float = 0.0,
                 smooth_seconds_accel: float = 0.8) -> Optional[Dict[str, Any]]:
    """
    Build the `estimated_drag` block from the worker's real pose_results.

    Returns None (so the caller simply omits estimated_drag) when:
      - height_cm or mass_kg is missing,
      - there is no scalable head+ankle frame, or
      - fewer than MIN_TRACK_FRAMES frames carry a hip midpoint.
    Never raises for ordinary "not enough pose" situations.
    """
    # Step 10: missing anthropometrics -> skip cleanly, never block analysis.
    if height_cm is None or mass_kg is None:
        return None
    if not pose_results:
        return None

    fps = float(fps) if fps and fps > 0 else 30.0
    axis = 0 if str(lane_axis).lower() == "x" else 1

    swimmer = build_swimmer(height_cm, mass_kg)          # cm -> m (+ unit guard)
    pose_key = _pose_key(stroke)
    model = AnthropometricDragModel(swimmer, pose=pose_key)

    # Step 6: scale from the most-extended frame.
    scale = estimate_scale(pose_results, swimmer.height_m, min_visibility)
    if scale is None:
        logger.info("estimated_drag skipped: no frame with nose + both ankles to scale from")
        return None
    s = scale["scale_m_per_unit"]

    # Step 7: hip-midpoint forward track (only frames where both hips are present).
    times: List[float] = []
    hip_axis: List[float] = []
    total_frames = len(pose_results)
    for i, fr in enumerate(pose_results):
        hl = _landmark(fr, HIP_L, min_visibility)
        hr = _landmark(fr, HIP_R, min_visibility)
        if hl is None or hr is None:
            continue
        hmid = _midpoint(hl, hr)
        times.append(int(fr.get("frame_idx", i)) / fps)
        hip_axis.append(hmid[axis])

    hip_frames = len(times)
    if hip_frames < MIN_TRACK_FRAMES:
        logger.info("estimated_drag skipped: only %d frames with a hip midpoint", hip_frames)
        return None

    t = np.asarray(times, dtype=float)
    order = np.argsort(t)                      # frame_idx should already be ordered, but be safe
    t = t[order]
    pos_units = np.asarray(hip_axis, dtype=float)[order]
    pos_m = (pos_units - pos_units[0]) * s     # displacement in metres

    vel, acc = AnthropometricDragModel.kinematics_from_positions(
        pos_m, t, smooth_seconds=smooth_seconds_accel)
    results = model.compute_series(t, vel, accelerations_m_s2=acc, accel_is_smoothed=True)

    # Trim entry/exit frames (swimmer part-in-view / truncated fit window).
    n = len(results)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    win = max(5, int(round(smooth_seconds_accel / dt))) if dt > 0 else 5
    trim = min(win // 2, n // 4)
    interior = slice(trim, n - trim) if (n - 2 * trim) >= 5 else slice(0, n)

    t_i = t[interior] - float(t[interior][0])
    vel_i = vel[interior]
    res_i = results[interior]

    reliable_frac = hip_frames / total_frames if total_frames else 0.0
    confidence_low = (
        any(r.confidence_low for r in res_i)
        or reliable_frac < MIN_RELIABLE_FRAME_FRAC
        or len(res_i) < 5
    )

    drag = np.array([r.drag_force_n for r in res_i])
    dwr = np.array([r.drag_to_weight_ratio for r in res_i])
    prop = np.array([r.propulsive_force_n for r in res_i])
    net = np.array([r.net_force_n for r in res_i])

    # Step 9: drag + drag-to-weight always; net/propulsive only when confident.
    payload: Dict[str, Any] = {
        "label": "estimated_drag",
        "basis": "estimated (monocular anthropometric scale from MediaPipe pose) -- not measured",
        "pose_source": "mediapipe_pose",
        "stroke": pose_key,
        "confidence_low": bool(confidence_low),
        "reliable_frame_fraction": round(reliable_frac, 3),
        "scale_m_per_unit": round(s, 5),
        "scale_reference_frame": scale["extended_frame"],
        "frames_analysed": int(len(res_i)),
        "summary": {
            "mean_drag_force_n": round(float(np.mean(drag)), 2),
            "peak_drag_force_n": round(float(np.max(drag)), 2),
            "mean_drag_to_weight_ratio": round(float(np.mean(dwr)), 4),
            "peak_velocity_m_s": round(float(np.max(vel_i)), 3),
        },
        "series": {
            "timestamp_s": [round(float(x), 4) for x in t_i],
            "velocity_m_s": [round(float(x), 4) for x in vel_i],
            "drag_force_n": [round(float(x), 2) for x in drag],
            "drag_to_weight_ratio": [round(float(x), 4) for x in dwr],
        },
    }
    if not confidence_low:
        payload["summary"]["mean_propulsive_force_n"] = round(float(np.mean(prop)), 2)
        payload["summary"]["peak_propulsive_force_n"] = round(float(np.max(prop)), 2)
        payload["series"]["propulsive_force_n"] = [round(float(x), 2) for x in prop]
        payload["series"]["net_force_n"] = [round(float(x), 2) for x in net]

    return payload


# ===========================================================================
# Synthetic helper for tests: build real-schema pose_results.
# ===========================================================================
def synthetic_pose_results(fps: float = 30.0,
                           seconds: float = 6.0,
                           frame_step: int = 1,
                           true_v: float = 1.6,
                           height_cm: float = 180.0,
                           body_len_units: float = 0.5,
                           jitter_m: float = 0.005,
                           visibility: float = 0.9,
                           detected: bool = True,
                           seed: int = 11) -> List[Dict[str, Any]]:
    """A swimmer of known height crossing a fixed frame at true_v, emitted in the
    exact shape run_pose_estimation produces."""
    rng = np.random.default_rng(seed)
    height_m = height_cm / 100.0
    scale_true = height_m / body_len_units
    half = body_len_units / 2.0
    jitter_u = jitter_m / scale_true

    frame_indices = list(range(0, int(fps * seconds), max(1, frame_step)))
    out: List[Dict[str, Any]] = []
    for fi in frame_indices:
        t = fi / fps
        hip_x = 0.15 + (true_v * t) / scale_true + rng.normal(0, jitter_u)
        y = 0.5 + rng.normal(0, jitter_u)

        def lm(xx):
            return {"x": float(xx + rng.normal(0, jitter_u)), "y": float(y), "visibility": visibility}

        landmarks = {
            HEAD: lm(hip_x - half),
            ANKLE_L: lm(hip_x + half), ANKLE_R: lm(hip_x + half),
            HIP_L: lm(hip_x), HIP_R: lm(hip_x),
            # extra realistic landmarks (unused by drag, present in real output)
            "left_shoulder": lm(hip_x - half * 0.4), "right_shoulder": lm(hip_x - half * 0.4),
            "left_knee": lm(hip_x + half * 0.55), "right_knee": lm(hip_x + half * 0.55),
        }
        out.append({
            "frame_idx": fi,
            "pose_detected": detected,
            "keypoint_count": len(landmarks),
            "landmarks": landmarks,
        })
    return out


def _demo() -> None:
    import json

    print("pose_worker_integration self-test (real MediaPipe schema)")
    print("-" * 60)

    # 1) Clean tracked clip -> believable drag, edge-fix holds, profile not leaked.
    pr = synthetic_pose_results(fps=30, seconds=6.0, true_v=1.6, height_cm=180.0)
    payload = analyse_clip(pr, fps=30, height_cm=180.0, mass_kg=75.0, stroke="Freestyle")
    assert payload is not None, "expected a payload for a clean clip"
    blob = json.dumps(payload)
    assert "height_cm" not in blob and "mass_kg" not in blob and "height_m" not in blob, "PROFILE LEAK"
    gap = max(abs(d - p) for d, p in zip(
        payload["series"]["drag_force_n"],
        payload["series"].get("propulsive_force_n", payload["series"]["drag_force_n"])))
    print(f"  clean clip: drag_mean={payload['summary']['mean_drag_force_n']} N, "
          f"v_peak={payload['summary']['peak_velocity_m_s']} m/s, "
          f"scale={payload['scale_m_per_unit']} (true 3.6)")
    print(f"  confidence_low={payload['confidence_low']}, max|prop-drag|={gap:.1f} N, "
          f"profile_leak={'mass_kg' in blob}")

    # 2) Missing profile -> None (never blocks the rest of the pipeline).
    none_payload = analyse_clip(pr, fps=30, height_cm=None, mass_kg=75.0)
    print(f"  missing height -> {none_payload} (expect None)")

    # 3) Sparse/low-visibility clip -> drag stays, propulsive hidden or None.
    sparse = synthetic_pose_results(fps=30, seconds=1.6, true_v=1.6)
    low = analyse_clip(sparse, fps=30, height_cm=180.0, mass_kg=75.0)
    if low is None:
        print("  short clip -> None (too few hip frames)")
    else:
        print(f"  short clip: confidence_low={low['confidence_low']}, "
              f"propulsive_exposed={'propulsive_force_n' in low['series']}")


if __name__ == "__main__":
    _demo()
