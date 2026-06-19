"""
app/findings_robust.py - more conservative peak detection for draft findings.

The default _series_peak in swim_analyzer fires on the SINGLE maximum frame of a
per-frame signal, so one noisy pose frame can trigger a coach-facing finding.

When ROBUST_FINDINGS is on, a candidate must be SUSTAINED -- its signal has to
stay within `sustain_frac` of the peak for at least `sustain` frames -- and the
reported strength is a high PERCENTILE rather than the raw max, which is far less
sensitive to a single outlier. One-off spikes return None (no finding).

Pure NumPy. OFF unless ROBUST_FINDINGS is truthy.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

_TRUTHY = {"1", "true", "yes", "on"}


def robust_findings_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    src = os.environ if env is None else env
    return str(src.get("ROBUST_FINDINGS", "false")).strip().lower() in _TRUTHY


def robust_peak(values: List[Dict[str, float]],
                sustain: int = 3,
                percentile: float = 90.0,
                sustain_frac: float = 0.6,
                max_gap_seconds: float = 1.5) -> Optional[Dict[str, Any]]:
    """
    values: list of {"value": float, "timestamp": float} (same shape the existing
    _series_peak receives -- these are already threshold-passing frames).

    Returns None when fewer than `sustain` frames are within `sustain_frac` of the
    peak (i.e. a one-off spike). Otherwise returns a peak dict whose strength is
    the `percentile` of the values and whose timestamp is the max frame.
    """
    if not values:
        return None
    vals = np.array([float(v["value"]) for v in values], dtype=float)
    peak_val = float(vals.max())
    if peak_val <= 0:
        return None
    high_items = sorted(
        (item for item in values if float(item["value"]) >= sustain_frac * peak_val),
        key=lambda item: float(item["timestamp"]),
    )
    runs: List[List[Dict[str, float]]] = []
    for item in high_items:
        if not runs or float(item["timestamp"]) - float(runs[-1][-1]["timestamp"]) > max_gap_seconds:
            runs.append([item])
        else:
            runs[-1].append(item)
    sustained_runs = [run for run in runs if len(run) >= sustain]
    if not sustained_runs:
        return None

    best_run = max(
        sustained_runs,
        key=lambda run: (len(run), max(float(item["value"]) for item in run)),
    )
    run_values = np.array([float(item["value"]) for item in best_run], dtype=float)
    strength = float(np.percentile(run_values, percentile))
    peak_item = max(best_run, key=lambda it: it["value"])
    return {
        "max": strength,
        "strength": strength,
        "timestamp": round(float(peak_item["timestamp"]), 2),
        "count": int(vals.size),
        "sustained_count": len(best_run),
        "sustained_duration_seconds": round(
            float(best_run[-1]["timestamp"]) - float(best_run[0]["timestamp"]),
            2,
        ),
    }
