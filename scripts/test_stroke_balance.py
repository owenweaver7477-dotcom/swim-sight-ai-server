"""Tests for experimental backstroke/butterfly findings (EXTENDED_STROKE_FINDINGS).

Verifies:
  - flag OFF: current backstroke/butterfly behaviour is unchanged (no new tags)
  - flag ON: each new finding can fire from synthetic pose data
  - all findings are coach_review_required with safe confidence
  - new findings keep the locked callback finding shape (no drift)
  - no starts/turns/underwater tags are ever produced

Pure/offline: synthetic landmarks, no video, no network, no deploy.
Run:  python3 scripts/test_stroke_balance.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.swim_analyzer import (  # noqa: E402
    analyze_pose_data,
    _backstroke_extended_findings,
    _butterfly_extended_findings,
)

FPS = 30.0
NEW_BACKSTROKE = {"backstroke_dropped_catch", "backstroke_short_extension"}
NEW_BUTTERFLY = {"butterfly_body_line_loss", "butterfly_breath_timing"}
FORBIDDEN_TAG_PARTS = (
    "start", "turn", "underwater", "wall", "dive", "push",
    "velocity", "split", "distance", "dps",
)


def _lm(x, y):
    return {"x": float(x), "y": float(y), "z": 0.0, "visibility": 0.9}


def _frame(idx, landmarks):
    return {"frame_idx": idx, "pose_detected": True, "keypoint_count": 12,
            "landmark_count_total": 15, "landmarks": landmarks}


def backstroke_frames(n=14):
    """Fires hip_sink (body line) + dropped_catch (elbow below wrist) +
    short_extension (wrist near shoulder)."""
    out = []
    for i in range(n):
        out.append(_frame(i, {
            "left_shoulder": _lm(0.45, 0.40), "right_shoulder": _lm(0.55, 0.40),
            "left_hip": _lm(0.45, 0.80), "right_hip": _lm(0.55, 0.80),
            "left_wrist": _lm(0.47, 0.60), "right_wrist": _lm(0.57, 0.60),
            "left_elbow": _lm(0.47, 0.72), "right_elbow": _lm(0.57, 0.72),
        }))
    return out


def butterfly_frames(n=14):
    """Fires body_line_loss (low hips) + breath_timing (head high). Constant hip
    offset means the rhythm signal (variability) does NOT fire."""
    out = []
    for i in range(n):
        out.append(_frame(i, {
            "left_shoulder": _lm(0.45, 0.40), "right_shoulder": _lm(0.55, 0.40),
            "left_hip": _lm(0.45, 0.80), "right_hip": _lm(0.55, 0.80),
            "nose": _lm(0.50, 0.25),
        }))
    return out


def freestyle_frames(n=14):
    """Baseline freestyle body-line-loss finding, for shape comparison."""
    out = []
    for i in range(n):
        out.append(_frame(i, {
            "left_shoulder": _lm(0.40, 0.42), "right_shoulder": _lm(0.60, 0.42),
            "left_hip": _lm(0.44, 0.80), "right_hip": _lm(0.56, 0.80),
        }))
    return out


def _check(cond, label):
    if not cond:
        raise AssertionError(label)


def _tags(findings):
    return {f.get("fault_tag") for f in findings}


def _assert_finding_safe(f, label, locked_keys):
    _check(set(f) == locked_keys, f"{label}: finding key set drifted "
           f"(missing {sorted(locked_keys - set(f))}, extra {sorted(set(f) - locked_keys)})")
    _check(f.get("coach_review_required") is True, f"{label}: coach_review_required must be True")
    cs = f.get("confidence_score")
    _check(isinstance(cs, (int, float)) and 0.0 < cs <= 0.88, f"{label}: confidence {cs} out of safe range")
    _check(f.get("severity") in ("High", "Medium"), f"{label}: severity {f.get('severity')} invalid")


def _assert_no_forbidden(findings, label):
    for f in findings:
        tag = str(f.get("fault_tag", "")).lower()
        for part in FORBIDDEN_TAG_PARTS:
            _check(part not in tag, f"{label}: forbidden tag part '{part}' in '{tag}'")


def _analyze(frames, stroke):
    return analyze_pose_data(
        pose_results=frames, frames=[], fps=FPS, total_duration=9.8,
        stroke_type=stroke, camera_angle="Side", video_upload_id="synthetic",
    )


def _set_flag(value):
    if value is None:
        os.environ.pop("EXTENDED_STROKE_FINDINGS", None)
    else:
        os.environ["EXTENDED_STROKE_FINDINGS"] = value


def main() -> int:
    # Locked finding shape derived from the real freestyle path.
    _set_flag(None)
    base = _analyze(freestyle_frames(), "Freestyle")
    _check(base.get("findings"), "freestyle baseline must produce a finding")
    locked_keys = set(base["findings"][0])

    try:
        # 1. Direct helpers: each new fault can fire, shape-safe
        bk = _backstroke_extended_findings(backstroke_frames(), FPS)
        _check(NEW_BACKSTROKE.issubset(_tags(bk)), f"backstroke extended should fire both: {_tags(bk)}")
        for f in bk:
            _assert_finding_safe(f, "backstroke_extended", locked_keys)

        fly = _butterfly_extended_findings(butterfly_frames(), FPS)
        _check(NEW_BUTTERFLY.issubset(_tags(fly)), f"butterfly extended should fire both: {_tags(fly)}")
        for f in fly:
            _assert_finding_safe(f, "butterfly_extended", locked_keys)

        # 2. Flag OFF: no new tags via the full analysis path
        _set_flag("false")
        bk_off = _analyze(backstroke_frames(), "Backstroke")
        _check(not (_tags(bk_off["findings"]) & NEW_BACKSTROKE),
               f"flag off must not emit new backstroke tags: {_tags(bk_off['findings'])}")
        fly_off = _analyze(butterfly_frames(), "Butterfly")
        _check(not (_tags(fly_off["findings"]) & NEW_BUTTERFLY),
               f"flag off must not emit new butterfly tags: {_tags(fly_off['findings'])}")

        # 3. Flag ON: new tags can appear via the full analysis path
        _set_flag("true")
        bk_on = _analyze(backstroke_frames(), "Backstroke")
        _check(_tags(bk_on["findings"]) & NEW_BACKSTROKE,
               f"flag on should emit new backstroke tags: {_tags(bk_on['findings'])}")
        fly_on = _analyze(butterfly_frames(), "Butterfly")
        _check(_tags(fly_on["findings"]) & NEW_BUTTERFLY,
               f"flag on should emit new butterfly tags: {_tags(fly_on['findings'])}")

        # 4. Shape + safety on every analysis-path finding; no forbidden tags
        for label, res in (("bk_on", bk_on), ("fly_on", fly_on), ("bk_off", bk_off), ("fly_off", fly_off)):
            _assert_no_forbidden(res["findings"], label)
            for f in res["findings"]:
                _assert_finding_safe(f, label, locked_keys)
    finally:
        _set_flag(None)

    print("stroke balance tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
