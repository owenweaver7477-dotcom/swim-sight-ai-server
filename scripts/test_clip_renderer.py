#!/usr/bin/env python3
"""Tests for app/clip_renderer.py.

The assembly logic (build_render_plan) is the safety-critical part and is fully
checked here: only coach-approved + public items appear, drag blocks and PII are
excluded, callouts are capped and sorted, and nothing but {frame_idx, text}
survives. The actual encode is smoke-tested and skipped if no encoder exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.clip_renderer import (  # noqa: E402
    ClipRenderUnavailable,
    build_render_plan,
    render_clip,
    share_clip_enabled,
)

_passed = 0
_failed = 0


def check(name, condition):
    global _passed, _failed
    print(("PASS" if condition else "FAIL") + " - " + name)
    if condition:
        _passed += 1
    else:
        _failed += 1


# Flag defaults.
check("flag default OFF", share_clip_enabled({}) is False)
check("flag on when truthy", share_clip_enabled({"ENABLE_SHARE_CLIP": "yes"}) is True)

items = [
    {"status": "approved", "is_public": True, "key_frame": 30, "coach_text": "Lengthen the glide",
     "swimmer_name": "Jane Doe", "internal_notes": "parent emailed", "email": "jane@example.com"},
    {"status": "draft", "is_public": True, "key_frame": 10, "coach_text": "should be excluded (unapproved)"},
    {"status": "approved", "is_public": False, "key_frame": 12, "coach_text": "excluded (private)"},
    {"status": "approved", "is_public": True, "category": "drag", "key_frame": 14, "coach_text": "excluded (drag)"},
    {"status": "approved", "is_public": True, "key_frame": 18},  # no text -> excluded
    {"status": "approved", "is_public": True, "coach_text": "no frame -> excluded"},  # no frame
]
plan = build_render_plan(items)

check("approved+public finding included", plan["included_count"] == 1)
check("unapproved excluded", plan["excluded"]["unapproved"] == 1)
check("private excluded", plan["excluded"]["private"] == 1)
check("drag excluded", plan["excluded"]["drag"] == 1)
check("missing frame/text excluded", plan["excluded"]["no_frame_or_text"] == 2)

callout = plan["callouts"][0]
check("callout carries only {frame_idx, text}", set(callout.keys()) == {"frame_idx", "text"})
leaked = "Jane" in str(callout) or "example.com" in str(callout) or "parent emailed" in str(callout)
check("no PII / internal notes leak into callout", not leaked)

# Cap + sort.
many = [
    {"status": "approved", "is_public": True, "key_frame": 50, "coach_text": "d"},
    {"status": "approved", "is_public": True, "key_frame": 10, "coach_text": "a"},
    {"status": "approved", "is_public": True, "key_frame": 30, "coach_text": "c"},
    {"status": "approved", "is_public": True, "key_frame": 20, "coach_text": "b"},
]
capped = build_render_plan(many, max_callouts=3)
check("capped to max_callouts", capped["included_count"] == 3)
check("over-cap counted", capped["excluded"]["over_cap"] == 1)
check("callouts sorted by frame_idx", [c["frame_idx"] for c in capped["callouts"]] == [10, 20, 30])

# Branding sanitisation.
branded = build_render_plan(
    [{"status": "approved", "is_public": True, "key_frame": 5, "coach_text": "x"}],
    branding={"club_name": "Harbour Swim", "link": "https://swimsight.example",
              "secret_token": "should-not-survive"},
)
check("branding keeps only safe fields",
      set(branded["branding"].keys()) <= {"club_name", "logo_path", "link"} and "secret_token" not in str(branded["branding"]))
check("footer attribution present", "SwimSight" in plan["footer"])

# Smoke encode (skipped cleanly if no encoder).
try:
    import numpy as np
    frames = [(i, np.zeros((120, 160, 3), dtype=np.uint8)) for i in range(12)]
    lm = {i: {"left_shoulder": {"x": 60, "y": 40}, "right_shoulder": {"x": 100, "y": 40},
              "left_hip": {"x": 64, "y": 80}, "right_hip": {"x": 96, "y": 80}} for i in range(12)}
    out = render_clip(frames, lm, build_render_plan(
        [{"status": "approved", "is_public": True, "key_frame": 5, "coach_text": "Lengthen glide"}]),
        "/tmp/_swimsight_smoke.mp4", fps=12, max_seconds=2)
    size = Path(out).stat().st_size if Path(out).exists() else 0
    check("smoke encode produced a non-empty mp4", size > 0)
except ClipRenderUnavailable:
    print("SKIP - smoke encode (no opencv/ffmpeg encoder in this env)")
except Exception as exc:  # pragma: no cover
    check(f"smoke encode raised unexpectedly: {exc}", False)

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
