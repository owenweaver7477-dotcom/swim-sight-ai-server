#!/usr/bin/env python3
"""Safe, dependency-free tests for structured AI report output requests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.report_outputs import (  # noqa: E402
    attach_report_output_metadata,
    build_report_output_plan,
    filter_findings_for_outputs,
)


def request(**overrides):
    values = {
        "selected_report_outputs": ["body_line_analysis"],
        "camera_angle": "Side",
        "athlete_profile_readiness": {},
        "swimmer_height_cm": None,
        "swimmer_mass_kg": None,
        "calibration_available": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


core = build_report_output_plan(request(), env={})
assert core["accepted_outputs"] == ["body_line_analysis"]
assert core["skipped_outputs"] == []

unknown = build_report_output_plan(
    request(selected_report_outputs=["body_line_analysis", "not_a_real_output"]),
    env={},
)
assert unknown["accepted_outputs"] == ["body_line_analysis"]
assert unknown["skipped_outputs"][0]["id"] == "not_a_real_output"

missing_drag = build_report_output_plan(
    request(selected_report_outputs=["estimated_drag_force"]),
    env={"ENABLE_ESTIMATED_DRAG": "true"},
)
assert missing_drag["accepted_outputs"] == []
assert "body mass" in missing_drag["skipped_outputs"][0]["reason"]

eligible_drag = build_report_output_plan(
    request(
        selected_report_outputs=["estimated_drag_force"],
        athlete_profile_readiness={
            "body_mass_available": True,
            "height_available": True,
            "calibration_available": True,
        },
        swimmer_height_cm=180,
        swimmer_mass_kg=72,
        calibration_available=True,
    ),
    env={"ENABLE_ESTIMATED_DRAG": "true"},
)
assert eligible_drag["accepted_outputs"] == ["estimated_drag_force"]

payload = attach_report_output_metadata({
    "analysis_mode": "real_pose",
    "real_pose_detected": True,
    "findings": [],
    "phase_breakdown": {},
    "estimated_drag": {"label": "Estimated drag force - coach review required"},
}, eligible_drag)
assert payload["completed_outputs"] == ["estimated_drag_force"]
assert payload["estimate_only_outputs"] == ["estimated_drag_force"]

manual = attach_report_output_metadata({
    "analysis_mode": "manual_review",
    "real_pose_detected": False,
    "findings": [],
}, core)
assert manual["completed_outputs"] == []
assert manual["skipped_outputs"][0]["id"] == "body_line_analysis"

filtered = filter_findings_for_outputs({
    "findings": [
        {"finding_title": "Body line needs coach review"},
        {"finding_title": "Breathing timing needs coach review"},
    ],
}, core)
assert len(filtered["findings"]) == 1
assert "Body line" in filtered["findings"][0]["finding_title"]

serialized = json.dumps(payload).lower()
for private_field in ("swimmer_height_cm", "swimmer_mass_kg", "body_mass", "approximate_height"):
    assert private_field not in serialized

legacy = build_report_output_plan(request(selected_report_outputs=[]), env={})
assert legacy["legacy_request"] is True
assert attach_report_output_metadata({"analysis_mode": "real_pose"}, legacy) == {"analysis_mode": "real_pose"}

print("report output selection tests passed")
