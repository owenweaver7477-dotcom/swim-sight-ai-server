#!/usr/bin/env python3
"""Tests for app/analysis_options.py — the pre-analysis checklist contract.

Verifies the trust-boundary behaviour: tier gating, data-dependency resolution,
select-all expansion, invalid/unknown rejection, manual-only mode, and that the
AI instruction block carries the honesty rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis_options import (  # noqa: E402
    analysis_options_enabled,
    index_options,
    load_catalog,
    plan_analysis,
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


def rejected_for(result, option_id, reason):
    return any(r["id"] == option_id and r["reason"] == reason for r in result["rejected"])


catalog = load_catalog()
options = index_options(catalog)

check("flag default OFF", analysis_options_enabled({}) is False)
check("flag on when truthy", analysis_options_enabled({"ENABLE_ANALYSIS_OPTIONS": "1"}) is True)
check("catalog has 6 panels", len(catalog["panels"]) == 6)
check("every option has id/label/kind/control/tier",
      all(all(k in o for k in ("id", "label", "kind", "control", "tier")) for o in options.values()))

# Free plan cannot reach a premium metric.
r = plan_analysis({"stroke": "freestyle", "camera_angle": "side", "metric_dps": True}, plan="coach_studio")
check("free: context accepted", "stroke" in r["accepted"] and "camera_angle" in r["accepted"])
check("free: premium metric tier_locked", rejected_for(r, "metric_dps", "tier_locked"))

# Free metric works when selected.
r = plan_analysis({"stroke": "freestyle", "camera_angle": "side", "metric_stroke_rate": True}, plan="coach_studio")
check("free: stroke rate accepted", any(m["id"] == "metric_stroke_rate" for m in r["metrics"]))

# AI Assist: DPS needs a scale.
r = plan_analysis({"stroke": "freestyle", "camera_angle": "side", "metric_dps": True}, plan="ai_assist")
check("ai: dps missing scale", rejected_for(r, "metric_dps", "missing_data"))
r = plan_analysis({"stroke": "freestyle", "camera_angle": "side",
                   "calibrate_known_distance": True, "metric_dps": True}, plan="ai_assist")
check("ai: dps accepted once calibrated", any(m["id"] == "metric_dps" for m in r["metrics"]))

# Force estimate needs scale + mass.
r = plan_analysis({"stroke": "freestyle", "camera_angle": "side",
                   "calibrate_known_distance": True, "metric_force": True}, plan="ai_assist")
check("ai: force missing mass", any(rr["id"] == "metric_force" and "mass_kg" in rr.get("missing", [])
                                    for rr in r["rejected"]))
r = plan_analysis({"stroke": "freestyle", "camera_angle": "side", "calibrate_known_distance": True,
                   "swimmer_mass_kg": 72, "metric_force": True}, plan="ai_assist")
check("ai: force accepted + flagged estimate",
      any(m["id"] == "metric_force" and m["estimate"] for m in r["metrics"]))

# Elite metric gated; unlocked by pool calibration on elite plan.
r = plan_analysis({"stroke": "freestyle", "camera_angle": "top_front",
                   "metric_absolute_position": True}, plan="ai_assist")
check("ai: elite metric tier_locked", rejected_for(r, "metric_absolute_position", "tier_locked"))
r = plan_analysis({"stroke": "freestyle", "camera_angle": "top_front", "pool_line_autocalibrate": True,
                   "metric_absolute_position": True}, plan="elite_lab")
check("elite: absolute position accepted w/ pool calib",
      any(m["id"] == "metric_absolute_position" for m in r["metrics"]))

# Select-all expands within the plan tier only.
r = plan_analysis({"stroke": "breaststroke", "camera_angle": "side", "faults_select_all": True}, plan="ai_assist")
check("select-all (ai): includes premium + free faults",
      "fault_catch" in r["faults"] and "fault_body_line" in r["faults"])
r = plan_analysis({"stroke": "breaststroke", "camera_angle": "side", "faults_select_all": True}, plan="coach_studio")
check("select-all (free): only free faults", "fault_body_line" in r["faults"] and "fault_catch" not in r["faults"])

# Invalid + unknown.
r = plan_analysis({"stroke": "flyyy", "camera_angle": "side", "made_up": True}, plan="coach_studio")
check("invalid stroke choice rejected", rejected_for(r, "stroke", "invalid_choice"))
check("unknown option rejected", rejected_for(r, "made_up", "unknown_option"))

# Manual-only: no AI faults, metrics still computed.
r = plan_analysis({"stroke": "freestyle", "camera_angle": "side", "mode": "manual_only",
                   "fault_body_line": True, "metric_stroke_rate": True}, plan="coach_studio")
check("manual: ai_instructions flagged manual", r["ai_instructions"]["manual_only"] is True)
check("manual: no faults sent to AI", r["ai_instructions"]["check_faults"] == [])
check("manual: metric still computed", any(m["id"] == "metric_stroke_rate" for m in r["metrics"]))

# Honesty rule + locked-on defaults + required-context warning.
check("honesty rule present", "looks fine" in r["ai_instructions"]["prompt_text"].lower())
check("confidence labels locked on", r["context"].get("show_confidence") in (True, "true"))
r = plan_analysis({"camera_angle": "side"}, plan="coach_studio")
check("missing stroke warns", any("stroke" in w for w in r["warnings"]))

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
