"""
Acceptance tests for app/pose_worker_integration.py against the REAL MediaPipe
pose schema. Run: python3 scripts/test_drag_integration.py
Exits non-zero on any failure.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import re  # noqa: E402
import numpy as np  # noqa: E402
from app.pose_worker_integration import (  # noqa: E402
    analyse_clip, synthetic_pose_results, HIP_L, HIP_R,
    estimated_drag_enabled, should_emit_estimated_drag,
    estimate_scale, froude_wave_factor, wave_drag_enabled,
)

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def max_gap(payload):
    return max(abs(d - p) for d, p in zip(
        payload["series"]["drag_force_n"],
        payload["series"].get("propulsive_force_n", payload["series"]["drag_force_n"])))


# 1) Realistic continuous ~6 s clip ---------------------------------------
pr = synthetic_pose_results(fps=30, seconds=6.0, true_v=1.6, height_cm=180.0)
p = analyse_clip(pr, fps=30, height_cm=180.0, mass_kg=75.0, stroke="Freestyle")
check("1. produces estimated_drag for a tracked clip", p is not None)
mean_drag = p["summary"]["mean_drag_force_n"]
# believable: anthropometric_drag demo is 78 N at 2.0 m/s; ~50 N expected at 1.6
check("1a. drag in believable range (20-120 N, order of demo)", 20 <= mean_drag <= 120,
      f"mean_drag={mean_drag} N @ v_peak={p['summary']['peak_velocity_m_s']}")
check("1b. edge-fix: max|propulsive-drag| <= 20 N", max_gap(p) <= 20.0, f"{max_gap(p):.1f} N")
check("1c. scale recovered within 10% of truth (3.6)",
      abs(p["scale_m_per_unit"] - 3.6) / 3.6 <= 0.10, f"scale={p['scale_m_per_unit']}")
blob = json.dumps(p)
check("1d. privacy: no height_cm/mass_kg/height_m in payload",
      ("height_cm" not in blob) and ("mass_kg" not in blob) and ("height_m" not in blob))
check("1e. confident clip exposes propulsive fields",
      (p["confidence_low"] is False) and ("propulsive_force_n" in p["series"]))


# 2) Task-3 length: 10 s clip, sampled every 3rd frame --------------------
pr10 = synthetic_pose_results(fps=30, seconds=10.0, frame_step=3, true_v=1.5,
                              height_cm=172.0, body_len_units=0.48, seed=4)
p10 = analyse_clip(pr10, fps=30, height_cm=172.0, mass_kg=68.0, stroke="Freestyle")
check("2. 10 s sampled clip produces drag", p10 is not None,
      None if p10 is None else f"frames_analysed={p10['frames_analysed']}")
if p10:
    check("2a. drag believable on 10 s clip (20-120 N)",
          20 <= p10["summary"]["mean_drag_force_n"] <= 120,
          f"mean_drag={p10['summary']['mean_drag_force_n']} N")
    check("2b. edge-fix holds on 10 s clip", max_gap(p10) <= 20.0, f"{max_gap(p10):.1f} N")


# 3) Sparse hips -> low confidence, drag stays, propulsive hidden ----------
sparse = synthetic_pose_results(fps=30, seconds=6.0, true_v=1.6, height_cm=180.0, seed=2)
for i, fr in enumerate(sparse):
    if i % 5 >= 2:                       # keep hips in only ~40% of frames
        fr["landmarks"].pop(HIP_L, None)
        fr["landmarks"].pop(HIP_R, None)
psp = analyse_clip(sparse, fps=30, height_cm=180.0, mass_kg=75.0)
check("3. sparse-hip clip still returns drag", psp is not None)
if psp:
    check("3a. sparse clip flagged confidence_low", psp["confidence_low"] is True,
          f"reliable_frac={psp['reliable_frame_fraction']}")
    check("3b. drag_force_n still present when low-confidence",
          "drag_force_n" in psp["series"])
    check("3c. propulsive/net OMITTED when low-confidence",
          ("propulsive_force_n" not in psp["series"]) and ("net_force_n" not in psp["series"]))


# 4) Missing profile -> None (never blocks the pipeline) ------------------
check("4. missing height_cm -> None",
      analyse_clip(pr, fps=30, height_cm=None, mass_kg=75.0) is None)
check("4b. missing mass_kg -> None",
      analyse_clip(pr, fps=30, height_cm=180.0, mass_kg=None) is None)


# 5) No scalable frame (no nose/ankles) -> None --------------------------
no_scale = synthetic_pose_results(fps=30, seconds=4.0)
for fr in no_scale:
    for k in ("nose", "left_ankle", "right_ankle"):
        fr["landmarks"].pop(k, None)
check("5. no nose+ankle frame -> None (graceful skip)",
      analyse_clip(no_scale, fps=30, height_cm=180.0, mass_kg=75.0) is None)


# 6) Feature flag: OFF by default --------------------------------------------
check("6. flag OFF when env missing", estimated_drag_enabled({}) is False)
check("6a. flag OFF when 'false'",
      estimated_drag_enabled({"ENABLE_ESTIMATED_DRAG": "false"}) is False)
check("6b. flag ON for true/TRUE/1/yes/on",
      all(estimated_drag_enabled({"ENABLE_ESTIMATED_DRAG": v})
          for v in ["true", "TRUE", "1", "yes", "on", "On"]))
check("6c. flag OFF for junk/empty values",
      not any(estimated_drag_enabled({"ENABLE_ESTIMATED_DRAG": v})
              for v in ["0", "no", "maybe", ""]))

# 7) Single-source gate combining flag + pose + profile + mode ---------------
ON = {"ENABLE_ESTIMATED_DRAG": "true"}
OFF = {"ENABLE_ESTIMATED_DRAG": "false"}
def emit(env, mode="real_pose", real=True, h=180.0, m=75.0):
    return should_emit_estimated_drag(
        analysis_mode=mode, real_pose_detected=real, height_cm=h, mass_kg=m, env=env)
check("7. disabled flag -> no emit even with valid pose+profile", emit(OFF) is False)
check("7a. enabled + valid pose + profile -> emit", emit(ON) is True)
check("7b. enabled + missing height -> no emit", emit(ON, h=None) is False)
check("7c. enabled + missing mass -> no emit", emit(ON, m=None) is False)
check("7d. enabled + manual-review fallback -> no emit",
      emit(ON, mode="manual_review", real=False) is False)
check("7e. enabled + suppressed (mode not real_pose) -> no fake force fields",
      emit(ON, mode="manual_review", real=True) is False)

# 8) Fixtures: canonical success default-off; pilot fixture clean ------------
FIX = ROOT / "fixtures"
success = json.load(open(FIX / "callback_success.example.json"))
check("8. canonical callback_success has NO estimated_drag (default off)",
      "estimated_drag" not in success)
pilot = json.load(open(FIX / "callback_estimated_drag_pilot.example.json"))
check("8a. pilot fixture HAS estimated_drag", "estimated_drag" in pilot)
pblob = json.dumps(pilot)
check("8b. pilot fixture never leaks height/mass/height_m",
      ("height_cm" not in pblob) and ("mass_kg" not in pblob) and ("height_m" not in pblob))
CALLBACK_REQUIRED = {"job_id", "server_job_id", "video_upload_id", "engine", "status",
                     "analysis_mode", "real_pose_detected", "findings", "overall_score",
                     "phase_breakdown", "quality_flags", "recommended_next_action"}
check("8c. pilot fixture keeps all required callback keys", CALLBACK_REQUIRED.issubset(pilot))
_UNSAFE = [re.compile(p, re.I) for p in
           [r"token=", r"ai_webhook_secret", r"/Users/|/home/|/var/folders/|/tmp/", r"owen\s+weaver"]]
def _walk(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _walk(x)
    elif isinstance(v, list):
        for x in v:
            yield from _walk(x)
check("8d. pilot fixture has no unsafe strings",
      not any(p.search(s) for s in _walk(pilot) for p in _UNSAFE))


# 9) Drag/metrics upgrades --------------------------------------------------
# Robust scale: a single outlier frame (ankle mis-detected far away) must NOT
# fool the body-length estimate the way the old single-max did.
pr_out = synthetic_pose_results(fps=30, seconds=6.0, true_v=1.6, height_cm=180.0, seed=5)
pr_out[len(pr_out) // 2]["landmarks"]["right_ankle"] = {"x": 5.0, "y": 5.0, "visibility": 0.9}
sc = estimate_scale(pr_out, 1.80)
check("9. robust scale ignores a single outlier frame",
      sc is not None and abs(sc["scale_m_per_unit"] - 3.6) / 3.6 <= 0.12,
      None if sc is None else round(sc["scale_m_per_unit"], 3))

# Calibration override is used verbatim.
pcal = analyse_clip(pr, fps=30, height_cm=180.0, mass_kg=75.0, scale_m_per_unit_override=3.0)
check("9a. calibration override used", pcal is not None and pcal["scale_m_per_unit"] == 3.0)
check("9b. override basis says 'calibrated'", pcal is not None and "calibrated" in pcal["basis"])

# Intra-cycle velocity metrics present.
check("9c. intra-cycle velocity fields present",
      p is not None and "velocity_drop_ratio" in p["summary"] and "min_velocity_m_s" in p["summary"])

# Froude wave factor: identity at v=0, >1 and capped for v>0.
check("9d. froude factor = 1.0 at v=0", froude_wave_factor(0.0, 0.5) == 1.0)
wf = froude_wave_factor(2.0, 0.5)
check("9e. froude factor > 1 and capped <= 1.6", 1.0 < wf <= 1.6, round(wf, 3))
check("9f. ENABLE_WAVE_DRAG off by default", wave_drag_enabled({}) is False)


print("\n" + "=" * 56)
nfail = results.count(False)
print(f"{results.count(True)}/{len(results)} checks passed")
print("ALL DRAG-INTEGRATION TESTS PASSED" if nfail == 0 else f"{nfail} FAILED")
sys.exit(1 if nfail else 0)
