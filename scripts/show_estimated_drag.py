#!/usr/bin/env python3
"""
Pretty-print the `estimated_drag` block from a worker callback payload, so the
pilot is easy to eyeball on a real clip (step 4 of the rollout).

Usage:
    python3 scripts/show_estimated_drag.py path/to/callback.json
    pbpaste | python3 scripts/show_estimated_drag.py        # from clipboard
    cat callback.json | python3 scripts/show_estimated_drag.py

Dependency-free (stdlib only). Also sanity-checks that no swimmer height/mass
leaked into the block and flags an implausible drag magnitude.
"""
import json
import sys

_RAMP = " .:-=+*#%@"


def spark(values, width=46):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    n = len(vals)
    cols = min(width, n)
    out = []
    for i in range(cols):
        v = vals[int(i * n / cols)]
        if hi - lo < 1e-9:
            out.append(_RAMP[0])
        else:
            out.append(_RAMP[int((v - lo) / (hi - lo) * (len(_RAMP) - 1))])
    return "".join(out)


def main():
    try:
        raw = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
        payload = json.loads(raw)
    except Exception as e:
        print(f"Could not read/parse callback JSON: {e}")
        return 2

    ed = payload.get("estimated_drag")
    if ed is None:
        print("No `estimated_drag` in this callback.")
        print("Expected when: ENABLE_ESTIMATED_DRAG is off, no swimmer height/mass was")
        print("sent, pose was unreliable, or analysis fell back to manual review.")
        print(f"  analysis_mode      : {payload.get('analysis_mode')}")
        print(f"  real_pose_detected : {payload.get('real_pose_detected')}")
        print(f"  pose_reliability   : {payload.get('pose_reliability')}")
        return 0

    summary = ed.get("summary", {})
    series = ed.get("series", {})
    leak = [k for k in ("height_cm", "mass_kg", "height_m") if k in json.dumps(ed)]

    print("=" * 60)
    print("  ESTIMATED DRAG  (pilot estimate -- NOT a measurement)")
    print("=" * 60)
    print(f"  stroke               : {ed.get('stroke')}")
    print(f"  pose source          : {ed.get('pose_source')}")
    print(f"  confidence_low       : {ed.get('confidence_low')}")
    print(f"  reliable frame frac  : {ed.get('reliable_frame_fraction')}")
    print(f"  scale (m / unit)     : {ed.get('scale_m_per_unit')}")
    print(f"  ref frame / analysed : {ed.get('scale_reference_frame')} / {ed.get('frames_analysed')}")
    print("  " + "-" * 56)
    print(f"  mean drag            : {summary.get('mean_drag_force_n')} N")
    print(f"  peak drag            : {summary.get('peak_drag_force_n')} N")
    print(f"  mean drag / weight   : {summary.get('mean_drag_to_weight_ratio')}")
    print(f"  peak velocity        : {summary.get('peak_velocity_m_s')} m/s")
    if "mean_propulsive_force_n" in summary:
        print(f"  mean propulsive      : {summary.get('mean_propulsive_force_n')} N")
        print(f"  peak propulsive      : {summary.get('peak_propulsive_force_n')} N")
    else:
        print("  propulsive / net     : (omitted -- confidence_low)")

    v = series.get("velocity_m_s") or []
    d = series.get("drag_force_n") or []
    if v:
        print("  " + "-" * 56)
        print(f"  velocity {min(v):4.2f}-{max(v):4.2f} m/s |{spark(v)}|")
    if d:
        print(f"  drag     {min(d):4.0f}-{max(d):4.0f} N   |{spark(d)}|")
    print("=" * 60)

    md = summary.get("mean_drag_force_n")
    if isinstance(md, (int, float)) and (md < 10 or md > 200):
        print(f"  [!] mean drag {md} N is outside the usual ~20-120 N band -- "
              "check the scale and the footage/camera angle.")
    print(f"  [{'X' if leak else 'ok'}] profile leak check: "
          f"{('LEAKED ' + str(leak)) if leak else 'no height/mass in block'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
