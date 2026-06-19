"""Unit tests for app/findings_robust.py (pure NumPy)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.findings_robust import robust_peak, robust_findings_enabled  # noqa: E402

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def series(vals):
    return [{"value": v, "timestamp": i * 0.1} for i, v in enumerate(vals)]

# 1. One-off spike -> suppressed (None)
spike = series([0.1, 0.1, 0.1, 2.0, 0.1, 0.1, 0.1, 0.1])
check("1. one-off spike returns None (rejected)", robust_peak(spike) is None)

# 2. Sustained signal -> accepted, strength is a percentile (<= max)
sustained = series([1.7, 1.85, 1.9, 2.0, 1.8, 1.75])
rp = robust_peak(sustained)
check("2. sustained signal accepted", rp is not None)
check("2a. strength is percentile (<= raw max 2.0)", rp and rp["strength"] <= 2.0,
      None if rp is None else round(rp["strength"], 3))
check("2b. strength uses the high values (>1.5)", rp and rp["strength"] > 1.5,
      None if rp is None else round(rp["strength"], 3))

# 3. Borderline: exactly 'sustain' high frames accepted
two_high = series([0.1, 2.0, 2.0, 0.1, 0.1])      # only 2 within 0.6*peak
check("3. fewer than sustain(3) high frames -> None", robust_peak(two_high) is None)
three_high = series([0.1, 2.0, 2.0, 2.0, 0.1])    # 3 within 0.6*peak
check("3a. >= sustain high frames -> accepted", robust_peak(three_high) is not None)

# 3b. Three isolated spikes are not a sustained temporal signal.
separated = [
    {"value": 2.0, "timestamp": 0.0},
    {"value": 2.0, "timestamp": 3.0},
    {"value": 2.0, "timestamp": 6.0},
]
check("3b. separated high frames -> None", robust_peak(separated) is None)

# 4. Empty / all-zero -> None
check("4. empty -> None", robust_peak([]) is None)
check("4a. all-zero -> None", robust_peak(series([0, 0, 0, 0])) is None)

# 5. flag default off
check("5. ROBUST_FINDINGS off by default", robust_findings_enabled({}) is False)
check("5a. flag on when truthy", robust_findings_enabled({"ROBUST_FINDINGS": "1"}) is True)

print("\n" + "=" * 50)
nfail = results.count(False)
print(f"{results.count(True)}/{len(results)} passed",
      "ALL PASS" if nfail == 0 else f"{nfail} FAILED")
sys.exit(1 if nfail else 0)
