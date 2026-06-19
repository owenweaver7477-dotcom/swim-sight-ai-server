"""Unit tests for app/pose_postprocess.py (pure NumPy)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_postprocess import smooth_pose_results, pose_smoothing_enabled  # noqa: E402

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

NAMES = ["nose", "left_hip", "right_hip", "left_ankle", "right_ankle",
         "left_shoulder", "right_shoulder"]

def frame(i, drop=(), nose_x=0.5):
    lm = {}
    for nm in NAMES:
        if nm in drop:
            continue
        x = nose_x if nm == "nose" else 0.5
        lm[nm] = {"x": x, "y": 0.5, "visibility": 0.9}
    return {"frame_idx": i, "pose_detected": len(lm) >= 4, "keypoint_count": len(lm), "landmarks": lm}

# left_hip: short gap at frame 4 (present 3 & 5). right_ankle: long gap 4..8.
# nose: single-frame x outlier at frame 5.
pr = []
for i in range(10):
    drop = []
    if i == 4:
        drop.append("left_hip")
    if 4 <= i <= 8:
        drop.append("right_ankle")
    pr.append(frame(i, drop=drop, nose_x=(0.95 if i == 5 else 0.5)))

out = smooth_pose_results(pr, window=3, max_gap=2)

# 1. short gap filled + marked interpolated + below-gate visibility
lh4 = out[4]["landmarks"].get("left_hip")
check("1. short gap (left_hip@4) interpolated", lh4 is not None and lh4.get("interpolated") is True,
      str(lh4))
check("1a. interpolated visibility below the 0.45 count gate",
      lh4 is not None and lh4["visibility"] < 0.45, None if lh4 is None else lh4["visibility"])

# 2. long gap NOT filled
check("2. long gap (right_ankle@4-8) NOT filled",
      all("right_ankle" not in out[i]["landmarks"] for i in (4, 5, 6, 7, 8)))

# 3. single-frame outlier pulled back toward the median
nose5 = out[5]["landmarks"]["nose"]["x"]
check("3. nose outlier @5 reduced toward 0.5", nose5 < 0.7, f"nose_x={nose5:.3f} (raw was 0.95)")

# 4. keypoint_count counts only observed (interpolated excluded)
check("4. keypoint_count@4 excludes interpolated", out[4]["keypoint_count"] == 5,
      f"kc={out[4]['keypoint_count']}")
check("4a. but interpolated landmark still available for geometry",
      "left_hip" in out[4]["landmarks"])

# 5. pose_detected recomputed
check("5. pose_detected@4 True (5 observed >= 4)", out[4]["pose_detected"] is True)

# 6. flag default off
check("6. ENABLE_POSE_SMOOTHING off by default", pose_smoothing_enabled({}) is False)
check("6a. flag on when truthy", pose_smoothing_enabled({"ENABLE_POSE_SMOOTHING": "true"}) is True)

# 7. clean clip is preserved (no spurious changes to observed counts)
clean = [frame(i) for i in range(8)]
cout = smooth_pose_results(clean)
check("7. clean clip keeps all 7 keypoints each frame",
      all(c["keypoint_count"] == 7 for c in cout))

print("\n" + "=" * 50)
nfail = results.count(False)
print(f"{results.count(True)}/{len(results)} passed",
      "ALL PASS" if nfail == 0 else f"{nfail} FAILED")
sys.exit(1 if nfail else 0)
