"""Unit tests for app/swimxyz_adapter.py (pure NumPy; footage-free end-to-end)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from app.swimxyz_adapter import (  # noqa: E402
    joints_to_pose_results, keypoint_errors, COCO17_TO_WORKER,
)
from app.pose_worker_integration import analyse_clip  # noqa: E402
from app.swim_analyzer import analyze_pose_data  # noqa: E402  (mediapipe-free now)

results = []
def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def synth_coco17(frames=40, w=1920, h=1080, step_px=16):
    """A horizontal swimmer (body along x) crossing the frame, in COCO-17 order."""
    # x offsets from hip-centre (pixels), y around mid-frame.
    off = {0: (300, 0), 1: (310, -5), 2: (310, 5), 3: (290, -10), 4: (290, 10),
           5: (150, -15), 6: (150, 15), 7: (50, -20), 8: (50, 20),
           9: (-50, -25), 10: (-50, 25), 11: (-30, -5), 12: (-30, 5),
           13: (-180, 0), 14: (-180, 0), 15: (-300, 0), 16: (-300, 0)}
    arr = np.zeros((frames, 17, 2))
    for k in range(frames):
        cx = 400 + k * step_px
        cy = h / 2
        for j, (dx, dy) in off.items():
            arr[k, j, 0] = cx + dx
            arr[k, j, 1] = cy + dy
    return arr


coco = synth_coco17()
pr = joints_to_pose_results(coco, image_size=(1920, 1080), fps=30)

# 1. shape + naming + normalisation
check("1. one pose_results entry per frame", len(pr) == 40)
names = set(pr[0]["landmarks"].keys())
check("1a. worker landmark names (not COCO indices)",
      {"nose", "left_shoulder", "right_ankle", "left_hip"} <= names)
check("1b. eyes dropped (COCO 1,2 not mapped)",
      "left_eye" not in names and "right_eye" not in names)
nx = pr[0]["landmarks"]["nose"]["x"]
check("1c. coords normalised to 0..1", 0.0 <= nx <= 1.0, f"nose.x={nx:.3f}")
check("1d. keypoint_count = 15 mapped joints", pr[0]["keypoint_count"] == 15,
      str(pr[0]["keypoint_count"]))

# 2. flows through the drag pipeline (footage-free)
drag = analyse_clip(pr, fps=30, height_cm=180.0, mass_kg=75.0, stroke="Freestyle")
check("2. analyse_clip produces drag from ground-truth joints", drag is not None)
check("2a. drag magnitude positive", drag and drag["summary"]["mean_drag_force_n"] > 0,
      None if drag is None else drag["summary"]["mean_drag_force_n"])

# 3. flows through the findings engine (no mediapipe)
analysis = analyze_pose_data(pose_results=pr, frames=list(range(len(pr))), fps=30,
                             total_duration=len(pr) / 30, stroke_type="Freestyle",
                             camera_angle="Side", video_upload_id="swimxyz-test")
check("3. analyze_pose_data runs on adapted joints",
      isinstance(analysis, dict) and "findings" in analysis,
      f"mode={analysis.get('analysis_mode')}")

# 4. visibility gating: occlude both ankles -> dropped
visib = np.ones((40, 17))
visib[:, 15] = 0.0
visib[:, 16] = 0.0
pr_occ = joints_to_pose_results(coco, image_size=(1920, 1080), visibility=visib)
check("4. occluded joints omitted", "left_ankle" not in pr_occ[0]["landmarks"]
      and pr_occ[0]["keypoint_count"] == 13)

# 5. accuracy scorer: identical -> 0 error / PCK 1.0; perturbed -> >0 error
same = keypoint_errors(pr, pr)
check("5. identical pose -> mean_error 0", same["mean_error"] == 0.0)
check("5a. identical pose -> PCK@0.05 = 1.0", same["pck_0.05"] == 1.0)

pred = joints_to_pose_results(coco + 40.0, image_size=(1920, 1080))  # shift ~40px
err = keypoint_errors(pred, pr)
check("5b. shifted prediction -> positive mean_error", err["mean_error"] > 0,
      f"mean_error={err['mean_error']}")

print("\n" + "=" * 52)
nfail = results.count(False)
print(f"{results.count(True)}/{len(results)} passed",
      "ALL PASS" if nfail == 0 else f"{nfail} FAILED")
sys.exit(1 if nfail else 0)
