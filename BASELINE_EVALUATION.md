# Swim Sight 3D Baseline Evaluation

Phase 15C adds script-based contract tests and a local baseline harness for the current `pose-mvp-0.5` worker.

This is not a model upgrade. It protects the current worker contract before future RTMPose, CLAHE, Kalman smoothing, ONNX, or other AI upgrades.

## Pose Evaluation Baseline — MediaPipe vs SwimXYZ ground truth (Roadmap Phase 1)

This is the technical baseline the roadmap calls for: how the current pose backend
(MediaPipe, `POSE_BACKEND` unset) compares with labelled SwimXYZ synthetic swimming
frames. It is an evaluation floor, not a public product or coaching claim. Future
RTMPose/ViTPose ONNX candidates must improve it on representative evaluation data
before they are considered for production.

**Prepared sequence** (Breaststroke, side-above-water view):
`Side_above_water/Swimmer_Skin_0,25_Muscle_2/Water_Quantity_0,25_Height_0,6/Lighting_rotx_140_roty_280/Speed_3/position_1,75`

- 301 frames @ 1920×1080, 60 fps (from `Side_above.webm`).
- Ground truth: `baseline_data/breast_side_above/seq_joints.npy` — `(302, 17, 2)` COCO-17, **image-space pixels**.
- Frames: `baseline_data/breast_side_above/frames/` (git-ignored).

**Critical convention finding.** SwimXYZ 2D labels are in **Unity screen space (origin
bottom-left, y-UP)**. They must be flipped to image space (`image_y = 1080 − y`) or the
ground truth is upside-down and the score is meaningless. This is now built into
`scripts/swimxyz_labels_to_npy.py --flip-y --image-height 1080`, and the prepared
`.npy` already has the flip applied — verified by overlaying the skeleton on the real
frames (head/shoulders track the swimmer across the whole pass).

**Run it** (needs the worker environment with `mediapipe` and OpenCV):

```bash
python3 scripts/measure_pose_baseline.py \
  --joints baseline_data/breast_side_above/seq_joints.npy \
  --frames-dir baseline_data/breast_side_above/frames \
  --fps 60
```

The command prints a concise JSON summary and writes a dated JSON report plus a
Markdown report to `baseline_reports/`. Those generated reports and all
`baseline_data/` inputs are ignored by Git.

The same prepared-folder run is also supported:

```bash
python3 scripts/measure_pose_baseline.py \
  --sequence-dir baseline_data/breast_side_above \
  --fps 60
```

**Current recorded baseline:**

| date | backend | frames | matched keypoints | mean error | median error | PCK@0.05 | recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 | mediapipe | 301 | 1026 | 0.5081 | 0.4076 | 0.0000 | 0.2272 |

The report also contains mean error, median error, PCK@0.05, and recall for each
mapped joint.

### Compare two backends

```bash
python3 scripts/measure_pose_baseline.py \
  --sequence-dir baseline_data/breast_side_above \
  --fps 60 \
  --compare mediapipe,onnx
```

Comparison output records both backends, backend-B-minus-backend-A deltas for
mean error, PCK@0.05, and recall, and labels each change as improved, worsened,
or unchanged. `POSE_BACKEND` still defaults to MediaPipe. ONNX comparison requires
a readable `POSE_ONNX_PATH`; otherwise the command exits with a clear message
before inference starts.

### Metric definitions

- **Mean error:** mean Euclidean landmark error in normalised image-coordinate
  units across matched keypoints. Lower is better.
- **Median error:** median of the same matched-keypoint errors. Lower is better.
- **PCK@0.05:** fraction of matched keypoints no more than 0.05 normalised units
  from their labelled point. Higher is better.
- **Recall:** matched predicted keypoints divided by all mapped visible
  ground-truth keypoints. Missing frames and joints reduce recall.
- **Per-joint metrics:** the same values grouped by mapped landmark so weak body
  regions are visible instead of being hidden inside one aggregate.

**Read it honestly.** This one synthetic, monocular sequence does not establish
real-world pose quality or coaching usefulness. Its low recall and large error
make it a useful floor for later swim-specific backends, while consented real
clips and coach review remain necessary.

**Caveats / confidence.**

- The sequence was auto-identified by matching the swimmer's motion against all 576
  label sequences, then visually verified. The visible upper body aligns tightly;
  submerged torso/legs are geometrically correct but hard to verify against the
  refracted image, so treat the ground truth as approximate with tens-of-pixel uncertainty.
- Synthetic, monocular, one view, one stroke. This is a technical floor, not a
  coaching claim. Real coach-labelled clips (see below) are still required.
- To add sequences (the Aerial view, other strokes), repeat with the matching video +
  `COCO/2D_cam.txt` and `--flip-y`.

## Roadmap Phase 2 — Evaluate a candidate backend

The Phase 1 MediaPipe baseline to beat on the prepared sequence is:

- mean error: `0.5081`
- median error: `0.4076`
- PCK@0.05: `0.0000`
- recall: `0.2272`

After exporting a candidate ONNX model, run both backends against exactly the
same labels and frames:

```bash
POSE_ONNX_PATH=/path/to/rtmpose-m-swimxyz.onnx \
python3 scripts/eval_backends_against_truth.py \
  --sequence-dir baseline_data/breast_side_above \
  --fps 60 \
  --baseline mediapipe \
  --candidate onnx \
  --overlay-frames 5
```

The evaluator writes JSON and Markdown reports to `backend_eval_reports/` and
overlay images to `backend_eval_reports/overlays/`. Reports contain overall and
per-joint candidate-minus-baseline deltas:

- negative mean/median error delta: improved;
- positive PCK@0.05 or recall delta: improved;
- the opposite sign: worsened; and
- a zero delta at report precision: unchanged.

Do not trust a numeric improvement until the overlays confirm that frame
indices, coordinate direction, image scaling, and landmark names align. A model
can produce attractive numbers for the wrong mapping. This remains internal
evaluation, not a public product claim, and coach approval remains required for
product findings. `POSE_BACKEND` remains MediaPipe by default.

## What This Measures

Use the baseline harness to record:

- processing time
- video duration
- source resolution
- processing tier
- frames sampled
- frames with usable pose
- pose detection rate
- average visible keypoints
- average visible landmarks from the full internal 33-point schema
- finding count
- fallback/manual-review state
- quality flags

These metrics give future upgrades something honest to beat.

## Coach-Labelled Clip Evaluation

Copy the example manifest without committing athlete data:

```bash
cp fixtures/labelled_clip_manifest.example.json samples/labels.local.json
```

Edit `samples/labels.local.json` so each entry references a local file in
`samples/videos/` and records the coach-expected analysis mode, expected fault
tags, and any explicitly forbidden fault tags. Then run:

```bash
python3 scripts/evaluate_labelled_clips.py
```

The report records matched, missed, unexpected, and forbidden findings, plus
per-clip precision and recall. It is written to
`baseline_reports/labelled_evaluation_<timestamp>.json`, which is ignored by
git. Passing contract tests is not evidence that a feature flag improves real
coaching accuracy; labelled representative clips are required.

## Combined Upgrade Test Runner

Run every safe local upgrade check, followed by a compile check:

```bash
python3 scripts/test_upgrades.py
```

The runner executes fixture validation, worker contract tests, drag integration,
pose post-processing, robust findings, synthetic pose evaluation, temporal
metrics, labelled-evaluation logic, durable-queue configuration, and Python
compilation in order. It does not hide failed checks. Missing worker dependencies
are listed before testing.

## Synthetic Pose Logic Checks

Run the footage-free synthetic harness directly:

```bash
python3 scripts/synth_eval.py --fault none
python3 scripts/synth_eval.py --fault hip_drop
python3 scripts/synth_eval.py --fault head_lift
python3 scripts/synth_eval.py --fault dropped_elbow
python3 scripts/synth_eval.py --inject-noise --compare-flag ENABLE_POSE_SMOOTHING
```

Run its assertions with:

```bash
python3 scripts/test_synth_eval.py
```

These tests exercise downstream pose smoothing and finding logic without video
or MediaPipe. They are logic checks only. They do not measure pose-detection
quality, validate findings on real swimmers, or prove product accuracy.

Real Swim Pro-exported clips are still required to evaluate detection quality,
fallback behaviour, false positives, and missed coach-observed faults. Stock
footage may help with early detection testing only when its licence permits this
use. Neither synthetic poses nor stock footage should be presented as
coach-validated product evidence.

## Fixture Validation

Run:

```bash
python3 scripts/validate_contract_fixtures.py
```

This checks every JSON fixture in `fixtures/` for:

- valid JSON
- required contract fields
- fake/redacted signed URL placeholders
- no obvious secrets
- no real Supabase signed URL tokens
- no local absolute paths
- manual-review fixtures with zero findings

## Worker Contract Tests

Run:

```bash
python3 scripts/test_worker_contract.py
```

This uses FastAPI `TestClient` and mocks the background processing task. It verifies:

- `/health` returns the current engine
- `/process-video` accepts the documented request fixture
- accepted response shape remains stable
- `/jobs/{job_id}` returns the queued job shape
- manual-review callback fixture has zero findings

It does not download a real video and does not call Vercel.

## Baseline Evaluation With Local Clips

Put local sample clips here:

```text
samples/videos/
```

Then run:

```bash
python3 scripts/evaluate_baseline.py --stroke Freestyle --camera-angle Side
```

Optional custom paths:

```bash
python3 scripts/evaluate_baseline.py \
  --samples-dir /path/to/local/videos \
  --output-dir /path/to/local/reports \
  --stroke Breaststroke \
  --camera-angle Side
```

If no clips exist, the script exits cleanly with:

```text
No sample clips found. Add local clips to samples/videos/ to run baseline evaluation.
```

## Compare Default-Off Upgrade Flags

Put local sample clips in `samples/videos/`, then run:

```bash
python3 scripts/compare_upgrade_flags.py \
  --stroke Breaststroke \
  --camera-angle Side
```

The comparison runs each clip in a fresh Python process using:

1. baseline/default flags
2. `ENABLE_CLAHE=true`
3. `ENABLE_POSE_SMOOTHING=true`
4. `POSE_MODEL_COMPLEXITY=1`
5. `ROBUST_FINDINGS=true`
6. `SEQUENTIAL_FRAME_READ=true`
7. all safe flags together, excluding estimated drag

Reports are written locally to:

```text
baseline_reports/upgrade_comparison_<timestamp>.json
```

Each result records the clip, stroke, camera angle, flags, processing time,
sampled and detected frames, detection rate, average keypoints, fallback state,
finding count, and quality flags. `ENABLE_ESTIMATED_DRAG` is explicitly false
for every standard comparison.

Test one change at a time before trusting the combined run. A practical order is:

1. `SEQUENTIAL_FRAME_READ` for decode reliability and processing time
2. `ENABLE_CLAHE` for low-contrast pose detection
3. `ENABLE_POSE_SMOOTHING` for landmark stability
4. `POSE_MODEL_COMPLEXITY=1` for accuracy versus processing cost
5. `ROBUST_FINDINGS` for finding precision and suppression behaviour

Do not enable every flag in production merely because tests pass. Compare real,
representative swim clips first and enable only upgrades that improve useful pose
evidence or reliability without unacceptable processing cost or extra fallbacks.

## Git Safety

Real videos and generated baseline reports are ignored by git:

- `samples/videos/*`
- `baseline_reports/*`
- `backend_eval_reports/*`

Only `.gitkeep` files are committed.

Do not commit swimmer footage, signed URLs, private storage paths, or real athlete identifiers.

## What Future Upgrades Should Improve

Future AI work should aim to improve:

- higher pose detection rate on real swim footage
- better handling of water distortion and splash
- fewer manual-review fallbacks on suitable clips
- lower processing time for normal clips
- more stable keypoint visibility across sampled frames
- stronger stroke-specific draft findings for coach review

Future upgrades must preserve:

- `/process-video` contract compatibility
- short-lived signed URL privacy
- callback secret verification
- app-side quality gate
- zero fake findings on weak evidence
- Coach Studio/manual review fallback

## What Still Requires Owen's Real Clips

Quantitative accuracy cannot be claimed until Owen provides labelled test clips with:

- stroke
- camera angle
- timestamped phases
- known coach-observed faults
- expected drill/cue targets
- acceptable/manual-review judgement

Until then, this harness measures contract safety and baseline technical behaviour, not coaching accuracy.

## Real SwimXYZ baseline — Side-above breaststroke

Date: 2026-06-21  
Backend: MediaPipe  
Clip: `baseline_data/breast_side_above`  
Frames compared: 301  
Matched keypoints: 1026  

| Metric | Result |
|---|---:|
| Mean error | 0.5081 |
| Median error | 0.4076 |
| PCK@0.05 | 0.0000 |
| Recall | 0.2272 |

Interpretation: generic MediaPipe performs poorly on this side-above swimming view. This is now the floor that swim-specific pose backends must beat.
