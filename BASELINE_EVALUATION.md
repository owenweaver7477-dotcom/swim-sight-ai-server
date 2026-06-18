# Swim Sight 3D Baseline Evaluation

Phase 15C adds script-based contract tests and a local baseline harness for the current `pose-mvp-0.5` worker.

This is not a model upgrade. It protects the current worker contract before future RTMPose, CLAHE, Kalman smoothing, ONNX, or other AI upgrades.

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
- finding count
- fallback/manual-review state
- quality flags

These metrics give future upgrades something honest to beat.

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

## Git Safety

Real videos and generated baseline reports are ignored by git:

- `samples/videos/*`
- `baseline_reports/*`

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
