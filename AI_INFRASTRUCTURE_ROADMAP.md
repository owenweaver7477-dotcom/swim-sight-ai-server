# Swim Sight 3D AI Infrastructure Roadmap

This roadmap separates deployed worker code from production-activated
capabilities and research-stage work. The operating rule remains:

> AI suggests. Coaches decide.

No AI output becomes shared swimmer or parent content until a coach reviews and
approves it. Relative 2D cues are not exact biomechanics, estimated drag is not
measured drag, and single-camera pose is not validated 3D.

## 1. Current Deployed State

| Item | Current state |
| --- | --- |
| Last known healthy worker commit | `517279cd7303372f7a503c4059129603a64e13b5` |
| Render health | `{"status":"ok","engine":"pose-mvp-0.5"}` |
| Public website | HTTP 200 |
| Worker regression suite | 9/9 test groups passed |
| App callback | Backward compatible with additive telemetry fields |
| Production endpoint | `POST /process-video` |
| Current engine | `pose-mvp-0.5` |
| Primary review workflow | Manual Coach Studio with coach approval |

Supabase and the Vercel application remain the canonical sources of truth for
jobs, videos, reports, findings, and shared-report safety. The worker performs
pose-assisted analysis and sends a sanitized callback.

## 2. Already Deployed Capabilities

Deployed means the code is present on the production worker. It does not mean
that every optional flag is enabled.

### Active, Internal, and Production-Safe

- All 33 MediaPipe Pose landmarks are retained internally when visible.
- The established 15-core-landmark count still controls the quality gate.
- Relative 2D temporal metrics are calculated in image space.
- Heuristic stroke-phase segments support internal evidence review.
- Pipeline stages cover metadata, frame extraction, pose detection, phase
  analysis, finding generation, output preparation, and callback delivery.
- Processing telemetry includes sampled frames, frames with pose, pose coverage,
  average core keypoints, average visible landmarks, extraction failures,
  fallback state, processing tier, and quality flags.
- Workload classification adapts frame count, processing width, and processing
  window using resolution, duration, FPS, decoded workload, file size, and
  screen-recording risk.
- Weak or unsafe evidence returns manual review with zero fake findings.
- The labelled-clip and upgrade-comparison tools are available for local use.

### Deployed but Inactive

- Redis Streams durability code is deployed but inactive because
  `ENABLE_DURABLE_QUEUE=false`.
- Persistent-signal finding filtering is deployed but inactive because
  `ROBUST_FINDINGS=false`.
- CLAHE frame enhancement is deployed but inactive because
  `ENABLE_CLAHE=false`.
- Pose smoothing is deployed but inactive because
  `ENABLE_POSE_SMOOTHING=false`.
- Sequential frame reading is deployed but inactive because
  `SEQUENTIAL_FRAME_READ=false`.
- Higher MediaPipe complexity is inactive because `POSE_MODEL_COMPLEXITY=0`.
- Heuristic stroke-cycle segmentation telemetry is deployed but inactive because
  `PHASE_ANALYSIS=false`. When enabled it adds a sanitized 2D-heuristic summary
  (cycle count, mean duration, regularity, confidence, quality flags) to the
  internal `processing_telemetry` only. It is `public_safe: false`, is not a
  biomechanical metric, and must stay internal until coach-validated.
- Estimated drag remains an internal prototype and is inactive because
  `ENABLE_ESTIMATED_DRAG=false`.

## 3. Inactive Capabilities and Prerequisites

### Redis Durability

Required before activation:

- Provision a private, access-controlled Redis instance with encrypted transit.
- Configure `REDIS_URL` in a staging worker only.
- Set `ENABLE_DURABLE_QUEUE=true` in staging only.
- Submit a real private test job and restart the worker during processing.
- Confirm the pending Redis Stream message is reclaimed after the lease.
- Verify the job is not processed twice after callback completion.
- Verify deduplication and terminal-job acknowledgement.
- Verify expired signed URLs fail safely and leave manual review or retry
  available in the application.
- Confirm no signed URL, Redis URL, or token appears in logs or callbacks.

Do not activate Redis durability in production until this restart test passes.

### Robust Findings, CLAHE, Smoothing, and Model Complexity

Required before enabling any quality flag:

- Collect representative, permissioned swim clips.
- Label stroke, camera angle, expected mode, coach-observed faults, and faults
  that must not be generated.
- Generate baseline and flag-comparison reports.
- Review false positives, missed faults, manual-review frequency, pose coverage,
  and processing time.
- Ask a qualified coach to review the generated draft findings against video.
- Enable one flag at a time in staging.
- Keep a flag only when it improves useful evidence without unacceptable cost or
  additional false findings.

Never enable every quality flag together first. A combined result cannot reveal
which change helped or caused a regression.

### Custom Underwater Pose Model

Required before model development or replacement:

- Permissioned underwater and above-water swimming frames.
- Labels for the required joints through splash, bubbles, occlusion, and glare.
- Training, validation, and holdout splits that avoid swimmer leakage.
- Coverage across all four strokes, camera angles, body types, lighting, pool
  backgrounds, and recording devices.
- Evaluation against the current MediaPipe baseline using the same holdout clips.
- Documented pose coverage, keypoint error, processing cost, and fallback rate.
- No production replacement until the model beats the baseline on unseen data.

### Camera Calibration and Refraction Correction

Required before implementation:

- The actual camera, lens, housing, mounting position, and pool setup.
- A calibration board, known pool geometry, or measured reference object.
- Intrinsic and extrinsic camera calibration data.
- Measured assumptions for the housing and water interface.
- Reprojection validation using known points in the real pool.

Do not apply a generic Snell's Law correction with guessed glass or housing
values. That can make coordinates less accurate while appearing more precise.

### Multi-Camera 3D

Required before any true-3D claim:

- Two or more synchronized views of the same swim sequence.
- Stable frame timestamps or a verified synchronization signal.
- A calibration profile for every camera.
- View-to-view athlete association and keypoint correspondence.
- Triangulation with per-point confidence.
- Reprojection-error tracking and a rejection threshold.
- Validation against known geometry or an accepted reference system.

No output should be called true 3D until these conditions are validated.

### SMPL and 3D Mesh Fitting

Required before implementation:

- Reliable calibrated 3D keypoints or a validated multi-view reconstruction.
- A licensed and technically suitable body model.
- GPU-capable fitting infrastructure.
- Occlusion and water-surface handling.
- Validation of fit stability on swimming poses.
- Clear positioning as a visual coaching reference, not a measured digital twin.

### GPU Workers and Autoscaling

Required before infrastructure expansion:

- Queue-depth and concurrent-job evidence.
- Processing-time and memory telemetry from representative pilot use.
- A model or workload that demonstrably benefits from GPU execution.
- Cost-per-review targets and acceptable cold-start latency.
- Autoscaling limits, retry rules, monitoring, and spend alerts.
- A rollback path to the current CPU worker and manual review.

Do not add GPU infrastructure solely because it sounds premium. Add it when
measured workload, latency, or a validated model requires it.

## 4. Website Integration Rules

### Safe Website Outputs

The authenticated coaching application may display:

- AI job status.
- A coach-readable processing-stage label.
- Pose quality and reliability summary.
- Manual-review recommendation.
- Number of frames analysed.
- Pose coverage or detection rate.
- Coach-safe confidence language.
- A sanitized fallback reason.
- Coach-reviewable draft findings.
- Coach-approved findings, cues, drills, and summaries.

### Unsafe Website Outputs

Do not expose:

- Raw pose landmarks or model tensors.
- Signed video URLs or private storage paths.
- Redis URLs, webhook secrets, tokens, or service-role credentials.
- Swimmer height or mass from worker payloads.
- Estimated drag in production or public reports.
- Unapproved or rejected findings.
- Raw AI callback payloads.
- Exact-biomechanics claims from relative 2D cues.
- Calibration internals or reprojection debug data.
- Private worker diagnostics in swimmer or parent views.

### Shared Report Rule

Shared reports may contain only coach-approved findings, coach-approved cues and
drills, report-safe comments, selected public annotations or key moments, and
careful summary language. Manual-review recommendations and uncertainty should
remain honest. Internal telemetry must not become public report content.

## 5. Internal-Only Capabilities

The following must remain internal until separately validated and approved:

- Relative 2D metric arrays and phase-segmentation diagnostics.
- Raw 33-landmark pose tracks.
- Quality-gate diagnostics and rejected finding candidates.
- Labelled-clip comparison reports.
- Redis queue payloads and operational keys.
- Estimated drag and anthropometric scaling prototypes.
- Calibration matrices and reprojection diagnostics.
- Experimental multi-camera or SMPL outputs.
- GPU cost, queue-depth, and autoscaling diagnostics.

## 6. Activation Sequence

1. Collect 3-5 short, permissioned local baseline clips.
2. Include more than one stroke and recording condition where possible.
3. Run the default and one-flag-at-a-time comparison.
4. Add entries to the local labelled-clip manifest.
5. Run the labelled evaluation harness.
6. Review false positives, missed faults, fallback rate, pose coverage, and
   processing time with a qualified coach.
7. Enable one AI quality flag in staging only.
8. Repeat the same clips and compare with the untouched baseline.
9. Keep only flags that produce a defensible improvement.
10. Test Redis durability separately using a real private Redis instance and a
    deliberate worker restart.
11. Only after these tests consider production environment changes.
12. Keep advanced metrics out of Coach Studio and shared reports until they are
    coach-validated and product wording is approved.

## 7. Required Commands

Run the worker regression suite:

```bash
cd "/Users/owen_weaver/Documents/GitHub/swim-sight-ai-server"
python3 scripts/test_upgrades.py
```

Compare default-off upgrade flags with local clips:

```bash
python3 scripts/compare_upgrade_flags.py \
  --stroke Breaststroke \
  --camera-angle Side
```

Run the coach-labelled evaluation:

```bash
python3 scripts/evaluate_labelled_clips.py
```

Production rules:

- Do not set `ENABLE_DURABLE_QUEUE=true` until the Redis staging restart test
  passes.
- Do not set `ENABLE_ESTIMATED_DRAG=true`.
- Do not enable all quality flags together.
- Do not commit local clips, local labels, or generated evaluation reports.

## 8. Rollback Plan

If an optional upgrade causes errors, regressions, increased false findings, or
unacceptable processing time:

1. Set all optional flags to their conservative defaults:

   ```text
   ENABLE_DURABLE_QUEUE=false
   ENABLE_CLAHE=false
   ENABLE_POSE_SMOOTHING=false
   ROBUST_FINDINGS=false
   SEQUENTIAL_FRAME_READ=false
   ENABLE_ESTIMATED_DRAG=false
   POSE_MODEL_COMPLEXITY=0
   ```

2. Remove `REDIS_URL` from the worker or leave it unused with durability off.
3. Return to direct `/process-video` background processing.
4. Preserve uploaded videos and the Coach Studio manual-review path.
5. Preserve zero-finding manual review whenever evidence is weak.
6. Re-run the worker contract and upgrade suites.
7. If necessary, redeploy the last known healthy worker commit:
   `517279cd7303372f7a503c4059129603a64e13b5`.

## 9. Stop Conditions

Stop an activation or research phase when any of the following occurs:

- A flag increases false positives or unsupported confidence.
- The callback contract or manual-review fallback regresses.
- Signed URLs, private paths, credentials, or private data appear in logs or
  outputs.
- Redis restart recovery duplicates completed work or loses a pending job.
- A custom model does not beat the MediaPipe baseline on holdout clips.
- Calibration or triangulation error cannot be measured reliably.
- GPU cost or latency cannot be justified by pilot workload.
- A proposed output cannot be explained honestly to a coach or swimmer.

At that point, disable the experimental capability, preserve manual review, and
return to the last known healthy configuration.
