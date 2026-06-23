# Swim Sight 3D AI Worker Contract

Phase 15B contract freeze for the Render AI worker and the Vercel/Supabase app.

Current engine: `pose-mvp-0.5`

## Contract Boundary

The worker accepts private video jobs, performs adaptive pose-assisted analysis, and sends a sanitized callback to the Swim Sight 3D Vercel app.

The worker does not own:

- Supabase job durability
- report finalisation
- coach approval
- shared report filtering
- AI credit enforcement
- billing or plan state
- public report rendering

The Vercel app and Supabase database remain the source of truth.

## Current Routes

### `GET /`

Returns route metadata and the current engine value.

### `GET /health`

Returns:

```json
{
  "ok": true,
  "service": "swim-sight-ai-server",
  "version": "pose-mvp-0.5",
  "timestamp": "2026-06-23T00:00:00+00:00",
  "heavy_models_loaded": false,
  "status": "ok",
  "engine": "pose-mvp-0.5"
}
```

This route is deliberately lightweight. It must not import pose backends,
OpenCV, ONNX Runtime, baseline data, or video-processing modules.

### `POST /process-video`

Current production analysis entrypoint. Vercel calls this route after creating an `ai_processing_jobs` row and generating a short-lived Supabase signed URL.

Required request fields:

- `video_upload_id`
- `signed_video_url`
- `callback_url`

Recommended request fields:

- `job_id`
- `app_job_id`
- `club_id`
- `swimmer_id`
- `uploaded_by_user_id`
- `stroke_type`
- `analysis_type`
- `camera_angle`
- `capture_source`
- `original_filename`
- `file_size_bytes`
- `file_size_mb`
- `duration_seconds`
- `review_context`
- `max_sampled_frames`
- `downscale_frames`
- `swimmer_height_cm` (optional; server-side only, drives drag estimate)
- `swimmer_mass_kg` (optional; server-side only, drives drag estimate)

Accepted response:

- `accepted`
- `job_id`
- `server_job_id`
- `video_upload_id`
- `status`
- `stage`
- `engine`

The endpoint must return quickly with `202 Accepted`. Heavy processing runs in the background.

### `GET /jobs/{job_id}`

Returns the worker job snapshot for local/debug visibility. With the default
configuration this remains in memory. When the optional durable queue is
enabled, the sanitized status snapshot is also read from Redis. Supabase remains
the canonical application job record.

### `POST /jobs/{job_id}/cancel`

Internal server-to-server cancellation route protected by
`x-ai-worker-secret`. Queued jobs become `cancelled`; running jobs become
`cancel_requested`. Late completion updates cannot overwrite `cancelled` or
`timed_out` terminal states. The response contains only job id, status, and a
safe coach-facing message.

## Reliability Timeout

`AI_JOB_TIMEOUT_SECONDS` controls the worker processing deadline and defaults
to 600 seconds. Timeout, cancellation, and worker failure callbacks contain a
safe reason code, coach message, `manual_review_available: true`, and zero
findings. They never include signed URLs, frames, landmarks, stack traces,
private paths, anthropometrics, or secrets.

## Optional Durable Redis Queue

The default execution path remains FastAPI background tasks. Set both:

```text
ENABLE_DURABLE_QUEUE=true
REDIS_URL=<private Redis connection URL>
```

to use Redis Streams with a consumer group. Accepted jobs are stored in the
stream, unacknowledged jobs remain pending, and stale pending jobs are reclaimed
after `AI_JOB_LEASE_MS` (default five minutes). Completed messages are
acknowledged and deleted. The `/process-video` accepted response is unchanged.

The private queue payload contains the short-lived signed URL because the worker
needs it to download the video after dequeue. Redis must therefore be private,
encrypted in transit, access-controlled, and treated as sensitive
infrastructure. Signed URLs are never stored in the sanitized job-status key,
logs, callbacks, or reports.

Do not enable this flag until Redis is provisioned and restart/reclaim behavior
has been tested in a staging worker.

## `/process-video` vs `/analyse`

`/process-video` is the current production route.

Future model upgrades may add `/analyse` as a cleaner alias, but it should not replace `/process-video` until both Vercel and Render are migrated together. If `/analyse` is added later, keep `/process-video` backwards compatible for existing deployments.

## Request Privacy

`signed_video_url` is intentionally present in the worker request, but it must never be logged, echoed, persisted, or returned in a callback.

Safe logs may include:

- `job_id`
- `video_upload_id`
- `stroke_type`
- `camera_angle`
- file size
- duration
- source resolution
- processing tier
- sampled frame count
- stage/progress

Unsafe logs include:

- full signed URLs
- URL tokens
- webhook secret values
- private storage paths
- auth tokens

## Adaptive Processing Tiers

The worker inspects video metadata before heavy processing and selects one of:

- `standard_ai`
- `reduced_ai`
- `minimal_ai`
- `manual_review_required`

The selected tier may change sampling width, sampled frame count, and processing window. File size alone is not the only signal; resolution, duration, FPS, frame count, and metadata reliability matter.

If the tier is `manual_review_required`, or extraction/pose processing is not reliable enough, the worker must send manual review with zero AI findings.

## Callback Route

The worker calls the Vercel app callback URL supplied in the request.

Required security header:

```http
x-ai-webhook-secret: <redacted>
```

Callback payloads must not include:

- signed video URL
- private storage path
- auth token
- webhook secret
- raw local temp file path

## Callback Payload Fields

The callback may include:

- `job_id`
- `server_job_id`
- `video_upload_id`
- `engine`
- `status`
- `analysis_mode`
- `real_pose_detected`
- `findings`
- `overall_score`
- `phase_breakdown`
- `drag_analysis`
- `estimated_drag` (optional; anthropometric drag estimate, see below)
- `key_frames`
- `technical_summary`
- `error_message`
- `stage_history`
- `processing_duration_seconds`
- `video_duration_seconds`
- `video_fps`
- `detection_ratio`
- `pose_reliability`
- `quality_flags`
- `recommended_next_action`
- `frame_count_processed`
- `detected_pose_frames`
- `detected_keypoints_count`
- `processing_tier`
- `source_width`
- `source_height`
- `processed_width`
- `processed_height`
- `processing_window_seconds`
- `sampled_frame_count`
- `processing_telemetry`
- `temporal_metrics`

`processing_telemetry` contains sanitized operational measurements such as
requested/sample frame counts, pose detection rate, average core keypoints,
average visible landmarks, failed frame reads, fallback state, and quality
flags. It never contains video URLs, tokens, private paths, or athlete profile
data.

`temporal_metrics` contains heuristic relative-2D image-space summaries and
phase segments. These are not calibrated 3D angles, distances, velocities, or
hydrodynamic measurements and must not be presented as such.

## Internal Pose Landmark Schema

The worker retains all 33 MediaPipe Pose landmarks internally, including face,
hand, heel, and foot-index points when visible. The stable `keypoint_count`
quality gate still counts only the original 15 core body landmarks, so richer
landmark availability cannot make weak pose evidence pass the gate by itself.

## `estimated_drag` (internal pilot prototype — disabled by default)

`estimated_drag` is an **internal pilot-only prototype**. It is **disabled by
default** and gated behind the `ENABLE_ESTIMATED_DRAG` environment variable
(default `false`). It is **not a live measurement tool**, **not part of shared
reports**, and **not part of the stable public callback contract yet** — its
shape may change and consumers must not depend on it.

When `ENABLE_ESTIMATED_DRAG` is unset or false, the worker behaves exactly as
before: no `estimated_drag` field, no height/mass output, no extra error path,
no blocked analysis, and no change to the manual-review fallback.

When `ENABLE_ESTIMATED_DRAG=true`, the block is included only when real pose was
detected, the analysis is `real_pose` (not a manual-review fallback), and the
request supplied `swimmer_height_cm` and `swimmer_mass_kg`. It is best-effort and
never blocks the callback. Values are an ESTIMATE derived from monocular pose
scale, not a measurement.

Always present when the block is included:

- `summary.mean_drag_force_n`, `summary.peak_drag_force_n`
- `summary.mean_drag_to_weight_ratio`
- per-frame `series.drag_force_n`, `series.drag_to_weight_ratio`

Included only when `confidence_low` is `false`:

- `summary.mean_propulsive_force_n`, `summary.peak_propulsive_force_n`
- per-frame `series.propulsive_force_n`, `series.net_force_n`

The block never contains swimmer height, mass, or any identifying profile value.

## Tuning & Pilot Flags (all default OFF / conservative)

These environment flags gate optional worker upgrades. With none set, the worker
behaves exactly as `pose-mvp-0.5` did. Enable one at a time and compare against
the baseline harness (`scripts/evaluate_baseline.py`).

| Env var | Default | Effect |
| --- | --- | --- |
| `POSE_MODEL_COMPLEXITY` | `0` | MediaPipe BlazePose complexity (0/1/2). `1` markedly improves keypoint accuracy/stability; slower per frame. |
| `ENABLE_CLAHE` | `false` | CLAHE contrast enhancement before pose (helps underwater/low-contrast). `CLAHE_CLIP_LIMIT` (2.0), `CLAHE_TILE` (8) tune it. |
| `ENABLE_POSE_SMOOTHING` | `false` | Temporal stabilisation of sampled tracks: short-gap interpolation, single-frame outlier removal, jitter smoothing. Interpolated points do not inflate the detection count. |
| `ROBUST_FINDINGS` | `false` | Draft findings must be sustained across frames (not a single spike); strength reported as a percentile. Fewer false positives. |
| `SEQUENTIAL_FRAME_READ` | `false` | Decode the processing window in one forward pass instead of per-frame seeks. Faster on long-GOP video; no accuracy change. |
| `ENABLE_ESTIMATED_DRAG` | `false` | Pilot anthropometric drag block (see above). Requires `swimmer_height_cm` + `swimmer_mass_kg`. |
| `ENABLE_WAVE_DRAG` | `false` | Experimental near-surface Froude wave-drag. Helper math only — NOT auto-applied, because reliable depth needs calibration a monocular camera lacks. |

All of these preserve the `/process-video` contract, the quality gate, and the
manual-review fallback. Each is wrapped so a failure falls back to current
behaviour rather than breaking analysis.

## Callback Statuses

Expected final callback statuses:

- `completed`
- `manual_review_recommended`
- `error`

Progress/status-stage values may include:

- `queued`
- `running`
- `downloading_video`
- `downloaded_video`
- `reading_video_metadata`
- `metadata_read`
- `processing_tier_selected`
- `extracting_frames`
- `frames_extracted`
- `running_pose_detection`
- `analysing_stroke_phases`
- `generating_findings`
- `generating_outputs`
- `callback_sending`
- `completed`
- `manual_review_fallback`
- `error`

## Manual Review Fallback Contract

When the worker cannot produce coach-grade pose evidence, it must return manual review with zero findings.

Required manual-review shape:

```json
{
  "status": "manual_review_recommended",
  "analysis_mode": "manual_review",
  "real_pose_detected": false,
  "findings": [],
  "overall_score": null,
  "phase_breakdown": {},
  "pose_reliability": "failed",
  "recommended_next_action": "manual_review_recommended"
}
```

The app should then keep the uploaded video available for Coach Studio/manual review.

## AI Finding Quality Gate

The worker may send `findings` only when:

- `analysis_mode` is `real_pose`
- `real_pose_detected` is true
- pose reliability is acceptable
- detection ratio and keypoint count are above current thresholds
- findings are medium/high confidence and stroke-specific enough for coach review

The Vercel callback route performs its own quality gate again. If the app-side gate fails, the app suppresses AI findings and creates a manual-review state.

## App-Side Canonical Tables

Current canonical storage is split across:

- `video_uploads`
- `ai_processing_jobs`
- `reports`
- `findings`
- `key_frames`
- `video_annotations`
- `ai_finding_feedback`
- `ai_credit_ledger`

## `video_analysis` Decision

Do not add a `video_analysis` table yet.

If future RTMPose/MMPose, labelling, or richer pose artifacts require a raw analysis table, add it as an additive table such as `video_analysis_runs` or `video_analysis_artifacts`. It should not replace `ai_processing_jobs`, `reports`, or `findings`.

Future raw analysis storage should preserve:

- immutable raw worker output
- model/engine version
- quality flags
- sampled-frame metadata
- normalized summary fields for reports
- strict server-side access control

## Future `SwimmerSkeleton3D` Decision

The current app may expose an Elite Lab preview, but it is not a real analysis-fed 3D system.

Future `SwimmerSkeleton3D` work should wait until there is reliable labelled pose data, a model contract, and a clear product scope. Do not claim live reference matching, measured movement truth, or production 3D analysis until the underlying pipeline exists.

## Fixtures

Safe example payloads live in `fixtures/`:

- `process_video_request.example.json`
- `process_video_accepted.example.json`
- `callback_success.example.json`
- `callback_manual_review.example.json`
- `callback_failed.example.json`
- `job_status.example.json`

The fixtures use placeholders and redacted URLs. They are not secrets and should not contain real swimmers, private file paths, or signed URL tokens.
