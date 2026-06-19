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
  "status": "ok",
  "engine": "pose-mvp-0.5"
}
```

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

Returns the in-memory worker job snapshot for local/debug visibility. This is not durable storage and must not be treated as the canonical job record.

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
- `metadata_read`
- `processing_tier_selected`
- `extracting_frames`
- `running_pose_detection`
- `analysing_stroke`
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
