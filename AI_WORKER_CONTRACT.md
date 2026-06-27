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

Current production analysis entrypoint. Vercel calls this route after creating
an `ai_processing_jobs` row. The current Render-compatible path still sends a
short-lived Supabase signed URL. The worker also accepts provider/key storage
identifiers so future Modal/RunPod workers can use direct private object access.

Required request fields:

- `video_upload_id`
- `callback_url`

One usable private video access method is required:

- `signed_video_url` for the current short-lived Render compatibility path, or
- `storage_provider` plus `video_key` for provider-based storage access.

Recommended request fields:

- `job_id`
- `app_job_id`
- `club_id`
- `swimmer_id`
- `uploaded_by_user_id`
- `storage_provider`
- `video_key`
- `signed_video_url_expires_in_seconds`
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
- `analysis_mode` (the coach-selected report preset)
- `selected_report_outputs`
- `athlete_profile_readiness`
- `estimate_only_outputs`
- `estimated_credit_cost`
- `coach_confirmed_draft_ai`
- `calibration_available`
- `pool_length_m`
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

## Job-Based Async Processing Contract

The Vercel app creates the durable `ai_processing_jobs` row before calling the
worker. The app/database state is the source of truth for coach UI, retries,
cancellation, report linking, and manual fallback.

Current request payload shape:

```json
{
  "job_id": "app-job-uuid",
  "app_job_id": "app-job-uuid",
  "video_upload_id": "video-upload-uuid",
  "storage_provider": "supabase_private",
  "video_key": "private-storage-key-or-file-id",
  "signed_video_url": "short-lived-private-url",
  "signed_video_url_expires_in_seconds": 900,
  "stroke_type": "Freestyle",
  "camera_angle": "Side",
  "review_context": {},
  "selected_report_outputs": [],
  "callback_url": "https://app.example/api/ai/callback"
}
```

Future provider-compatible payload shape:

```json
{
  "job_id": "app-job-uuid",
  "storage_provider": "supabase_private",
  "video_key": "private-storage-key-or-file-id",
  "analysis_options": {
    "model_tier": "mediapipe_fast",
    "extract_fps_cap": 60,
    "detect_stroke_phases": true,
    "generate_pdf_draft": false
  },
  "callback": {
    "target": "vercel_app",
    "action": "ai_callback"
  }
}
```

Accepted response:

```json
{
  "accepted": true,
  "job_id": "app-job-uuid",
  "server_job_id": "worker-job-id",
  "status": "queued",
  "stage": "queued"
}
```

The worker may expose local status through `GET /jobs/{job_id}`, but the app
continues polling Supabase job state. Completion callbacks must include
`job_id` or `app_job_id`, `video_upload_id`, `status`, safe output metadata, and
coach-review framing. Failure callbacks must include a safe reason code, a
coach-safe message, and `manual_review_available: true`.

Worker logs and callbacks must never include signed URLs, private storage
paths, callback secrets, environment variables, raw landmarks, frame arrays,
height/mass in public-facing payloads, stack traces, or internal calibration
details.

Modal, RunPod, S3, SQS, and other GPU/storage providers can plug into this
contract later by accepting the same `job_id`, returning a fast accepted
response, and reporting completion/failure through the same callback shape.

### Structured report-output selection

New app requests can name the report outputs the coach selected. The worker
revalidates those IDs and returns safe metadata in final/manual/failure
callbacks:

- `requested_outputs`
- `completed_outputs`
- `skipped_outputs` (`id` and safe reason only)
- `estimate_only_outputs`

Unknown and unavailable outputs are skipped explicitly. Legacy requests with no
selection preserve the existing analysis path. Athlete profile inputs and
calibration details are never echoed in this metadata or in callbacks.

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

`signed_video_url` may be present in the worker request, but it must never be
logged, echoed, persisted, or returned in a callback. `video_key` is also an
internal private object identifier; logs may show only the storage provider and a
redacted key/hash.

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
- storage provider
- redacted storage key/hash

Unsafe logs include:

- full signed URLs
- URL tokens
- webhook secret values
- private storage paths
- full `video_key` values
- auth tokens

## Worker Storage Access Adapter

The worker accepts these storage inputs:

```json
{
  "storage_provider": "supabase_private",
  "video_key": "club/swimmer/video/object.mp4",
  "signed_video_url": "optional-short-lived-fallback"
}
```

Current behavior:

- If `signed_video_url` is present, the worker uses it as the active Render path.
- If `storage_provider` and `video_key` are present, the request is accepted.
- `supabase_private` can use provider-native access when `SUPABASE_URL` and a
  service key are configured on the worker.
- `s3_private` and `gcs_private` are reserved adapter labels for future
  Modal/RunPod/S3/GCS work and fail safely until configured.

Provider/key-only failures must return safe failure/manual-review callbacks. They
must not expose private keys, service credentials, signed URLs, local paths, or
storage internals to the app UI or shared reports.

## Adaptive Processing Tiers

The worker inspects video metadata before heavy processing and selects one of:

- `standard_ai`
- `reduced_ai`
- `minimal_ai`
- `manual_review_required`

The selected tier may change sampling width, sampled frame count, and processing window. File size alone is not the only signal; resolution, duration, FPS, frame count, and metadata reliability matter.

If the tier is `manual_review_required`, or extraction/pose processing is not reliable enough, the worker must send manual review with zero AI findings.

## Video Probe Progress Callbacks

Before pose processing, the worker sends additive progress callbacks when video
metadata can be inspected. These callbacks are safe worker/app coordination
messages; they are not public report output and they do not contain raw video,
signed URLs, local paths, raw frames, pose landmarks, 3D data, force values, or
drag estimates.

The worker first sends `metadata_ready` after probing the downloaded temporary
video with `ffprobe` when available, falling back to OpenCV metadata when
needed:

```json
{
  "job_id": "worker-or-app-job-id",
  "app_job_id": "app-job-id",
  "server_job_id": "worker-or-app-job-id",
  "video_upload_id": "video-upload-id",
  "engine": "pose-mvp-0.5",
  "status": "metadata_ready",
  "video_metadata": {
    "duration_seconds": 42.5,
    "fps": 60.0,
    "frame_count": 2550,
    "width": 1920,
    "height": 1080,
    "aspect_ratio": 1.7778,
    "codec": "h264",
    "container": "mp4",
    "file_size_mb": 146.2,
    "orientation": "landscape",
    "metadata_complete": true,
    "probe_source": "ffprobe",
    "warnings": [],
    "errors": []
  },
  "warnings": [],
  "errors": []
}
```

It then sends `frames_sampled` after building a timestamp-only sampling plan.
This does not extract or upload frame images:

```json
{
  "job_id": "worker-or-app-job-id",
  "app_job_id": "app-job-id",
  "server_job_id": "worker-or-app-job-id",
  "video_upload_id": "video-upload-id",
  "engine": "pose-mvp-0.5",
  "status": "frames_sampled",
  "video_metadata": { "...": "same safe metadata shape" },
  "frame_sampling": {
    "ok": true,
    "sampling_rate_fps": 5.0,
    "source_fps": 60.0,
    "max_sampled_frames": 300,
    "total_sampled_frames": 213,
    "duration_seconds": 42.5,
    "samples": [
      {
        "sampleIndex": 0,
        "sourceFrameIndex": 0,
        "timestampMs": 0,
        "status": "scheduled"
      }
    ],
    "warnings": [],
    "errors": []
  },
  "warnings": [],
  "errors": []
}
```

Progress callback delivery is best-effort. If the app rejects one of these
non-terminal callbacks, the worker continues the existing analysis/manual-review
pipeline and the final callback remains authoritative.

## 2D Pose Progress Callback

After sampled frames are read and the configured pose backend runs, the worker
sends a `pose_2d_ready` progress callback. The current model path is the default
MediaPipe Pose backend (`mediapipe_pose`, BlazePose 33 landmarks). This callback
contains only a safe summary and private artifact metadata; it does not contain
raw pose frame arrays.

Stable 2D pose frame shape inside the private worker artifact:

```json
{
  "timestamp_ms": 4000,
  "source_frame_index": 240,
  "sample_index": 20,
  "view_type": "side",
  "pose_model": "mediapipe_pose",
  "joints_2d": {
    "left_shoulder": {
      "x": 0.42,
      "y": 0.31,
      "confidence": 0.91,
      "visibility": 0.91,
      "status": "tracked"
    }
  },
  "frame_confidence": 0.76,
  "tracking_status": "partial"
}
```

Joint status values are `tracked`, `low_confidence`, `missing`,
`interpolated`, and `not_visible`. Phase 5 does not interpolate; missing joints
remain missing. Frame tracking statuses are `tracked`, `partial`, `failed`, and
`no_person_detected`.

Safe `pose_2d_ready` callback shape:

```json
{
  "job_id": "worker-or-app-job-id",
  "app_job_id": "app-job-id",
  "server_job_id": "worker-or-app-job-id",
  "video_upload_id": "video-upload-id",
  "engine": "pose-mvp-0.5",
  "status": "pose_2d_ready",
  "stage": "pose_2d_ready",
  "progress_percent": 62,
  "pose_2d_summary": {
    "availabilityState": "pose_2d_ready",
    "ok": true,
    "model": "mediapipe_pose",
    "modelVersion": "blazepose_33",
    "sampledFrames": 213,
    "processedFrames": 213,
    "trackedFrames": 184,
    "partialFrames": 22,
    "failedFrames": 7,
    "averageFrameConfidence": 0.74,
    "lowConfidenceJointRate": 0.18,
    "viewType": "side",
    "warnings": []
  },
  "pose_artifact": {
    "artifact_type": "pose_2d_timeseries",
    "storage_visibility": "private",
    "format": "json",
    "frame_count": 213,
    "contains_raw_pose": true,
    "contains_video_pixels": false,
    "public_safe": false
  },
  "warnings": []
}
```

Until private artifact storage is added, the worker may write raw pose
timeseries locally for tests or internal debugging. The callback must never
include the local path. Shared/public reports must not expose raw pose arrays or
pose artifacts.

## Monocular Estimated 3D Pose Callback

After the private 2D pose timeseries is available, the worker may build a
single-view estimated 3D pose timeseries using the heuristic lifter. This is not
calibrated multi-view 3D, not measured world geometry, and not a public report
output. It is a private internal artifact for future biomechanics work.

Current method:

- `source: "monocular_estimate"`
- `method: "anatomical_heuristic_lift"`
- `measurementType: "estimated"`
- `pose3dModel: "anatomical_heuristic_lift_v1"`
- coordinate system: `hip_centered_relative`
- scale: `relative_body_units`

Private 3D pose frame shape inside the worker artifact:

```json
{
  "timestamp_ms": 4000,
  "source_frame_index": 240,
  "sample_index": 20,
  "view_type": "Side",
  "source": "monocular_estimate",
  "method": "anatomical_heuristic_lift",
  "measurementType": "estimated",
  "pose_model": "mediapipe_pose",
  "pose_3d_model": "anatomical_heuristic_lift_v1",
  "joints_3d": {
    "left_shoulder": {
      "x": 0.12,
      "y": 1.42,
      "z": -0.08,
      "confidence": 0.76,
      "source_2d_confidence": 0.91,
      "status": "estimated"
    }
  },
  "frame_confidence": 0.7,
  "tracking_status": "partial"
}
```

Safe `pose_3d_estimated` callback shape:

```json
{
  "job_id": "worker-or-app-job-id",
  "app_job_id": "app-job-id",
  "server_job_id": "worker-or-app-job-id",
  "video_upload_id": "video-upload-id",
  "engine": "pose-mvp-0.5",
  "status": "pose_3d_estimated",
  "stage": "pose_3d_estimated",
  "progress_percent": 66,
  "pose_3d_summary": {
    "availabilityState": "pose_3d_estimated",
    "ok": true,
    "source": "monocular_estimate",
    "method": "anatomical_heuristic_lift",
    "measurementType": "estimated",
    "pose3dModel": "anatomical_heuristic_lift_v1",
    "inputPoseModel": "mediapipe_pose",
    "inputFrames": 213,
    "estimatedFrames": 184,
    "partialFrames": 22,
    "failedFrames": 7,
    "averageFrameConfidence": 0.68,
    "coordinateSystem": "hip_centered_relative",
    "scale": "relative_body_units",
    "calibration": {
      "cameraCalibrated": false,
      "worldScaleKnown": false,
      "multiView": false
    },
    "assumptions": [
      "single-view depth estimated from 2D pose sequence",
      "coordinates are relative body units, not measured metres",
      "z-depth is inferred from anatomical constraints and temporal smoothing"
    ],
    "warnings": []
  },
  "pose_3d_artifact": {
    "artifact_type": "pose_3d_timeseries",
    "storage_visibility": "private",
    "format": "json",
    "frame_count": 213,
    "contains_raw_pose": true,
    "contains_video_pixels": false,
    "public_safe": false,
    "source": "monocular_estimate",
    "measurementType": "estimated"
  },
  "warnings": []
}
```

The callback must never include raw `joints_3d`, raw `joints_2d`, frame images,
signed URLs, local artifact paths, storage paths, biomechanics frames, force
frames, or drag estimates. Phase 7 can use the private 3D artifact as input for
biomechanics calculations, but public report exposure still requires a separate
safe summary and coach approval.

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

When `ENABLE_ESTIMATED_DRAG=true`, the block is included only when the coach
explicitly requested `estimated_drag_force`, real pose was detected, the
analysis is `real_pose` (not a manual-review fallback), the request supplied
`swimmer_height_cm` and `swimmer_mass_kg`, a scale-calibration readiness flag is
present, and the clip is side view. It is best-effort and never blocks the
callback. Values remain an ESTIMATE requiring coach interpretation, not a
measurement.

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
| `ENABLE_ESTIMATED_DRAG` | `false` | Pilot anthropometric drag block (see above). Requires explicit output selection, side view, athlete inputs, and calibration readiness. |
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
- `probing_video_metadata`
- `metadata_ready`
- `frames_sampled`
- `reading_video_metadata`
- `metadata_read`
- `processing_tier_selected`
- `extracting_frames`
- `frames_extracted`
- `running_pose_detection`
- `pose_2d_ready`
- `pose_3d_estimated`
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
