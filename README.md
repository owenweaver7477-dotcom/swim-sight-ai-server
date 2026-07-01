# Swim Sight 3D AI Worker

FastAPI worker for Swim Sight 3D pose-assisted video review.

Current engine: `pose-mvp-0.5`

This worker is part of the Vercel/Supabase/Render pipeline:

1. The Swim Sight 3D app creates an `ai_processing_jobs` row in Supabase.
2. Vercel generates a short-lived signed URL for the private video.
3. Vercel sends the job to this worker at `POST /process-video`.
4. This worker downloads the video temporarily, runs adaptive pose-assisted analysis, and calls back to Vercel at `/api/ai/callback`.
5. The app applies quality gates before creating coach-reviewable findings.

The app is the source of truth for jobs, reports, findings, sharing, and coach approval. The worker's in-memory job status endpoint is only a runtime convenience.

## Endpoints

### `GET /`

Returns service metadata and available route names.

### `GET /health`

Returns worker health and engine version.

Example:

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

The health route does not import OpenCV, MediaPipe, ONNX Runtime, video files,
or baseline tooling. Blocking frame and pose work runs away from the FastAPI
server loop so health checks remain responsive during analysis.

### `POST /process-video`

Accepts a job quickly and processes the video in the background.

The request must include:

- `video_upload_id`
- `signed_video_url`
- `callback_url`

The request normally also includes job, club, swimmer, stroke, camera angle, file metadata, and review context.

The accepted response is `202` with:

- `accepted`
- `job_id`
- `server_job_id`
- `video_upload_id`
- `status`
- `stage`
- `engine`

### `GET /jobs/{job_id}`

Returns the current in-memory worker status for a recently accepted job.

This endpoint is not durable storage. Supabase `ai_processing_jobs` remains the reliable job record.

### `POST /jobs/{job_id}/cancel`

Requests cancellation using the server-side `x-ai-worker-secret` header. Queued
jobs become cancelled immediately. Processing jobs become `cancel_requested`
and suppress completion after the current blocking stage returns. The app and
Supabase remain the canonical cancellation record.

## Safety Rules

- Never log signed video URLs, URL tokens, webhook secrets, auth tokens, or private storage paths.
- Never make private videos public.
- Use the signed URL only for temporary server-side download.
- Temporary video files are cleaned up after processing.
- If video evidence is not reliable enough, return manual review with zero AI findings.
- AI findings are draft evidence only. Coaches review, edit, approve, or reject findings before sharing reports.

## Adaptive Processing

The worker classifies videos before heavy processing. Normal clips receive fuller sampling. Heavier clips receive reduced or minimal sampling. Unsafe, corrupt, or too-heavy videos return a manual-review recommendation instead of crashing the worker.

Processing tiers:

- `standard_ai`
- `reduced_ai`
- `minimal_ai`
- `manual_review_required`

Recommended pilot capture:

- 5-15 seconds
- side view where possible
- MP4/MOV
- 720p or compressed 1080p
- normal camera footage preferred over high-resolution screen recordings

## Environment Variables

### `AI_WEBHOOK_SECRET`

Required for callbacks. Must match the Vercel app value. Sent as `x-ai-webhook-secret`.

This is the **outbound** worker → app secret only. It is **not** reused for
inbound authentication (see `AI_INBOUND_SECRET`).

### `AI_INBOUND_SECRET`

Optional. The **inbound** app → worker job-submission secret, sent by the app as
the `x-ai-inbound-secret` header on `POST /process-video`. Kept separate from
`AI_WEBHOOK_SECRET` so the two rotate independently. Never logged.

### `AI_INBOUND_AUTH_MODE`

Optional. One of `off` (default), `monitor`, or `enforce`.

- `off` — accept jobs without checking the header (current behaviour).
- `monitor` — check the header and log a safe outcome (`ok`/`missing`/`invalid`),
  but still accept the job.
- `enforce` — reject a missing/invalid header with `401`. If `AI_INBOUND_SECRET`
  is not set, enforce **fails closed** (rejects).

Unset (or an unknown value) behaves as `off`, so setting neither var changes
nothing. Roll out by deploying with `off`/`monitor`, having the app send the
header, then flipping to `enforce`.

### `AI_CALLBACK_ALLOWED_HOSTS`

Optional. Comma-separated list of exact hostnames the worker is allowed to send
the callback to (for example `swim-sight-3d-v1.vercel.app`). Matching requires
`https` and an exact host (no suffix/wildcard).

### `AI_CALLBACK_HOST_MODE`

Optional. One of `monitor` (default) or `enforce`.

- `monitor` — the callback still sends as today; the worker only logs whether the
  host would be allowed or blocked.
- `enforce` — a non-allowlisted `callback_url` is blocked **before** the callback
  is sent, so the outbound webhook secret is never delivered to an unapproved
  host. With no allowlist configured, enforce blocks all callbacks (fail closed).

Only the callback **host** and outcome are logged — never the full callback URL,
`signed_video_url`, or any secret.

### `PORT`

Optional. Render provides this automatically.

### `AI_JOB_TIMEOUT_SECONDS`

Optional. Maximum worker processing time in seconds. Defaults to `600` (10
minutes), with accepted values from 30 to 3600 seconds. Timed-out jobs return a
safe manual-review callback with zero findings.

Recommended Render service settings:

- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

## Local Development

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 10000
curl http://localhost:10000/health
```

## Contract Documentation

See `AI_WORKER_CONTRACT.md` for the frozen Phase 15B request, response, callback, status, and roadmap compatibility contract.

Example payloads live in `fixtures/`.

See `BASELINE_EVALUATION.md` for Phase 15C fixture validation, endpoint contract tests, and local sample-clip baseline evaluation commands.
