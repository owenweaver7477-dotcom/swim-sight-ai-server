from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import hmac
import sys
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, Any, List, Optional, TypeVar

from fastapi import FastAPI, BackgroundTasks, Header, HTTPException, status

from app.models import VideoProcessingRequest, HealthResponse
from app.durable_queue import DurableJobQueue
from app.job_reliability import (
    build_failure_callback,
    failure_payload_is_safe,
    job_timeout_seconds,
)
from app.report_outputs import (
    attach_report_output_metadata,
    build_report_output_plan,
    filter_findings_for_outputs,
)
from app.worker_auth import (
    inbound_auth_mode,
    inbound_secret_configured,
    verify_inbound_secret,
)
from app.callback_safety import callback_host_mode, is_callback_allowed


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

AI_ENGINE_VERSION = "pose-mvp-0.5"
AI_JOB_TIMEOUT_SECONDS = job_timeout_seconds()
T = TypeVar("T")

app = FastAPI(
    title="Swim Sight AI Server",
    version=AI_ENGINE_VERSION,
)


# ─────────────────────────────────────────────────────────────
# Root / health
# ─────────────────────────────────────────────────────────────

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "status": "ok",
        "service": "Swim Sight AI Server",
        "engine": AI_ENGINE_VERSION,
        "health": "/health",
        "docs": "/docs",
        "process_video": "/process-video",
        "job_status": "/jobs/{job_id}",
        "job_cancel": "/jobs/{job_id}/cancel",
    }


@app.head("/", status_code=status.HTTP_200_OK)
def root_head():
    return None


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    return HealthResponse(
        ok=True,
        service="swim-sight-ai-server",
        version=AI_ENGINE_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        heavy_models_loaded="app.pose_estimator" in sys.modules,
        status="ok",
        engine=AI_ENGINE_VERSION,
    )


# ─────────────────────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────────────────────
# This is only a Render-side convenience store.
# Supabase/Vercel remains the source of truth.
# If Render restarts, this resets.
# ─────────────────────────────────────────────────────────────

JOBS: Dict[str, Dict[str, Any]] = {}
DURABLE_QUEUE = DurableJobQueue()
ACTIVE_BLOCKING_TASKS: Dict[str, asyncio.Task] = {}
PROTECTED_TERMINAL_STATUSES = {"cancelled", "timed_out"}


class JobCancelledError(RuntimeError):
    """Raised between worker stages after a coach requests cancellation."""


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_request_summary(request: VideoProcessingRequest) -> Dict[str, Any]:
    """Safe log summary containing no private URL fields."""
    return {
        "video_upload_id": getattr(request, "video_upload_id", None),
        "stroke_type": getattr(request, "stroke_type", None),
        "camera_angle": getattr(request, "camera_angle", None),
        "callback_url_present": bool(getattr(request, "callback_url", None)),
    }


def get_or_create_job_id(request: VideoProcessingRequest) -> str:
    """
    Prefer the Vercel/Supabase job id if the trigger route sends one.
    Fall back to Python-side UUID for backwards compatibility.
    """
    incoming_job_id = getattr(request, "job_id", None)

    if incoming_job_id:
        return str(incoming_job_id)

    return str(uuid.uuid4())


def create_job(job_id: str, video_upload_id: str) -> None:
    JOBS[job_id] = {
        "job_id": job_id,
        "server_job_id": job_id,
        "video_upload_id": video_upload_id,
        "status": "queued",
        "stage": "queued",
        "progress_percent": 0,
        "message": "Queued for pose-assisted review",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    schedule_job_persistence(job_id)


def schedule_job_persistence(job_id: str) -> None:
    if not DURABLE_QUEUE.ready or job_id not in JOBS:
        return
    try:
        asyncio.get_running_loop().create_task(
            DURABLE_QUEUE.persist_job(dict(JOBS[job_id]))
        )
    except RuntimeError:
        return


def update_job(
    job_id: str,
    status_value: str,
    stage: str,
    progress_percent: int,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if job_id not in JOBS:
        JOBS[job_id] = {
            "job_id": job_id,
            "server_job_id": job_id,
            "created_at": now_iso(),
        }

    existing_status = JOBS[job_id].get("status")
    if existing_status in PROTECTED_TERMINAL_STATUSES and status_value != existing_status:
        logger.info(
            "[job=%s] Ignored late status update %s after %s",
            job_id,
            status_value,
            existing_status,
        )
        return
    if existing_status == "cancel_requested" and status_value not in {"cancel_requested", "cancelled"}:
        logger.info("[job=%s] Ignored late status update after cancellation request", job_id)
        return

    payload = {
        "status": status_value,
        "stage": stage,
        "progress_percent": progress_percent,
        "message": message,
        "updated_at": now_iso(),
    }

    if extra:
        payload.update(extra)

    JOBS[job_id].update(payload)
    schedule_job_persistence(job_id)

    logger.info(
        f"[job={job_id}] stage={stage} status={status_value} "
        f"progress={progress_percent}% message={message}"
    )


def add_stage(
    stage_history: List[Dict[str, Any]],
    job_id: str,
    status_value: str,
    stage: str,
    progress_percent: int,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    stage_entry = {
        "stage": stage,
        "status": status_value,
        "progress_percent": progress_percent,
        "message": message,
        "timestamp": now_iso(),
    }

    if extra:
        stage_entry.update(extra)

    stage_history.append(stage_entry)

    update_job(
        job_id=job_id,
        status_value=status_value,
        stage=stage,
        progress_percent=progress_percent,
        message=message,
        extra=extra,
    )


@app.get("/jobs/{job_id}", status_code=status.HTTP_200_OK)
async def get_job(job_id: str):
    job = JOBS.get(job_id)

    if not job and DURABLE_QUEUE.ready:
        job = await DURABLE_QUEUE.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


def job_cancel_requested(job_id: str) -> bool:
    return JOBS.get(job_id, {}).get("status") in {"cancel_requested", "cancelled"}


def ensure_job_active(job_id: str) -> None:
    if job_cancel_requested(job_id):
        raise JobCancelledError("AI cancellation requested")


async def run_blocking(job_id: str, function: Callable[..., T], *args, **kwargs) -> T:
    """Run CPU/blocking work away from the server loop so health stays responsive."""
    ensure_job_active(job_id)
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    ACTIVE_BLOCKING_TASKS[job_id] = task
    try:
        result = await asyncio.shield(task)
    finally:
        if task.done():
            ACTIVE_BLOCKING_TASKS.pop(job_id, None)
    ensure_job_active(job_id)
    return result


def _worker_secret_valid(value: Optional[str]) -> bool:
    expected = os.getenv("AI_WEBHOOK_SECRET", "")
    return bool(expected and value and hmac.compare_digest(expected, value))


def _apply_inbound_auth(provided_secret: Optional[str], video_upload_id: Optional[str]) -> None:
    """Optional inbound auth for /process-video. Default OFF (no-op).

    off     -> no check (current behaviour).
    monitor -> log the outcome, still accept.
    enforce -> reject missing/invalid with 401; if AI_INBOUND_SECRET is not
               configured, fail closed (401).

    Never logs the provided or configured secret value -- only an outcome code.
    """
    mode = inbound_auth_mode()
    if mode == "off":
        return

    configured = inbound_secret_configured()
    outcome = verify_inbound_secret(provided_secret)
    tag = video_upload_id or "unknown-video"

    if mode == "enforce":
        if not configured:
            logger.error(
                "[%s] Inbound auth enforce requested but AI_INBOUND_SECRET is not set; failing closed",
                tag,
            )
            raise HTTPException(status_code=401, detail="Unauthorized")
        if outcome != "ok":
            logger.warning("[%s] Inbound auth rejected: outcome=%s", tag, outcome)
            raise HTTPException(status_code=401, detail="Unauthorized")
        return

    # monitor
    logger.info(
        "[%s] Inbound auth monitor: outcome=%s secret_configured=%s",
        tag,
        outcome,
        configured,
    )


def _guard_callback_host(callback_url: str, job_id: str) -> bool:
    """Decide whether the outbound callback may be sent to callback_url.

    Returns True to proceed with send. In 'enforce' a non-allowlisted host is
    blocked (returns False) BEFORE send_callback, so the webhook secret is never
    delivered to an unapproved host. In 'monitor' the callback still sends, but a
    would-block is logged. Logs host + outcome only -- never the full URL or any
    secret.
    """
    allowed, host, reason = is_callback_allowed(callback_url)
    mode = callback_host_mode()
    safe_host = host or "unknown"

    if allowed:
        if mode == "monitor":
            logger.info("[job=%s] Callback host allowed (monitor): host=%s", job_id, safe_host)
        return True

    if mode == "enforce":
        logger.error(
            "[job=%s] Callback blocked (enforce): host=%s reason=%s",
            job_id,
            safe_host,
            reason,
        )
        return False

    logger.warning(
        "[job=%s] Callback host would be blocked (monitor): host=%s reason=%s",
        job_id,
        safe_host,
        reason,
    )
    return True


@app.post("/jobs/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job(
    job_id: str,
    x_ai_worker_secret: Optional[str] = Header(default=None),
):
    if not _worker_secret_valid(x_ai_worker_secret):
        raise HTTPException(status_code=403, detail="Forbidden")

    job = JOBS.get(job_id)
    if not job and DURABLE_QUEUE.ready:
        job = await DURABLE_QUEUE.get_job(job_id)
        if job:
            JOBS[job_id] = job
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    current_status = job.get("status")
    if current_status in {"completed", "manual_review_recommended", "failed", "error", "timed_out", "cancelled"}:
        return {
            "job_id": job_id,
            "status": current_status,
            "message": "This AI job is already in a final state.",
        }

    next_status = "cancelled" if current_status == "queued" else "cancel_requested"
    update_job(
        job_id=job_id,
        status_value=next_status,
        stage=next_status,
        progress_percent=100 if next_status == "cancelled" else int(job.get("progress_percent") or 0),
        message="AI cancellation requested. Continue with manual coach review.",
        extra={"recommended_next_action": "manual_review_recommended"},
    )
    if DURABLE_QUEUE.ready:
        await DURABLE_QUEUE.persist_job(dict(JOBS[job_id]))

    return {
        "job_id": job_id,
        "status": next_status,
        "message": "AI cancellation requested. Continue with manual coach review.",
    }


# ─────────────────────────────────────────────────────────────
# Main Vercel entrypoint
# ─────────────────────────────────────────────────────────────

@app.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(
    request: VideoProcessingRequest,
    background_tasks: BackgroundTasks,
    x_ai_inbound_secret: Optional[str] = Header(default=None),
):
    """
    Vercel calls this endpoint from /api/ai/trigger.

    This endpoint must accept quickly so Vercel does not time out.
    Heavy processing happens in the background.
    """
    # Optional inbound authentication. Default OFF: no-op unless AI_INBOUND_AUTH_MODE
    # is set to monitor/enforce. Runs before request validation and dispatch.
    _apply_inbound_auth(x_ai_inbound_secret, request.video_upload_id)

    if not request.video_upload_id:
        raise HTTPException(status_code=400, detail="Missing video_upload_id")

    if not request.signed_video_url:
        raise HTTPException(status_code=400, detail="Missing signed_video_url")

    if not request.callback_url:
        raise HTTPException(status_code=400, detail="Missing callback_url")

    job_id = get_or_create_job_id(request)
    create_job(job_id=job_id, video_upload_id=request.video_upload_id)

    logger.info(
        f"[{request.video_upload_id}] Accepted AI job {job_id}: "
        f"{safe_request_summary(request)}"
    )

    if DURABLE_QUEUE.requested:
        if not DURABLE_QUEUE.ready:
            raise HTTPException(
                status_code=503,
                detail="Durable AI queue is configured but unavailable",
            )
        await DURABLE_QUEUE.persist_job(dict(JOBS[job_id]))
        await DURABLE_QUEUE.enqueue(request, job_id)
    else:
        background_tasks.add_task(run_analysis_pipeline, request, job_id)

    return {
        "accepted": True,
        "job_id": job_id,
        "server_job_id": job_id,
        "video_upload_id": request.video_upload_id,
        "status": "queued",
        "stage": "queued",
        "engine": AI_ENGINE_VERSION,
    }


# ─────────────────────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────────────────────

async def run_analysis_pipeline(
    request: VideoProcessingRequest,
    job_id: str,
):
    started_at = time.time()
    stage_history: List[Dict[str, Any]] = []
    try:
        await asyncio.wait_for(
            _run_analysis_pipeline(request, job_id, started_at, stage_history),
            timeout=AI_JOB_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "[job=%s] AI job exceeded timeout_seconds=%s",
            job_id,
            AI_JOB_TIMEOUT_SECONDS,
        )
        await send_failure_result(
            request=request,
            job_id=job_id,
            stage_history=stage_history,
            started_at=started_at,
            status_value="timed_out",
            reason_code="worker_timeout",
        )
    except JobCancelledError:
        await send_failure_result(
            request=request,
            job_id=job_id,
            stage_history=stage_history,
            started_at=started_at,
            status_value="cancelled",
            reason_code="cancelled_by_user",
        )


async def _run_analysis_pipeline(
    request: VideoProcessingRequest,
    job_id: str,
    started_at: float,
    stage_history: List[Dict[str, Any]],
):
    video_path = None
    output_plan = build_report_output_plan(request)

    try:
        from app.pose_backends import run_pose_estimation_backend
        from app.swim_analyzer import analyze_pose_data
        from app.video_processor import (
            cleanup_temp_file,
            download_video,
            extract_frames,
            inspect_video,
        )

        ensure_job_active(job_id)
        add_stage(
            stage_history,
            job_id,
            "running",
            "downloading_video",
            10,
            "Downloading video securely from private storage",
            {
                "video_upload_id": request.video_upload_id,
                "stroke_type": request.stroke_type,
                "camera_angle": request.camera_angle,
                "engine": AI_ENGINE_VERSION,
            },
        )

        download_result = await download_video(
            video_upload_id=request.video_upload_id,
            signed_url=request.signed_video_url,
        )
        video_path = download_result.path

        if download_result.manual_review_reason:
            await send_manual_review_result(
                request=request,
                job_id=job_id,
                stage_history=stage_history,
                started_at=started_at,
                stage_message="Video skipped before AI processing",
                quality_flags=download_result.quality_flags or ["video_too_large_for_worker"],
                video_metadata={
                    "file_size_mb": download_result.file_size_mb,
                    "processing_tier": "manual_review_required",
                    "manual_review_reason": download_result.manual_review_reason,
                    "quality_flags": download_result.quality_flags,
                },
                error_message=(
                    "This video is larger than the current AI worker can download safely. "
                    "Manual coach review is recommended."
                ),
            )
            return

        if not video_path:
            await send_failure_result(
                request=request,
                job_id=job_id,
                stage_history=stage_history,
                started_at=started_at,
                status_value="failed",
                reason_code="video_download_failed",
            )
            return

        add_stage(
            stage_history,
            job_id,
            "running",
            "downloaded_video",
            20,
            "Private video downloaded",
        )

        add_stage(
            stage_history,
            job_id,
            "running",
            "reading_video_metadata",
            25,
            "Reading video metadata and workload risk",
        )

        video_metadata = await run_blocking(
            job_id,
            inspect_video,
            video_path,
            video_upload_id=request.video_upload_id,
            filename=getattr(request, "original_filename", None),
            capture_source=getattr(request, "capture_source", None),
        )
        quality_flags_from_video = list(video_metadata.get("quality_flags") or [])

        add_stage(
            stage_history,
            job_id,
            "running",
            "metadata_read",
            28,
            "Video metadata and workload risk read",
            {
                "file_size_mb": video_metadata.get("file_size_mb"),
                "source_width": video_metadata.get("source_width"),
                "source_height": video_metadata.get("source_height"),
                "video_fps": video_metadata.get("fps"),
                "video_duration_seconds": video_metadata.get("duration_seconds"),
            },
        )

        add_stage(
            stage_history,
            job_id,
            "running",
            "processing_tier_selected",
            30,
            f"Selected {video_metadata.get('processing_tier', 'unknown')} processing tier",
            {
                "processing_tier": video_metadata.get("processing_tier"),
                "quality_flags": quality_flags_from_video,
                "manual_review_reason": video_metadata.get("manual_review_reason"),
            },
        )

        if video_metadata.get("processing_tier") == "manual_review_required":
            await send_manual_review_result(
                request=request,
                job_id=job_id,
                stage_history=stage_history,
                started_at=started_at,
                stage_message="Manual review selected before frame extraction",
                quality_flags=quality_flags_from_video or ["video_too_heavy_for_ai_processing"],
                video_metadata=video_metadata,
                error_message=(
                    "This video is too risky for the current AI worker to process safely. "
                    "Manual coach review is recommended."
                ),
            )
            return

        add_stage(
            stage_history,
            job_id,
            "running",
            "extracting_frames",
            35,
            "Sampling resized review frames",
            {"processing_tier": video_metadata.get("processing_tier")},
        )

        extraction = await run_blocking(
            job_id,
            extract_frames,
            video_path,
            video_upload_id=request.video_upload_id,
            filename=getattr(request, "original_filename", None),
            capture_source=getattr(request, "capture_source", None),
            inspected_metadata=video_metadata,
        )
        frames = extraction.frames
        video_metadata = extraction.metadata or {}
        fps = float(video_metadata.get("fps") or 0.0)
        total_duration = float(video_metadata.get("duration_seconds") or 0.0)
        quality_flags_from_video = list(video_metadata.get("quality_flags") or [])

        if not frames:
            await send_manual_review_result(
                request=request,
                job_id=job_id,
                stage_history=stage_history,
                started_at=started_at,
                stage_message="Frame extraction could not produce usable frames",
                quality_flags=list(dict.fromkeys([*quality_flags_from_video, "frame_extraction_failed"])),
                video_metadata=video_metadata,
                error_message=(
                    "The worker could not extract usable frames from this video. "
                    "Manual coach review is recommended."
                ),
            )
            return

        logger.info(
            f"[{request.video_upload_id}] Extracted {len(frames)} sampled frames "
            f"fps={fps:.2f}, duration={total_duration:.2f}s, "
            f"tier={video_metadata.get('processing_tier')}"
        )

        add_stage(
            stage_history,
            job_id,
            "running",
            "frames_extracted",
            42,
            f"Extracted {len(frames)} review frames",
            {
                "frame_count_processed": len(frames),
                "video_fps": round(fps, 2),
                "video_duration_seconds": round(total_duration, 2),
                "processing_tier": video_metadata.get("processing_tier"),
                "processed_width": video_metadata.get("processed_width"),
                "processed_height": video_metadata.get("processed_height"),
            },
        )

        add_stage(
            stage_history,
            job_id,
            "running",
            "running_pose_detection",
            55,
            f"Checking swimmer visibility across {len(frames)} sampled frames",
            {
                "frame_count_processed": len(frames),
                "video_fps": round(fps, 2),
                "video_duration_seconds": round(total_duration, 2),
                "processing_tier": video_metadata.get("processing_tier"),
            },
        )

        pose_results = await run_blocking(job_id, run_pose_estimation_backend, frames)

        if not pose_results:
            await send_manual_review_result(
                request=request,
                job_id=job_id,
                stage_history=stage_history,
                started_at=started_at,
                stage_message="Pose processing did not return usable results",
                quality_flags=list(dict.fromkeys([*quality_flags_from_video, "pose_processing_failed"])),
                video_metadata=video_metadata,
                error_message=(
                    "Pose processing did not return usable swimmer evidence. "
                    "Manual coach review is recommended."
                ),
            )
            return

        # Optional temporal stabilisation of the sampled pose tracks
        # (ENABLE_POSE_SMOOTHING): interpolate short gaps, drop single-frame
        # outliers, smooth jitter. OFF by default; falls back to raw on error.
        try:
            from app.pose_postprocess import pose_smoothing_enabled, smooth_pose_results
            if pose_smoothing_enabled():
                pose_results = smooth_pose_results(pose_results)
        except Exception as smooth_err:
            logger.warning(f"[{request.video_upload_id}] pose smoothing skipped: {smooth_err}")

        frame_count_processed = len(pose_results)
        detected_frames = [r for r in pose_results if r.get("pose_detected")]
        detected_count = len(detected_frames)

        detection_ratio = (
            detected_count / frame_count_processed
            if frame_count_processed > 0
            else 0.0
        )

        detected_keypoints_count = 0.0
        visible_landmarks_average = 0.0
        if detected_frames:
            detected_keypoints_count = round(
                sum(r.get("keypoint_count", 0) for r in detected_frames)
                / len(detected_frames),
                1,
            )
            visible_landmarks_average = round(
                sum(r.get("landmark_count_total", r.get("keypoint_count", 0)) for r in detected_frames)
                / len(detected_frames),
                1,
            )

        add_stage(
            stage_history,
            job_id,
            "running",
            "analysing_stroke_phases",
            68,
            f"Analysing {request.stroke_type or 'swim'} phases and relative 2D signals",
            {
                "detected_pose_frames": detected_count,
                "detection_ratio": round(detection_ratio, 3),
                "detected_keypoints_count": detected_keypoints_count,
                "visible_landmarks_average": visible_landmarks_average,
            },
        )

        analysis_payload = await run_blocking(
            job_id,
            analyze_pose_data,
            pose_results=pose_results,
            frames=frames,
            fps=fps,
            total_duration=total_duration,
            stroke_type=request.stroke_type,
            camera_angle=request.camera_angle or "Unknown",
            video_upload_id=request.video_upload_id,
        )
        analysis_payload = filter_findings_for_outputs(analysis_payload, output_plan)

        analysis_mode = analysis_payload.get("analysis_mode", "placeholder")
        real_pose_detected = bool(analysis_payload.get("real_pose_detected"))

        temporal_metrics = analysis_payload.get("temporal_metrics") or {}
        add_stage(
            stage_history,
            job_id,
            "running",
            "generating_findings",
            78,
            "Applying sustained-evidence checks to coach-draft findings",
            {
                "temporal_sample_count": temporal_metrics.get("usable_sample_count", 0),
                "phase_segment_count": len(temporal_metrics.get("phase_segments") or []),
                "candidate_finding_count": len(analysis_payload.get("findings") or []),
            },
        )

        pose_reliability = classify_pose_reliability(
            analysis_mode=analysis_mode,
            real_pose_detected=real_pose_detected,
            detection_ratio=detection_ratio,
            detected_keypoints_count=detected_keypoints_count,
            frame_count_processed=frame_count_processed,
        )

        quality_flags = build_quality_flags(
            detection_ratio=detection_ratio,
            detected_keypoints_count=detected_keypoints_count,
            frame_count_processed=frame_count_processed,
            total_duration=total_duration,
            camera_angle=request.camera_angle or "",
        )
        quality_flags = list(dict.fromkeys([
            *quality_flags_from_video,
            *quality_flags,
            *(temporal_metrics.get("quality_flags") or []),
        ]))

        recommended_next_action = get_recommended_next_action(
            analysis_mode=analysis_mode,
            pose_reliability=pose_reliability,
            quality_flags=quality_flags,
        )

        should_allow_ai_findings = can_emit_ai_findings(
            analysis_mode=analysis_mode,
            real_pose_detected=real_pose_detected,
            pose_reliability=pose_reliability,
            detection_ratio=detection_ratio,
            detected_keypoints_count=detected_keypoints_count,
            findings=analysis_payload.get("findings") or [],
        )

        if not should_allow_ai_findings:
            logger.info(
                f"[{request.video_upload_id}] AI findings suppressed: "
                f"mode={analysis_mode}, real_pose={real_pose_detected}, "
                f"reliability={pose_reliability}, ratio={detection_ratio:.3f}, "
                f"kps={detected_keypoints_count}"
            )

            analysis_payload["analysis_mode"] = "manual_review"
            analysis_payload["real_pose_detected"] = False
            analysis_payload["findings"] = []
            analysis_payload["overall_score"] = None
            analysis_payload["phase_breakdown"] = {}
            analysis_payload["drag_analysis"] = []
            recommended_next_action = "manual_review_recommended"

        processing_duration = round(time.time() - started_at, 2)
        processing_telemetry = {
            "frames_requested": video_metadata.get("requested_frame_count", frame_count_processed),
            "frames_sampled": frame_count_processed,
            "frames_with_pose": detected_count,
            "pose_frame_failures": sum(1 for result in pose_results if result.get("error")),
            "pose_detection_rate": round(detection_ratio, 4),
            "average_core_keypoints": detected_keypoints_count,
            "average_visible_landmarks": visible_landmarks_average,
            "fallback_triggered": not should_allow_ai_findings,
            "processing_tier": video_metadata.get("processing_tier"),
            "failed_frame_reads": video_metadata.get("failed_frame_reads", 0),
            "quality_flags": quality_flags,
        }

        analysis_payload.update({
            "job_id": job_id,
            "server_job_id": job_id,
            "video_upload_id": request.video_upload_id,
            "engine": AI_ENGINE_VERSION,
            "stage_history": stage_history,
            "processing_duration_seconds": processing_duration,
            "video_duration_seconds": round(total_duration, 2),
            "video_fps": round(fps, 2),
            "detection_ratio": round(detection_ratio, 3),
            "pose_reliability": pose_reliability,
            "quality_flags": quality_flags,
            "recommended_next_action": recommended_next_action,
            "frame_count_processed": frame_count_processed,
            "detected_pose_frames": detected_count,
            "detected_keypoints_count": detected_keypoints_count,
            "processing_tier": video_metadata.get("processing_tier"),
            "source_width": video_metadata.get("source_width"),
            "source_height": video_metadata.get("source_height"),
            "processed_width": video_metadata.get("processed_width"),
            "processed_height": video_metadata.get("processed_height"),
            "processing_window_seconds": video_metadata.get("processing_window_seconds"),
            "sampled_frame_count": video_metadata.get("sampled_frame_count", frame_count_processed),
            "processing_telemetry": processing_telemetry,
        })

        # Estimated anthropometric drag is an INTERNAL PILOT prototype, OFF by
        # default. It is controlled by the ENABLE_ESTIMATED_DRAG env flag (see
        # AI_WORKER_CONTRACT.md). When the flag is unset/false this entire block
        # is skipped, so the worker behaves EXACTLY as before: no estimated_drag
        # field, no height/mass output, no extra error path, no blocked analysis,
        # and no change to the manual-review fallback.
        if "estimated_drag_force" in output_plan.get("accepted_outputs", []):
            try:
                from app.pose_worker_integration import (
                    should_emit_estimated_drag,
                    analyse_clip,
                )

                # Single-source gate: flag ON + real pose + real_pose mode
                # (not manual-review fallback) + both anthropometrics present.
                if should_emit_estimated_drag(
                    analysis_mode=analysis_payload.get("analysis_mode"),
                    real_pose_detected=analysis_payload.get("real_pose_detected"),
                    height_cm=request.swimmer_height_cm,
                    mass_kg=request.swimmer_mass_kg,
                ):
                    estimated_drag = analyse_clip(
                        pose_results,
                        fps=fps,
                        height_cm=request.swimmer_height_cm,
                        mass_kg=request.swimmer_mass_kg,
                        stroke=request.stroke_type or "Freestyle",
                    )
                    if estimated_drag:
                        analysis_payload["estimated_drag"] = estimated_drag
                        logger.info(
                            f"[{request.video_upload_id}] estimated_drag attached (pilot): "
                            f"mean_drag={estimated_drag['summary']['mean_drag_force_n']}N, "
                            f"confidence_low={estimated_drag['confidence_low']}"
                        )
            except Exception as drag_error:
                logger.warning(
                    f"[{request.video_upload_id}] estimated_drag skipped: {drag_error}"
                )

        analysis_payload = attach_report_output_metadata(analysis_payload, output_plan)

        add_stage(
            stage_history,
            job_id,
            "running",
            "generating_outputs",
            88,
            "Preparing coach-review result",
            {
                "analysis_mode": analysis_payload.get("analysis_mode"),
                "real_pose_detected": analysis_payload.get("real_pose_detected"),
                "pose_reliability": pose_reliability,
                "quality_flags": quality_flags,
                "recommended_next_action": recommended_next_action,
                "finding_count": len(analysis_payload.get("findings") or []),
            },
        )

        add_stage(
            stage_history,
            job_id,
            "running",
            "callback_sending",
            95,
            "Sending result back to Swim Sight 3D",
        )

        ensure_job_active(job_id)
        from app.callback_client import send_callback
        if _guard_callback_host(request.callback_url, job_id):
            callback_ok = await send_callback(request.callback_url, analysis_payload)
        else:
            callback_ok = False

        final_status = (
            "completed"
            if analysis_payload.get("analysis_mode") == "real_pose"
            and analysis_payload.get("real_pose_detected")
            else "manual_review_recommended"
        )

        if not callback_ok:
            final_status = "callback_failed"

        final_message = (
            "AI review ready for coach approval"
            if final_status == "completed"
            else "Manual review recommended"
            if final_status == "manual_review_recommended"
            else "Callback failed"
        )

        add_stage(
            stage_history,
            job_id,
            final_status,
            final_status,
            100,
            final_message,
            {
                "callback_sent": callback_ok,
                "processing_duration_seconds": processing_duration,
            },
        )

        logger.info(
            f"[{request.video_upload_id}] Processing finished: "
            f"job_id={job_id}, status={final_status}, "
            f"mode={analysis_payload.get('analysis_mode')}, "
            f"real_pose={analysis_payload.get('real_pose_detected')}, "
            f"findings={len(analysis_payload.get('findings') or [])}, "
            f"callback_ok={callback_ok}, duration={processing_duration}s"
        )

    except JobCancelledError:
        raise
    except Exception as error:
        logger.error(
            "[%s] Pipeline failed with %s",
            request.video_upload_id,
            type(error).__name__,
        )

        await send_failure_result(
            request=request,
            job_id=job_id,
            stage_history=stage_history,
            started_at=started_at,
            status_value="failed",
            reason_code="unknown_worker_failure",
        )

    finally:
        if video_path:
            blocking_task = ACTIVE_BLOCKING_TASKS.get(job_id)
            if blocking_task and not blocking_task.done():
                def cleanup_after_blocking(_task):
                    ACTIVE_BLOCKING_TASKS.pop(job_id, None)
                    cleanup_temp_file(video_path)
                blocking_task.add_done_callback(cleanup_after_blocking)
            else:
                ACTIVE_BLOCKING_TASKS.pop(job_id, None)
                cleanup_temp_file(video_path)


@app.on_event("startup")
async def start_durable_queue() -> None:
    logger.info(
        "Worker ready: engine=%s timeout_seconds=%s durable_queue_requested=%s pose_backend=%s",
        AI_ENGINE_VERSION,
        AI_JOB_TIMEOUT_SECONDS,
        DURABLE_QUEUE.requested,
        os.getenv("POSE_BACKEND", "mediapipe"),
    )
    if DURABLE_QUEUE.requested:
        await DURABLE_QUEUE.start(run_analysis_pipeline)


@app.on_event("shutdown")
async def stop_durable_queue() -> None:
    await DURABLE_QUEUE.stop()


# ─────────────────────────────────────────────────────────────
# Callback helpers
# ─────────────────────────────────────────────────────────────

async def send_failure_result(
    request: VideoProcessingRequest,
    job_id: str,
    stage_history: List[Dict[str, Any]],
    started_at: float,
    status_value: str,
    reason_code: str,
):
    processing_duration = round(time.time() - started_at, 2)
    failure_payload = build_failure_callback(
        video_upload_id=request.video_upload_id,
        job_id=job_id,
        status=status_value,
        reason_code=reason_code,
        engine=AI_ENGINE_VERSION,
        processing_duration_seconds=processing_duration,
        stage_history=stage_history,
    )
    add_stage(
        stage_history,
        job_id,
        status_value,
        status_value,
        100,
        failure_payload["coach_message"],
        {
            "error_message": failure_payload["coach_message"],
            "reason_code": failure_payload["reason_code"],
            "processing_duration_seconds": processing_duration,
            "quality_flags": failure_payload["quality_flags"],
        },
    )
    failure_payload = build_failure_callback(
        video_upload_id=request.video_upload_id,
        job_id=job_id,
        status=status_value,
        reason_code=reason_code,
        engine=AI_ENGINE_VERSION,
        processing_duration_seconds=processing_duration,
        stage_history=stage_history,
    )
    failure_payload = attach_report_output_metadata(
        failure_payload,
        build_report_output_plan(request),
    )
    if not failure_payload_is_safe(failure_payload):
        logger.error("[job=%s] Unsafe failure callback blocked", job_id)
        callback_ok = False
    elif not _guard_callback_host(request.callback_url, job_id):
        callback_ok = False
    else:
        from app.callback_client import send_callback
        callback_ok = await send_callback(request.callback_url, failure_payload)

    update_job(
        job_id=job_id,
        status_value=status_value,
        stage=status_value,
        progress_percent=100,
        message="Failure callback sent" if callback_ok else "Failure callback failed",
        extra={
            "callback_sent": callback_ok,
            "error_message": failure_payload["coach_message"],
            "reason_code": failure_payload["reason_code"],
            "manual_review_available": True,
        },
    )


async def send_manual_review_result(
    request: VideoProcessingRequest,
    job_id: str,
    stage_history: List[Dict[str, Any]],
    started_at: float,
    stage_message: str,
    quality_flags: Optional[List[str]] = None,
    video_metadata: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
):
    processing_duration = round(time.time() - started_at, 2)
    metadata = video_metadata or {}
    flags = list(dict.fromkeys(quality_flags or ["manual_review_recommended"]))
    if "manual_review_recommended" not in flags:
        flags.append("manual_review_recommended")

    add_stage(
        stage_history,
        job_id,
        "manual_review_recommended",
        "manual_review_fallback",
        100,
        stage_message,
        {
            "processing_duration_seconds": processing_duration,
            "processing_tier": metadata.get("processing_tier"),
            "quality_flags": flags,
            "manual_review_reason": metadata.get("manual_review_reason"),
        },
    )

    payload = {
        "job_id": job_id,
        "server_job_id": job_id,
        "video_upload_id": request.video_upload_id,
        "engine": AI_ENGINE_VERSION,
        "status": "manual_review_recommended",
        "reason_code": "insufficient_pose_confidence",
        "coach_message": (
            error_message
            or "The system did not find enough reliable evidence to produce a coach-ready AI draft."
        ),
        "manual_review_available": True,
        "analysis_mode": "manual_review",
        "real_pose_detected": False,
        "findings": [],
        "overall_score": None,
        "phase_breakdown": {},
        "drag_analysis": [],
        "key_frames": [],
        "technical_summary": (
            error_message
            or "The video was not safe enough for reliable pose-assisted analysis. Manual coach review is recommended."
        ),
        "error_message": error_message,
        "stage_history": stage_history,
        "processing_duration_seconds": processing_duration,
        "video_duration_seconds": metadata.get("duration_seconds"),
        "video_fps": metadata.get("fps"),
        "detection_ratio": 0,
        "pose_reliability": "failed",
        "quality_flags": flags,
        "recommended_next_action": "manual_review_recommended",
        "frame_count_processed": metadata.get("sampled_frame_count", 0) or 0,
        "detected_pose_frames": 0,
        "detected_keypoints_count": 0,
        "processing_tier": metadata.get("processing_tier"),
        "source_width": metadata.get("source_width"),
        "source_height": metadata.get("source_height"),
        "processed_width": metadata.get("processed_width"),
        "processed_height": metadata.get("processed_height"),
        "processing_window_seconds": metadata.get("processing_window_seconds"),
        "sampled_frame_count": metadata.get("sampled_frame_count", 0) or 0,
        "processing_telemetry": {
            "frames_requested": metadata.get("requested_frame_count", 0) or 0,
            "frames_sampled": metadata.get("sampled_frame_count", 0) or 0,
            "frames_with_pose": 0,
            "pose_frame_failures": 0,
            "pose_detection_rate": 0,
            "average_core_keypoints": 0,
            "average_visible_landmarks": 0,
            "fallback_triggered": True,
            "processing_tier": metadata.get("processing_tier"),
            "failed_frame_reads": metadata.get("failed_frame_reads", 0) or 0,
            "quality_flags": flags,
        },
    }
    payload = attach_report_output_metadata(payload, build_report_output_plan(request))

    from app.callback_client import send_callback
    if _guard_callback_host(request.callback_url, job_id):
        callback_ok = await send_callback(request.callback_url, payload)
    else:
        callback_ok = False

    update_job(
        job_id=job_id,
        status_value="manual_review_recommended",
        stage="manual_review_fallback",
        progress_percent=100,
        message="Manual review callback sent" if callback_ok else "Manual review callback failed",
        extra={
            "callback_sent": callback_ok,
            "quality_flags": flags,
            "processing_tier": metadata.get("processing_tier"),
            "manual_review_reason": metadata.get("manual_review_reason"),
        },
    )


# ─────────────────────────────────────────────────────────────
# Quality helpers
# ─────────────────────────────────────────────────────────────

def classify_pose_reliability(
    analysis_mode: str,
    real_pose_detected: bool,
    detection_ratio: float,
    detected_keypoints_count: float,
    frame_count_processed: int,
) -> str:
    """
    Coaching-facing reliability label.

    This is not a certified biomechanical accuracy score.
    """
    if frame_count_processed <= 0:
        return "failed"

    if analysis_mode == "real_pose" and real_pose_detected:
        if detection_ratio >= 0.65 and detected_keypoints_count >= 8:
            return "reliable"

        if detection_ratio >= 0.45 and detected_keypoints_count >= 7:
            return "partial"

        if detection_ratio >= 0.25 and detected_keypoints_count >= 5:
            return "weak"

    if detection_ratio > 0:
        return "weak"

    return "failed"


def can_emit_ai_findings(
    analysis_mode: str,
    real_pose_detected: bool,
    pose_reliability: str,
    detection_ratio: float,
    detected_keypoints_count: float,
    findings: List[Dict[str, Any]],
) -> bool:
    """
    Final AI output gate.

    If this returns False, the app gets manual-review state with zero AI findings.
    """
    if analysis_mode != "real_pose":
        return False

    if not real_pose_detected:
        return False

    if pose_reliability != "reliable":
        return False

    if detection_ratio < 0.65:
        return False

    if detected_keypoints_count < 8:
        return False

    if not findings:
        return False

    return True


def build_quality_flags(
    detection_ratio: float,
    detected_keypoints_count: float,
    frame_count_processed: int,
    total_duration: float,
    camera_angle: str,
) -> List[str]:
    flags: List[str] = []

    if frame_count_processed <= 0:
        flags.append("no_frames_processed")

    if detection_ratio < 0.30:
        flags.append("too_few_pose_frames")
        flags.append("low_visibility")

    if detected_keypoints_count < 6:
        flags.append("low_keypoint_count")

    if total_duration and total_duration < 4:
        flags.append("short_clip")

    if total_duration and total_duration > 15:
        flags.append("long_clip")

    camera = (camera_angle or "").lower()

    if "underwater" in camera:
        flags.append("underwater_distortion")

    if "screen" in camera:
        flags.append("screen_recording_possible")

    if "front" in camera or "head" in camera:
        flags.append("non_side_angle")

    return list(dict.fromkeys(flags))


def get_recommended_next_action(
    analysis_mode: str,
    pose_reliability: str,
    quality_flags: List[str],
) -> str:
    if analysis_mode == "real_pose" and pose_reliability in ["reliable", "partial"]:
        return "real_pose_review_ready"

    if "underwater_distortion" in quality_flags:
        return "try_above_water_side_angle"

    if "screen_recording_possible" in quality_flags:
        return "use_original_video_export"

    if "too_few_pose_frames" in quality_flags or "low_visibility" in quality_flags:
        return "try_clearer_side_angle"

    if "long_clip" in quality_flags:
        return "trim_to_5_10_seconds"

    return "manual_review_recommended"
