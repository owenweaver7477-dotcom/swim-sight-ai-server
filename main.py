from dotenv import load_dotenv
load_dotenv()

import uuid
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, BackgroundTasks, HTTPException, status

from app.models import VideoProcessingRequest, HealthResponse
from app.video_processor import download_video, extract_frames, cleanup_temp_file
from app.pose_estimator import run_pose_estimation
from app.swim_analyzer import analyze_pose_data
from app.callback_client import send_callback
from app.utils import build_error_callback


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────

AI_ENGINE_VERSION = "pose-mvp-0.2"

app = FastAPI(
    title="Swim Sight AI Server",
    version=AI_ENGINE_VERSION,
)


# ─────────────────────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────────────────────
# This is good enough for the current Render MVP.
#
# Important:
# - Jobs are stored in memory.
# - If Render restarts, this store resets.
# - Base44 still remains the source of truth through AIProcessingJob.
#
# Future upgrade:
# Replace this with Redis/Celery/Postgres if you need persistent job tracking.
# ─────────────────────────────────────────────────────────────

JOBS: Dict[str, Dict[str, Any]] = {}


def now_iso() -> str:
    """Return UTC timestamp in ISO format for Base44/debug logs."""
    return datetime.utcnow().isoformat() + "Z"


def create_job(video_upload_id: str) -> str:
    """Create a Python-side job record and return job_id."""
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "video_upload_id": video_upload_id,
        "status": "queued",
        "stage": "queued",
        "progress_percent": 0,
        "message": "Queued for pose-assisted review",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }

    return job_id


def update_job(
    job_id: str,
    status_value: str,
    stage: str,
    progress_percent: int,
    message: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Update Python-side job state.

    Base44 can poll GET /jobs/{job_id} in future.
    Current Base44 mostly relies on callback, but this makes the server production-ready.
    """
    if job_id not in JOBS:
        JOBS[job_id] = {"job_id": job_id}

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
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Record a stage both in memory and in the callback payload.

    Base44 stores this on AIProcessingJob so the app can show a real timeline.
    """
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


# ─────────────────────────────────────────────────────────────
# Health / job status endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    """
    Render health check.

    Base44 does not need this for analysis, but Render uses it to confirm the server is live.
    """
    return HealthResponse(status="ok", engine=AI_ENGINE_VERSION)


@app.get("/jobs/{job_id}", status_code=status.HTTP_200_OK)
def get_job(job_id: str):
    """
    Optional status endpoint.

    Base44 can use this later to poll job stage/progress while processing.
    """
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


# ─────────────────────────────────────────────────────────────
# Main Base44 entrypoint
# ─────────────────────────────────────────────────────────────

@app.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(
    request: VideoProcessingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Base44 calls this endpoint from triggerPoseAnalysis.

    New Phase 9C behaviour:
    - accept immediately
    - return job_id immediately
    - process in background
    - callback to Base44 when done

    Base44 expects this response shape:
    {
      "accepted": true,
      "job_id": "...",
      "video_upload_id": "...",
      "status": "queued"
    }
    """
    job_id = create_job(request.video_upload_id)

    logger.info(
        f"[{request.video_upload_id}] Accepted AI job {job_id} — "
        f"stroke={request.stroke_type}, camera_angle={request.camera_angle}"
    )

    background_tasks.add_task(run_analysis_pipeline, request, job_id)

    return {
        "accepted": True,
        "job_id": job_id,
        "video_upload_id": request.video_upload_id,
        "status": "queued",
    }


# ─────────────────────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────────────────────

async def run_analysis_pipeline(
    request: VideoProcessingRequest,
    job_id: str,
):
    """
    Full AI processing pipeline.

    Important:
    - This must never create fake findings.
    - If pose detection is weak, return placeholder/unreliable with zero findings.
    - Coach approval in Base44 remains required.
    """
    video_path = None
    started_at = time.time()
    stage_history: List[Dict[str, Any]] = []

    try:
        # ─────────────────────────────────────────────────────
        # Stage 1 — Download video from Base44 signed URL
        # ─────────────────────────────────────────────────────

        add_stage(
            stage_history,
            job_id,
            "running",
            "downloading_video",
            10,
            "Downloading video securely from Base44 storage",
            {
                "video_upload_id": request.video_upload_id,
                "stroke_type": request.stroke_type,
                "camera_angle": request.camera_angle,
            },
        )

        video_path = await download_video(
            video_upload_id=request.video_upload_id,
            signed_url=request.signed_video_url,
        )

        if not video_path:
            await send_error_result(
                request=request,
                job_id=job_id,
                stage_history=stage_history,
                started_at=started_at,
                error_message="Failed to download video from signed URL.",
                stage_message="Video download failed",
            )
            return

        # ─────────────────────────────────────────────────────
        # Stage 2 — Extract sampled frames
        # ─────────────────────────────────────────────────────

        add_stage(
            stage_history,
            job_id,
            "running",
            "extracting_frames",
            30,
            "Extracting review frames",
        )

        frames, fps, total_duration = extract_frames(video_path)

        if not frames:
            await send_error_result(
                request=request,
                job_id=job_id,
                stage_history=stage_history,
                started_at=started_at,
                error_message="Could not extract frames from video file.",
                stage_message="Frame extraction failed",
            )
            return

        logger.info(
            f"[{request.video_upload_id}] Extracted {len(frames)} sampled frames "
            f"at fps={fps:.2f}, duration={total_duration:.2f}s"
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
            },
        )

        # ─────────────────────────────────────────────────────
        # Stage 3 — Pose estimation
        # ─────────────────────────────────────────────────────

        pose_results = run_pose_estimation(frames)

        detected_frames = [
            result for result in pose_results
            if result.get("pose_detected")
        ]

        frame_count_processed = len(pose_results)
        detected_count = len(detected_frames)
        detection_ratio = (
            detected_count / frame_count_processed
            if frame_count_processed > 0
            else 0
        )

        detected_keypoints_count = 0
        if detected_frames:
            detected_keypoints_count = round(
                sum(r.get("keypoint_count", 0) for r in detected_frames)
                / len(detected_frames),
                1,
            )

        add_stage(
            stage_history,
            job_id,
            "running",
            "analysing_stroke",
            75,
            f"Running {request.stroke_type} stroke-specific checks",
            {
                "detected_pose_frames": detected_count,
                "detection_ratio": round(detection_ratio, 3),
                "detected_keypoints_count": detected_keypoints_count,
            },
        )

        # ─────────────────────────────────────────────────────
        # Stage 4 — Stroke analysis rules
        # ─────────────────────────────────────────────────────

        analysis_payload = analyze_pose_data(
            pose_results=pose_results,
            frames=frames,
            fps=fps,
            total_duration=total_duration,
            stroke_type=request.stroke_type,
            camera_angle=request.camera_angle or "Unknown",
            video_upload_id=request.video_upload_id,
        )

        # ─────────────────────────────────────────────────────
        # Stage 5 — Reliability classification
        # ─────────────────────────────────────────────────────

        analysis_mode = analysis_payload.get("analysis_mode", "placeholder")
        real_pose_detected = bool(analysis_payload.get("real_pose_detected"))

        pose_reliability = classify_pose_reliability(
            analysis_mode=analysis_mode,
            real_pose_detected=real_pose_detected,
            detection_ratio=detection_ratio,
            detected_keypoints_count=detected_keypoints_count,
        )

        quality_flags = build_quality_flags(
            detection_ratio=detection_ratio,
            detected_keypoints_count=detected_keypoints_count,
            camera_angle=request.camera_angle or "",
            signed_video_url=request.signed_video_url or "",
        )

        recommended_next_action = get_recommended_next_action(
            analysis_mode=analysis_mode,
            pose_reliability=pose_reliability,
            quality_flags=quality_flags,
        )

        add_stage(
            stage_history,
            job_id,
            "running",
            "generating_outputs",
            88,
            "Preparing AI review result for Base44",
            {
                "analysis_mode": analysis_mode,
                "real_pose_detected": real_pose_detected,
                "pose_reliability": pose_reliability,
                "quality_flags": quality_flags,
                "recommended_next_action": recommended_next_action,
            },
        )

        # ─────────────────────────────────────────────────────
        # Stage 6 — Enrich callback payload for Base44
        # ─────────────────────────────────────────────────────

        processing_duration = round(time.time() - started_at, 2)

        analysis_payload.update({
            "job_id": job_id,
            "stage_history": stage_history,
            "processing_duration_seconds": processing_duration,
            "video_duration_seconds": round(total_duration, 2),
            "video_fps": round(fps, 2),
            "detection_ratio": round(detection_ratio, 3),
            "pose_reliability": pose_reliability,
            "quality_flags": quality_flags,
            "recommended_next_action": recommended_next_action,
            "frame_count_processed": frame_count_processed,
            "detected_keypoints_count": detected_keypoints_count,
        })

        # Extra safety:
        # If not real_pose, never allow fake findings / scores / drag analysis through.
        if analysis_mode != "real_pose" or not real_pose_detected:
            analysis_payload["analysis_mode"] = "placeholder"
            analysis_payload["real_pose_detected"] = False
            analysis_payload["findings"] = []
            analysis_payload["overall_score"] = None
            analysis_payload["phase_breakdown"] = {}
            analysis_payload["drag_analysis"] = []

        add_stage(
            stage_history,
            job_id,
            "running",
            "callback_sending",
            95,
            "Sending result back to Base44",
        )

        # ─────────────────────────────────────────────────────
        # Stage 7 — Callback to Base44
        # ─────────────────────────────────────────────────────

        await send_callback(request.callback_url, analysis_payload)

        final_status = (
            "completed"
            if analysis_payload.get("analysis_mode") == "real_pose"
            else "unreliable_pose"
        )

        final_message = (
            "AI review ready for coach approval"
            if final_status == "completed"
            else "Pose unreliable — manual review recommended"
        )

        add_stage(
            stage_history,
            job_id,
            final_status,
            final_status,
            100,
            final_message,
            {
                "callback_sent": True,
                "processing_duration_seconds": processing_duration,
            },
        )

        logger.info(
            f"[{request.video_upload_id}] Callback sent — "
            f"job_id={job_id}, mode={analysis_payload.get('analysis_mode')}, "
            f"real_pose_detected={analysis_payload.get('real_pose_detected')}, "
            f"findings={len(analysis_payload.get('findings', []))}, "
            f"duration={processing_duration}s"
        )

    except Exception as error:
        logger.exception(f"[{request.video_upload_id}] Pipeline failed")

        await send_error_result(
            request=request,
            job_id=job_id,
            stage_history=stage_history,
            started_at=started_at,
            error_message=f"Internal processing error: {str(error)}",
            stage_message="Internal processing error",
        )

    finally:
        if video_path:
            cleanup_temp_file(video_path)


# ─────────────────────────────────────────────────────────────
# Callback helpers
# ─────────────────────────────────────────────────────────────

async def send_error_result(
    request: VideoProcessingRequest,
    job_id: str,
    stage_history: List[Dict[str, Any]],
    started_at: float,
    error_message: str,
    stage_message: str,
):
    """
    Send safe error callback to Base44.

    This ensures the app does not remain stuck in processing forever.
    """
    processing_duration = round(time.time() - started_at, 2)

    add_stage(
        stage_history,
        job_id,
        "error",
        "error",
        100,
        stage_message,
        {
            "error_message": error_message,
            "processing_duration_seconds": processing_duration,
        },
    )

    error_payload = build_error_callback(
        request.video_upload_id,
        error_message,
    )

    error_payload.update({
        "job_id": job_id,
        "stage_history": stage_history,
        "processing_duration_seconds": processing_duration,
        "pose_reliability": "failed",
        "quality_flags": ["processing_error"],
        "recommended_next_action": "manual_review_recommended",
        "detection_ratio": 0,
    })

    try:
        await send_callback(request.callback_url, error_payload)
    except Exception:
        logger.exception(f"[{request.video_upload_id}] Error callback failed")


def classify_pose_reliability(
    analysis_mode: str,
    real_pose_detected: bool,
    detection_ratio: float,
    detected_keypoints_count: float,
) -> str:
    """
    Convert raw pose stats into a simple reliability label for Base44.

    This is a coaching-facing reliability classification,
    not a biomechanical certainty score.
    """
    if analysis_mode == "real_pose" and real_pose_detected:
        if detection_ratio >= 0.60 and detected_keypoints_count >= 8:
            return "reliable"
        if detection_ratio >= 0.35 and detected_keypoints_count >= 6:
            return "partial"
        return "weak"

    if detection_ratio > 0:
        return "weak"

    return "failed"


def build_quality_flags(
    detection_ratio: float,
    detected_keypoints_count: float,
    camera_angle: str,
    signed_video_url: str,
) -> List[str]:
    """
    Build non-judgemental quality flags for the coach.

    These flags explain why AI may have failed or why manual review is needed.
    """
    flags: List[str] = []

    if detection_ratio < 0.30:
        flags.append("too_few_keypoints")
        flags.append("low_visibility")

    if detected_keypoints_count < 6:
        flags.append("low_keypoint_count")

    camera = camera_angle.lower()
    if "underwater" in camera:
        flags.append("underwater_distortion")

    url = signed_video_url.lower()
    if "screenrecording" in url or "screen-recording" in url or "screen" in url:
        flags.append("screen_recording_possible")

    return list(dict.fromkeys(flags))


def get_recommended_next_action(
    analysis_mode: str,
    pose_reliability: str,
    quality_flags: List[str],
) -> str:
    """
    Give Base44 a clear next action to display.

    This is guidance only and should not create fake findings.
    """
    if analysis_mode == "real_pose" and pose_reliability in ["reliable", "partial"]:
        return "real_pose_review_ready"

    if "underwater_distortion" in quality_flags:
        return "try_above_water_angle"

    if "screen_recording_possible" in quality_flags:
        return "use_clearer_video"

    if "too_few_keypoints" in quality_flags or "low_visibility" in quality_flags:
        return "try_side_angle"

    return "manual_review_recommended"