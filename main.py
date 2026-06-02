from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks, status
from app.models import VideoProcessingRequest, HealthResponse
from app.video_processor import download_video, extract_frames, cleanup_temp_file
from app.pose_estimator import run_pose_estimation
from app.swim_analyzer import analyze_pose_data
from app.callback_client import send_callback
from app.utils import build_error_callback

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Swim Sight AI Server", version="pose-mvp-0.1")


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    return HealthResponse(status="ok", engine="pose-mvp-0.1")


@app.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
    logger.info(f"[{request.video_upload_id}] Received processing request")
    background_tasks.add_task(run_analysis_pipeline, request)

    return {
        "message": "Video processing initiated",
        "video_upload_id": request.video_upload_id,
    }


async def run_analysis_pipeline(request: VideoProcessingRequest):
    video_path = None

    try:
        logger.info(
            f"[{request.video_upload_id}] Starting pose pipeline — "
            f"stroke={request.stroke_type}, camera_angle={request.camera_angle}"
        )

        video_path = await download_video(
            video_upload_id=request.video_upload_id,
            signed_url=request.signed_video_url,
        )

        if not video_path:
            await send_callback(
                request.callback_url,
                build_error_callback(
                    request.video_upload_id,
                    "Failed to download video from signed URL."
                ),
            )
            return

        frames, fps, total_duration = extract_frames(video_path)

        if not frames:
            await send_callback(
                request.callback_url,
                build_error_callback(
                    request.video_upload_id,
                    "Could not extract frames from video file."
                ),
            )
            return

        logger.info(
            f"[{request.video_upload_id}] Extracted {len(frames)} sampled frames "
            f"at fps={fps:.2f}, duration={total_duration:.2f}s"
        )

        pose_results = run_pose_estimation(frames)

        analysis_payload = analyze_pose_data(
            pose_results=pose_results,
            frames=frames,
            fps=fps,
            total_duration=total_duration,
            stroke_type=request.stroke_type,
            camera_angle=request.camera_angle or "Unknown",
            video_upload_id=request.video_upload_id,
        )

        await send_callback(request.callback_url, analysis_payload)

        logger.info(
            f"[{request.video_upload_id}] Callback sent — "
            f"mode={analysis_payload.get('analysis_mode')}, "
            f"real_pose_detected={analysis_payload.get('real_pose_detected')}, "
            f"findings={len(analysis_payload.get('findings', []))}"
        )

    except Exception as error:
        logger.exception(f"[{request.video_upload_id}] Pipeline failed")

        error_payload = build_error_callback(
            request.video_upload_id,
            f"Internal processing error: {str(error)}"
        )

        try:
            await send_callback(request.callback_url, error_payload)
        except Exception:
            logger.exception(f"[{request.video_upload_id}] Error callback failed")

    finally:
        if video_path:
            cleanup_temp_file(video_path)
