from typing import Dict, Any

AI_ENGINE_VERSION = "pose-mvp-0.4"
MODEL_NAME = "mediapipe_pose"


def build_error_callback(video_upload_id: str, error_message: str) -> Dict[str, Any]:
    return {
        "video_upload_id": video_upload_id,
        "status": "error",
        "analysis_mode": "error",
        "ai_engine_version": AI_ENGINE_VERSION,
        "model_name": MODEL_NAME,
        "real_pose_detected": False,
        "frame_count_processed": 0,
        "detected_keypoints_count": 0,
        "processed_video_url": "",
        "pose_data_file_url": "",
        "overall_score": None,
        "technical_summary": (
            "Video processing failed before analysis could complete. "
            "No findings have been generated. Please retry or contact support."
        ),
        "phase_breakdown": {},
        "findings": [],
        "key_frames": [],
        "error_message": error_message,
    }
