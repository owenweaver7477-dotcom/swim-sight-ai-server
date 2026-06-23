from pydantic import BaseModel
from typing import Any, Dict, Optional


class VideoProcessingRequest(BaseModel):
    job_id: Optional[str] = None
    app_job_id: Optional[str] = None
    video_upload_id: str
    club_id: Optional[str] = None
    swimmer_id: Optional[str] = None
    uploaded_by_user_id: Optional[str] = None
    signed_video_url: str
    stroke_type: str = "Freestyle"
    analysis_type: Optional[str] = "Technique Review"
    camera_angle: Optional[str] = "Side"
    frame_rate: Optional[float] = 30.0
    callback_url: str
    capture_source: Optional[str] = None
    original_filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_size_mb: Optional[float] = None
    duration_seconds: Optional[float] = None
    review_context: Optional[Dict[str, Any]] = None
    max_sampled_frames: Optional[int] = None
    downscale_frames: Optional[bool] = None
    # Optional coach-entered anthropometrics (server-side only). Used to scale
    # pose tracking and estimate drag/force. Never echoed in the callback.
    swimmer_height_cm: Optional[float] = None
    swimmer_mass_kg: Optional[float] = None


class HealthResponse(BaseModel):
    ok: bool
    service: str
    version: str
    timestamp: str
    heavy_models_loaded: bool
    status: str
    engine: str
