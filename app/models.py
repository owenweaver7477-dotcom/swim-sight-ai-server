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
    review_context: Optional[Dict[str, Any]] = None
    max_sampled_frames: Optional[int] = None
    downscale_frames: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    engine: str
