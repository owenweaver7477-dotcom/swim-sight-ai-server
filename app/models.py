from pydantic import BaseModel
from typing import Optional


class VideoProcessingRequest(BaseModel):
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


class HealthResponse(BaseModel):
    status: str
    engine: str
