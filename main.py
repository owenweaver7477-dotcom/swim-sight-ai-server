from fastapi import FastAPI, BackgroundTasks, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Literal, List, Dict, Union
import httpx
import os
import tempfile
import random
import asyncio

load_dotenv()

app = FastAPI(title="Swim Sight 3D AI Processor")

AI_WEBHOOK_SECRET = os.getenv("AI_WEBHOOK_SECRET")

if not AI_WEBHOOK_SECRET:
    raise ValueError("AI_WEBHOOK_SECRET environment variable is not set.")


class VideoProcessingRequest(BaseModel):
    video_upload_id: str
    club_id: str
    swimmer_id: str
    uploaded_by_user_id: str
    signed_video_url: str
    stroke_type: str
    analysis_type: str
    camera_angle: str
    frame_rate: Union[float, str]
    callback_url: str


class Finding(BaseModel):
    finding_title: str
    finding_description: str
    why_it_matters: str
    recommended_correction: str
    drill: str
    next_focus: str
    stroke_phase: str
    severity: Literal["Low", "Medium", "High", "Critical"]
    timestamp_start: float
    timestamp_end: float
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    source: str = "AI Suggested"
    approval_status: str = "Pending Coach Review"


class KeyFrame(BaseModel):
    timestamp: float
    label: str
    description: str

class CallbackPayload(BaseModel):
    video_upload_id: str
    status: Literal["completed", "error"]
    processed_video_url: str = ""
    pose_data_file_url: str = ""
    overall_score: Union[int, None] = None
    technical_summary: Union[str, None] = None
    phase_breakdown: Union[Dict[str, int], None] = {}
    findings: List[Finding] = []
    key_frames: List[KeyFrame] = []
    error_message: Union[str, None] = None


def normalize_stroke_type(stroke: str) -> str:
    value = (stroke or "").strip().lower()

    mapping = {
        "free": "Freestyle",
        "freestyle": "Freestyle",
        "front crawl": "Freestyle",
        "breast": "Breaststroke",
        "breaststroke": "Breaststroke",
        "back": "Backstroke",
        "backstroke": "Backstroke",
        "fly": "Butterfly",
        "butterfly": "Butterfly",
        "im": "IM",
        "individual medley": "IM",
    }

    return mapping.get(value, "Freestyle")


PLACEHOLDER_FINDINGS_DATA = {
    "Freestyle": [
        {
            "finding_title": "Dropped Elbow During Catch",
            "finding_description": "The elbow appears to drop early during the catch phase, reducing the swimmer’s ability to hold water.",
            "why_it_matters": "A dropped elbow reduces propulsion and can increase drag through the front of the stroke.",
            "recommended_correction": "Cue fingertips down, elbow high, and press water back rather than down.",
            "drill": "Single-arm freestyle with catch pause.",
            "next_focus": "Establish an early vertical forearm before increasing stroke rate.",
            "severity": "High",
            "stroke_phase": "Catch",
        },
        {
            "finding_title": "Breathing Disrupts Body Line",
            "finding_description": "The head appears to lift or rotate late during breathing, affecting the swimmer’s body alignment.",
            "why_it_matters": "Late or high breathing can drop the hips and interrupt forward momentum.",
            "recommended_correction": "Cue one goggle in the water and return the head before the recovering hand enters.",
            "drill": "Six-kick switch drill with controlled breathing.",
            "next_focus": "Keep breathing low and connected to body rotation.",
            "severity": "Medium",
            "stroke_phase": "Breathing",
        },
    ],
    "Breaststroke": [
        {
            "finding_title": "Knees Recovering Too Wide",
            "finding_description": "The knees separate wider than ideal during the kick recovery.",
            "why_it_matters": "Wide knee recovery increases frontal drag and delays the propulsive snap of the kick.",
            "recommended_correction": "Cue heels up, narrow knees, late foot turn, then straight-back piston drive.",
            "drill": "Narrow-knee breaststroke kick on back.",
            "next_focus": "Recover the heels without letting the knees flare wide.",
            "severity": "High",
            "stroke_phase": "Kick Recovery",
        },
        {
            "finding_title": "Glide Body Line Drops",
            "finding_description": "The swimmer appears to lose hip height during the glide phase.",
            "why_it_matters": "Dropped hips increase drag and reduce the benefit of the kick finish.",
            "recommended_correction": "Finish the kick into a long narrow line with hips high and eyes down.",
            "drill": "Breaststroke kick into streamline glide count.",
            "next_focus": "Hold a short, controlled line after each kick.",
            "severity": "Medium",
            "stroke_phase": "Glide",
        },
    ],
    "Backstroke": [
        {
            "finding_title": "Crossover Entry",
            "finding_description": "The hand appears to enter across the centre line of the body.",
            "why_it_matters": "Crossover entry can destabilise rotation and reduce the quality of the catch.",
            "recommended_correction": "Cue hand entry in line with the shoulder and keep the head still.",
            "drill": "Single-arm backstroke with shoulder-line focus.",
            "next_focus": "Enter wide enough to protect shoulder line and rotation.",
            "severity": "High",
            "stroke_phase": "Entry",
        },
        {
            "finding_title": "Low Hip Position",
            "finding_description": "The hips appear to sit low relative to the shoulders during the stroke.",
            "why_it_matters": "Low hips increase drag and make the kick work harder than needed.",
            "recommended_correction": "Cue ribs up, head still, and hips close to the surface.",
            "drill": "Backstroke body-line kick with arms streamlined.",
            "next_focus": "Keep a flatter, higher body line.",
            "severity": "Medium",
            "stroke_phase": "Body Line",
        },
    ],
    "Butterfly": [
        {
            "finding_title": "Late Second Kick",
            "finding_description": "The second kick appears late relative to the arm exit and recovery.",
            "why_it_matters": "Late kick timing breaks rhythm and reduces forward drive during recovery.",
            "recommended_correction": "Cue the second kick to drive the hands forward and finish the stroke.",
            "drill": "3-3-3 butterfly drill with kick timing focus.",
            "next_focus": "Connect the second kick to arm recovery.",
            "severity": "High",
            "stroke_phase": "Kick Timing",
        },
        {
            "finding_title": "Flat Body Wave",
            "finding_description": "The body wave appears too flat through the stroke cycle.",
            "why_it_matters": "A flat body wave reduces rhythm and makes breathing and recovery harder.",
            "recommended_correction": "Cue chest press, hips release, and kick finishing the wave.",
            "drill": "Body dolphin drill.",
            "next_focus": "Create a smoother chest-to-hip wave.",
            "severity": "Medium",
            "stroke_phase": "Body Wave",
        },
    ],
    "IM": [
        {
            "finding_title": "Stroke Transition Timing Issue",
            "finding_description": "The transition between strokes appears rushed or inconsistent.",
            "why_it_matters": "Poor transition timing can break rhythm and lose speed between strokes.",
            "recommended_correction": "Cue a clean finish into each transition and establish rhythm early.",
            "drill": "IM transition drill with controlled tempo.",
            "next_focus": "Make the first three strokes after each transition stable.",
            "severity": "High",
            "stroke_phase": "Stroke Transition",
        },
        {
            "finding_title": "Weak Underwater Breakout",
            "finding_description": "The swimmer appears to lose body line or speed through the underwater breakout.",
            "why_it_matters": "Weak breakout position reduces speed carried off the wall.",
            "recommended_correction": "Cue tight streamline, fast underwater rhythm, and breakout before speed fades.",
            "drill": "Push-off underwater breakout timing drill.",
            "next_focus": "Hold speed from the wall into the first stroke.",
            "severity": "Medium",
            "stroke_phase": "Breakout",
        },
    ],
}


STROKE_PHASE_BREAKDOWNS = {
    "Freestyle": ["Body Line", "Catch", "Pull", "Breathing", "Kick Timing"],
    "Breaststroke": ["Body Line", "Catch", "Kick Recovery", "Kick Drive", "Glide", "Timing"],
    "Backstroke": ["Entry", "Rotation", "Catch", "Kick", "Head Position", "Body Line"],
    "Butterfly": ["Body Wave", "Catch", "Pull", "Recovery", "Kick Timing", "Breath Timing", "Body Line"],
    "IM": ["Stroke Transition", "Turn", "Underwater", "Breakout", "Rhythm Control", "Fatigue Management"],
}


STROKE_KEY_FRAMES = {
    "Freestyle": [
        {"label": "Catch Check", "description": "Reviewing early catch shape and elbow position."},
        {"label": "Breathing Window", "description": "Reviewing head timing and body rotation during breath."},
        {"label": "Body-Line Check", "description": "Reviewing hip position and alignment."},
    ],
    "Breaststroke": [
        {"label": "Kick Recovery", "description": "Reviewing knee path and heel recovery."},
        {"label": "Kick Drive", "description": "Reviewing propulsive snap and foot turn."},
        {"label": "Glide Position", "description": "Reviewing body line after the kick finish."},
    ],
    "Backstroke": [
        {"label": "Entry Check", "description": "Reviewing hand entry relative to the shoulder."},
        {"label": "Rotation Check", "description": "Reviewing body roll and shoulder line."},
        {"label": "Body-Line Check", "description": "Reviewing hip height and head position."},
    ],
    "Butterfly": [
        {"label": "Body Wave", "description": "Reviewing chest press and hip timing."},
        {"label": "Kick Timing", "description": "Reviewing second kick timing with the arm cycle."},
        {"label": "Breath Timing", "description": "Reviewing head return before hand entry."},
    ],
    "IM": [
        {"label": "Stroke Transition", "description": "Reviewing rhythm between strokes."},
        {"label": "Underwater Breakout", "description": "Reviewing body line and breakout timing."},
        {"label": "Turn Check", "description": "Reviewing speed carried through the wall."},
    ],
}


async def send_callback(url: str, payload: dict):
    headers = {"X-AI-WEBHOOK-SECRET": AI_WEBHOOK_SECRET}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=60.0)
        print("Callback status:", response.status_code)
        print("Callback response:", response.text)
        response.raise_for_status()


async def process_video_task(request_data: VideoProcessingRequest):
    temp_video_path = None

    try:
        stroke_type = normalize_stroke_type(request_data.stroke_type)

        print(f"Processing started for video_upload_id: {request_data.video_upload_id}")
        print(f"Stroke type: {stroke_type}")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                request_data.signed_video_url,
                follow_redirects=True,
                timeout=300.0,
            )
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(response.content)
                temp_video_path = tmp_file.name

        print(f"Video downloaded to: {temp_video_path}")

        await asyncio.sleep(random.uniform(3, 8))

        generated_findings = []

        for f_data in PLACEHOLDER_FINDINGS_DATA.get(stroke_type, []):
            start = round(random.uniform(5.0, 30.0), 2)
            end = round(start + random.uniform(1.2, 3.0), 2)

            generated_findings.append(
                Finding(
                    finding_title=f_data["finding_title"],
                    finding_description=f_data["finding_description"],
                    why_it_matters=f_data["why_it_matters"],
                    recommended_correction=f_data["recommended_correction"],
                    drill=f_data["drill"],
                    next_focus=f_data["next_focus"],
                    stroke_phase=f_data["stroke_phase"],
                    severity=f_data["severity"],
                    timestamp_start=start,
                    timestamp_end=end,
                    confidence_score=round(random.uniform(0.72, 0.94), 2),
                )
            )

        generated_key_frames = []

        for kf in STROKE_KEY_FRAMES.get(stroke_type, []):
            generated_key_frames.append(
                KeyFrame(
                    timestamp=round(random.uniform(5.0, 45.0), 2),
                    label=kf["label"],
                    description=kf["description"],
                )
            )

        generated_phase_breakdown = {
            phase: random.randint(62, 96)
            for phase in STROKE_PHASE_BREAKDOWNS.get(stroke_type, [])
        }

        payload = CallbackPayload(
            video_upload_id=request_data.video_upload_id,
            status="completed",
            overall_score=random.randint(72, 91),
            technical_summary=(
                f"AI-suggested {stroke_type} technical analysis generated by the external Python server. "
                f"These findings require coach review before being shared with the swimmer or parent."
            ),
            phase_breakdown=generated_phase_breakdown,
            findings=generated_findings,
            key_frames=generated_key_frames,
        )

        await send_callback(request_data.callback_url, payload.model_dump())

    except Exception as error:
        print("Processing failed:", str(error))

        error_payload = CallbackPayload(
            video_upload_id=request_data.video_upload_id,
            status="error",
            error_message=str(error),
        )

        try:
            await send_callback(request_data.callback_url, error_payload.model_dump())
        except Exception as callback_error:
            print("Error callback failed:", str(callback_error))

    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Deleted temporary video: {temp_video_path}")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


@app.post("/process-video", status_code=status.HTTP_202_ACCEPTED)
async def process_video(request: VideoProcessingRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_video_task, request)
    return {
        "message": "Video processing started",
        "video_upload_id": request.video_upload_id,
    }