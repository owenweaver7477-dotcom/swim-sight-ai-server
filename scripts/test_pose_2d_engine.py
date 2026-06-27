"""2D pose contract normalisation tests."""

import json
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models import VideoProcessingRequest  # noqa: E402
from app.pose_2d_engine import (  # noqa: E402
    build_pose_2d_callback_payload,
    build_pose_2d_summary,
    pose_2d_callback_payload_is_safe,
    pose_results_to_pose_2d_frames,
    write_pose_2d_artifact,
)


FRAME_SAMPLING = {
    "samples": [
        {"sampleIndex": 0, "sourceFrameIndex": 0, "timestampMs": 0, "status": "scheduled"},
        {"sampleIndex": 1, "sourceFrameIndex": 12, "timestampMs": 200, "status": "scheduled"},
        {"sampleIndex": 2, "sourceFrameIndex": 24, "timestampMs": 400, "status": "scheduled"},
    ]
}


POSE_RESULTS = [
    {
        "frame_idx": 0,
        "pose_detected": True,
        "keypoint_count": 10,
        "landmark_count_total": 12,
        "landmarks": {
            "nose": {"x": 0.5, "y": 0.2, "visibility": 0.92},
            "left_shoulder": {"x": 0.42, "y": 0.31, "visibility": 0.91},
            "right_shoulder": {"x": 0.58, "y": 0.31, "visibility": 0.89},
            "left_elbow": {"x": 0.38, "y": 0.42, "visibility": 0.76},
            "right_elbow": {"x": 0.63, "y": 0.42, "visibility": 0.73},
            "left_wrist": {"x": 0.35, "y": 0.52, "visibility": 0.41},
            "right_wrist": {"x": 0.66, "y": 0.52, "visibility": 0.82},
            "left_hip": {"x": 0.45, "y": 0.58, "visibility": 0.8},
            "right_hip": {"x": 0.55, "y": 0.58, "visibility": 0.79},
            "left_knee": {"x": 0.45, "y": 0.75, "visibility": 0.78},
            "right_knee": {"x": 0.55, "y": 0.75, "visibility": 0.77},
            "left_ankle": {"x": 0.44, "y": 0.9, "visibility": 0.74},
            "right_ankle": {"x": 0.56, "y": 0.9, "visibility": 0.72},
        },
    },
    {
        "frame_idx": 12,
        "pose_detected": True,
        "keypoint_count": 4,
        "landmarks": {
            "left_shoulder": {"x": 0.4, "y": 0.33, "visibility": 0.62},
            "right_shoulder": {"x": 0.6, "y": 0.33, "visibility": 0.64},
            "left_hip": {"x": 0.44, "y": 0.58, "visibility": 0.61},
            "right_hip": {"x": 0.56, "y": 0.58, "visibility": 0.63},
        },
    },
    {
        "frame_idx": 24,
        "pose_detected": False,
        "keypoint_count": 0,
        "landmarks": {},
    },
]


def test_pose_frames_are_timestamped_and_classified():
    frames = pose_results_to_pose_2d_frames(POSE_RESULTS, FRAME_SAMPLING, fps=60, view_type="Side")
    assert len(frames) == 3
    assert frames[0]["timestamp_ms"] == 0
    assert frames[1]["timestamp_ms"] == 200
    assert frames[2]["timestamp_ms"] == 400
    assert frames[0]["joints_2d"]["left_shoulder"]["status"] == "tracked"
    assert frames[0]["joints_2d"]["left_wrist"]["status"] == "low_confidence"
    assert frames[0]["joints_2d"]["left_eye"]["status"] == "missing"
    assert frames[0]["tracking_status"] == "tracked"
    assert frames[1]["tracking_status"] == "partial"
    assert frames[2]["tracking_status"] == "no_person_detected"


def test_summary_and_artifact_are_safe():
    frames = pose_results_to_pose_2d_frames(POSE_RESULTS, FRAME_SAMPLING, fps=60, view_type="Side")
    summary = build_pose_2d_summary(frames, view_type="Side", sampled_frames=3)
    assert summary["availabilityState"] == "pose_2d_ready"
    assert summary["processedFrames"] == 3
    assert summary["trackedFrames"] == 1
    assert summary["partialFrames"] == 1
    assert summary["failedFrames"] == 1
    assert summary["lowConfidenceJointRate"] > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = write_pose_2d_artifact(frames, "job-1", output_dir=tmpdir)
        assert artifact["artifact_type"] == "pose_2d_timeseries"
        assert artifact["storage_visibility"] == "private"
        assert artifact["contains_raw_pose"] is True
        assert artifact["contains_video_pixels"] is False
        assert artifact["public_safe"] is False
        assert "path" not in artifact
        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 1
        written = json.loads(files[0].read_text(encoding="utf-8"))
        assert written["frames"][0]["joints_2d"]["left_shoulder"]["status"] == "tracked"


def test_callback_payload_contains_summary_not_raw_pose():
    frames = pose_results_to_pose_2d_frames(POSE_RESULTS, FRAME_SAMPLING, fps=60, view_type="Side")
    summary = build_pose_2d_summary(frames, view_type="Side", sampled_frames=3)
    artifact = {
        "artifact_type": "pose_2d_timeseries",
        "artifact_id": "job-1-pose-2d",
        "storage_visibility": "private",
        "format": "json",
        "frame_count": 3,
        "contains_raw_pose": True,
        "contains_video_pixels": False,
        "public_safe": False,
    }
    request = VideoProcessingRequest(
        job_id="job-1",
        app_job_id="app-job-1",
        video_upload_id="video-1",
        signed_video_url="https://signed-url-redacted.example/private-video.mp4",
        callback_url="https://app.example/api/ai/callback",
    )
    payload = build_pose_2d_callback_payload(request, "job-1", summary, artifact)
    assert payload["status"] == "pose_2d_ready"
    assert payload["pose_2d_summary"]["processedFrames"] == 3
    assert payload["pose_artifact"]["public_safe"] is False
    assert pose_2d_callback_payload_is_safe(payload)
    text = json.dumps(payload).lower()
    assert "joints_2d" not in text
    assert "landmarks" not in text
    assert "signed_video_url" not in text
    assert "private-video" not in text
    assert "joints_3d" not in text
    assert "estimated_drag" not in text


def test_callback_safety_blocks_raw_or_private_values():
    assert not pose_2d_callback_payload_is_safe({
        "status": "pose_2d_ready",
        "pose_2d_frames": [{"joints_2d": {}}],
    })
    assert not pose_2d_callback_payload_is_safe({
        "status": "pose_2d_ready",
        "pose_artifact": {"local_path": "/tmp/private-pose.json"},
    })
    assert not pose_2d_callback_payload_is_safe({
        "status": "pose_2d_ready",
        "pose_artifact": {"url": "https://host.example/pose.json?token=secret"},
    })


if __name__ == "__main__":
    test_pose_frames_are_timestamped_and_classified()
    test_summary_and_artifact_are_safe()
    test_callback_payload_contains_summary_not_raw_pose()
    test_callback_safety_blocks_raw_or_private_values()
    print("pose 2d engine checks passed")
