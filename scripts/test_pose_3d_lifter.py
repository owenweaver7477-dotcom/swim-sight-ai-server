"""Monocular 3D lifter contract checks."""

import json
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models import VideoProcessingRequest  # noqa: E402
from app.pose_3d_lifter import (  # noqa: E402
    build_pose_3d_callback_payload,
    build_pose_3d_summary,
    lift_pose_2d_frames_to_3d,
    pose_3d_callback_payload_is_safe,
    write_pose_3d_artifact,
)


POSE_2D_FRAMES = [
    {
        "timestamp_ms": 0,
        "source_frame_index": 0,
        "sample_index": 0,
        "view_type": "Side",
        "pose_model": "mediapipe_pose",
        "joints_2d": {
            "left_shoulder": {"x": 0.42, "y": 0.31, "confidence": 0.91, "visibility": 0.91, "status": "tracked"},
            "right_shoulder": {"x": 0.58, "y": 0.31, "confidence": 0.89, "visibility": 0.89, "status": "tracked"},
            "left_elbow": {"x": 0.38, "y": 0.42, "confidence": 0.76, "visibility": 0.76, "status": "tracked"},
            "right_elbow": {"x": 0.63, "y": 0.42, "confidence": 0.73, "visibility": 0.73, "status": "tracked"},
            "left_wrist": {"x": 0.35, "y": 0.52, "confidence": 0.41, "visibility": 0.41, "status": "low_confidence"},
            "right_wrist": {"x": 0.66, "y": 0.52, "confidence": 0.82, "visibility": 0.82, "status": "tracked"},
            "left_hip": {"x": 0.45, "y": 0.58, "confidence": 0.8, "visibility": 0.8, "status": "tracked"},
            "right_hip": {"x": 0.55, "y": 0.58, "confidence": 0.79, "visibility": 0.79, "status": "tracked"},
            "left_knee": {"x": 0.45, "y": 0.75, "confidence": 0.78, "visibility": 0.78, "status": "tracked"},
            "right_knee": {"x": 0.55, "y": 0.75, "confidence": 0.77, "visibility": 0.77, "status": "tracked"},
            "left_ankle": {"x": 0.44, "y": 0.9, "confidence": 0.74, "visibility": 0.74, "status": "tracked"},
            "right_ankle": {"x": 0.56, "y": 0.9, "confidence": 0.72, "visibility": 0.72, "status": "tracked"},
        },
        "frame_confidence": 0.76,
        "tracking_status": "tracked",
    },
    {
        "timestamp_ms": 200,
        "source_frame_index": 12,
        "sample_index": 1,
        "view_type": "Side",
        "pose_model": "mediapipe_pose",
        "joints_2d": {
            "left_shoulder": {"x": 0.4, "y": 0.33, "confidence": 0.62, "visibility": 0.62, "status": "low_confidence"},
            "right_shoulder": {"x": 0.6, "y": 0.33, "confidence": 0.64, "visibility": 0.64, "status": "low_confidence"},
            "left_hip": {"x": 0.44, "y": 0.58, "confidence": 0.61, "visibility": 0.61, "status": "low_confidence"},
            "right_hip": {"x": 0.56, "y": 0.58, "confidence": 0.63, "visibility": 0.63, "status": "low_confidence"},
            "left_ankle": {"x": None, "y": None, "confidence": 0.0, "visibility": None, "status": "missing"},
        },
        "frame_confidence": 0.62,
        "tracking_status": "partial",
    },
    {
        "timestamp_ms": 400,
        "source_frame_index": 24,
        "sample_index": 2,
        "view_type": "Side",
        "pose_model": "mediapipe_pose",
        "joints_2d": {},
        "frame_confidence": 0.0,
        "tracking_status": "no_person_detected",
    },
]


def test_lift_valid_frames_to_estimated_3d():
    frames = lift_pose_2d_frames_to_3d(POSE_2D_FRAMES)
    assert len(frames) == 3
    first = frames[0]
    assert first["timestamp_ms"] == 0
    assert first["source_frame_index"] == 0
    assert first["sample_index"] == 0
    assert first["source"] == "monocular_estimate"
    assert first["method"] == "anatomical_heuristic_lift"
    assert first["measurementType"] == "estimated"
    assert first["pose_3d_model"] == "anatomical_heuristic_lift_v1"
    shoulder = first["joints_3d"]["left_shoulder"]
    assert shoulder["status"] == "estimated"
    assert shoulder["source_2d_confidence"] == 0.91
    assert shoulder["confidence"] > 0
    assert first["tracking_status"] == "estimated"


def test_missing_and_low_confidence_are_preserved_safely():
    frames = lift_pose_2d_frames_to_3d(POSE_2D_FRAMES)
    low = frames[1]["joints_3d"]["left_shoulder"]
    missing = frames[1]["joints_3d"]["left_ankle"]
    assert low["status"] == "estimated"
    assert low["confidence"] < low["source_2d_confidence"]
    assert missing["status"] == "missing"
    assert missing["x"] is None
    assert frames[2]["tracking_status"] == "no_person_detected"


def test_summary_and_artifact_are_private_and_estimated():
    frames = lift_pose_2d_frames_to_3d(POSE_2D_FRAMES)
    summary = build_pose_3d_summary(frames)
    assert summary["availabilityState"] == "pose_3d_estimated"
    assert summary["ok"] is True
    assert summary["source"] == "monocular_estimate"
    assert summary["measurementType"] == "estimated"
    assert summary["calibration"]["cameraCalibrated"] is False
    assert summary["calibration"]["worldScaleKnown"] is False
    assert summary["calibration"]["multiView"] is False
    assert "not measured metres" in " ".join(summary["assumptions"])

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = write_pose_3d_artifact(frames, "job-3d", output_dir=tmpdir)
        assert artifact["artifact_type"] == "pose_3d_timeseries"
        assert artifact["storage_visibility"] == "private"
        assert artifact["contains_raw_pose"] is True
        assert artifact["contains_video_pixels"] is False
        assert artifact["public_safe"] is False
        assert artifact["source"] == "monocular_estimate"
        assert artifact["measurementType"] == "estimated"
        assert "path" not in artifact
        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 1
        written = json.loads(files[0].read_text(encoding="utf-8"))
        assert written["source"] == "monocular_estimate"
        assert written["measurementType"] == "estimated"


def test_callback_payload_is_summary_only():
    frames = lift_pose_2d_frames_to_3d(POSE_2D_FRAMES)
    summary = build_pose_3d_summary(frames)
    artifact = {
        "artifact_type": "pose_3d_timeseries",
        "artifact_id": "job-3d-pose-3d",
        "storage_visibility": "private",
        "format": "json",
        "frame_count": 3,
        "contains_raw_pose": True,
        "contains_video_pixels": False,
        "public_safe": False,
        "source": "monocular_estimate",
        "measurementType": "estimated",
    }
    request = VideoProcessingRequest(
        job_id="job-3d",
        app_job_id="app-job-3d",
        video_upload_id="video-3d",
        signed_video_url="https://signed-url-redacted.example/private-video.mp4",
        callback_url="https://app.example/api/ai/callback",
    )
    payload = build_pose_3d_callback_payload(request, "job-3d", summary, artifact)
    assert payload["status"] == "pose_3d_estimated"
    assert payload["pose_3d_summary"]["measurementType"] == "estimated"
    assert payload["pose_3d_artifact"]["public_safe"] is False
    assert pose_3d_callback_payload_is_safe(payload)
    text = json.dumps(payload).lower()
    assert "joints_3d" not in text
    assert "joints_2d" not in text
    assert "signed_video_url" not in text
    assert "private-video" not in text
    assert "estimated_drag" not in text
    assert "drag_force" not in text
    assert "biomechanics" not in text


def test_empty_input_returns_structured_non_throwing_summary():
    frames = lift_pose_2d_frames_to_3d([])
    summary = build_pose_3d_summary(frames)
    assert frames == []
    assert summary["ok"] is False
    assert summary["inputFrames"] == 0
    assert summary["measurementType"] == "estimated"


def test_callback_safety_blocks_raw_or_private_values():
    assert not pose_3d_callback_payload_is_safe({
        "status": "pose_3d_estimated",
        "pose_3d_frames": [{"joints_3d": {}}],
    })
    assert not pose_3d_callback_payload_is_safe({
        "status": "pose_3d_estimated",
        "pose_3d_artifact": {"local_path": "/tmp/private-3d-pose.json"},
    })
    assert not pose_3d_callback_payload_is_safe({
        "status": "pose_3d_estimated",
        "pose_3d_artifact": {"url": "https://host.example/pose.json?token=secret"},
    })


if __name__ == "__main__":
    test_lift_valid_frames_to_estimated_3d()
    test_missing_and_low_confidence_are_preserved_safely()
    test_summary_and_artifact_are_private_and_estimated()
    test_callback_payload_is_summary_only()
    test_empty_input_returns_structured_non_throwing_summary()
    test_callback_safety_blocks_raw_or_private_values()
    print("pose 3d lifter checks passed")
