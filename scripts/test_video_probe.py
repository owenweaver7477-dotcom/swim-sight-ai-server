"""Video metadata probe and timestamp sampling contract checks."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.models import VideoProcessingRequest  # noqa: E402
from app.video_probe import (  # noqa: E402
    build_frame_sampling_plan,
    build_video_probe_callback_payload,
    callback_payload_is_safe,
    metadata_from_ffprobe_json,
    probe_video_file,
)


def fake_ffprobe_payload(**stream_overrides):
    stream = {
        "codec_type": "video",
        "codec_name": "h264",
        "avg_frame_rate": "60/1",
        "r_frame_rate": "60/1",
        "duration": "42.5",
        "nb_frames": "2550",
        "width": 1920,
        "height": 1080,
    }
    stream.update(stream_overrides)
    return {
        "streams": [stream],
        "format": {
            "duration": "42.5",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "size": str(int(146.2 * 1024 * 1024)),
        },
    }


def test_valid_ffprobe_metadata():
    metadata = metadata_from_ffprobe_json(fake_ffprobe_payload(), file_size_mb=146.2)
    assert metadata["duration_seconds"] == 42.5
    assert metadata["fps"] == 60.0
    assert metadata["frame_count"] == 2550
    assert metadata["width"] == 1920
    assert metadata["height"] == 1080
    assert metadata["codec"] == "h264"
    assert metadata["container"] == "mov"
    assert metadata["orientation"] == "landscape"
    assert metadata["metadata_complete"] is True
    assert metadata["errors"] == []


def test_sampling_plan_default_and_cap():
    metadata = metadata_from_ffprobe_json(fake_ffprobe_payload(duration="120", nb_frames="7200"))
    plan = build_frame_sampling_plan(metadata, sampling_rate_fps=5, max_sampled_frames=300)
    assert plan["ok"] is True
    assert plan["sampling_rate_fps"] == 5.0
    assert plan["source_fps"] == 60.0
    assert plan["total_sampled_frames"] == 300
    assert "sample_count_capped" in plan["warnings"]
    assert plan["samples"][1]["timestampMs"] == 200
    assert plan["samples"][1]["sourceFrameIndex"] == 12


def test_unknown_fps_warns_not_crashes():
    metadata = metadata_from_ffprobe_json(
        fake_ffprobe_payload(avg_frame_rate="0/0", r_frame_rate="0/0", nb_frames=None),
        file_size_mb=1.5,
    )
    plan = build_frame_sampling_plan(metadata, sampling_rate_fps=5, max_sampled_frames=10)
    assert "fps_unavailable" in metadata["warnings"]
    assert "source_fps_unavailable" in plan["warnings"]
    assert plan["ok"] is True
    assert plan["samples"][0]["sourceFrameIndex"] is None


def test_missing_file_returns_safe_error():
    result = probe_video_file("/private/non-existent/swim-video.mp4", file_size_mb=2.0)
    assert result["ok"] is False
    assert "video_file_missing" in result["errors"]
    assert "/private/non-existent" not in json.dumps(result)


def test_probe_callbacks_are_safe_and_statused():
    request = VideoProcessingRequest(
        job_id="job-1",
        app_job_id="app-job-1",
        video_upload_id="video-1",
        signed_video_url="https://signed-url-redacted.example/private-video.mp4",
        callback_url="https://app.example/api/ai/callback",
        stroke_type="breaststroke",
        camera_angle="Side",
    )
    metadata = metadata_from_ffprobe_json(fake_ffprobe_payload(), file_size_mb=10.0)
    probe = {"ok": True, "video_metadata": metadata, "warnings": [], "errors": []}
    plan = build_frame_sampling_plan(metadata)

    metadata_payload = build_video_probe_callback_payload(
        request,
        job_id="job-1",
        status_value="metadata_ready",
        video_probe=probe,
        engine="pose-mvp-0.5",
    )
    frames_payload = build_video_probe_callback_payload(
        request,
        job_id="job-1",
        status_value="frames_sampled",
        video_probe=probe,
        frame_sampling=plan,
        engine="pose-mvp-0.5",
    )

    assert metadata_payload["status"] == "metadata_ready"
    assert frames_payload["status"] == "frames_sampled"
    assert frames_payload["frame_sampling"]["total_sampled_frames"] > 0
    assert callback_payload_is_safe(metadata_payload)
    assert callback_payload_is_safe(frames_payload)
    payload_text = json.dumps(frames_payload).lower()
    assert "signed_video_url" not in payload_text
    assert "private-video" not in payload_text
    assert "landmarks" not in payload_text
    assert "joints_2d" not in payload_text
    assert "joints_3d" not in payload_text
    assert "estimated_drag" not in payload_text


def test_callback_safety_rejects_private_values():
    assert not callback_payload_is_safe({
        "status": "metadata_ready",
        "video_metadata": {"path": "/tmp/private-video.mp4"},
    })
    assert not callback_payload_is_safe({
        "status": "metadata_ready",
        "video_metadata": {"source": "https://host.example/video.mp4?token=secret"},
    })


if __name__ == "__main__":
    test_valid_ffprobe_metadata()
    test_sampling_plan_default_and_cap()
    test_unknown_fps_warns_not_crashes()
    test_missing_file_returns_safe_error()
    test_probe_callbacks_are_safe_and_statused()
    test_callback_safety_rejects_private_values()
    print("video probe contract checks passed")
