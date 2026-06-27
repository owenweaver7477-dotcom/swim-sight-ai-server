"""Endpoint checks for /process-video storage access inputs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import main as worker  # noqa: E402


async def fake_analysis_pipeline(_request, _job_id):
    return None


def post_process_video(client, payload):
    response = client.post("/process-video", json=payload)
    return response.status_code, response.json()


base_payload = {
    "job_id": "11111111-1111-4111-8111-111111111111",
    "app_job_id": "11111111-1111-4111-8111-111111111111",
    "video_upload_id": "22222222-2222-4222-8222-222222222222",
    "callback_url": "https://swim-sight-3d-v1.vercel.app/api/ai/callback",
    "stroke_type": "Breaststroke",
    "camera_angle": "Side",
}

original_pipeline = worker.run_analysis_pipeline
worker.run_analysis_pipeline = fake_analysis_pipeline
worker.JOBS.clear()

try:
    with TestClient(worker.app) as client:
        status_code, payload = post_process_video(client, {
            **base_payload,
            "job_id": "job-signed-url",
            "signed_video_url": "https://signed-url-redacted.example/private-video.mp4",
        })
        assert status_code == 202
        assert payload["accepted"] is True
        assert payload["job_id"] == "job-signed-url"

        status_code, payload = post_process_video(client, {
            **base_payload,
            "job_id": "job-provider-key",
            "storage_provider": "supabase_private",
            "video_key": "club/swimmer/video/private-video.mp4",
        })
        assert status_code == 202
        assert payload["accepted"] is True
        assert payload["job_id"] == "job-provider-key"

        status_code, payload = post_process_video(client, {
            **base_payload,
            "job_id": "job-missing-source",
        })
        assert status_code == 400
        assert payload["detail"] == "Missing private video access method"

        signed_summary = worker.safe_request_summary(type("Req", (), {
            "video_upload_id": "video-1",
            "stroke_type": "Freestyle",
            "camera_angle": "Side",
            "callback_url": "https://callback.example",
            "signed_video_url": "https://example.test/video?token=secret",
            "storage_provider": None,
            "video_key": None,
        })())
        assert "token=secret" not in str(signed_summary)

        provider_summary = worker.safe_request_summary(type("Req", (), {
            "video_upload_id": "video-2",
            "stroke_type": "Freestyle",
            "camera_angle": "Side",
            "callback_url": "https://callback.example",
            "signed_video_url": None,
            "storage_provider": "supabase_private",
            "video_key": "club/swimmer/video/private-video.mp4",
        })())
        assert "club/swimmer/video" not in str(provider_summary)
        assert "redacted-key" in str(provider_summary)
finally:
    worker.run_analysis_pipeline = original_pipeline
    worker.JOBS.clear()

print("worker storage access endpoint tests passed")
