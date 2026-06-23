"""The pipeline wrapper must turn an overlong job into a safe timeout result."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as worker  # noqa: E402
from app.job_reliability import job_timeout_seconds  # noqa: E402
from app.models import VideoProcessingRequest  # noqa: E402


assert job_timeout_seconds(None) == 600
assert job_timeout_seconds("480") == 480
assert job_timeout_seconds("invalid") == 600
assert job_timeout_seconds("2") == 600


async def check_timeout():
    captured = []

    async def slow_pipeline(_request, _job_id, _started_at, _stage_history):
        await asyncio.sleep(1)

    async def capture_failure(**kwargs):
        captured.append(kwargs)

    request = VideoProcessingRequest(
        job_id="timeout-job",
        video_upload_id="timeout-video",
        signed_video_url="https://private.example.invalid/video",
        callback_url="https://callback.example.invalid/api/ai/callback",
    )
    original_pipeline = worker._run_analysis_pipeline
    original_failure = worker.send_failure_result
    original_timeout = worker.AI_JOB_TIMEOUT_SECONDS
    worker._run_analysis_pipeline = slow_pipeline
    worker.send_failure_result = capture_failure
    worker.AI_JOB_TIMEOUT_SECONDS = 0.01
    try:
        await worker.run_analysis_pipeline(request, "timeout-job")
    finally:
        worker._run_analysis_pipeline = original_pipeline
        worker.send_failure_result = original_failure
        worker.AI_JOB_TIMEOUT_SECONDS = original_timeout

    assert len(captured) == 1
    assert captured[0]["status_value"] == "timed_out"
    assert captured[0]["reason_code"] == "worker_timeout"


asyncio.run(check_timeout())
print("job timeout test passed")
