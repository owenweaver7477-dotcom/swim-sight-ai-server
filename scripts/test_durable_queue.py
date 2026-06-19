"""Configuration and enqueue safety tests for the optional durable queue."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.durable_queue import DurableJobQueue, durable_queue_requested  # noqa: E402
from app.models import VideoProcessingRequest  # noqa: E402


assert durable_queue_requested({}) is False
assert durable_queue_requested({"ENABLE_DURABLE_QUEUE": "false"}) is False
assert durable_queue_requested({"ENABLE_DURABLE_QUEUE": "true"}) is True
assert durable_queue_requested({"ENABLE_DURABLE_QUEUE": "1"}) is True


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.messages = []

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)

    async def xadd(self, stream, fields, **kwargs):
        self.messages.append((stream, fields))
        return "1-0"


async def check_queue_behaviour():
    queue = DurableJobQueue()
    queue._client = FakeRedis()
    queue._consumer_task = object()
    request = VideoProcessingRequest(
        job_id="job-1",
        video_upload_id="video-1",
        signed_video_url="https://signed-url-redacted.example/video",
        callback_url="https://callback.example/api/ai/callback",
        stroke_type="Breaststroke",
    )

    assert await queue.enqueue(request, "job-1") is True
    assert len(queue._client.messages) == 1
    assert await queue.enqueue(request, "job-1") is False
    assert len(queue._client.messages) == 1

    snapshot = {"job_id": "job-1", "status": "queued"}
    await queue.persist_job(snapshot)
    assert await queue.get_job("job-1") == snapshot


asyncio.run(check_queue_behaviour())
print("durable queue configuration tests passed")
