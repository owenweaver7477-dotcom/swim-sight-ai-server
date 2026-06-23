"""Health endpoint must stay fast and avoid importing heavy pose modules."""

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import main as worker  # noqa: E402


assert "app.pose_estimator" not in sys.modules
assert "app.video_processor" not in sys.modules


async def verify_blocking_stage_does_not_starve_health():
    worker.JOBS.clear()
    worker.create_job("health-during-work", "synthetic-video")
    blocking = asyncio.create_task(
        worker.run_blocking("health-during-work", time.sleep, 0.15)
    )
    await asyncio.sleep(0.01)
    started = time.perf_counter()
    payload = worker.health_check().model_dump()
    elapsed = time.perf_counter() - started
    assert payload["ok"] is True
    assert elapsed < 0.1, f"Health was starved by blocking work: {elapsed:.4f}s"
    await blocking
    worker.JOBS.clear()


with TestClient(worker.app) as client:
    started = time.perf_counter()
    response = client.get("/health")
    elapsed = time.perf_counter() - started

assert response.status_code == 200
payload = response.json()
assert payload["ok"] is True
assert payload["service"] == "swim-sight-ai-server"
assert payload["version"] == worker.AI_ENGINE_VERSION
assert payload["status"] == "ok"
assert payload["engine"] == worker.AI_ENGINE_VERSION
assert payload["heavy_models_loaded"] is False
assert elapsed < 0.1, f"Health request took {elapsed:.4f}s"
assert "app.pose_estimator" not in sys.modules
assert "app.video_processor" not in sys.modules
asyncio.run(verify_blocking_stage_does_not_starve_health())

print(f"worker health test passed in {elapsed * 1000:.2f}ms")
