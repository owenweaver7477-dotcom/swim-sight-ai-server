"""Cancellation must prevent queued work and block late completion updates."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import main as worker  # noqa: E402


secret = "synthetic-worker-secret"
original_secret = os.environ.get("AI_WEBHOOK_SECRET")
os.environ["AI_WEBHOOK_SECRET"] = secret
worker.JOBS.clear()

try:
    worker.create_job("queued-job", "video-1")
    with TestClient(worker.app) as client:
        denied = client.post("/jobs/queued-job/cancel")
        assert denied.status_code == 403
        response = client.post(
            "/jobs/queued-job/cancel",
            headers={"x-ai-worker-secret": secret},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    worker.update_job("queued-job", "completed", "completed", 100, "late completion")
    assert worker.JOBS["queued-job"]["status"] == "cancelled"

    worker.create_job("running-job", "video-2")
    worker.update_job("running-job", "running", "running_pose_detection", 55, "running")
    with TestClient(worker.app) as client:
        response = client.post(
            "/jobs/running-job/cancel",
            headers={"x-ai-worker-secret": secret},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "cancel_requested"
    worker.update_job("running-job", "completed", "completed", 100, "late completion")
    assert worker.JOBS["running-job"]["status"] == "cancel_requested"
    try:
        worker.ensure_job_active("running-job")
    except worker.JobCancelledError:
        pass
    else:
        raise AssertionError("Cancellation did not stop the next worker stage")
finally:
    worker.JOBS.clear()
    if original_secret is None:
        os.environ.pop("AI_WEBHOOK_SECRET", None)
    else:
        os.environ["AI_WEBHOOK_SECRET"] = original_secret

print("job cancellation test passed")
