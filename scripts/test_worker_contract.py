import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import main as worker  # noqa: E402


def load_fixture(name: str):
    with (ROOT / "fixtures" / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def fake_analysis_pipeline(_request, _job_id):
    return None


def assert_keys(payload: dict, required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise AssertionError(f"{label} missing keys: {', '.join(missing)}")


def check_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["engine"] == worker.AI_ENGINE_VERSION


def check_process_video_accepts_documented_request(client: TestClient) -> str:
    request_payload = load_fixture("process_video_request.example.json")
    response = client.post("/process-video", json=request_payload)
    assert response.status_code == 202

    payload = response.json()
    assert_keys(
        payload,
        {"accepted", "job_id", "server_job_id", "video_upload_id", "status", "stage", "engine"},
        "process-video response",
    )
    assert payload["accepted"] is True
    assert payload["job_id"] == request_payload["job_id"]
    assert payload["server_job_id"] == request_payload["job_id"]
    assert payload["video_upload_id"] == request_payload["video_upload_id"]
    assert payload["status"] == "queued"
    assert payload["stage"] == "queued"
    assert payload["engine"] == worker.AI_ENGINE_VERSION
    return payload["job_id"]


def check_job_status(client: TestClient, job_id: str) -> None:
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    payload = response.json()
    assert_keys(
        payload,
        {"job_id", "server_job_id", "video_upload_id", "status", "stage", "progress_percent", "message"},
        "job status response",
    )
    assert payload["job_id"] == job_id
    assert payload["status"] == "queued"
    assert payload["stage"] == "queued"


def check_manual_review_fixture_contract() -> None:
    payload = load_fixture("callback_manual_review.example.json")
    assert payload["analysis_mode"] == "manual_review"
    assert payload["real_pose_detected"] is False
    assert payload["findings"] == []
    assert payload["overall_score"] is None
    assert payload["recommended_next_action"] == "manual_review_recommended"


def main() -> int:
    original_pipeline = worker.run_analysis_pipeline
    worker.run_analysis_pipeline = fake_analysis_pipeline
    worker.JOBS.clear()

    try:
        with TestClient(worker.app) as client:
            check_health(client)
            job_id = check_process_video_accepts_documented_request(client)
            check_job_status(client, job_id)
            check_manual_review_fixture_contract()
    finally:
        worker.run_analysis_pipeline = original_pipeline
        worker.JOBS.clear()

    print("worker contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
