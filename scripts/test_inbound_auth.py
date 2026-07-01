"""Inbound authentication tests for POST /process-video.

Exercises off / monitor / enforce modes with the FastAPI TestClient. No network,
no real pipeline (it is mocked), no deploy. Also asserts the inbound secret value
never appears in a response body.

Run:  python3 scripts/test_inbound_auth.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import main as worker  # noqa: E402

SECRET = "unit-inbound-secret-abc123"
WRONG = "unit-inbound-secret-WRONG"
HEADER = "x-ai-inbound-secret"


def load_request():
    with (ROOT / "fixtures" / "process_video_request.example.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def fake_pipeline(_request, _job_id):
    return None


def set_mode(mode, secret=None):
    os.environ["AI_INBOUND_AUTH_MODE"] = mode
    if secret is None:
        os.environ.pop("AI_INBOUND_SECRET", None)
    else:
        os.environ["AI_INBOUND_SECRET"] = secret


def assert_status(response, expected, label):
    if response.status_code != expected:
        raise AssertionError(f"{label}: expected {expected}, got {response.status_code} ({response.text[:200]})")


def assert_no_secret(response, label):
    if SECRET in response.text or WRONG in response.text:
        raise AssertionError(f"{label}: secret value leaked into response body")


def main() -> int:
    original_pipeline = worker.run_analysis_pipeline
    worker.run_analysis_pipeline = fake_pipeline
    worker.JOBS.clear()
    body = load_request()

    try:
        with TestClient(worker.app) as client:
            # off: no header -> accepted (current behaviour preserved)
            set_mode("off")
            r = client.post("/process-video", json=body)
            assert_status(r, 202, "off/no-header")

            # off: unknown mode value also falls back to off
            set_mode("gibberish")
            r = client.post("/process-video", json=body)
            assert_status(r, 202, "unknown-mode/no-header")

            # monitor: accepts with or without a header
            set_mode("monitor", SECRET)
            assert_status(client.post("/process-video", json=body), 202, "monitor/no-header")
            r = client.post("/process-video", json=body, headers={HEADER: SECRET})
            assert_status(r, 202, "monitor/valid-header")
            assert_no_secret(r, "monitor/valid-header")
            assert_status(
                client.post("/process-video", json=body, headers={HEADER: WRONG}),
                202,
                "monitor/wrong-header",
            )

            # enforce (secret configured): reject missing/invalid, accept valid
            set_mode("enforce", SECRET)
            r = client.post("/process-video", json=body)
            assert_status(r, 401, "enforce/no-header")
            assert_no_secret(r, "enforce/no-header")
            assert_status(
                client.post("/process-video", json=body, headers={HEADER: WRONG}),
                401,
                "enforce/wrong-header",
            )
            r = client.post("/process-video", json=body, headers={HEADER: SECRET})
            assert_status(r, 202, "enforce/valid-header")
            assert_no_secret(r, "enforce/valid-header")

            # enforce but AI_INBOUND_SECRET missing -> fail closed (401) either way
            set_mode("enforce", None)
            assert_status(client.post("/process-video", json=body), 401, "enforce/no-secret/no-header")
            assert_status(
                client.post("/process-video", json=body, headers={HEADER: SECRET}),
                401,
                "enforce/no-secret/with-header",
            )
    finally:
        worker.run_analysis_pipeline = original_pipeline
        worker.JOBS.clear()
        os.environ.pop("AI_INBOUND_AUTH_MODE", None)
        os.environ.pop("AI_INBOUND_SECRET", None)

    print("inbound auth tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
