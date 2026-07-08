"""Tests for the Batch 1 pilot-hardening fixes.

Covers: duplicate-job guard, safe 422 validation responses, callback retry
with backoff, and temp-file cleanup on failed downloads. Offline: pipeline and
HTTP clients are faked; no network, no real videos, no deploy.

Run:  python3 scripts/test_pilot_hardening.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main as worker  # noqa: E402
import app.callback_client as cc  # noqa: E402
import app.video_processor as vp  # noqa: E402

FAKE_TOKEN = "SECRET_TOKEN_ABC123_DO_NOT_ECHO"


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def load_request():
    with (ROOT / "fixtures" / "process_video_request.example.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ── 1. Duplicate-job guard ────────────────────────────────────────────────────

def test_duplicate_job_guard():
    dispatches = []

    async def fake_pipeline(request, job_id):
        dispatches.append(job_id)

    original = worker.run_analysis_pipeline
    worker.run_analysis_pipeline = fake_pipeline
    worker.JOBS.clear()
    body = load_request()

    try:
        with TestClient(worker.app) as client:
            first = client.post("/process-video", json=body)
            _check(first.status_code == 202, f"first POST must be 202, got {first.status_code}")
            _check("duplicate_suppressed" not in first.json(), "first POST must not be marked duplicate")
            job_id = first.json()["job_id"]
            # Fake pipeline leaves status 'queued' => job counts as in-flight.
            _check(len(dispatches) == 1, f"first POST must dispatch once, got {len(dispatches)}")

            second = client.post("/process-video", json=body)
            _check(second.status_code == 202, f"duplicate POST must still be 202, got {second.status_code}")
            payload = second.json()
            _check(payload.get("duplicate_suppressed") is True, "duplicate must be flagged suppressed")
            _check(payload["accepted"] is True, "duplicate response must remain accepted=True")
            _check(payload["job_id"] == job_id, "duplicate must echo the same job_id")
            for key in ("server_job_id", "video_upload_id", "status", "stage", "engine"):
                _check(key in payload, f"duplicate response missing contract key {key}")
            _check(len(dispatches) == 1, f"duplicate must NOT dispatch again, got {len(dispatches)}")

            # Terminal job may re-run (existing safe recovery path).
            worker.JOBS[job_id]["status"] = "callback_failed"
            third = client.post("/process-video", json=body)
            _check(third.status_code == 202, "terminal re-run must be accepted")
            _check("duplicate_suppressed" not in third.json(), "terminal re-run must not be suppressed")
            _check(len(dispatches) == 2, f"terminal re-run must dispatch, got {len(dispatches)}")
    finally:
        worker.run_analysis_pipeline = original
        worker.JOBS.clear()

    print("  duplicate-job guard ok")


# ── 2. Safe 422 validation responses ─────────────────────────────────────────

def test_422_does_not_echo_input():
    body = load_request()
    body["signed_video_url"] = f"https://storage.example/video.mp4?token={FAKE_TOKEN}"
    body.pop("callback_url", None)  # missing required field => FastAPI 422

    worker.JOBS.clear()
    with TestClient(worker.app) as client:
        response = client.post("/process-video", json=body)

    _check(response.status_code == 422, f"expected 422, got {response.status_code}")
    text = response.text
    _check(FAKE_TOKEN not in text, "422 body leaked the signed URL token")
    _check("storage.example" not in text, "422 body leaked the signed URL host")
    _check("input" not in response.json().get("errors", [{}])[0], "422 errors must not carry input echo")
    payload = response.json()
    _check(payload.get("detail") == "Request validation failed", "422 detail wording changed")
    fields = [item.get("field") for item in payload.get("errors", [])]
    _check("callback_url" in fields, f"422 must name the missing field, got {fields}")
    print("  422 redaction ok")


# ── 3. Callback retry with backoff ────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, code):
        self.status_code = code
        self.text = "upstream says no"


def _fake_httpx(script, calls):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None, headers=None):
            calls.append(1)
            step = script.pop(0)
            if step == "timeout":
                raise httpx.TimeoutException("simulated timeout")
            if step == "exc":
                raise RuntimeError("simulated network error")
            return _FakeResponse(step)

    return types.SimpleNamespace(AsyncClient=FakeClient, TimeoutException=httpx.TimeoutException)


async def _no_sleep(_seconds):
    return None


def _run_send(script):
    calls = []
    orig_httpx, orig_asyncio = cc.httpx, cc.asyncio
    os.environ["AI_WEBHOOK_SECRET"] = "unit-test-secret"
    cc.httpx = _fake_httpx(list(script), calls)
    cc.asyncio = types.SimpleNamespace(sleep=_no_sleep)
    try:
        ok = asyncio.run(cc.send_callback("https://app.example/api/ai/callback", {"job_id": "j1"}))
    finally:
        cc.httpx, cc.asyncio = orig_httpx, orig_asyncio
        os.environ.pop("AI_WEBHOOK_SECRET", None)
    return ok, len(calls)


def test_callback_retry():
    ok, n = _run_send([500, 500, 200])
    _check(ok is True and n == 3, f"5xx,5xx,200 must succeed on 3rd attempt (ok={ok}, calls={n})")

    ok, n = _run_send(["timeout", 200])
    _check(ok is True and n == 2, f"timeout then 200 must succeed on retry (ok={ok}, calls={n})")

    ok, n = _run_send(["exc", 200])
    _check(ok is True and n == 2, f"network error then 200 must succeed on retry (ok={ok}, calls={n})")

    ok, n = _run_send([500, 500, 500])
    _check(ok is False and n == 3, f"persistent 5xx must fail after 3 attempts (ok={ok}, calls={n})")

    ok, n = _run_send([401])
    _check(ok is False and n == 1, f"4xx must NOT retry (ok={ok}, calls={n})")

    ok, n = _run_send([200])
    _check(ok is True and n == 1, f"happy path unchanged (ok={ok}, calls={n})")
    print("  callback retry ok")


# ── 4. Temp-file cleanup on failed downloads ─────────────────────────────────

def _spy_tempfile(created):
    import tempfile as real_tempfile

    def spy_named(*args, **kwargs):
        handle = real_tempfile.NamedTemporaryFile(*args, **kwargs)
        created.append(handle.name)
        return handle

    return types.SimpleNamespace(NamedTemporaryFile=spy_named)


def _fake_stream_httpx(status_code=None, raise_error=False, chunks=()):
    class FakeStreamResponse:
        def __init__(self):
            self.status_code = status_code

        async def aiter_bytes(self, chunk_size=None):
            for chunk in chunks:
                yield chunk

    class FakeStreamCM:
        async def __aenter__(self):
            if raise_error:
                raise RuntimeError("simulated download error")
            return FakeStreamResponse()

        async def __aexit__(self, *args):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url):
            return FakeStreamCM()

    return types.SimpleNamespace(AsyncClient=FakeClient)


def _run_download(fake_httpx):
    created = []
    orig_httpx, orig_tempfile = vp.httpx, vp.tempfile
    vp.httpx = fake_httpx
    vp.tempfile = _spy_tempfile(created)
    try:
        result = asyncio.run(vp.download_video("unit-video", "https://signed.example/v.mp4?token=x"))
    finally:
        vp.httpx, vp.tempfile = orig_httpx, orig_tempfile
    return result, created


def test_temp_cleanup_on_failed_download():
    # Non-200 response: temp file must be deleted.
    result, created = _run_download(_fake_stream_httpx(status_code=403))
    _check(result.path is None and result.failed_reason == "download_http_403",
           f"403 must fail with download_http_403, got {result.failed_reason}")
    _check(created and not os.path.exists(created[0]), "temp file must be deleted on non-200 download")

    # Exception during download: temp file must be deleted.
    result, created = _run_download(_fake_stream_httpx(raise_error=True))
    _check(result.path is None and result.failed_reason == "download_error",
           f"exception must fail with download_error, got {result.failed_reason}")
    _check(created and not os.path.exists(created[0]), "temp file must be deleted on download exception")

    # Happy path unchanged: file exists with content, then clean up.
    result, created = _run_download(_fake_stream_httpx(status_code=200, chunks=(b"fakebytes",)))
    _check(result.path is not None and os.path.exists(result.path), "successful download must keep the file")
    vp.cleanup_temp_file(result.path)
    _check(not os.path.exists(result.path), "manual cleanup must remove the file")
    print("  temp cleanup ok")


def main() -> int:
    test_duplicate_job_guard()
    test_422_does_not_echo_input()
    test_callback_retry()
    test_temp_cleanup_on_failed_download()
    print("pilot hardening tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
