"""Tests for the AI_MAX_CONCURRENT_JOBS concurrency cap.

Verifies the pure env parsers and that run_analysis_pipeline actually limits how
many heavy pipelines run at once (the heavy stage is stubbed). No network, no
real analysis, no deploy.

Run:  python3 scripts/test_concurrency_cap.py
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as worker  # noqa: E402
from app.concurrency import max_concurrent_jobs, post_timeout_drain_seconds  # noqa: E402


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_parsers():
    _check(max_concurrent_jobs(env={}) == 0, "unset -> 0 (disabled)")
    _check(max_concurrent_jobs(env={"AI_MAX_CONCURRENT_JOBS": "0"}) == 0, "0 -> 0")
    _check(max_concurrent_jobs(env={"AI_MAX_CONCURRENT_JOBS": "3"}) == 3, "3 -> 3")
    _check(max_concurrent_jobs(env={"AI_MAX_CONCURRENT_JOBS": "-2"}) == 0, "-2 -> 0")
    _check(max_concurrent_jobs(env={"AI_MAX_CONCURRENT_JOBS": "abc"}) == 0, "abc -> 0")
    _check(max_concurrent_jobs(env={"AI_MAX_CONCURRENT_JOBS": " 2 "}) == 2, "' 2 ' -> 2")

    _check(post_timeout_drain_seconds(env={}) == 0.0, "drain unset -> 0")
    _check(post_timeout_drain_seconds(env={"AI_POST_TIMEOUT_DRAIN_SECONDS": "5"}) == 5.0, "5 -> 5")
    _check(post_timeout_drain_seconds(env={"AI_POST_TIMEOUT_DRAIN_SECONDS": "-1"}) == 0.0, "-1 -> 0")
    _check(post_timeout_drain_seconds(env={"AI_POST_TIMEOUT_DRAIN_SECONDS": "abc"}) == 0.0, "abc -> 0")
    _check(post_timeout_drain_seconds(env={"AI_POST_TIMEOUT_DRAIN_SECONDS": "99999"}) == 600.0, "clamp -> 600")


class _Req:
    video_upload_id = "synthetic"
    stroke_type = "Freestyle"
    callback_url = "https://example/cb"


async def _run_scenario(cap, n, hold=0.05):
    """Launch n pipelines with the given cap and return the peak concurrency."""
    if cap is None:
        os.environ.pop("AI_MAX_CONCURRENT_JOBS", None)
    else:
        os.environ["AI_MAX_CONCURRENT_JOBS"] = str(cap)
    worker._JOB_SEMAPHORE = None
    worker._JOB_SEMAPHORE_CAP = 0
    worker.JOBS.clear()

    state = {"active": 0, "peak": 0}
    lock = asyncio.Lock()

    async def fake_pipeline(request, job_id, started_at, stage_history):
        async with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(hold)
        async with lock:
            state["active"] -= 1

    original = worker._run_analysis_pipeline
    worker._run_analysis_pipeline = fake_pipeline
    try:
        await asyncio.gather(*[worker.run_analysis_pipeline(_Req(), f"job-{i}") for i in range(n)])
    finally:
        worker._run_analysis_pipeline = original
        os.environ.pop("AI_MAX_CONCURRENT_JOBS", None)
        worker._JOB_SEMAPHORE = None
        worker._JOB_SEMAPHORE_CAP = 0
        worker.JOBS.clear()
    return state["peak"]


def test_cap_enforced():
    peak2 = asyncio.run(_run_scenario(cap=2, n=6))
    _check(peak2 <= 2, f"cap=2 must limit concurrency to <=2, peak={peak2}")
    _check(peak2 >= 1, f"cap=2 must still run jobs, peak={peak2}")

    peak1 = asyncio.run(_run_scenario(cap=1, n=4))
    _check(peak1 == 1, f"cap=1 must serialise heavy jobs, peak={peak1}")

    peak_off = asyncio.run(_run_scenario(cap=None, n=6))
    _check(peak_off >= 3, f"disabled cap must allow full concurrency, peak={peak_off}")


def main() -> int:
    test_parsers()
    test_cap_enforced()
    print("concurrency cap tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
