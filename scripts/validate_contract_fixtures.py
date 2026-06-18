import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"

sys.path.insert(0, str(ROOT))

from app.models import VideoProcessingRequest  # noqa: E402


EXPECTED_FIXTURES = {
    "process_video_request.example.json",
    "process_video_accepted.example.json",
    "callback_success.example.json",
    "callback_manual_review.example.json",
    "callback_failed.example.json",
    "job_status.example.json",
}

REQUEST_REQUIRED = {
    "video_upload_id",
    "signed_video_url",
    "callback_url",
}

ACCEPTED_REQUIRED = {
    "accepted",
    "job_id",
    "server_job_id",
    "video_upload_id",
    "status",
    "stage",
    "engine",
}

CALLBACK_REQUIRED = {
    "job_id",
    "server_job_id",
    "video_upload_id",
    "engine",
    "status",
    "analysis_mode",
    "real_pose_detected",
    "findings",
    "overall_score",
    "phase_breakdown",
    "quality_flags",
    "recommended_next_action",
}

JOB_REQUIRED = {
    "job_id",
    "server_job_id",
    "video_upload_id",
    "status",
    "stage",
    "progress_percent",
    "message",
}

UNSAFE_PATTERNS = [
    re.compile(r"token=", re.IGNORECASE),
    re.compile(r"access_token", re.IGNORECASE),
    re.compile(r"ai_webhook_secret", re.IGNORECASE),
    re.compile(r"service_role", re.IGNORECASE),
    re.compile(r"supabase\.co/storage", re.IGNORECASE),
    re.compile(r"/Users/|/home/|/var/folders/|/tmp/", re.IGNORECASE),
    re.compile(r"[A-Z]:\\\\", re.IGNORECASE),
    re.compile(r"owen\s+weaver", re.IGNORECASE),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from walk_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from walk_strings(nested)


def assert_required(name: str, payload: dict, required: set[str]) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise AssertionError(f"{name} missing required fields: {', '.join(missing)}")


def assert_safe_fixture(name: str, payload: Any) -> None:
    for text in walk_strings(payload):
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(text):
                raise AssertionError(f"{name} contains unsafe fixture text: {text}")


def validate_request(payload: dict) -> None:
    assert_required("process_video_request", payload, REQUEST_REQUIRED)
    VideoProcessingRequest(**payload)
    signed_url = payload["signed_video_url"]
    if "redacted" not in signed_url or "signed-url-redacted.example" not in signed_url:
        raise AssertionError("process_video_request signed_video_url must be fake/redacted")


def validate_accepted(payload: dict) -> None:
    assert_required("process_video_accepted", payload, ACCEPTED_REQUIRED)
    if payload["accepted"] is not True:
        raise AssertionError("process_video_accepted accepted must be true")
    if payload["status"] != "queued" or payload["stage"] != "queued":
        raise AssertionError("process_video_accepted must start queued")


def validate_callback(name: str, payload: dict) -> None:
    assert_required(name, payload, CALLBACK_REQUIRED)
    if not isinstance(payload.get("findings"), list):
        raise AssertionError(f"{name} findings must be a list")
    if payload["analysis_mode"] == "manual_review":
        if payload["findings"]:
            raise AssertionError(f"{name} manual-review payload must not include findings")
        if payload.get("real_pose_detected") is not False:
            raise AssertionError(f"{name} manual-review payload must set real_pose_detected false")
        if payload.get("recommended_next_action") != "manual_review_recommended":
            raise AssertionError(f"{name} manual-review payload must recommend manual review")


def validate_job_status(payload: dict) -> None:
    assert_required("job_status", payload, JOB_REQUIRED)
    progress = payload.get("progress_percent")
    if not isinstance(progress, int) or progress < 0 or progress > 100:
        raise AssertionError("job_status progress_percent must be an integer from 0 to 100")


def main() -> int:
    found = {path.name for path in FIXTURES.glob("*.json")}
    missing = sorted(EXPECTED_FIXTURES - found)
    if missing:
        raise AssertionError(f"Missing fixture files: {', '.join(missing)}")

    for name in sorted(EXPECTED_FIXTURES):
        payload = load_json(FIXTURES / name)
        assert_safe_fixture(name, payload)

        if name == "process_video_request.example.json":
            validate_request(payload)
        elif name == "process_video_accepted.example.json":
            validate_accepted(payload)
        elif name.startswith("callback_"):
            validate_callback(name, payload)
        elif name == "job_status.example.json":
            validate_job_status(payload)

        print(f"valid fixture: {name}")

    print("fixture validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
