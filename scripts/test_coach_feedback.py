#!/usr/bin/env python3
"""Synthetic privacy and metric checks for coach feedback evaluation."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.coach_feedback import (  # noqa: E402
    FeedbackValidationError,
    append_feedback_record,
    normalize_coach_feedback,
    validate_feedback_privacy,
)
from app.feedback_metrics import (  # noqa: E402
    compute_feedback_metrics,
    load_feedback_jsonl,
    summarise_feedback_records,
)


def run() -> None:
    fixture_dir = ROOT / "fixtures" / "coach_feedback"
    payload = json.loads(
        (fixture_dir / "approved_rejected_edited.example.json").read_text(encoding="utf-8")
    )
    record = normalize_coach_feedback(payload)
    assert record["review_status"] == "reviewed"
    assert len(record["items"]) == 5

    by_id = {item["finding_id"]: item for item in record["items"]}
    assert by_id["body_line_001"]["coach_decision"] == "approved"
    assert by_id["body_line_001"]["decision_type"] == "approved"
    assert "coach_note_summary" not in by_id["body_line_001"]
    assert by_id["head_lift_002"]["coach_decision"] == "rejected"
    assert by_id["kick_width_003"]["decision_type"] == "severity_edited"
    assert by_id["timing_004"]["coach_decision"] == "unsafe_to_use"
    assert by_id["line_reset_005"]["coach_decision"] == "needs_more_context"
    assert by_id["body_line_001"]["evidence_frame_count"] == 3
    assert "evidence_frames" not in by_id["body_line_001"]

    unknown = copy.deepcopy(payload)
    unknown["coach_decisions"][0]["decision"] = "auto_accept"
    try:
        normalize_coach_feedback(unknown)
    except FeedbackValidationError as exc:
        assert "Unknown coach decision" in str(exc)
    else:
        raise AssertionError("Unknown decision must fail clearly")

    unsafe_payload = json.loads(
        (fixture_dir / "unsafe_fields.example.json").read_text(encoding="utf-8")
    )
    stripped = normalize_coach_feedback(unsafe_payload)
    stripped_blob = json.dumps(stripped)
    for forbidden in (
        "signed_video_url",
        "height_cm",
        "mass_kg",
        "swimmer_name",
        "video_upload_id",
        "review_id",
    ):
        assert forbidden not in stripped_blob
    assert "REDACTED_UNSAFE_FIELD" not in stripped_blob
    assert all("raw_landmarks" not in item for item in stripped["items"])
    assert stripped["privacy"]["stripped_unsafe_field_count"] >= 7
    privacy_valid, privacy_issues = validate_feedback_privacy(stripped)
    assert privacy_valid, privacy_issues

    synthetic_url_payload = copy.deepcopy(payload)
    synthetic_url_payload["signed_video_url"] = (
        "https://" + "redacted.invalid/private?token=synthetic-test-only"
    )
    url_stripped = normalize_coach_feedback(synthetic_url_payload)
    assert "redacted.invalid" not in json.dumps(url_stripped)

    metrics = compute_feedback_metrics(record)
    assert metrics["total_findings"] == 5
    assert metrics["approved_count"] == 1
    assert metrics["rejected_count"] == 1
    assert metrics["edited_count"] == 1
    assert metrics["unsafe_count"] == 1
    assert metrics["needs_more_context_count"] == 1
    assert metrics["approval_rate"] == 0.2
    assert metrics["rejection_rate"] == 0.2
    assert metrics["edited_rate"] == 0.2
    assert metrics["severity_agreement_rate"] == 0.6667
    assert metrics["phase_context_available_rate"] == 0.8
    assert metrics["high_confidence_rejection_rate"] == 0.5

    aggregate = summarise_feedback_records([record])
    assert aggregate["most_rejected_findings"][0]["finding"] == "head_lift"
    assert aggregate["high_confidence_rejections"][0]["rejection_rate"] == 1.0
    assert aggregate["edits_by_stroke_camera"] == {"breaststroke | Side": 1}

    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "feedback.jsonl"
        written = append_feedback_record(record, output)
        assert written == output
        loaded = load_feedback_jsonl(output)
        assert loaded == [record]

        invalid = copy.deepcopy(record)
        invalid["signed_video_url"] = "REDACTED_UNSAFE_FIELD"
        try:
            append_feedback_record(invalid, output)
        except FeedbackValidationError as exc:
            assert "Refusing to write privacy-invalid" in str(exc)
        else:
            raise AssertionError("Writer must reject privacy-invalid records")

    summary_cli = subprocess.run(
        [
            sys.executable,
            "scripts/summarise_coach_feedback.py",
            "--input",
            str(fixture_dir / "sample_feedback.jsonl"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert summary_cli.returncode == 0, summary_cli.stderr
    assert "INTERNAL EVALUATION DATA ONLY" in summary_cli.stdout
    assert '"approval_rate": 0.2' in summary_cli.stdout

    ingest_cli = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_coach_feedback.py",
            "--input",
            str(fixture_dir / "approved_rejected_edited.example.json"),
            "--dry-run",
            "--summary",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert ingest_cli.returncode == 0, ingest_cli.stderr
    assert "Privacy validation: PASS" in ingest_cli.stdout
    assert "Dry run: nothing written" in ingest_cli.stdout

    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app.coach_feedback, app.feedback_metrics; "
                "assert 'mediapipe' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules; print('pure feedback imports ok')"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "coach_feedback_exports/" in gitignore
    print("Coach feedback checks passed.")


if __name__ == "__main__":
    run()
