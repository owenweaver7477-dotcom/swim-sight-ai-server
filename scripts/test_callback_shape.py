"""Golden shape test for the real success callback structure.

This runs the ACTUAL worker analysis path (``app.swim_analyzer.analyze_pose_data``)
on synthetic in-memory pose results -- no video, no network, no secrets -- and
derives the authoritative key sets the worker emits for a real-pose result. It
then asserts ``fixtures/callback_success.example.json`` conforms to those exact
key sets, so any future drift between the emitted payload and the documented
fixture fails here.

It is intentionally read-only w.r.t. runtime code: it imports the analysis path
but changes nothing and asserts nothing about scoring/pose values -- only the
callback *shape*.

Run:  python scripts/test_callback_shape.py   (python == python3)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.swim_analyzer import analyze_pose_data  # noqa: E402


SUCCESS_FIXTURE = ROOT / "fixtures" / "callback_success.example.json"

# Top-level fields the app-facing callback contract requires to be present
# (kept in sync with scripts/validate_contract_fixtures.CALLBACK_REQUIRED).
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

# Field names that indicate the historical fixture drift and must never reappear.
FORBIDDEN_FINDING_KEYS = {"impact", "coach_cue"}
FORBIDDEN_PHASE_KEYS = {"score", "finding_count"}
FORBIDDEN_KEYFRAME_KEYS = {"frame_label"}


def _synthetic_pose_results(detected: int = 40, undetected: int = 14):
    """Freestyle body-line-loss trigger: hips sit well below the shoulder line.

    Only shoulders + hips are provided so exactly one deterministic finding
    (``body_line_loss``) is produced, giving a stable golden shape.
    """
    results = []
    idx = 0
    for _ in range(detected):
        results.append({
            "frame_idx": idx,
            "pose_detected": True,
            "keypoint_count": 12,
            "landmark_count_total": 15,
            "landmarks": {
                "left_shoulder": {"x": 0.40, "y": 0.42, "z": 0.0, "visibility": 0.9},
                "right_shoulder": {"x": 0.60, "y": 0.42, "z": 0.0, "visibility": 0.9},
                "left_hip": {"x": 0.44, "y": 0.80, "z": 0.0, "visibility": 0.9},
                "right_hip": {"x": 0.56, "y": 0.80, "z": 0.0, "visibility": 0.9},
            },
        })
        idx += 5
    for _ in range(undetected):
        results.append({
            "frame_idx": idx,
            "pose_detected": False,
            "keypoint_count": 0,
            "landmarks": {},
        })
        idx += 5
    return results


def _real_analysis():
    return analyze_pose_data(
        pose_results=_synthetic_pose_results(),
        frames=[],
        fps=30.0,
        total_duration=9.8,
        stroke_type="Freestyle",
        camera_angle="Side",
        video_upload_id="synthetic-video-id",
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_real_path_invariants(real: dict) -> None:
    _assert(real.get("analysis_mode") == "real_pose",
            f"analysis path did not produce real_pose: {real.get('analysis_mode')}")
    _assert(real.get("real_pose_detected") is True, "real_pose_detected must be True")
    _assert(isinstance(real.get("overall_score"), int), "overall_score must be an int on the real path")
    _assert(bool(real.get("findings")), "expected at least one draft finding from synthetic pose")
    for finding in real["findings"]:
        _assert(finding.get("coach_review_required") is True,
                "every finding must set coach_review_required True")


def check_fixture_conforms(real: dict, fixture: dict) -> None:
    # Top-level app contract present.
    missing = sorted(CALLBACK_REQUIRED - set(fixture))
    _assert(not missing, f"success fixture missing required top-level keys: {missing}")

    # Fixture must actually represent a real-pose success example.
    _assert(fixture.get("analysis_mode") == "real_pose", "success fixture analysis_mode must be real_pose")
    _assert(fixture.get("real_pose_detected") is True, "success fixture real_pose_detected must be true")
    _assert(bool(fixture.get("findings")), "success fixture must include at least one finding")

    # Finding shape must equal the worker's real emitted finding shape.
    real_finding_keys = set(real["findings"][0].keys())
    real_evidence_keys = set(real["findings"][0]["evidence"].keys())
    for i, finding in enumerate(fixture["findings"]):
        keys = set(finding)
        _assert(keys == real_finding_keys,
                f"finding[{i}] key mismatch.\n  missing: {sorted(real_finding_keys - keys)}\n  extra:   {sorted(keys - real_finding_keys)}")
        _assert(not (keys & FORBIDDEN_FINDING_KEYS),
                f"finding[{i}] contains forbidden drifted keys: {sorted(keys & FORBIDDEN_FINDING_KEYS)}")
        _assert("evidence" in finding and isinstance(finding["evidence"], dict),
                f"finding[{i}] must include an evidence object")
        ev_keys = set(finding["evidence"])
        _assert(ev_keys == real_evidence_keys,
                f"finding[{i}].evidence key mismatch.\n  missing: {sorted(real_evidence_keys - ev_keys)}\n  extra:   {sorted(ev_keys - real_evidence_keys)}")

    # phase_breakdown entries must match one of the real variants (flagged / not_flagged).
    real_phase_variants = {frozenset(v.keys()) for v in real["phase_breakdown"].values()}
    for phase, entry in fixture["phase_breakdown"].items():
        keys = set(entry)
        _assert(frozenset(keys) in real_phase_variants,
                f"phase_breakdown[{phase}] key set {sorted(keys)} is not a real variant {[sorted(v) for v in real_phase_variants]}")
        _assert(not (keys & FORBIDDEN_PHASE_KEYS),
                f"phase_breakdown[{phase}] contains forbidden drifted keys: {sorted(keys & FORBIDDEN_PHASE_KEYS)}")

    # key_frames shape must equal the worker's real key-frame shape.
    real_keyframe_keys = set(real["key_frames"][0].keys())
    for i, frame in enumerate(fixture.get("key_frames", [])):
        keys = set(frame)
        _assert(keys == real_keyframe_keys,
                f"key_frames[{i}] key mismatch.\n  missing: {sorted(real_keyframe_keys - keys)}\n  extra:   {sorted(keys - real_keyframe_keys)}")
        _assert(not (keys & FORBIDDEN_KEYFRAME_KEYS),
                f"key_frames[{i}] contains forbidden drifted keys: {sorted(keys & FORBIDDEN_KEYFRAME_KEYS)}")


def main() -> int:
    real = _real_analysis()
    check_real_path_invariants(real)

    with SUCCESS_FIXTURE.open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)

    check_fixture_conforms(real, fixture)

    print("callback shape golden test passed")
    print(f"  real finding keys locked: {len(set(real['findings'][0]))}")
    print(f"  real evidence keys locked: {len(set(real['findings'][0]['evidence']))}")
    print(f"  fixture findings checked: {len(fixture['findings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
