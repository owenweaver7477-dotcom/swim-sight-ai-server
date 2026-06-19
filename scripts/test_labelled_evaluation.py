"""Unit tests for labelled clip comparison without real swimmer footage."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_labelled_clips import compare_labels  # noqa: E402


label = {
    "expected_analysis_mode": "real_pose",
    "expected_fault_tags": ["wide_breaststroke_kick"],
    "forbidden_fault_tags": ["head_lift"],
}
evaluation = {
    "analysis_mode": "real_pose",
    "findings": [{"fault_tag": "wide_breaststroke_kick"}],
}
result = compare_labels(label, evaluation)
assert result["passed"] is True
assert result["precision"] == 1.0
assert result["recall"] == 1.0

bad = compare_labels(
    label,
    {"analysis_mode": "manual_review", "findings": [{"fault_tag": "head_lift"}]},
)
assert bad["passed"] is False
assert bad["forbidden_fault_tags_found"] == ["head_lift"]
print("labelled evaluation tests passed")
