import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.swim_analyzer import analyze_pose_data


def landmark(x, y, visibility=0.9):
    return {"x": x, "y": y, "visibility": visibility}


def synthetic_frame(frame_idx):
    return {
        "frame_idx": frame_idx,
        "pose_detected": True,
        "keypoint_count": 13,
        "landmarks": {
            "nose": landmark(0.50, 0.24),
            "left_shoulder": landmark(0.40, 0.40),
            "right_shoulder": landmark(0.60, 0.40),
            "left_elbow": landmark(0.39, 0.48),
            "right_elbow": landmark(0.61, 0.48),
            "left_wrist": landmark(0.34, 0.44),
            "right_wrist": landmark(0.66, 0.44),
            "left_hip": landmark(0.45, 0.80),
            "right_hip": landmark(0.55, 0.80),
            "left_knee": landmark(0.28, 0.86),
            "right_knee": landmark(0.72, 0.86),
            "left_ankle": landmark(0.32, 0.94),
            "right_ankle": landmark(0.68, 0.94),
        },
    }


def assert_strong_breaststroke_analysis():
    pose_results = [synthetic_frame(i * 8) for i in range(14)]
    result = analyze_pose_data(
        pose_results=pose_results,
        frames=[],
        fps=24.0,
        total_duration=5.0,
        stroke_type="Breaststroke",
        camera_angle="Side",
        video_upload_id="sample-video",
    )

    findings = result["findings"]
    assert result["analysis_mode"] == "real_pose"
    assert result["real_pose_detected"] is True
    assert findings, "Expected strong synthetic pose to emit draft findings"

    for finding in findings:
        assert finding["source"] == "ai_pose"
        assert finding["coach_review_required"] is True
        assert finding["severity"] in ["High", "Medium"]
        assert finding["confidence"] in ["high", "medium"]
        assert finding["confidence_score"] >= 0.62
        assert finding["observation"]
        assert finding["correction_cue"]
        assert finding["drill"]
        assert finding["evidence"]["evidence_note"]
        assert "signed" not in str(finding).lower()
        assert "file_path" not in str(finding).lower()


def assert_weak_pose_suppresses_findings():
    pose_results = [synthetic_frame(i * 8) for i in range(4)]
    result = analyze_pose_data(
        pose_results=pose_results,
        frames=[],
        fps=24.0,
        total_duration=5.0,
        stroke_type="Freestyle",
        camera_angle="Side",
        video_upload_id="weak-video",
    )

    assert result["analysis_mode"] == "placeholder"
    assert result["real_pose_detected"] is False
    assert result["findings"] == []
    assert result["overall_score"] is None


if __name__ == "__main__":
    assert_strong_breaststroke_analysis()
    assert_weak_pose_suppresses_findings()
    print("sample analysis checks passed")
