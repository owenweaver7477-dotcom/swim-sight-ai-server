import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.video_processor import classify_video_workload


def base_metadata(**overrides):
    metadata = {
        "file_size_mb": 18,
        "duration_seconds": 10,
        "fps": 30,
        "frame_count_total": 300,
        "source_width": 1280,
        "source_height": 720,
        "filename": "side-view.mp4",
        "capture_source": "standard_camera",
        "quality_flags": [],
    }
    metadata.update(overrides)
    return metadata


def assert_tier(name, metadata, expected):
    result = classify_video_workload(metadata)
    assert result["processing_tier"] == expected, (
        f"{name}: expected {expected}, got {result['processing_tier']} "
        f"flags={result.get('quality_flags')}"
    )
    return result


def test_normal_720p_short_clip():
    result = assert_tier("720p short", base_metadata(), "standard_ai")
    assert result["max_sampled_frames"] >= 45


def test_normal_1080p_short_clip():
    result = assert_tier(
        "1080p short",
        base_metadata(source_width=1920, source_height=1080, file_size_mb=55),
        "standard_ai",
    )
    assert result["max_processing_width"] == 640


def test_high_res_screen_recording_is_reduced_not_crash():
    result = classify_video_workload(
        base_metadata(
            file_size_mb=44,
            duration_seconds=14,
            fps=30,
            frame_count_total=420,
            source_width=2940,
            source_height=1912,
            filename="Screen Recording.mov",
            capture_source="screen_recording",
        )
    )
    assert result["processing_tier"] in {"reduced_ai", "minimal_ai"}
    assert result["processing_tier"] != "manual_review_required"
    assert "screen_recording_possible" in result["quality_flags"]


def test_unreadable_metadata_goes_manual_review():
    result = assert_tier(
        "unreadable",
        base_metadata(source_width=0, source_height=0, frame_count_total=0),
        "manual_review_required",
    )
    assert "metadata_unreadable" in result["quality_flags"]


def test_long_video_uses_sampled_window():
    result = classify_video_workload(
        base_metadata(duration_seconds=70, frame_count_total=2100, file_size_mb=90)
    )
    assert result["processing_tier"] in {"reduced_ai", "minimal_ai"}
    assert "sampled_processing_window" in result["quality_flags"]
    assert result["processing_window_seconds"] < 70


def test_high_fps_reduces_sampling():
    result = classify_video_workload(
        base_metadata(fps=100, frame_count_total=1000, duration_seconds=10)
    )
    assert result["processing_tier"] in {"reduced_ai", "minimal_ai"}
    assert "high_fps_video" in result["quality_flags"]


if __name__ == "__main__":
    test_normal_720p_short_clip()
    test_normal_1080p_short_clip()
    test_high_res_screen_recording_is_reduced_not_crash()
    test_unreadable_metadata_goes_manual_review()
    test_long_video_uses_sampled_window()
    test_high_fps_reduces_sampling()
    print("video workload classifier checks passed")
