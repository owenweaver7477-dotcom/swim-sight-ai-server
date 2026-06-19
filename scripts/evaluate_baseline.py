import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pose_estimator import run_pose_estimation  # noqa: E402
from app.pose_postprocess import pose_smoothing_enabled, smooth_pose_results  # noqa: E402
from app.swim_analyzer import AI_ENGINE_VERSION, analyze_pose_data  # noqa: E402
from app.video_processor import extract_frames  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


def find_videos(samples_dir: Path) -> List[Path]:
    if not samples_dir.exists():
        return []
    return sorted(
        path
        for path in samples_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def evaluate_video(video_path: Path, stroke: str, camera_angle: str) -> Dict[str, Any]:
    started = time.perf_counter()
    extraction = extract_frames(
        str(video_path),
        video_upload_id=f"baseline-{video_path.stem}",
        filename=video_path.name,
        capture_source="local_baseline_sample",
    )

    frames = extraction.frames
    metadata = extraction.metadata
    fallback_triggered = metadata.get("processing_tier") == "manual_review_required" or not frames

    pose_results = []
    analysis = {
        "analysis_mode": "manual_review",
        "real_pose_detected": False,
        "findings": [],
        "overall_score": None,
        "phase_breakdown": {},
    }

    if frames:
        pose_results = run_pose_estimation(frames)
        if pose_smoothing_enabled():
            pose_results = smooth_pose_results(pose_results)
        analysis = analyze_pose_data(
            pose_results=pose_results,
            frames=frames,
            fps=float(metadata.get("fps") or 30.0),
            total_duration=float(metadata.get("duration_seconds") or 0.0),
            stroke_type=stroke,
            camera_angle=camera_angle,
            video_upload_id=f"baseline-{video_path.stem}",
        )

    frames_sampled = len(frames)
    frames_with_pose = sum(1 for result in pose_results if result.get("pose_detected"))
    pose_detection_rate = (
        round(frames_with_pose / frames_sampled, 4)
        if frames_sampled
        else 0.0
    )

    if not analysis.get("real_pose_detected"):
        fallback_triggered = True

    temporal_metrics = analysis.get("temporal_metrics") or {}
    quality_flags = list(dict.fromkeys([
        *(metadata.get("quality_flags") or []),
        *(temporal_metrics.get("quality_flags") or []),
    ]))

    return {
        "engine": AI_ENGINE_VERSION,
        "video": video_path.name,
        "stroke": stroke,
        "camera_angle": camera_angle,
        "duration_seconds": metadata.get("duration_seconds"),
        "fps": metadata.get("fps"),
        "source_width": metadata.get("source_width"),
        "source_height": metadata.get("source_height"),
        "processing_tier": metadata.get("processing_tier"),
        "processing_window_seconds": metadata.get("processing_window_seconds"),
        "frames_sampled": frames_sampled,
        "requested_frame_count": metadata.get("requested_frame_count"),
        "frames_with_pose": frames_with_pose,
        "pose_detection_rate": pose_detection_rate,
        "detected_keypoints_average": (
            round(
                sum(result.get("keypoint_count", 0) for result in pose_results) / len(pose_results),
                2,
            )
            if pose_results
            else 0
        ),
        "visible_landmarks_average": (
            round(
                sum(
                    result.get("landmark_count_total", result.get("keypoint_count", 0))
                    for result in pose_results
                ) / len(pose_results),
                2,
            )
            if pose_results
            else 0
        ),
        "analysis_mode": analysis.get("analysis_mode"),
        "real_pose_detected": bool(analysis.get("real_pose_detected")),
        "finding_count": len(analysis.get("findings") or []),
        "finding_fault_tags": sorted({
            finding.get("fault_tag")
            for finding in (analysis.get("findings") or [])
            if finding.get("fault_tag")
        }),
        "overall_score": analysis.get("overall_score"),
        "fallback_triggered": fallback_triggered,
        "quality_flags": quality_flags,
        "temporal_metrics": temporal_metrics,
        "processing_seconds": round(time.perf_counter() - started, 2),
        "notes": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local baseline evaluation on uncommitted sample swim clips."
    )
    parser.add_argument("--samples-dir", default=str(ROOT / "samples" / "videos"))
    parser.add_argument("--output-dir", default=str(ROOT / "baseline_reports"))
    parser.add_argument("--stroke", default="Freestyle")
    parser.add_argument("--camera-angle", default="Side")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    videos = find_videos(samples_dir)

    if not videos:
        print(
            f"No sample clips found. Add local clips to {samples_dir} to run baseline evaluation."
        )
        print("Do not commit real swimmer footage.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "engine": AI_ENGINE_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "samples_dir": str(samples_dir),
        "video_count": len(videos),
        "results": [
            evaluate_video(video_path, args.stroke, args.camera_angle)
            for video_path in videos
        ],
    }

    output_path = output_dir / f"baseline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Baseline report written locally: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
