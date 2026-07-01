"""Compare default-off AI worker upgrades against local sample clips."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}

FLAG_NAMES = (
    "ENABLE_CLAHE",
    "ENABLE_POSE_SMOOTHING",
    "POSE_MODEL_COMPLEXITY",
    "ROBUST_FINDINGS",
    "SEQUENTIAL_FRAME_READ",
    "ENABLE_ESTIMATED_DRAG",
    "PHASE_ANALYSIS",
    "EXTENDED_STROKE_FINDINGS",
)


def flag_set(**overrides: str) -> Dict[str, str]:
    flags = {
        "ENABLE_CLAHE": "false",
        "ENABLE_POSE_SMOOTHING": "false",
        "POSE_MODEL_COMPLEXITY": "0",
        "ROBUST_FINDINGS": "false",
        "SEQUENTIAL_FRAME_READ": "false",
        "ENABLE_ESTIMATED_DRAG": "false",
        "PHASE_ANALYSIS": "false",
        "EXTENDED_STROKE_FINDINGS": "false",
    }
    flags.update(overrides)
    return flags


VARIANTS = (
    ("baseline", flag_set()),
    ("clahe", flag_set(ENABLE_CLAHE="true")),
    ("pose_smoothing", flag_set(ENABLE_POSE_SMOOTHING="true")),
    ("model_complexity_1", flag_set(POSE_MODEL_COMPLEXITY="1")),
    ("robust_findings", flag_set(ROBUST_FINDINGS="true")),
    ("phase_analysis", flag_set(PHASE_ANALYSIS="true")),
    ("extended_stroke_findings", flag_set(EXTENDED_STROKE_FINDINGS="true")),
    ("sequential_frame_read", flag_set(SEQUENTIAL_FRAME_READ="true")),
    (
        "all_safe_except_estimated_drag",
        flag_set(
            ENABLE_CLAHE="true",
            ENABLE_POSE_SMOOTHING="true",
            POSE_MODEL_COMPLEXITY="1",
            ROBUST_FINDINGS="true",
            SEQUENTIAL_FRAME_READ="true",
        ),
    ),
)


def find_videos(samples_dir: Path) -> List[Path]:
    if not samples_dir.exists():
        return []
    return sorted(
        path
        for path in samples_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def child_result(video_path: Path, stroke: str, camera_angle: str) -> int:
    # Imported only after the child process has received its isolated env flags.
    from scripts.evaluate_baseline import evaluate_video

    result = evaluate_video(video_path, stroke, camera_angle)
    print(json.dumps(result))
    return 0


def run_variant(
    video_path: Path,
    stroke: str,
    camera_angle: str,
    variant_name: str,
    flags: Dict[str, str],
) -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(flags)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        "--video-path",
        str(video_path),
        "--stroke",
        stroke,
        "--camera-angle",
        camera_angle,
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"Comparison failed for {video_path.name} with variant {variant_name}"
        )

    try:
        raw = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(
            f"Comparison returned invalid JSON for {video_path.name} ({variant_name})"
        ) from error

    return {
        "clip_name": raw.get("video", video_path.name),
        "stroke": raw.get("stroke", stroke),
        "camera_angle": raw.get("camera_angle", camera_angle),
        "variant": variant_name,
        "flags_used": {name: flags[name] for name in FLAG_NAMES},
        "processing_seconds": raw.get("processing_seconds"),
        "frames_sampled": raw.get("frames_sampled"),
        "frames_with_pose": raw.get("frames_with_pose"),
        "pose_detection_rate": raw.get("pose_detection_rate"),
        "average_keypoints": raw.get("detected_keypoints_average"),
        "fallback_triggered": raw.get("fallback_triggered"),
        "finding_count": raw.get("finding_count"),
        "finding_fault_tags": raw.get("finding_fault_tags") or [],
        "overall_score": raw.get("overall_score"),
        "stroke_cycles": raw.get("stroke_cycles"),
        "quality_flags": raw.get("quality_flags") or [],
        "processing_tier": raw.get("processing_tier"),
        "analysis_mode": raw.get("analysis_mode"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare default-off AI worker upgrades using local sample clips."
    )
    parser.add_argument("--samples-dir", default=str(ROOT / "samples" / "videos"))
    parser.add_argument("--output-dir", default=str(ROOT / "baseline_reports"))
    parser.add_argument("--stroke", default="Freestyle")
    parser.add_argument("--camera-angle", default="Side")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--video-path", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_one:
        if not args.video_path:
            parser.error("--video-path is required with --run-one")
        return child_result(Path(args.video_path), args.stroke, args.camera_angle)

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    videos = find_videos(samples_dir)
    if not videos:
        print(f"No sample clips found in {samples_dir}.")
        print("Add local clips there, then rerun this command. Do not commit real swimmer footage.")
        return 0

    results: List[Dict[str, Any]] = []
    total_runs = len(videos) * len(VARIANTS)
    run_number = 0
    for video_path in videos:
        for variant_name, flags in VARIANTS:
            run_number += 1
            print(
                f"[{run_number}/{total_runs}] {video_path.name}: {variant_name}",
                flush=True,
            )
            results.append(
                run_variant(
                    video_path,
                    args.stroke,
                    args.camera_angle,
                    variant_name,
                    flags,
                )
            )

    created_at = datetime.now(timezone.utc)
    report = {
        "created_at": created_at.isoformat(),
        "samples_dir": str(samples_dir),
        "video_count": len(videos),
        "variant_count": len(VARIANTS),
        "estimated_drag_included": False,
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"upgrade_comparison_{created_at.strftime('%Y%m%d_%H%M%S')}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nCompleted {len(results)} comparison runs.")
    print(f"Comparison report written locally: {output_path}")
    print("ENABLE_ESTIMATED_DRAG remained false for every comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
