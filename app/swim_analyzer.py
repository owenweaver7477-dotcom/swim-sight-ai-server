import numpy as np
import logging
from typing import List, Dict, Any, Optional
from app.pose_estimator import get_midpoint, vertical_distance, horizontal_distance

logger = logging.getLogger(__name__)

AI_ENGINE_VERSION = "pose-mvp-0.1"
MODEL_NAME = "mediapipe_pose"

MIN_DETECTION_RATIO = 0.30
MIN_FRAMES_FOR_ANALYSIS = 10


def normalize_stroke_type(stroke_type: str) -> str:
    value = (stroke_type or "").strip().lower()

    mapping = {
        "free": "Freestyle",
        "freestyle": "Freestyle",
        "front crawl": "Freestyle",
        "breast": "Breaststroke",
        "breaststroke": "Breaststroke",
        "back": "Backstroke",
        "backstroke": "Backstroke",
        "fly": "Butterfly",
        "butterfly": "Butterfly",
        "im": "IM",
        "individual medley": "IM",
    }

    return mapping.get(value, "Freestyle")


def analyze_pose_data(
    pose_results: List[Dict],
    frames: list,
    fps: float,
    total_duration: float,
    stroke_type: str,
    camera_angle: str,
    video_upload_id: str,
) -> Dict[str, Any]:

    normalized_stroke = normalize_stroke_type(stroke_type)

    total_sampled = len(pose_results)
    detected_frames = [r for r in pose_results if r["pose_detected"]]
    detection_count = len(detected_frames)
    detection_ratio = detection_count / total_sampled if total_sampled > 0 else 0.0

    avg_keypoints = (
        float(np.mean([r["keypoint_count"] for r in detected_frames]))
        if detected_frames
        else 0.0
    )

    base = {
        "video_upload_id": video_upload_id,
        "status": "completed",
        "ai_engine_version": AI_ENGINE_VERSION,
        "model_name": MODEL_NAME,
        "frame_count_processed": total_sampled,
        "detected_keypoints_count": round(avg_keypoints, 1),
        "processed_video_url": "",
        "pose_data_file_url": "",
        "phase_breakdown": {},
        "findings": [],
        "key_frames": [],
        "error_message": None,
    }

    if detection_count < MIN_FRAMES_FOR_ANALYSIS or detection_ratio < MIN_DETECTION_RATIO:
        logger.warning(
            f"[{video_upload_id}] Pose detection too weak: "
            f"{detection_count}/{total_sampled} frames ({detection_ratio:.0%})"
        )

        return {
            **base,
            "analysis_mode": "placeholder",
            "real_pose_detected": False,
            "overall_score": None,
            "technical_summary": (
                f"Pose detection was not reliable enough for real pose analysis. "
                f"Only {detection_count} of {total_sampled} sampled frames had usable pose data "
                f"({detection_ratio:.0%} detection rate). Coach manual review is required."
            ),
            "error_message": (
                f"Pose detection rate was {detection_ratio:.0%} "
                f"({detection_count}/{total_sampled} sampled frames)."
            ),
        }

    fps_for_timestamps = fps if fps > 0 else 30.0

    findings = []
    key_frames = []

    hip_finding, hip_keyframes = _check_hip_drop(detected_frames, fps_for_timestamps)
    if hip_finding:
        findings.append(hip_finding)
    key_frames.extend(hip_keyframes)

    head_finding = _check_head_instability(
        detected_frames,
        fps_for_timestamps,
        normalized_stroke,
    )
    if head_finding:
        findings.append(head_finding)

    if normalized_stroke == "Breaststroke":
        kick_finding = _check_breaststroke_kick_width(detected_frames, fps_for_timestamps)
        if kick_finding:
            findings.append(kick_finding)

    rhythm_finding = _check_stroke_rhythm(
        detected_frames,
        fps_for_timestamps,
        normalized_stroke,
    )
    if rhythm_finding:
        findings.append(rhythm_finding)

    key_frames = _generate_structural_keyframes(
        detected_frames,
        fps_for_timestamps,
        normalized_stroke,
    ) + key_frames

    overall_score = _calculate_score(findings)

    technical_summary = _build_technical_summary(
        total_sampled=total_sampled,
        detection_count=detection_count,
        detection_ratio=detection_ratio,
        findings=findings,
        stroke_type=normalized_stroke,
    )

    phase_breakdown = _build_phase_breakdown(findings)

    return {
        **base,
        "analysis_mode": "real_pose",
        "real_pose_detected": True,
        "overall_score": overall_score,
        "technical_summary": technical_summary,
        "phase_breakdown": phase_breakdown,
        "findings": findings,
        "key_frames": key_frames,
    }


def _check_hip_drop(detected_frames: List[Dict], fps: float):
    offsets = []
    frame_timestamps = []

    for frame in detected_frames:
        lm = frame["landmarks"]

        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(k in lm for k in required):
            continue

        shoulder_mid = get_midpoint(lm["left_shoulder"], lm["right_shoulder"])
        hip_mid = get_midpoint(lm["left_hip"], lm["right_hip"])

        raw_offset = vertical_distance(shoulder_mid, hip_mid)
        torso_length = abs(raw_offset) if abs(raw_offset) > 0.01 else 0.30
        normalized = raw_offset / torso_length if torso_length > 0 else 0

        offsets.append(normalized)
        frame_timestamps.append(frame["frame_idx"] / fps)

    if len(offsets) < 8:
        return None, []

    mean_offset = float(np.mean(offsets))
    std_offset = float(np.std(offsets))

    HIP_DROP_THRESHOLD = 1.55

    if mean_offset < HIP_DROP_THRESHOLD:
        return None, []

    worst_indices = np.argsort(offsets)[-5:]
    worst_timestamps = sorted([frame_timestamps[i] for i in worst_indices])

    t_start = round(worst_timestamps[0], 2) if worst_timestamps else 0.0
    t_end = round(worst_timestamps[-1], 2) if worst_timestamps else 0.0

    confidence = min(0.85, 0.50 + (len(offsets) / 60) * 0.35)

    finding = {
        "finding_title": "Possible Hip Drop / Body Line Loss",
        "finding_description": (
            f"Hip position appears consistently lower than shoulder line across "
            f"{len(offsets)} analyzed frames. Average normalized hip offset was "
            f"{mean_offset:.2f}."
        ),
        "why_it_matters": (
            "A dropped hip position increases drag and reduces efficiency. "
            "A flatter body line helps the swimmer carry speed with less resistance."
        ),
        "recommended_correction": (
            "Focus on core engagement and a longer body line. Cue: press the chest slightly down "
            "to help the hips ride higher."
        ),
        "drill": "Kicking on back drill, Superman streamline kick, body-line kick",
        "next_focus": "Body line and core engagement",
        "stroke_phase": "Body Line",
        "severity": "High" if mean_offset > 1.8 else "Medium",
        "timestamp_start": t_start,
        "timestamp_end": t_end,
        "confidence_score": round(confidence, 2),
        "evidence_type": "pose_measurement",
        "measurement_summary": (
            f"Mean normalized hip-shoulder vertical offset: {mean_offset:.2f}; "
            f"std: {std_offset:.2f}; frames used: {len(offsets)}"
        ),
        "frame_reference": "",
    }

    keyframes = []
    if worst_timestamps:
        keyframes.append({
            "timestamp": round(worst_timestamps[len(worst_timestamps) // 2], 2),
            "label": "Hip Drop Check",
            "description": f"Frame range with highest detected hip drop pattern.",
        })

    return finding, keyframes


def _check_head_instability(detected_frames: List[Dict], fps: float, stroke_type: str):
    relative_positions = []

    for frame in detected_frames:
        lm = frame["landmarks"]

        if not all(k in lm for k in ["left_shoulder", "right_shoulder"]):
            continue

        shoulder_mid = get_midpoint(lm["left_shoulder"], lm["right_shoulder"])

        if "nose" in lm:
            head_point = lm["nose"]
        elif "left_ear" in lm and "right_ear" in lm:
            head_point = get_midpoint(lm["left_ear"], lm["right_ear"])
        else:
            continue

        relative_positions.append((
            head_point["x"] - shoulder_mid["x"],
            head_point["y"] - shoulder_mid["y"],
        ))

    if len(relative_positions) < 10:
        return None

    xs = [p[0] for p in relative_positions]
    ys = [p[1] for p in relative_positions]

    x_std = float(np.std(xs))
    y_std = float(np.std(ys))
    combined_instability = float(np.sqrt(x_std ** 2 + y_std ** 2))

    threshold = 0.12 if stroke_type == "Breaststroke" else 0.065

    if combined_instability < threshold:
        return None

    confidence = min(0.80, 0.45 + (len(relative_positions) / 60) * 0.35)

    return {
        "finding_title": "Unstable Head Position",
        "finding_description": (
            f"Head position relative to the shoulder line moved noticeably across "
            f"{len(relative_positions)} frames. Combined instability score: "
            f"{combined_instability:.3f}."
        ),
        "why_it_matters": (
            "An unstable head can disturb body line, breathing rhythm, and streamline."
        ),
        "recommended_correction": (
            "Keep the head aligned with the spine. Cue: steady head, quiet eyes, long neck."
        ),
        "drill": "Head-lead body line drill, controlled breathing drill",
        "next_focus": "Head alignment and breathing control",
        "stroke_phase": "Head Position",
        "severity": "Medium" if combined_instability < 0.10 else "High",
        "timestamp_start": 0.0,
        "timestamp_end": 0.0,
        "confidence_score": round(confidence, 2),
        "evidence_type": "pose_measurement",
        "measurement_summary": (
            f"Head movement relative to shoulders: combined std={combined_instability:.3f}; "
            f"x std={x_std:.3f}; y std={y_std:.3f}; frames used={len(relative_positions)}"
        ),
        "frame_reference": "",
    }


def _check_breaststroke_kick_width(detected_frames: List[Dict], fps: float):
    ankle_ratios = []
    frame_timestamps = []

    for frame in detected_frames:
        lm = frame["landmarks"]

        required = ["left_hip", "right_hip", "left_ankle", "right_ankle"]
        if not all(k in lm for k in required):
            continue

        hip_width = horizontal_distance(lm["left_hip"], lm["right_hip"])
        ankle_sep = horizontal_distance(lm["left_ankle"], lm["right_ankle"])

        if hip_width < 0.01:
            continue

        ankle_ratios.append(ankle_sep / hip_width)
        frame_timestamps.append(frame["frame_idx"] / fps)

    if len(ankle_ratios) < 6:
        return None

    max_ratio = float(np.max(ankle_ratios))
    mean_ratio = float(np.mean(ankle_ratios))

    KICK_WIDTH_THRESHOLD = 2.5

    if max_ratio < KICK_WIDTH_THRESHOLD:
        return None

    confidence = min(0.80, 0.45 + (len(ankle_ratios) / 40) * 0.35)

    worst_idx = int(np.argmax(ankle_ratios))
    t_worst = round(frame_timestamps[worst_idx], 2) if frame_timestamps else 0.0

    return {
        "finding_title": "Wide Breaststroke Kick Pattern",
        "finding_description": (
            f"Ankle separation during kick was up to {max_ratio:.1f}x hip width across "
            f"{len(ankle_ratios)} frames."
        ),
        "why_it_matters": (
            "An excessively wide breaststroke kick can increase frontal drag and reduce efficiency."
        ),
        "recommended_correction": (
            "Keep knees controlled during recovery. Cue: heels up, knees narrow, late foot turn."
        ),
        "drill": "Narrow-knee breaststroke kick on back, hands-at-side breaststroke kick",
        "next_focus": "Knee and ankle tracking during kick recovery",
        "stroke_phase": "Kick Recovery",
        "severity": "Medium" if max_ratio < 3.5 else "High",
        "timestamp_start": max(0.0, t_worst - 0.5),
        "timestamp_end": round(t_worst + 0.5, 2),
        "confidence_score": round(confidence, 2),
        "evidence_type": "pose_measurement",
        "measurement_summary": (
            f"Max ankle/hip-width ratio: {max_ratio:.2f}; mean: {mean_ratio:.2f}; "
            f"frames used={len(ankle_ratios)}"
        ),
        "frame_reference": "",
    }


def _check_stroke_rhythm(detected_frames: List[Dict], fps: float, stroke_type: str):
    wrist_positions = []

    for frame in detected_frames:
        lm = frame["landmarks"]

        if "left_wrist" in lm and "right_wrist" in lm:
            avg_wrist_y = (lm["left_wrist"]["y"] + lm["right_wrist"]["y"]) / 2
            wrist_positions.append((frame["frame_idx"], avg_wrist_y))
        elif "left_wrist" in lm:
            wrist_positions.append((frame["frame_idx"], lm["left_wrist"]["y"]))
        elif "right_wrist" in lm:
            wrist_positions.append((frame["frame_idx"], lm["right_wrist"]["y"]))

    if len(wrist_positions) < 20:
        return None

    positions = np.array([p[1] for p in wrist_positions])
    mean_pos = np.mean(positions)

    peaks = []

    for i in range(1, len(positions) - 1):
        if positions[i] < mean_pos and positions[i] < positions[i - 1] and positions[i] < positions[i + 1]:
            peaks.append(wrist_positions[i][0])

    if len(peaks) < 3:
        return None

    intervals = [(peaks[i + 1] - peaks[i]) / fps for i in range(len(peaks) - 1)]

    interval_std = float(np.std(intervals))
    interval_mean = float(np.mean(intervals))
    cv = interval_std / interval_mean if interval_mean > 0 else 0

    if cv < 0.25:
        return None

    confidence = min(0.60, 0.30 + (len(peaks) / 10) * 0.15)

    return {
        "finding_title": "Inconsistent Stroke Rhythm",
        "finding_description": (
            f"Wrist movement peaks showed timing variation across {len(peaks)} detected cycles. "
            f"Mean interval: {interval_mean:.2f}s; variability coefficient: {cv:.2f}."
        ),
        "why_it_matters": (
            "Consistent rhythm helps maintain propulsion and reduces speed loss between strokes."
        ),
        "recommended_correction": (
            "Use controlled tempo work. Cue: hold the same rhythm even as fatigue builds."
        ),
        "drill": "Tempo trainer sets, catch-up drill with deliberate timing",
        "next_focus": "Stroke rate consistency",
        "stroke_phase": "Timing",
        "severity": "Low" if cv < 0.40 else "Medium",
        "timestamp_start": 0.0,
        "timestamp_end": 0.0,
        "confidence_score": round(confidence, 2),
        "evidence_type": "video_observation",
        "measurement_summary": (
            f"Wrist cycle variability: mean interval={interval_mean:.2f}s; "
            f"std={interval_std:.2f}s; CV={cv:.2f}; peaks={len(peaks)}"
        ),
        "frame_reference": "",
    }


def _generate_structural_keyframes(detected_frames: List[Dict], fps: float, stroke_type: str) -> List[Dict]:
    if not detected_frames:
        return []

    step = max(1, len(detected_frames) // 6)
    selected = detected_frames[::step][:6]

    labels = ["Start", "Mid-Stroke 1", "Mid-Stroke 2", "Mid-Stroke 3", "Mid-Stroke 4", "End"]

    keyframes = []
    for i, frame in enumerate(selected):
        keyframes.append({
            "timestamp": round(frame["frame_idx"] / fps, 2),
            "label": labels[i] if i < len(labels) else f"Frame {i + 1}",
            "description": f"Structural reference — {frame['keypoint_count']} keypoints detected",
        })

    return keyframes


def _calculate_score(findings: List[Dict]) -> Optional[int]:
    if not findings:
        return 82

    score = 85
    penalties = {
        "Critical": 12,
        "High": 8,
        "Medium": 5,
        "Low": 2,
    }

    for finding in findings:
        score -= penalties.get(finding.get("severity", "Medium"), 4)

    return max(50, min(95, score))


def _build_technical_summary(
    total_sampled: int,
    detection_count: int,
    detection_ratio: float,
    findings: List[Dict],
    stroke_type: str,
) -> str:
    if not findings:
        return (
            f"Pose-assisted analysis processed {total_sampled} sampled frames and detected usable "
            f"body landmarks in {detection_count} frames ({detection_ratio:.0%} detection rate). "
            f"No strong pose-based issues were detected in the sampled frames. Coach review is still required."
        )

    finding_titles = "; ".join([f["finding_title"] for f in findings])

    return (
        f"Pose-assisted analysis processed {total_sampled} sampled frames and detected usable body "
        f"landmarks in {detection_count} frames ({detection_ratio:.0%} detection rate). "
        f"The following areas were flagged from pose evidence: {finding_titles}. "
        f"All findings are estimates and require coach review before sharing."
    )


def _build_phase_breakdown(findings: List[Dict]) -> Dict[str, int]:
    phases = {
        "Body Line": 80,
        "Head Position": 80,
        "Timing": 80,
        "Kick": 80,
        "Arm Pull": 80,
    }

    severity_impact = {
        "Critical": 25,
        "High": 18,
        "Medium": 10,
        "Low": 5,
    }

    for finding in findings:
        phase = finding.get("stroke_phase", "")
        severity = finding.get("severity", "Medium")
        impact = severity_impact.get(severity, 8)

        if phase in phases:
            phases[phase] = max(40, phases[phase] - impact)
        elif "Kick" in phase:
            phases["Kick"] = max(40, phases["Kick"] - impact)
        elif "Body" in phase or "Hip" in phase:
            phases["Body Line"] = max(40, phases["Body Line"] - impact)

    return phases
