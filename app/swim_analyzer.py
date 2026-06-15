import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.pose_estimator import get_midpoint, horizontal_distance, vertical_distance

logger = logging.getLogger(__name__)

AI_ENGINE_VERSION = "pose-mvp-0.5"
MODEL_NAME = "mediapipe_pose"

MIN_DETECTION_RATIO = 0.30
MIN_FRAMES_FOR_ANALYSIS = 10
MIN_FINDING_CONFIDENCE = 0.62
MAX_FINDINGS = 5


PHASE_SEQUENCES = {
    "Breaststroke": [
        "streamline",
        "pull",
        "breath",
        "recovery",
        "kick_setup",
        "kick_drive",
        "line_reset",
    ],
    "Freestyle": [
        "entry_extension",
        "catch_setup",
        "pull",
        "breathing",
        "recovery",
        "body_line",
    ],
    "Backstroke": [
        "entry_extension",
        "catch_setup",
        "pull",
        "recovery",
        "body_line",
    ],
    "Butterfly": [
        "entry_extension",
        "catch_setup",
        "pull",
        "breathing",
        "recovery",
        "body_wave",
    ],
}


DRILL_CUE_MAP = {
    "body_line_loss": {
        "cue": "Hold a long line from head through hips before adding speed.",
        "drill": "Side-line kick or streamline kick with quiet head",
        "next_focus": "Body line before propulsion",
    },
    "head_lift": {
        "cue": "Breathe without lifting the chin away from the line.",
        "drill": "Breath timing drill with eyes down and long neck",
        "next_focus": "Head position through the breath",
    },
    "wide_breaststroke_kick": {
        "cue": "Recover the heels with knees controlled, then turn the feet late.",
        "drill": "Narrow-knee breaststroke kick on back",
        "next_focus": "Kick recovery width",
    },
    "breaststroke_timing_review": {
        "cue": "Finish the pull, breathe, then let the kick drive the line forward.",
        "drill": "Pull-breathe-kick-glide timing drill",
        "next_focus": "Pull-breath-kick sequence",
    },
    "breaststroke_line_reset": {
        "cue": "Reset to a narrow streamline before starting the next stroke.",
        "drill": "One-pull one-kick glide with a counted line reset",
        "next_focus": "Line reset after the kick",
    },
    "dropped_elbow_catch": {
        "cue": "Set the forearm early and keep pressure on the water.",
        "drill": "Scull to early vertical forearm progression",
        "next_focus": "Catch setup",
    },
    "short_entry_extension": {
        "cue": "Enter and extend forward before pressing into the catch.",
        "drill": "Single-arm freestyle with front-quadrant pause",
        "next_focus": "Entry and extension",
    },
    "breathing_line_break": {
        "cue": "Rotate to breathe without lifting the head out of the body line.",
        "drill": "Six-kick switch with low breath",
        "next_focus": "Breathing control",
    },
    "backstroke_hip_sink": {
        "cue": "Keep ribs and hips connected so the kick supports the body line.",
        "drill": "Backstroke streamline kick with hips high",
        "next_focus": "Backstroke hip position",
    },
    "butterfly_rhythm_break": {
        "cue": "Let the body wave set the timing before forcing the arms.",
        "drill": "Body dolphin into single-stroke butterfly",
        "next_focus": "Rhythm and body wave",
    },
}


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
    detected_frames = [r for r in pose_results if r.get("pose_detected")]
    detection_count = len(detected_frames)
    detection_ratio = detection_count / total_sampled if total_sampled > 0 else 0.0

    avg_keypoints = (
        float(np.mean([r.get("keypoint_count", 0) for r in detected_frames]))
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
            "[%s] Pose detection too weak: %s/%s frames (%.0f%%)",
            video_upload_id,
            detection_count,
            total_sampled,
            detection_ratio * 100,
        )

        return {
            **base,
            "analysis_mode": "placeholder",
            "real_pose_detected": False,
            "overall_score": None,
            "technical_summary": (
                "Pose detection was not reliable enough for real pose analysis. "
                f"Only {detection_count} of {total_sampled} sampled frames had usable pose data "
                f"({detection_ratio:.0%} detection rate). Coach manual review is required."
            ),
            "error_message": (
                f"Pose detection rate was {detection_ratio:.0%} "
                f"({detection_count}/{total_sampled} sampled frames)."
            ),
        }

    fps_for_timestamps = fps if fps > 0 else 30.0

    candidate_findings: List[Dict[str, Any]] = []
    candidate_findings.extend(
        _stroke_specific_findings(detected_frames, fps_for_timestamps, normalized_stroke)
    )

    findings = _filter_and_rank_findings(candidate_findings)
    key_frames = _generate_structural_keyframes(
        detected_frames,
        fps_for_timestamps,
        normalized_stroke,
    )

    technical_summary = _build_technical_summary(
        total_sampled=total_sampled,
        detection_count=detection_count,
        detection_ratio=detection_ratio,
        findings=findings,
        stroke_type=normalized_stroke,
    )

    return {
        **base,
        "analysis_mode": "real_pose",
        "real_pose_detected": True,
        "overall_score": _calculate_score(findings),
        "technical_summary": technical_summary,
        "phase_breakdown": _build_phase_breakdown(findings, normalized_stroke),
        "findings": findings,
        "key_frames": key_frames,
    }


def _stroke_specific_findings(
    detected_frames: List[Dict[str, Any]],
    fps: float,
    stroke: str,
) -> List[Dict[str, Any]]:
    if stroke == "Breaststroke":
        return _breaststroke_findings(detected_frames, fps)

    if stroke == "Freestyle":
        return _freestyle_findings(detected_frames, fps)

    if stroke == "Backstroke":
        return _backstroke_findings(detected_frames, fps)

    if stroke == "Butterfly":
        return _butterfly_findings(detected_frames, fps)

    return _freestyle_findings(detected_frames, fps)


def _breaststroke_findings(detected_frames: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    wide_kick = _ratio_series(
        detected_frames,
        fps,
        numerator=("left_knee", "right_knee"),
        denominator=("left_hip", "right_hip"),
        required=("left_knee", "right_knee", "left_hip", "right_hip"),
    )
    if wide_kick and wide_kick["max"] >= 1.35 and wide_kick["count"] >= 6:
        confidence = _confidence_from_strength(wide_kick["max"], 1.35, 1.95, wide_kick["count"])
        findings.append(_make_finding(
            fault_tag="wide_breaststroke_kick",
            stroke="Breaststroke",
            phase="kick_setup",
            severity="High" if wide_kick["max"] >= 1.75 else "Medium",
            confidence_score=confidence,
            frame=_frame_at_timestamp(detected_frames, wide_kick["timestamp"], fps),
            timestamp_seconds=wide_kick["timestamp"],
            observation=(
                "The knees appear to separate wider than the hips during the kick setup."
            ),
            why_it_matters=(
                "A wide kick recovery can create extra frontal resistance before the propulsive phase."
            ),
            keypoints_used=["left_knee", "right_knee", "left_hip", "right_hip"],
            evidence_note=(
                f"Peak knee-to-hip width ratio was {wide_kick['max']:.2f} across "
                f"{wide_kick['count']} usable pose frames."
            ),
            quality_flags=[],
        ))

    head_lift = _head_lift_signal(detected_frames, fps, threshold=-0.42)
    if head_lift and head_lift["count"] >= 8:
        findings.append(_make_finding(
            fault_tag="head_lift",
            stroke="Breaststroke",
            phase="breath",
            severity="High" if head_lift["strength"] >= 0.62 else "Medium",
            confidence_score=_confidence_from_strength(head_lift["strength"], 0.42, 0.72, head_lift["count"]),
            frame=_frame_at_timestamp(detected_frames, head_lift["timestamp"], fps),
            timestamp_seconds=head_lift["timestamp"],
            observation=(
                "The head appears to lift away from the shoulder line during the breath."
            ),
            why_it_matters=(
                "Lifting the head can interrupt the body line and make the recovery harder to reset."
            ),
            keypoints_used=["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            evidence_note=(
                f"Head position rose above the shoulder line by {head_lift['strength']:.2f} "
                "torso units in the strongest sampled frame."
            ),
            quality_flags=[],
        ))

    line_reset = _body_line_signal(detected_frames, fps, threshold=1.48)
    if line_reset and line_reset["count"] >= 8:
        findings.append(_make_finding(
            fault_tag="breaststroke_line_reset",
            stroke="Breaststroke",
            phase="line_reset",
            severity="High" if line_reset["strength"] >= 1.75 else "Medium",
            confidence_score=_confidence_from_strength(line_reset["strength"], 1.48, 1.95, line_reset["count"]),
            frame=_frame_at_timestamp(detected_frames, line_reset["timestamp"], fps),
            timestamp_seconds=line_reset["timestamp"],
            observation=(
                "The hips appear to sit lower than the shoulder line during the line reset."
            ),
            why_it_matters=(
                "The swimmer may be losing the narrow glide shape before the next pull begins."
            ),
            keypoints_used=["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            evidence_note=(
                f"Peak shoulder-to-hip vertical offset was {line_reset['strength']:.2f} "
                f"across {line_reset['count']} usable pose frames."
            ),
            quality_flags=[],
        ))

    timing = _breaststroke_timing_signal(detected_frames, fps)
    if timing and timing["count"] >= 6:
        findings.append(_make_finding(
            fault_tag="breaststroke_timing_review",
            stroke="Breaststroke",
            phase="recovery",
            severity="Medium",
            confidence_score=_confidence_from_strength(timing["strength"], 0.18, 0.34, timing["count"]),
            frame=_frame_at_timestamp(detected_frames, timing["timestamp"], fps),
            timestamp_seconds=timing["timestamp"],
            observation=(
                "Arm recovery and kick setup appear to overlap in a way that may rush the timing."
            ),
            why_it_matters=(
                "Breaststroke is easier to coach when pull, breath, kick, and line reset are clearly separated."
            ),
            keypoints_used=["left_wrist", "right_wrist", "left_knee", "right_knee"],
            evidence_note=(
                "The sampled pose showed wide knee setup while the wrists were still recovering forward."
            ),
            quality_flags=[],
        ))

    return findings


def _freestyle_findings(detected_frames: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    body_line = _body_line_signal(detected_frames, fps, threshold=1.55)
    if body_line and body_line["count"] >= 8:
        findings.append(_make_finding(
            fault_tag="body_line_loss",
            stroke="Freestyle",
            phase="body_line",
            severity="High" if body_line["strength"] >= 1.85 else "Medium",
            confidence_score=_confidence_from_strength(body_line["strength"], 1.55, 2.05, body_line["count"]),
            frame=_frame_at_timestamp(detected_frames, body_line["timestamp"], fps),
            timestamp_seconds=body_line["timestamp"],
            observation="The hips appear to drop below the shoulder line through the sampled stroke.",
            why_it_matters=(
                "A lower hip line can increase resistance and make the catch less effective."
            ),
            keypoints_used=["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            evidence_note=(
                f"Peak shoulder-to-hip vertical offset was {body_line['strength']:.2f} "
                f"across {body_line['count']} usable pose frames."
            ),
            quality_flags=[],
        ))

    catch = _dropped_elbow_signal(detected_frames, fps)
    if catch and catch["count"] >= 6:
        findings.append(_make_finding(
            fault_tag="dropped_elbow_catch",
            stroke="Freestyle",
            phase="catch_setup",
            severity="High" if catch["strength"] >= 0.16 else "Medium",
            confidence_score=_confidence_from_strength(catch["strength"], 0.10, 0.22, catch["count"]),
            frame=_frame_at_timestamp(detected_frames, catch["timestamp"], fps),
            timestamp_seconds=catch["timestamp"],
            observation="The elbow appears to drop close to or below the wrist during catch setup.",
            why_it_matters=(
                "A dropped elbow can reduce the early hold on the water and make the pull less connected."
            ),
            keypoints_used=["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
            evidence_note=(
                f"Peak elbow-wrist catch signal was {catch['strength']:.2f} in the sampled pose."
            ),
            quality_flags=[],
        ))

    extension = _short_extension_signal(detected_frames, fps)
    if extension and extension["count"] >= 6:
        findings.append(_make_finding(
            fault_tag="short_entry_extension",
            stroke="Freestyle",
            phase="entry_extension",
            severity="Medium",
            confidence_score=_confidence_from_strength(extension["strength"], 0.12, 0.28, extension["count"]),
            frame=_frame_at_timestamp(detected_frames, extension["timestamp"], fps),
            timestamp_seconds=extension["timestamp"],
            observation="The lead hand appears to start the catch before a clear forward extension.",
            why_it_matters=(
                "A rushed entry can shorten the stroke and reduce the time available to set the catch."
            ),
            keypoints_used=["left_shoulder", "right_shoulder", "left_wrist", "right_wrist"],
            evidence_note=(
                f"Lead wrist extension remained close to the shoulder line across {extension['count']} sampled frames."
            ),
            quality_flags=[],
        ))

    breath = _head_lift_signal(detected_frames, fps, threshold=-0.35)
    if breath and breath["count"] >= 8:
        findings.append(_make_finding(
            fault_tag="breathing_line_break",
            stroke="Freestyle",
            phase="breathing",
            severity="High" if breath["strength"] >= 0.58 else "Medium",
            confidence_score=_confidence_from_strength(breath["strength"], 0.35, 0.68, breath["count"]),
            frame=_frame_at_timestamp(detected_frames, breath["timestamp"], fps),
            timestamp_seconds=breath["timestamp"],
            observation="The head appears to lift out of the body line during the breath.",
            why_it_matters=(
                "Lifting to breathe can interrupt rotation and pull the hips down."
            ),
            keypoints_used=["nose", "left_shoulder", "right_shoulder"],
            evidence_note=(
                f"Head lift signal was {breath['strength']:.2f} torso units in the strongest sampled frame."
            ),
            quality_flags=[],
        ))

    return findings


def _backstroke_findings(detected_frames: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    body_line = _body_line_signal(detected_frames, fps, threshold=1.52)
    if not body_line or body_line["count"] < 8:
        return []

    return [_make_finding(
        fault_tag="backstroke_hip_sink",
        stroke="Backstroke",
        phase="body_line",
        severity="High" if body_line["strength"] >= 1.82 else "Medium",
        confidence_score=_confidence_from_strength(body_line["strength"], 1.52, 2.00, body_line["count"]),
        frame=_frame_at_timestamp(detected_frames, body_line["timestamp"], fps),
        timestamp_seconds=body_line["timestamp"],
        observation="The hips appear to sit low relative to the shoulder line in the sampled frames.",
        why_it_matters=(
            "Backstroke speed depends on a supported body line so the kick and rotation can stay connected."
        ),
        keypoints_used=["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        evidence_note=(
            f"Peak shoulder-to-hip vertical offset was {body_line['strength']:.2f} "
            f"across {body_line['count']} usable pose frames."
        ),
        quality_flags=[],
    )]


def _butterfly_findings(detected_frames: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
    rhythm = _butterfly_rhythm_signal(detected_frames, fps)
    if not rhythm or rhythm["count"] < 8:
        return []

    return [_make_finding(
        fault_tag="butterfly_rhythm_break",
        stroke="Butterfly",
        phase="body_wave",
        severity="High" if rhythm["strength"] >= 0.18 else "Medium",
        confidence_score=_confidence_from_strength(rhythm["strength"], 0.10, 0.24, rhythm["count"]),
        frame=_frame_at_timestamp(detected_frames, rhythm["timestamp"], fps),
        timestamp_seconds=rhythm["timestamp"],
        observation="The shoulder and hip line appears to change unevenly through the sampled rhythm.",
        why_it_matters=(
            "Butterfly timing is easier to sustain when the body wave and arm recovery stay connected."
        ),
        keypoints_used=["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        evidence_note=(
            f"Body-wave timing signal was {rhythm['strength']:.2f} across sampled pose frames."
        ),
        quality_flags=[],
    )]


def _make_finding(
    fault_tag: str,
    stroke: str,
    phase: str,
    severity: str,
    confidence_score: float,
    frame: Optional[Dict[str, Any]],
    timestamp_seconds: float,
    observation: str,
    why_it_matters: str,
    keypoints_used: Sequence[str],
    evidence_note: str,
    quality_flags: Sequence[str],
) -> Dict[str, Any]:
    drill_map = DRILL_CUE_MAP[fault_tag]
    frame_index = int(frame.get("frame_idx", 0)) if frame else 0
    confidence_score = round(float(confidence_score), 2)
    confidence_label = "high" if confidence_score >= 0.75 else "medium"
    title = _title_from_fault(fault_tag)

    evidence = {
        "timestamp_seconds": round(float(timestamp_seconds), 2),
        "frame_index": frame_index,
        "phase": phase,
        "keypoints_used": list(keypoints_used),
        "confidence": confidence_label,
        "confidence_score": confidence_score,
        "evidence_note": evidence_note,
    }

    return {
        "id": f"{fault_tag}-{frame_index}",
        "source": "ai_pose",
        "stroke": stroke,
        "phase": phase,
        "stroke_phase": phase,
        "fault_tag": fault_tag,
        "severity": severity,
        "confidence": confidence_label,
        "confidence_score": confidence_score,
        "ai_confidence": confidence_score,
        "timestamp_seconds": round(float(timestamp_seconds), 2),
        "timestamp_start": max(0.0, round(float(timestamp_seconds) - 0.4, 2)),
        "timestamp_end": round(float(timestamp_seconds) + 0.4, 2),
        "frame_index": frame_index,
        "finding_title": title,
        "finding_description": observation,
        "observation": observation,
        "why_it_matters": why_it_matters,
        "correction_cue": drill_map["cue"],
        "recommended_correction": drill_map["cue"],
        "cue": drill_map["cue"],
        "drill": drill_map["drill"],
        "recommended_drill": drill_map["drill"],
        "next_focus": drill_map["next_focus"],
        "evidence": evidence,
        "evidence_type": "pose_measurement",
        "measurement_summary": evidence_note,
        "quality_flags": list(quality_flags),
        "coach_review_required": True,
        "frame_reference": "",
    }


def _ratio_series(
    frames: List[Dict[str, Any]],
    fps: float,
    numerator: Tuple[str, str],
    denominator: Tuple[str, str],
    required: Sequence[str],
) -> Optional[Dict[str, Any]]:
    values = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        if not all(k in lm for k in required):
            continue

        denom = horizontal_distance(lm[denominator[0]], lm[denominator[1]])
        if denom < 0.01:
            continue

        values.append({
            "value": horizontal_distance(lm[numerator[0]], lm[numerator[1]]) / denom,
            "timestamp": frame.get("frame_idx", 0) / fps,
        })

    return _series_peak(values)


def _body_line_signal(
    frames: List[Dict[str, Any]],
    fps: float,
    threshold: float,
) -> Optional[Dict[str, Any]]:
    values = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(k in lm for k in required):
            continue

        shoulder_mid = get_midpoint(lm["left_shoulder"], lm["right_shoulder"])
        hip_mid = get_midpoint(lm["left_hip"], lm["right_hip"])
        raw_offset = vertical_distance(shoulder_mid, hip_mid)
        shoulder_width = horizontal_distance(lm["left_shoulder"], lm["right_shoulder"])
        scale = shoulder_width if shoulder_width > 0.02 else 0.22
        normalized = raw_offset / scale

        if normalized >= threshold:
            values.append({
                "value": normalized,
                "timestamp": frame.get("frame_idx", 0) / fps,
            })

    return _series_peak(values)


def _head_lift_signal(
    frames: List[Dict[str, Any]],
    fps: float,
    threshold: float,
) -> Optional[Dict[str, Any]]:
    values = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        required = ["nose", "left_shoulder", "right_shoulder"]
        if not all(k in lm for k in required):
            continue

        shoulder_mid = get_midpoint(lm["left_shoulder"], lm["right_shoulder"])
        shoulder_width = horizontal_distance(lm["left_shoulder"], lm["right_shoulder"])
        scale = shoulder_width if shoulder_width > 0.02 else 0.22
        relative_y = (lm["nose"]["y"] - shoulder_mid["y"]) / scale

        if relative_y <= threshold:
            values.append({
                "value": abs(relative_y),
                "timestamp": frame.get("frame_idx", 0) / fps,
            })

    return _series_peak(values)


def _dropped_elbow_signal(frames: List[Dict[str, Any]], fps: float) -> Optional[Dict[str, Any]]:
    values = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        for side in ["left", "right"]:
            required = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
            if not all(k in lm for k in required):
                continue

            shoulder = lm[f"{side}_shoulder"]
            elbow = lm[f"{side}_elbow"]
            wrist = lm[f"{side}_wrist"]
            arm_span = max(0.08, abs(wrist["x"] - shoulder["x"]) + abs(wrist["y"] - shoulder["y"]))
            elbow_below_wrist = (elbow["y"] - wrist["y"]) / arm_span

            if elbow_below_wrist >= 0.10:
                values.append({
                    "value": elbow_below_wrist,
                    "timestamp": frame.get("frame_idx", 0) / fps,
                })

    return _series_peak(values)


def _short_extension_signal(frames: List[Dict[str, Any]], fps: float) -> Optional[Dict[str, Any]]:
    values = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        usable_sides = []
        for side in ["left", "right"]:
            required = [f"{side}_shoulder", f"{side}_wrist"]
            if not all(k in lm for k in required):
                continue

            shoulder = lm[f"{side}_shoulder"]
            wrist = lm[f"{side}_wrist"]
            extension = abs(wrist["x"] - shoulder["x"])
            usable_sides.append(extension)

        if not usable_sides:
            continue

        max_extension = max(usable_sides)
        if max_extension <= 0.12:
            values.append({
                "value": 0.12 - max_extension,
                "timestamp": frame.get("frame_idx", 0) / fps,
            })

    return _series_peak(values)


def _breaststroke_timing_signal(frames: List[Dict[str, Any]], fps: float) -> Optional[Dict[str, Any]]:
    values = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        required = [
            "left_wrist",
            "right_wrist",
            "left_shoulder",
            "right_shoulder",
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
        ]
        if not all(k in lm for k in required):
            continue

        shoulder_width = horizontal_distance(lm["left_shoulder"], lm["right_shoulder"])
        hip_width = horizontal_distance(lm["left_hip"], lm["right_hip"])
        if shoulder_width < 0.02 or hip_width < 0.02:
            continue

        wrist_width = horizontal_distance(lm["left_wrist"], lm["right_wrist"]) / shoulder_width
        knee_width = horizontal_distance(lm["left_knee"], lm["right_knee"]) / hip_width
        overlap = max(0.0, knee_width - 1.25) * max(0.0, wrist_width - 1.00)

        if overlap >= 0.18:
            values.append({
                "value": overlap,
                "timestamp": frame.get("frame_idx", 0) / fps,
            })

    return _series_peak(values)


def _butterfly_rhythm_signal(frames: List[Dict[str, Any]], fps: float) -> Optional[Dict[str, Any]]:
    values = []

    for frame in frames:
        lm = frame.get("landmarks", {})
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not all(k in lm for k in required):
            continue

        shoulder_mid = get_midpoint(lm["left_shoulder"], lm["right_shoulder"])
        hip_mid = get_midpoint(lm["left_hip"], lm["right_hip"])
        values.append({
            "value": abs(vertical_distance(shoulder_mid, hip_mid)),
            "timestamp": frame.get("frame_idx", 0) / fps,
        })

    if len(values) < 8:
        return None

    series = [item["value"] for item in values]
    variability = float(np.std(series))
    peak = max(values, key=lambda item: item["value"])

    if variability < 0.10:
        return None

    return {
        "max": variability,
        "strength": variability,
        "timestamp": round(peak["timestamp"], 2),
        "count": len(values),
    }


def _series_peak(values: List[Dict[str, float]]) -> Optional[Dict[str, Any]]:
    if not values:
        return None

    peak = max(values, key=lambda item: item["value"])
    return {
        "max": float(peak["value"]),
        "strength": float(peak["value"]),
        "timestamp": round(float(peak["timestamp"]), 2),
        "count": len(values),
    }


def _confidence_from_strength(
    value: float,
    threshold: float,
    strong_threshold: float,
    frame_count: int,
) -> float:
    if strong_threshold <= threshold:
        strength_score = 0.0
    else:
        strength_score = min(1.0, max(0.0, (value - threshold) / (strong_threshold - threshold)))

    frame_score = min(1.0, frame_count / 14)
    confidence = 0.58 + (strength_score * 0.24) + (frame_score * 0.12)
    return min(0.88, confidence)


def _filter_and_rank_findings(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered = [
        finding
        for finding in findings
        if finding.get("severity") in ["High", "Medium"]
        and float(finding.get("confidence_score") or 0) >= MIN_FINDING_CONFIDENCE
        and finding.get("observation")
        and finding.get("correction_cue")
    ]

    severity_rank = {"High": 2, "Medium": 1}
    filtered.sort(
        key=lambda finding: (
            severity_rank.get(finding.get("severity"), 0),
            float(finding.get("confidence_score") or 0),
        ),
        reverse=True,
    )

    return filtered[:MAX_FINDINGS]


def _frame_at_timestamp(
    frames: List[Dict[str, Any]],
    timestamp: float,
    fps: float,
) -> Optional[Dict[str, Any]]:
    if not frames:
        return None

    safe_fps = fps if fps > 0 else 30.0
    return min(frames, key=lambda frame: abs((frame.get("frame_idx", 0) / safe_fps) - timestamp))


def _generate_structural_keyframes(
    detected_frames: List[Dict[str, Any]],
    fps: float,
    stroke_type: str,
) -> List[Dict[str, Any]]:
    if not detected_frames:
        return []

    sequence = PHASE_SEQUENCES.get(stroke_type, ["unknown"])
    step = max(1, len(detected_frames) // min(6, len(detected_frames)))
    selected = detected_frames[::step][:6]

    keyframes = []
    for i, frame in enumerate(selected):
        phase = sequence[min(i, len(sequence) - 1)] if sequence else "unknown"
        keyframes.append({
            "timestamp": round(frame.get("frame_idx", 0) / fps, 2),
            "label": _human_label(phase),
            "description": (
                f"Coach reference frame for {stroke_type.lower()} {phase}; "
                f"{frame.get('keypoint_count', 0)} visible keypoints."
            ),
        })

    return keyframes


def _calculate_score(findings: List[Dict[str, Any]]) -> Optional[int]:
    if not findings:
        return 84

    score = 86
    penalties = {"High": 8, "Medium": 5}

    for finding in findings:
        score -= penalties.get(finding.get("severity", "Medium"), 4)

    return max(55, min(94, score))


def _build_technical_summary(
    total_sampled: int,
    detection_count: int,
    detection_ratio: float,
    findings: List[Dict[str, Any]],
    stroke_type: str,
) -> str:
    if not findings:
        return (
            f"Pose-assisted {stroke_type.lower()} analysis processed {total_sampled} sampled frames "
            f"and detected usable body landmarks in {detection_count} frames "
            f"({detection_ratio:.0%} detection rate). No medium or high-confidence pose findings "
            "were emitted. Coach review is still required."
        )

    finding_titles = "; ".join([f["finding_title"] for f in findings])

    return (
        f"Pose-assisted {stroke_type.lower()} analysis processed {total_sampled} sampled frames "
        f"and detected usable body landmarks in {detection_count} frames "
        f"({detection_ratio:.0%} detection rate). Draft findings flagged for coach review: "
        f"{finding_titles}. These are evidence prompts, not automatic conclusions."
    )


def _build_phase_breakdown(findings: List[Dict[str, Any]], stroke_type: str) -> Dict[str, Dict[str, Any]]:
    phases = PHASE_SEQUENCES.get(stroke_type, ["body_line"])
    breakdown: Dict[str, Dict[str, Any]] = {
        phase: {
            "status": "not_flagged",
            "label": _human_label(phase),
            "coach_note": "No medium or high-confidence draft finding from sampled pose.",
        }
        for phase in phases
    }

    for finding in findings:
        phase = finding.get("phase") or finding.get("stroke_phase") or "unknown"
        severity = finding.get("severity", "Medium")
        breakdown[phase] = {
            "status": "review_required",
            "label": _human_label(phase),
            "severity": severity,
            "fault_tag": finding.get("fault_tag"),
            "coach_note": finding.get("observation"),
        }

    return breakdown


def _title_from_fault(fault_tag: str) -> str:
    return {
        "body_line_loss": "Body Line Needs Coach Review",
        "head_lift": "Head Lift During Breath",
        "wide_breaststroke_kick": "Breaststroke Kick Width",
        "breaststroke_timing_review": "Breaststroke Timing Review",
        "breaststroke_line_reset": "Line Reset After Kick",
        "dropped_elbow_catch": "Catch Setup Needs Review",
        "short_entry_extension": "Entry Extension Looks Short",
        "breathing_line_break": "Breathing Line Break",
        "backstroke_hip_sink": "Backstroke Hip Position",
        "butterfly_rhythm_break": "Butterfly Rhythm Review",
    }.get(fault_tag, "Coach Review Finding")


def _human_label(value: str) -> str:
    return (value or "unknown").replace("_", " ").title()
