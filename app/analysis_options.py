"""
app/analysis_options.py - the "set up analysis" contract.

A single catalog (analysis_options_catalog.json) drives BOTH the coach-facing
checklist (frontend) and this server-side planner. The server is the trust
boundary: it re-checks tier access and data prerequisites (never trust the
client), expands "check everything", resolves dependencies, and produces a
normalized analysis plan plus an explicit "what to look for" instruction block
for the AI.

The honesty rule is baked into that instruction block: a requested check the
video does not clearly support must come back as "looks fine / not visible",
never an invented fault; every metric is labelled with confidence and anything
marked an estimate is not a measurement.

Flag-gated behind ENABLE_ANALYSIS_OPTIONS (default OFF). With it off, the worker
keeps today's behaviour; callers may still load the catalog.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_CATALOG_PATH = Path(__file__).with_name("analysis_options_catalog.json")
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", "", "none", "null"}
_REQUIRED_CONTEXT = ("stroke", "camera_angle")


def analysis_options_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    src = os.environ if env is None else env
    return str(src.get("ENABLE_ANALYSIS_OPTIONS", "false")).strip().lower() in _TRUTHY


def _is_on(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in _FALSY
    return bool(value)


def load_catalog(path: Optional[str] = None) -> Dict[str, Any]:
    target = Path(path) if path else _CATALOG_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def index_options(catalog: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for panel in catalog.get("panels", []):
        for opt in panel.get("options", []):
            out[opt["id"]] = dict(opt, panel=panel["id"])
    return out


def plan_allows(catalog: Mapping[str, Any], plan: str, option_tier: str) -> bool:
    unlocks = catalog.get("plan_unlocks", {})
    allowed = set(unlocks.get(plan) or unlocks.get("coach_studio", []))
    return option_tier in allowed


def build_ai_instructions(
    context: Mapping[str, Any],
    faults: List[str],
    metrics: List[Dict[str, Any]],
    options: Mapping[str, Mapping[str, Any]],
    *,
    manual_only: bool = False,
) -> Dict[str, Any]:
    """Compose the structured + natural-language 'what to look for' block."""

    fault_labels = [options.get(f, {}).get("label", f) for f in faults]
    metric_labels = [m["label"] + (" (estimate)" if m.get("estimate") else "") for m in metrics]

    lines: List[str] = []
    for key in ("stroke", "camera_angle", "clip_start", "course"):
        if context.get(key):
            lines.append(f"{key.replace('_', ' ')}: {context[key]}.")

    check_block = (
        "Check and report on these technique points: " + ", ".join(fault_labels) + "."
        if fault_labels else "No specific technique points requested."
    )
    metric_block = (
        "Compute these metrics: " + ", ".join(metric_labels) + "."
        if metric_labels else "No metrics requested."
    )
    honesty = (
        "Rules: every finding is a DRAFT for coach approval. If a requested check is not "
        "clearly supported by the video, report it as 'looks fine / not visible' - never "
        "invent a fault. Label every metric with a confidence level; anything marked "
        "(estimate) is an estimate, not a measurement."
    )
    headline = (
        "MANUAL REVIEW ONLY - do not generate AI findings; provide computed metrics and context only."
        if manual_only else "AI-assisted draft review."
    )

    return {
        "headline": headline,
        "manual_only": manual_only,
        "context": {k: context.get(k) for k in
                    ("stroke", "camera_angle", "clip_start", "course", "review_type", "lap_note")
                    if context.get(k)},
        "check_faults": fault_labels,
        "compute_metrics": metric_labels,
        "honesty_rule": (
            "Unsupported requested checks must be reported as 'looks fine / not visible', "
            "never invented; estimates are labelled and are not measurements."
        ),
        "prompt_text": " ".join([headline] + lines + [check_block, metric_block, honesty]),
    }


def plan_analysis(
    selection: Optional[Mapping[str, Any]] = None,
    *,
    plan: str = "coach_studio",
    available_data: Optional[Mapping[str, Any]] = None,
    catalog: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn a coach's ticked selection into a validated, gated analysis plan.

    ``selection``: ``{option_id: value}`` (checkbox -> truthy, select/number -> value).
    ``plan``: the coach's plan key (coach_studio / ai_assist / club_pro / elite_lab).
    ``available_data``: prerequisite data already present, e.g.
        ``{"scale": True, "mass_kg": 72, "pool_calibration": False}``.

    Returns the plan with ``accepted`` / ``rejected`` (with reasons) and a
    normalized routing of ``context`` / ``faults`` / ``metrics`` / ``outputs`` plus
    the ``ai_instructions`` block. Tier and dependency checks are authoritative
    here regardless of what the client sent.
    """

    catalog = catalog or load_catalog()
    options = index_options(catalog)
    available: Dict[str, Any] = {k: v for k, v in (available_data or {}).items()
                                 if v not in (None, "", False)}
    selection = dict(selection or {})

    accepted: List[str] = []
    rejected: List[Dict[str, Any]] = []
    warnings: List[str] = []
    context: Dict[str, Any] = {}

    def reject(option_id: str, reason: str, **extra: Any) -> None:
        rejected.append({"id": option_id, "reason": reason, **extra})

    # --- pass 1: context / inputs / controls (populate available_data) ---
    for option_id in list(selection.keys()):
        opt = options.get(option_id)
        if opt is None:
            reject(option_id, "unknown_option")
            selection.pop(option_id, None)
            continue
        if opt.get("kind") not in ("context", "input", "control"):
            continue
        value = selection[option_id]
        control = opt.get("control")
        if control == "checkbox" and not _is_on(value):
            continue
        if control in ("select", "radio") and opt.get("choices") and value not in opt["choices"]:
            reject(option_id, "invalid_choice", value=value)
            continue
        if not plan_allows(catalog, plan, opt.get("tier", "coach_studio")):
            reject(option_id, "tier_locked", tier=opt.get("tier"))
            continue
        accepted.append(option_id)
        context[option_id] = value
        for key in opt.get("provides", []) or []:
            available[key] = value if opt.get("kind") == "input" else True

    # always-on controls (confidence labels, low-quality flagging)
    for option_id, opt in options.items():
        if opt.get("locked_on") and option_id not in context:
            context[option_id] = opt.get("default", True)

    for key in _REQUIRED_CONTEXT:
        if key not in context:
            warnings.append(f"missing_required_context:{key}")

    manual_only = context.get("mode") == "manual_only"

    # --- expand "check everything relevant" within the plan's tier ---
    select_all = options.get("faults_select_all")
    if select_all and _is_on(selection.get("faults_select_all")) and plan_allows(
        catalog, plan, select_all.get("tier", "ai_assist")
    ):
        for option_id, opt in options.items():
            if opt.get("kind") == "fault" and plan_allows(catalog, plan, opt.get("tier", "coach_studio")):
                selection.setdefault(option_id, True)

    # --- pass 2: faults / metrics / outputs (gate tier + dependencies) ---
    faults: List[str] = []
    metrics: List[Dict[str, Any]] = []
    outputs: List[str] = []
    for option_id, value in selection.items():
        opt = options.get(option_id)
        if opt is None or opt.get("kind") not in ("fault", "metric", "output"):
            continue
        if not _is_on(value):
            continue
        if not plan_allows(catalog, plan, opt.get("tier", "coach_studio")):
            reject(option_id, "tier_locked", tier=opt.get("tier"))
            continue
        missing = [k for k in (opt.get("requires") or []) if k not in available]
        if missing:
            reject(option_id, "missing_data", missing=missing)
            continue
        accepted.append(option_id)
        kind = opt["kind"]
        if kind == "fault":
            faults.append(option_id)
        elif kind == "metric":
            metrics.append({"id": option_id, "label": opt["label"], "estimate": bool(opt.get("estimate"))})
        else:
            outputs.append(option_id)

    faults_for_ai: List[str] = []
    if manual_only:
        warnings.append("manual_only_mode:ai_findings_disabled")
    else:
        faults_for_ai = faults

    instructions = build_ai_instructions(context, faults_for_ai, metrics, options, manual_only=manual_only)

    return {
        "enabled": True,
        "plan": plan,
        "accepted": accepted,
        "rejected": rejected,
        "warnings": warnings,
        "context": context,
        "faults": faults,
        "metrics": metrics,
        "outputs": outputs,
        "ai_instructions": instructions,
        "labels": catalog.get("labels"),
    }
