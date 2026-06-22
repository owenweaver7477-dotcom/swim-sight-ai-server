# Wiring the analysis-options checklist

One catalog drives the coach checklist and the server planner. Both are
flag-gated, default OFF — nothing changes until you turn them on.

## Pieces (built + tested)
- **Catalog (source of truth):** `app/analysis_options_catalog.json`, mirrored to
  `Swim Sight 3D V1/src/lib/analysisOptionsCatalog.json`. Keep the two in sync.
- **Worker planner:** `app/analysis_options.py` →
  `plan_analysis(selection, plan, available_data)` returns
  `{context, faults, metrics, outputs, rejected, warnings, ai_instructions}`.
  Tests: `scripts/test_analysis_options.py` (23/23 green).
- **Frontend:** `src/lib/analysisOptions.js` + `src/components/analysis/AnalysisSetup.jsx`.

## Flags
- Frontend: `VITE_ENABLE_ANALYSIS_OPTIONS=true`
- Worker: `ENABLE_ANALYSIS_OPTIONS=true`

## 1. Mount the checklist (after upload, before analysis)
```jsx
import AnalysisSetup from '@/components/analysis/AnalysisSetup';

<AnalysisSetup
  planKey={coachPlanKey}                                  // the coach's subscription tier
  externalData={{ height_cm: swimmer.height_cm, mass_kg: swimmer.mass_kg }}
  running={trigger.isPending}
  onRun={(selection) =>
    functions.triggerPoseAnalysis(videoUploadId, { analysis_options: selection, plan: coachPlanKey })}
/>
```

## 2. Payload
`triggerPoseAnalysis` already accepts a payload — add `analysis_options` + `plan`.
The base44 `triggerPoseAnalysis` function forwards them to `/process-video`.

## 3. Worker (`/process-video`, main.py) — guarded by the flag
```python
from app.analysis_options import analysis_options_enabled, plan_analysis

if analysis_options_enabled() and req.analysis_options:
    available = {
        "scale": bool(req.scale_m_per_unit_override),
        "mass_kg": req.swimmer_mass_kg,
        "height_cm": req.swimmer_height_cm,
        "pool_calibration": False,   # set True once pool_calibration.py exists
    }
    options_plan = plan_analysis(req.analysis_options, plan=server_plan, available_data=available)
    # route:
    #   options_plan["faults"]   -> which finding checks to run
    #   options_plan["metrics"]  -> stroke rate via stroke_cycles; dps/velocity/force via calibration+drag
    #   options_plan["outputs"]  -> comparison / share_clip / report / ...
    #   options_plan["ai_instructions"]["prompt_text"] -> the "what to look for" context for findings
    # echo options_plan["rejected"] / ["warnings"] back in the callback so the UI can say
    #   e.g. "force skipped — add swimmer mass".
```

## CRITICAL — tier comes from the server, not the client
Use the coach's **actual** subscription plan (looked up server-side) for
`server_plan`, NOT the `plan` field in the payload. A client could send
`plan: "elite_lab"`; the payload value is a UX hint only. `plan_analysis` is the
gate, but it can only gate correctly if you feed it the verified plan.

## Honesty rule (already in the plan)
`ai_instructions.prompt_text` instructs the model: a requested check the video
doesn't support comes back as "looks fine / not visible", never an invented
fault; estimates are labelled, not measurements. Pass that text into whatever
generates the draft findings.

## Verify
`python3 scripts/test_analysis_options.py` (worker) and `npm run lint && npm run build` (frontend).
