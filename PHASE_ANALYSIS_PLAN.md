# Phase Analysis Plan

## Status

The Phase 3 cycle-analysis layer is internal, optional, and disabled by default.

```text
PHASE_ANALYSIS=false
```

Setting `PHASE_ANALYSIS=true` adds approximate stroke-cycle, phase, and
phase-aware reference context to the worker analysis payload. With the flag
unset or false, the existing findings and output fields remain unchanged.

## Current Scope

The first version supports:

- Breaststroke: extension, pull, recovery, kick
- Freestyle: entry/catch, pull, push, recovery

Other strokes return an explicit unsupported result with no cycles. Sparse or
weak landmark tracks return empty or low-confidence results rather than guessed
phase data.

Cycle detection uses normalized 2D hand and hip periodicity. Phase boundaries
are approximate quarter-cycle estimates. Confidence accounts for landmark
coverage, visibility, interpolated samples, signal strength, and cycle
regularity.

## Provisional Reference Context

Reference files live in `app/reference_bands/`. They are labelled
`provisional_internal`, explicitly set `validated: false`, and are designed to
be replaced after coach-graded clip evaluation. Persistent evidence is required
within a phase before context is emitted, which prevents a single-frame signal
from becoming phase-aware advice.

This output is not a validated biomechanics model. It supports AI-assisted
draft explanation only. A coach must review the source video and approve, edit,
or reject every finding before a report is shared.

## Local Checks

```bash
python3 scripts/test_phase_analysis.py --synthetic breaststroke_clean
python3 scripts/test_phase_analysis.py --synthetic breaststroke_fault
python3 scripts/test_phase_analysis.py --synthetic freestyle_clean
python3 scripts/test_upgrades.py
```

The synthetic tracks test cycle and persistence logic only. They do not prove
pose-detection quality or real-swimmer technique interpretation.

## Validation Before Activation

1. Evaluate representative, permissioned swim clips across camera angles.
2. Compare cycle boundaries with coach-labelled phases.
3. Measure missed cycles and false cycle splits.
4. Replace provisional reference bands with reviewed data.
5. Confirm phase context improves draft explanations without changing coach
   approval or manual-review fallback.
6. Enable the flag only in a controlled pilot before any wider use.
