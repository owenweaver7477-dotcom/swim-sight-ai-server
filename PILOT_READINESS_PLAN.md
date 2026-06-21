# Pilot Readiness and Analysis QA

## Purpose

Phase 6 adds local safety checks around existing worker output. It helps detect
broken analysis, weak evidence, private-data leaks, experimental output, and
unsafe wording before a payload reaches coach review.

These checks are internal QA. They are not public performance claims and do not
prove model accuracy.

## What Phase 6 Does

- inspects callback or analysis JSON without changing it
- checks pose evidence and processed-frame counts
- checks that AI findings remain coach-review drafts
- detects private URLs, paths, raw pose/frame data, identity, height, and mass
- detects unsafe certainty and measurement wording
- checks estimated-drag and phase-analysis labelling when those optional blocks
  appear
- validates Phase 5 coach-feedback records against their privacy contract
- verifies local pilot defaults, fixtures, ignores, and tracked artefacts

## What Phase 6 Does Not Do

- enable ONNX, estimated drag, or phase analysis
- change `/process-video` or callback output
- add a production endpoint
- train or automatically update a model
- publish findings or reports
- replace coach review

## QA Before Coach Review

```bash
python3 scripts/qa_analysis_payload.py \
  --input path/to/local-callback.json
```

`pass` and `warn` exit successfully. A warning can still be safe for coach
review, for example when an internal worker callback is correctly blocked from
public sharing. `fail` exits non-zero.

## QA Before Shared or Public Output

Use the stricter target:

```bash
python3 scripts/qa_analysis_payload.py \
  --input path/to/proposed-public-report.json \
  --public-report
```

Public output must contain only deliberately coach-approved report content. A
raw worker callback is expected to fail this gate because it contains internal
IDs, telemetry, and draft findings.

Public/shared reports must never contain:

- signed or private video URLs
- storage or local file paths
- raw landmarks, pose results, or frame arrays
- swimmer/guardian identity fields not deliberately public
- height or mass
- internal job IDs and processing telemetry
- unapproved or rejected findings
- calibration internals or coach-feedback exports
- experimental drag/phase blocks
- certainty claims such as guaranteed, perfect, measured drag, or exact
  biomechanics

## Pilot Readiness

```bash
python3 scripts/pilot_readiness_check.py
```

This offline check confirms safe defaults, required synthetic fixtures, ignore
rules, optional calibration, and that no video/model/generated-report artefacts
are tracked.

## Connection to Coach Feedback

After a coach reviews draft findings, Phase 5 can convert their decisions into
privacy-safe local evaluation labels. Phase 6 validates those sanitised records
before evaluation. Feedback does not automatically train or modify the worker;
it is evidence for later controlled model-improvement work.

Coach review remains required at every stage.
