# Coach Feedback Evaluation Plan

## Purpose

Coach feedback records are internal evaluation labels for future model
improvement. They show where AI-assisted draft findings align with, differ from,
or are rejected by coach judgement.

Coach review remains the source of truth. This toolkit does not train a model,
update production behaviour, or create an automatic learning loop.

## Privacy Boundary

The normaliser removes or summarises input before local persistence. Feedback
records must not contain:

- video or signed URLs
- private storage/file paths
- raw landmarks, pose results, footage, or frame arrays
- swimmer, coach, or guardian identity/contact details
- swimmer height or mass
- source video/review identifiers

Evidence frames are reduced to a count. Coach notes are omitted by default and
are included only as a short `coach_note_summary` when explicitly marked
`coach_note_is_safe: true` and when they pass the unsafe-value checks.

The JSONL writer validates privacy a second time and refuses unsafe records.

## Decision Labels

Supported coach decisions are:

- `approved`
- `rejected`
- `edited`
- `needs_more_context`
- `unsafe_to_use`

An approved finding whose severity or cue changes is counted as edited rather
than approved unchanged. This keeps approval and edit rates mutually useful.

## Local Workflow

Dry-run and inspect a feedback file:

```bash
python3 scripts/ingest_coach_feedback.py \
  --input fixtures/coach_feedback/approved_rejected_edited.example.json \
  --dry-run \
  --summary
```

Write a sanitised local export:

```bash
python3 scripts/ingest_coach_feedback.py \
  --input path/to/local-feedback.json \
  --output coach_feedback_exports/feedback.local.jsonl
```

Summarise evaluation labels:

```bash
python3 scripts/summarise_coach_feedback.py \
  --input coach_feedback_exports/feedback.local.jsonl
```

`coach_feedback_exports/` is ignored by Git. Real exports must remain private.

## Interpreting Metrics

- Approval rate counts findings accepted without an edit.
- Edited rate identifies draft findings that were useful but needed correction.
- Rejection rate identifies draft findings the coach did not accept.
- Severity agreement compares AI and coach severity only where both are given.
- High-confidence rejection rate highlights confident drafts that most need
  investigation.
- Phase-context availability shows how often a reviewed finding had phase data.

A high rejection rate means the AI draft needs improvement; it does not mean
the coach is wrong. These are internal evaluation metrics, not public accuracy
claims.

## Endpoint Decision

Phase 5 intentionally does not add `/coach-feedback`. Render-local JSONL storage
is ephemeral and is not a durable production feedback system. A future endpoint
should be added only with authenticated app-side ingestion, durable private
storage, retention rules, and server-side access controls.
