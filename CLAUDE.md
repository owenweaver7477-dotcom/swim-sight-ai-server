# Swim Sight AI Server — Worker Repo Control File (CLAUDE.md)

> **This is the AI WORKER repository control file.**
> You are inside **swim-sight-ai-server** — the AI video-analysis worker.
> This is **NOT** the Swim Sight 3D app repo and **NOT** CoachSight Core (the private operations brain).
> Repo path: `/Users/owen_weaver/Documents/GitHub/swim-sight-ai-server`
> Stack: **Python / FastAPI**, deployed to **Render**. It receives jobs from the **app** (`Swim Sight 3D V1`) and reads private video from **Supabase Storage** via signed URLs.
>
> Vision, roadmap, and master safety policy live in **CoachSight Core**, not here. This file is the operating contract for working *inside the worker code*.

---

## 1. What the AI Worker Repo Is

The AI worker is a backend service that performs swimming video analysis off the main app. The app sends it an analysis job; the worker accesses the private video, processes it, produces **careful, reviewable suggestions** (never definitive conclusions), and returns them to the app via a secured callback. It does no UI and stores no coach-facing truth — it proposes; the coach (in the app) decides.

It is a **separate deployable** on Render so heavy video/ML work never blocks the user-facing app.

## 2. What Swim Sight 3D Uses the Worker For

Within the core pilot flow, the worker owns the "AI worker processing" stage:

```
app: coach uploads video & triggers analysis
  → WORKER receives job
  → WORKER accesses private/signed video
  → WORKER probes + processes (frames / pose)
  → WORKER generates a safe, AI-suggested summary
  → WORKER calls back to the app
  → app shows AI-suggested findings → coach review & approval
```

Already-built foundations to preserve (inspect before rewriting): storage access adapter, signed-URL fallback, provider/key payload handling, callback summary handling, early 2D/3D pose foundations, log-redaction foundations, storage/callback-safety test foundations.

## 3. The Worker Pipeline

Treat each stage as a checkpoint. Understand the whole chain before editing any link.

1. **Receive job** — accept an analysis job from the app (job id, video reference, provider/key payload, callback target). Validate the request; never trust arbitrary external input. `[UNCONFIRMED — verify entry point in-repo]`
2. **Access video** — resolve the private video via signed URL / storage provider, with signed-URL fallback. Treat the URL as a temporary secret. Never persist it beyond the job; never log it.
3. **Probe video** — inspect format/duration/streams before heavy work; fail fast and safely on unreadable or oversized input.
4. **Process frames / pose** — frame extraction → 2D pose foundations → optional 3D lifting. Target schema where applicable: MediaPipe BlazePose-style 33-landmark. `[UNCONFIRMED — verify in-repo]`
5. **Generate safe summary** — assemble an **AI-suggested**, clearly-hedged callback payload with confidence/estimation wording (Section 8). No definitive coaching or biomechanical conclusions.
6. **Callback to app** — deliver the summary to the app's callback endpoint using the agreed secret/validation. Classify and report errors safely; redact everything sensitive (Sections 5–7).

Reliability matters more than feature depth: clear failure modes, understandable error classification, and safe partial/failed results beat ambitious-but-fragile analysis.

## 4. Protected Worker Systems

Load-bearing. Inspect and preserve; do not weaken:

1. **Job intake / request validation** — only legitimate app jobs are processed.
2. **Private video access** — signed-URL + provider/key handling + signed-URL fallback (Section 5).
3. **Storage adapter** — how the worker reaches Supabase Storage; failure handling.
4. **Video probing & frame extraction** — safe handling of bad/large/unsupported input.
5. **Pose-analysis foundations** — existing 2D/3D logic.
6. **Callback generation, delivery & validation** — the secured handshake back to the app (Section 7).
7. **Logging with redaction** — secrets, signed URLs, keys, and private video access details never leak (Section 6).
8. **Render runtime limits** — memory/compute/timeout constraints; don't assume capacity that isn't provisioned.

## 5. Signed URL & Private Video Safety

Videos are private — often of minors. This is non-negotiable.

- Signed URLs are **temporary and secret**. Use them only for the duration of the job, then drop them.
- **Never log** a signed URL, storage path, bucket name, or provider key — not at any log level, not in error messages, not in stack traces, not in the callback payload.
- **Never return** a signed URL or raw storage path to the app callback or any external surface.
- **Never persist** the video or its URL beyond what the job needs; clean up temp files.
- Honor the **signed-URL fallback** path; don't remove it without understanding why it exists.
- If you can't access the video safely, **fail the job with a redacted, generic error** — never expose access details to make debugging easier.

## 6. Log Redaction Rules

- Logs must **never** contain: signed URLs, storage paths/bucket names, provider/service keys, API keys, callback secrets, or personally identifiable swimmer/coach data.
- Log **identifiers and outcomes**, not secrets (e.g. job id, stage, duration, error class) — never the sensitive value itself.
- Error messages and stack traces are subject to the same redaction. Assume logs may be read by someone who should not see private data.
- Preserve and route through the existing redaction helpers; do not bypass them with raw `print`/`logger` calls on sensitive objects. `[UNCONFIRMED — verify redaction utility in-repo]`

## 7. Callback Validation Rules

- The callback between app and worker is **protected by a secret / secure validation mechanism**. Never weaken, disable, or bypass it.
- **Never accept arbitrary external callbacks** or unauthenticated job submissions. Validate inbound requests; reject anything not from the trusted app path.
- Keep the callback secret out of code, logs, and the payload — read it from environment/config only. `[UNCONFIRMED — verify config source in-repo]`
- Callback payloads carry **AI-suggested** content only (Section 8) and **no** private access details (Section 5).
- If validation needs a fix, it goes through review (Owen) — see Section 14. Never "temporarily" loosen it for testing.

## 8. AI-Suggested / Estimated-Data Wording Rules

The worker proposes; the coach decides. All output must be framed as suggestions/estimates.

**Use:** "AI-suggested finding", "estimated pose data", "possible technical issue", "suggested cue", "suggested drill", "confidence score", "coach review required".

**Never:** "guaranteed", "diagnosed"/"diagnosis", "proven", "perfect", "medical cause", "injury cause", "definitive biomechanics conclusion".

No medical or injury claims. No unverified biomechanics certainty. Include confidence/estimation signals so the app can present findings as reviewable suggestions, never as final truth.

## 9. What Claude Must Never Touch

Hard limits in this repo. If a task heads toward one of these, stop and flag it.

- ❌ Read, print, or commit `.env`, provider keys, service-role keys, callback secrets, or any credential.
- ❌ Trigger a **Render redeploy**, change Render scaling/tier, or **enable GPU**.
- ❌ **Push, merge, or delete branches** (Owen does this after review).
- ❌ Write to, update, delete, or change schema/RLS on Supabase; delete any storage object/video.
- ❌ Log, return, or persist **signed URLs / private video access details**.
- ❌ Weaken, disable, or bypass **callback validation** or accept arbitrary external callbacks.
- ❌ Bypass or weaken **log redaction**.
- ❌ Emit definitive/medical/biomechanical claims (Section 8).
- ❌ Add a new paid dependency, service, or compute resource without approval.

## 10. How Claude Should Inspect Before Editing

Default mode is **understand first, edit second.** Before changing anything:

1. **Map the service** — entry points, FastAPI routes, the job lifecycle, where config/secrets are read. `[UNCONFIRMED — verify in-repo]`
2. **Trace the full pipeline** (Section 3) for the stage in question before touching it.
3. **Identify protected systems in scope** (Section 4) and what must not change.
4. **State the smallest change** that fixes the issue and the blast radius (including Render resource impact).
5. **Preserve foundations** — extend existing storage/callback/redaction code rather than rewriting.
6. **Check redaction & secrets** — confirm no sensitive value enters logs, callbacks, or test output.
7. **Mark uncertainty** — anything not directly inspected is `[UNCONFIRMED]`.
8. **Confirm risk level** (Section 11) and get approval where required (Section 14) **before** editing.

## 11. Risk Levels

| Level | Meaning | Examples | Gate |
|---|---|---|---|
| 🟢 **Low** | Isolated, no protected system, no secret/redaction path | Comments, docstrings, local refactor of pure helper, test additions that touch no secrets | Proceed; show diff for review |
| 🟡 **Medium** | Worker logic, not a protected system | Non-critical processing tweak, error-message wording (already redacted), isolated pose-helper change | Explain change + risk (incl. Render impact), then proceed on Owen's OK |
| 🔴 **High** | Touches a protected system (Section 4) or anything irreversible/external | Job intake/validation, signed-URL/video access, storage adapter, callback validation, redaction, secrets/config, Render deploy/scale/GPU, Supabase access | **Stop. Requires explicit `APPROVE:`** (Section 14). Draft only until approved |

If unsure, treat it as the **higher** level.

## 12. First Safe Non-Editing Worker Diagnosis Prompt

Paste this to run a **non-editing** diagnosis:

> "You are working inside the swim-sight-ai-server worker repo. Read this CLAUDE.md first. Do not edit anything, do not redeploy or scale Render, do not read .env or secrets, and do not weaken callback validation or log redaction. Produce a non-editing diagnosis covering: worker entry points and job lifecycle; video access via signed URLs, provider/key handling, and fallback; the storage adapter; video probing and frame extraction; pose-analysis foundations; callback payload structure, delivery, and validation; logging/redaction safety; and failure modes including timeouts and Render memory limits. Rank the highest-impact, lowest-risk reliability fixes. List anything I should not touch without caution, and mark anything you have not directly inspected as [UNCONFIRMED]."

## 13. Test / Check Commands

> ⚠️ `[UNCONFIRMED — verify in-repo]` — confirm the real setup before relying on these. Typical Python/FastAPI candidates:

```bash
# create / activate a virtual env
python -m venv .venv && source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run the service locally
uvicorn app.main:app --reload        # adjust module:path to the real entry point

# run tests
pytest

# lint / type-check (if configured)
ruff check .        # or: flake8
mypy .
```

Rules for checks:
- Tests, lint, and type-checks are **read-only verification** — encouraged before calling a change "done."
- Running locally is fine. **Redeploying Render is not** (Section 9).
- Never run anything that writes to Supabase, deletes storage, or sends a real callback to production.
- Tests must use **fixtures/mocks**, never real signed URLs, secrets, or private videos. Confirm test output is also redacted.
- After confirming the real commands, replace these placeholders with the exact ones.

## 14. Approval Rules

CoachSight Core's policy applies here: **Claude advises and drafts; Owen decides and acts.**

- 🟢 **Low risk:** proceed and show the diff for review.
- 🟡 **Medium risk:** explain the change and its risk (including Render impact), then proceed once Owen agrees.
- 🔴 **High risk** (any protected system, or anything irreversible/external): **draft only — do not apply — until Owen types an explicit approval.**

**Approval format:** Owen types `APPROVE: [action]` (e.g. `APPROVE: change callback validation in worker`). Without it, a high-risk change is not applied.

**Always requires `APPROVE:` (never silent):**
- Pushing to any branch, merging, or deleting a branch
- Redeploying Render, changing tier/scaling, or enabling GPU
- Changing environment variables / secrets / config sources
- Any Supabase write, delete, schema, RLS, or storage change
- Any change to job-intake validation, callback validation, signed-URL/video access, or log redaction
- Adding a new paid dependency, service, or compute resource

If an action isn't listed but feels irreversible, costs money, exposes private data, or affects real users — **treat it as high risk and ask first.**

---

*End of AI worker repo control file. Source of truth for vision and master safety policy: CoachSight Core (`coachsight-core`). The app's control file lives in the Swim Sight 3D V1 repo. Keep this file accurate; update the `[UNCONFIRMED]` placeholders once verified inside the repo.*
