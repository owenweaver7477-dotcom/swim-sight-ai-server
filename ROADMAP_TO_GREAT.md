# Swim Sight 3D — Roadmap to Great (with Codex prompts)

This is a roadmap to **great**, not perfect — perfect isn't a real endpoint for an
analysis product; *trusted and adopted* is. Two kinds of phase below:

- 🛠 **CODEX** — a coding agent can build it. A ready-to-paste prompt is included.
- 🌍 **REAL WORLD** — only you + real footage + real coaches can complete it. No
  prompt exists for these, by design. They are what actually make it great.

**Conventions every Codex prompt assumes** (state once): keep the `pose_results`
contract `{frame_idx, pose_detected, keypoint_count, landmarks:{name:{x,y,visibility}}}`;
everything new is **flag-gated and default-OFF** so the worker is unchanged unless
enabled; reuse `app/swimxyz_adapter.py` mappings and `keypoint_errors`; add unit
tests; standard-library/Apache-2.0 deps only (no AGPL); no over-claiming wording
("AI-assisted / coach-approved / draft / estimate", never "measured/accurate/true 3D");
leave work uncommitted for review.

---

## Phase 1 — Measure the truth 🌍 + 🛠  (do this FIRST)

You have never measured the current model on real swim footage. Everything else is
guesswork until you do. Code the harness, then run it for real.

**🛠 Codex (worker repo):**
```
Turn scripts/measure_pose_baseline.py into a complete runner: given a folder of a
SwimXYZ sequence (frames + ground-truth joints), build ground-truth pose_results
via joints_to_pose_results, run app.pose_backends.run_pose_estimation_backend, and
write a dated report to baseline_reports/ (markdown + JSON) with overall and
per-joint mean_error, PCK@0.05, and recall. Add a --compare flag to score two
backends side by side. Unit-test the report maths on synthetic data. Do not need
the real dataset to run the test.
```
**🌍 You:** download one stroke's SwimXYZ labels, run it, paste the number into
`BASELINE_EVALUATION.md`. That number is the thing every later phase must beat.

## Phase 2 — Better eyes: swim-specific pose 🌍 + 🛠

The #1 technical ceiling. The fine-tune path exists; it has never been run.

**🛠 Codex (worker repo):**
```
Add scripts/eval_backends_against_truth.py that runs mediapipe vs the onnx backend
on the same labelled clips and prints the delta in mean_error / PCK / recall, plus a
keypoint-overlay sanity image for the first N frames (so coordinate-mapping bugs are
caught). Harden scripts/swimxyz_to_mmpose.py and configs/rtmpose-m_swimxyz.py. Keep
POSE_BACKEND default mediapipe. Tests on synthetic pose only.
```
**🌍 You:** rent a GPU, run the training + ONNX export from FINE_TUNE_POSE_PLAN.md,
drop the model in, and confirm Phase-1's number actually improves.

## Phase 3 — From symptoms to understanding 🛠  (the product frontier)

Today it flags symptoms ("hips low"). Greatness is understanding the stroke.

**🛠 Codex (worker repo):**
```
Add app/stroke_cycles.py: detect stroke cycles and phases from keypoint kinematics
(hip/wrist vertical + horizontal periodicity), returning per-cycle phase segments
with timestamps. Add app/technique_reference.py: compare a swimmer's per-phase
metrics against a configurable "good-technique" reference band (JSON), producing
phase-aware context for findings instead of single-frame thresholds. Flag-gated
(PHASE_ANALYSIS), default off; when off, findings behave exactly as now. Pure numpy,
fully unit-tested with synthetic strokes (including a clean stroke and an injected
fault).
```
**🌍 You (later):** the reference bands should eventually come from real coach-graded
clips, not guesses — see Phase 5.

## Phase 4 — Real scale, honest metrics 🛠

Drag/velocity are monocular estimates today. A known distance makes them real.

**🛠 Codex (worker repo):**
```
Add app/calibration.py: accept a known real-world distance between two marked image
points (e.g. lane-rope spacing) to derive metres-per-pixel, and let analyse_clip use
it when present so velocity/drag are calibrated rather than body-scale estimates.
Label outputs "calibrated" vs "estimated" in the payload. Flag-gated; default keeps
today's estimate behaviour. Unit-tested.
```

## Phase 5 — The coaching brain & data flywheel 🌍 + 🛠  (the moat)

This is the revolutionary part — and the data only exists if real coaches use it.

**🛠 Codex (app repo + worker):**
```
Build the flywheel MACHINERY: capture every coach edit / approve / reject and the
exact wording they use, as a structured, consented, anonymised training signal
(extend the existing findings/ai_finding_feedback tables — do not store swimmer PII
or footage). Add a pipeline that aggregates coach phrasing into improved cue text and
suggests threshold adjustments, and an export of an anonymised "coach-judgment
dataset" for future model training. Only include records where consent + anonymisation
are satisfied. Tests + privacy checks (no PII, no footage paths) included.
```
**🌍 You:** the dataset is empty until real coaches use the tool. This phase's value
is created by adoption, not by the code. Codex builds the bucket; coaches fill it.

## Phase 6 — Reliability at scale 🛠

**🛠 Codex (worker + app):**
```
Harden production: finish the durable job queue, add health/metrics endpoints,
graceful degradation + manual-review fallback on any model error, idempotent retries,
and a per-clip latency budget with logging/alerts when exceeded. No new public
contract; existing tests stay green.
```

## Phase 7 — Trust & transparency 🛠  (trust is what makes it revolutionary)

**🛠 Codex (app repo):**
```
Surface honesty in the report/UI: per-finding confidence, keypoint/evidence overlays
on the key frames a finding cites, plain uncertainty language, and the
draft/coach-approved framing on every AI element. Add a "why this finding" evidence
panel. No accuracy/measurement claims anywhere.
```

## Phase 8 — Validate in the world 🌍  (the only test that matters — NO Codex)

There is no prompt for this phase, on purpose. Run the pilot with real coaches on
real (consented) footage. Measure accuracy on *real* clips (not synthetic), measure
whether a coach uses it again unprompted, and start the Phase-5 flywheel. This is the
phase that decides whether any of the above was worth it. Code cannot do it.

---

## The honest summary

Phases 1, 3, 4, 6, 7 are mostly Codex — they make the server **capable** of being
great. Phases 2, 5, and 8 have hard real-world dependencies — they're what actually
**make** it great, and Codex can only build the scaffolding around them. If you do
every Codex phase and skip the 🌍 ones, you'll have an immaculate, well-tested system
that no one has proven anyone wants. Do Phase 1 and Phase 8 first and last; let the
real world tell you which middle phases are worth the effort.
