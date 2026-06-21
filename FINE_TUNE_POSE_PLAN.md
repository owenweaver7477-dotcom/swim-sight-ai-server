# Fine-tune a swim-specific pose model (ViTPose / RTMPose on SwimXYZ)

**Goal:** evaluate a swimming-specific pose model against the current MediaPipe
baseline, behind a `POSE_BACKEND` flag, so any later swap is reversible and
supported by internal evidence.

**Why this is the real upgrade:** MediaPipe is trained on land humans and
struggles with the aquatic environment. SwimXYZ exists precisely to fix the
"no labelled swim data" gap, and its authors reported a fine-tuned ViTPose on
their synthetic swimming dataset. The candidate still needs independent overlay
inspection and representative evaluation before production use.

**Licensing (hand these names to your lawyer):**
- ViTPose, MMPose, MMDeploy, ONNX Runtime, MediaPipe -> **Apache-2.0** (commercial OK).
- SwimXYZ data -> **CC-BY-4.0**: commercial use allowed **with attribution** (cite below).
- Avoid **YOLOv8-Pose (AGPL-3.0)** for a closed SaaS.
- SMPL note: only relevant if you later animate **SMPL bodies** (3D avatar) from the
  motions — that needs a Meshcapade commercial licence. Training a 2D detector on
  the rendered images + joint labels does not.

---

## Phase 0 - Measure the baseline FIRST (do this before any training)

You cannot claim an upgrade without a number to beat. The adapter makes this easy.

1. Download one stroke's labels (~1.6 GB) + a few matching videos from Zenodo
   (`record/8399376` for labels; the per-stroke video records for the clips).
2. Load a sequence's ground-truth joints, convert to the worker's format:

   ```python
   from app.swimxyz_adapter import joints_to_pose_results, keypoint_errors
   truth = joints_to_pose_results(gt_joints, image_size=(W, H), fps=FPS)  # ground truth
   ```

3. Run the CURRENT detector on the same clip and score it:

   ```python
   from app.pose_backends import run_pose_estimation_backend   # POSE_BACKEND unset = mediapipe
   pred = run_pose_estimation_backend(frames)
   print(keypoint_errors(pred, truth))   # mean_error, PCK@0.05, recall
   ```

That `mean_error` / `pck_0.05` / `recall` on labelled evaluation footage is the baseline.
Record it (this is what `BASELINE_EVALUATION.md` wants).

Current prepared side-above breaststroke baseline (301 frames): mean error
`0.5081`, median error `0.4076`, PCK@0.05 `0.0000`, recall `0.2272`.

## Phase 1 - Pick the model

| Model | Expected evaluation quality | CPU speed | Fit |
| --- | --- | --- | --- |
| **RTMPose-m** | high | **fast** | best for a CPU Render box; recommended first |
| **ViTPose-base** | higher | slower | if you can run a small GPU instance |
| ViTPose-huge | highest | slow | overkill for production |

Both are Apache-2.0 and live in MMPose. Start with **RTMPose-m**: it is close to
ViTPose accuracy at a fraction of the inference cost, which matters on Render.

## Phase 2 - Data prep

- Keep COCO-17 keypoint format (SwimXYZ's 2D joints already follow it; confirm the
  exact index order against the SwimXYZ README and, if different, edit
  `COCO17_TO_WORKER` in `app/swimxyz_adapter.py`).
- Convert SwimXYZ annotations to an MMPose COCO-style JSON (`images`, `annotations`
  with `keypoints` = 17*3 [x,y,v]). Split per-stroke into train/val (e.g. 90/10),
  and hold out a few **real** web swim clips for honest qualitative eval.
- Start with ONE stroke (freestyle, side view) end-to-end before scaling to all four.

## Phase 3 - Environment

- `pip install mmpose mmcv mmengine mmdeploy` (Apache-2.0).
- GPU: fine-tuning needs one. Rent a single A10/A100 by the hour (Lambda, RunPod,
  Vast, or a Render GPU) - a freestyle-only finetune is a few GPU-hours, on the
  order of tens of dollars, not a cluster.

## Phase 4 - Train

Representative MMPose config deltas (RTMPose-m, COCO-17, swim dataset):

```python
# rtmpose-m_swimxyz.py  (inherits an MMPose rtmpose-m coco config)
_base_ = ['rtmpose-m_8xb256-420e_coco-256x192.py']
data_root = 'data/swimxyz/'
train_dataloader = dict(dataset=dict(ann_file='annotations/freestyle_train.json',
                                     data_prefix=dict(img='images/freestyle/')))
val_dataloader   = dict(dataset=dict(ann_file='annotations/freestyle_val.json',
                                     data_prefix=dict(img='images/freestyle/')))
load_from = 'https://download.openmmlab.com/.../rtmpose-m_coco.pth'  # pretrained
train_cfg = dict(max_epochs=60, val_interval=10)
optim_wrapper = dict(optimizer=dict(lr=1e-4))   # low LR for finetuning
```

### Run training on a GPU (do not run on the production worker)

Prepare the dataset locally first, then run these commands inside an MMPose GPU
environment. Replace the example paths with the checked-out repositories and
licensed local data locations:

```bash
python3 scripts/swimxyz_to_mmpose.py \
  --joints /data/swimxyz/freestyle/joints.npy \
  --images-dir /data/swimxyz/freestyle/images \
  --output-dir /data/swimxyz/annotations \
  --stroke freestyle \
  --joint-layout coco17

cd /workspace/mmpose
SWIMXYZ_DATA_ROOT=/data/swimxyz/ \
SWIMXYZ_STROKE=freestyle \
python tools/train.py \
  /workspace/swim-sight-ai-server/configs/rtmpose-m_swimxyz.py \
  --work-dir /workspace/work_dirs/rtmpose-m_swimxyz-freestyle
```

If the input array still uses raw SwimXYZ Unity screen coordinates (bottom-left
origin, y-up), add `--flip-y`. Do not add it when the array has already been
converted to image-space y-down. OpenPose COCO-18 and BODY-25 arrays can be
declared with `--joint-layout openpose_coco18` or `openpose_body25`.

Do not run training until the SwimXYZ files have been obtained under their
CC-BY-4.0 terms and the attribution below is retained.

## Phase 5 - Evaluate

- MMPose reports AP / AR on the SwimXYZ val split.
- Cross-check with YOUR metric: run the finetuned model on the held-out clips,
  `keypoint_errors(pred, truth)` against ground truth, and confirm `mean_error`
  drops and `pck_0.05` / `recall` rise vs the Phase 0 baseline.
- Inspect ground-truth/prediction overlays. Numeric improvement is not trusted
  if the visual mapping, Y axis, frame alignment, or joint names are wrong.

## Phase 6 - Export to ONNX

```bash
cd /workspace/mmdeploy
python tools/deploy.py \
  configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
  /workspace/swim-sight-ai-server/configs/rtmpose-m_swimxyz.py \
  /workspace/work_dirs/rtmpose-m_swimxyz-freestyle/best_coco_AP_epoch_*.pth \
  /data/swimxyz/freestyle/images/example.jpg \
  --work-dir /workspace/export/rtmpose-m-swimxyz \
  --device cuda:0 \
  --dump-info
```

Quantise to int8 for CPU if latency is tight. Keep the `.onnx` out of git (large);
store it on S3/Render disk and load at startup.

### Roadmap Phase 2 comparison gate

Once an exported model exists locally, compare it against MediaPipe on the same
prepared sequence:

```bash
POSE_ONNX_PATH=/models/rtmpose-m-swimxyz.onnx \
python3 scripts/eval_backends_against_truth.py \
  --sequence-dir baseline_data/breast_side_above \
  --fps 60 \
  --baseline mediapipe \
  --candidate onnx \
  --overlay-frames 5
```

The tool writes ignored JSON/Markdown reports and GT/prediction overlays under
`backend_eval_reports/`. Negative mean/median error deltas and positive
PCK/recall deltas indicate improvement. Overlay inspection is mandatory before
trusting those numbers. This is internal evaluation, not a public product claim;
coach approval remains required for findings.

## Phase 7 - Integrate behind POSE_BACKEND (default OFF)

1. `app/pose_onnx.py` provides `run_onnx_pose(frames) -> pose_results`:
   - run ONNX Runtime on each frame,
   - take the 17 COCO keypoints + scores,
   - emit the worker's landmark dict (reuse `swimxyz_adapter.COCO17_TO_WORKER` for
     names; set `visibility` from the model's per-keypoint score),
   - keep the exact same dict shape as `run_pose_estimation`.
2. The single call in `main.py` dispatches through the backend boundary:

   ```python
   # from:
   pose_results = run_pose_estimation(frames)
   # to:
   from app.pose_backends import run_pose_estimation_backend
   pose_results = run_pose_estimation_backend(frames)
   ```

   With `POSE_BACKEND` unset this is identical to today. Set `POSE_BACKEND=onnx`
   to use the swim model.

## Phase 8 - Validate in the worker

- `python3 scripts/synth_eval.py` still passes (findings/drag logic is unchanged).
- Drop free stock clips into `samples/videos/` and run
  `python3 scripts/compare_upgrade_flags.py` with `POSE_BACKEND=onnx` vs unset:
  detection rate and keypoint recall should climb on real swim footage.
- Watch Render latency. A ViT on CPU can be slow; if so use RTMPose-m + int8, or a
  small GPU instance. Keep the per-clip budget you already target.

## Effort & risk

- **Effort:** ~1-2 focused days for a freestyle-only proof (data prep + train +
  export + integrate), then repeat per stroke.
- **Main risk:** CPU inference latency on Render. Mitigations: RTMPose over ViT,
  int8 quantisation, fewer sampled frames, or a GPU instance.
- **Reversibility:** everything is behind `POSE_BACKEND`; unset = today's worker.

## Citation (required by CC-BY)

```bibtex
@inproceedings{fiche2023swimxyz,
  title={SwimXYZ: A large-scale dataset of synthetic swimming motions and videos},
  author={Fiche, Gu\'enol\'e and Sevestre, Vincent and Gonzalez-Barral, Camila
          and Leglaive, Simon and S\'eguier, Renaud},
  booktitle={Proceedings of the 16th ACM SIGGRAPH Conference on Motion,
             Interaction and Games}, pages={1--7}, year={2023}}
```
