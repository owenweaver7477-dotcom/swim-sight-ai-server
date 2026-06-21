"""RTMPose-m COCO-17 fine-tune config for prepared SwimXYZ data.

Training/export configuration only. The production worker never imports this
file. Before a real GPU run, prepare licensed images/COCO annotations, set the
environment variables below, visually inspect label overlays, and choose an
explicit pretrained checkpoint. No dataset or checkpoint is downloaded here.

SwimXYZ attribution: Fiche et al., "SwimXYZ", ACM MIG 2023 (CC-BY-4.0).
MMPose/MMEngine/MMCV/MMDeploy/RTMPose are Apache-2.0 projects.
"""
import os


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


_base_ = [
    "mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"
]

# Required before training: point these at locally prepared, licensed data.
# Defaults are repository-relative placeholders and are not expected to exist
# in a clean checkout.
data_root = os.getenv("SWIMXYZ_DATA_ROOT", "data/swimxyz/")
stroke = os.getenv("SWIMXYZ_STROKE", "freestyle")
train_ann_file = os.getenv("SWIMXYZ_TRAIN_ANN", f"annotations/{stroke}_train.json")
val_ann_file = os.getenv("SWIMXYZ_VAL_ANN", f"annotations/{stroke}_val.json")
test_ann_file = os.getenv("SWIMXYZ_TEST_ANN", val_ann_file)
train_image_prefix = os.getenv("SWIMXYZ_TRAIN_IMAGE_PREFIX", f"images/{stroke}/")
val_image_prefix = os.getenv("SWIMXYZ_VAL_IMAGE_PREFIX", train_image_prefix)
test_image_prefix = os.getenv("SWIMXYZ_TEST_IMAGE_PREFIX", val_image_prefix)

# Required decision before training: use a verified Apache-2.0-compatible
# RTMPose-m checkpoint. The URL is the upstream MMPose pretrained starting
# point, not a swim-specific model and not a production artifact.
load_from = os.getenv(
    "RTMPOSE_PRETRAINED",
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
)

# Generated checkpoints and logs stay under the ignored work_dirs/ tree.
work_dir = os.getenv("RTMPOSE_WORK_DIR", f"work_dirs/rtmpose-m-swimxyz-{stroke}")
resume = os.getenv("RTMPOSE_RESUME", "false").strip().lower() in {"1", "true", "yes", "on"}

train_cfg = dict(
    max_epochs=_env_int("RTMPOSE_MAX_EPOCHS", 60),
    val_interval=_env_int("RTMPOSE_VAL_INTERVAL", 10),
)
optim_wrapper = dict(
    optimizer=dict(lr=_env_float("RTMPOSE_LEARNING_RATE", 1e-4))
)

train_dataloader = dict(
    batch_size=_env_int("RTMPOSE_TRAIN_BATCH_SIZE", 32),
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_image_prefix),
    ),
)
val_dataloader = dict(
    batch_size=_env_int("RTMPOSE_EVAL_BATCH_SIZE", 32),
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_image_prefix),
    ),
)
test_dataloader = dict(
    batch_size=_env_int("RTMPOSE_EVAL_BATCH_SIZE", 32),
    dataset=dict(
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(img=test_image_prefix),
    ),
)

val_evaluator = dict(ann_file=os.path.join(data_root, val_ann_file))
test_evaluator = dict(ann_file=os.path.join(data_root, test_ann_file))

# Save validation-selected checkpoints for later evaluation/export. A checkpoint
# must still beat the Phase 1 baseline and pass overlay inspection before ONNX
# export or any separate production enablement decision.
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=_env_int("RTMPOSE_CHECKPOINT_INTERVAL", 10),
        save_best="coco/AP",
        rule="greater",
        max_keep_ckpts=_env_int("RTMPOSE_MAX_CHECKPOINTS", 3),
    )
)
