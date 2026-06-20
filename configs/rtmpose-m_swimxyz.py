"""RTMPose-m COCO-17 fine-tune config for local SwimXYZ data.

This config is intentionally parameterised through environment variables so no
private dataset path is committed. It is a training configuration only and is
not imported by the production worker.
"""
import os

_base_ = [
    "mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"
]

data_root = os.getenv("SWIMXYZ_DATA_ROOT", "data/swimxyz/")
stroke = os.getenv("SWIMXYZ_STROKE", "freestyle")
image_prefix = os.getenv("SWIMXYZ_IMAGE_PREFIX", f"images/{stroke}/")

train_ann_file = os.getenv(
    "SWIMXYZ_TRAIN_ANN", f"annotations/{stroke}_train.json"
)
val_ann_file = os.getenv(
    "SWIMXYZ_VAL_ANN", f"annotations/{stroke}_val.json"
)

load_from = os.getenv(
    "RTMPOSE_PRETRAINED",
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
)

train_cfg = dict(max_epochs=60, val_interval=10)
optim_wrapper = dict(optimizer=dict(lr=1e-4))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=image_prefix),
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=image_prefix),
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=os.path.join(data_root, val_ann_file),
)
test_evaluator = val_evaluator
