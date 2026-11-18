# mmdetection 3.x config of the released Cascade R-CNN R101-FPN character detector (rcnn-res101-5-10.pth).
# Single class, anchors sized for small characters, random 512 px crops of pages resized to 1024 px, 38 epochs at
# batch size 1 on one Tesla T4. Train with
#     python mmdetection/tools/train.py cascade_rcnn_r101.py --work-dir runs/cascade
# after writing train.json / val.json with convert_to_coco.py and setting data_root to the CODH folder.
_base_ = "mmdetection/configs/cascade_rcnn/cascade-rcnn_r101_fpn_1x_coco.py"

CLASSES = ["character"]
data_root = "./"                       # folder that holds train.json, val.json and the CODH book folders
total_epochs = 38

bbox_head = lambda stds: dict(         # noqa: E731
    type="Shared2FCBBoxHead", in_channels=256, fc_out_channels=1024, roi_feat_size=7, num_classes=len(CLASSES),
    bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=stds),
    reg_class_agnostic=True,
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0))

model = dict(
    rpn_head=dict(anchor_generator=dict(
        type="AnchorGenerator", scales=[1], ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64], base_sizes=[4, 8, 16, 32, 64])),
    roi_head=dict(bbox_head=[
        bbox_head([0.1, 0.1, 0.2, 0.2]),
        bbox_head([0.05, 0.05, 0.1, 0.1]),
        bbox_head([0.033, 0.033, 0.067, 0.067]),
    ]),
    test_cfg=dict(
        rpn=dict(nms_pre=5000, max_per_img=5000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.2), max_per_img=5000)),
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(512, 512)),
    dict(type="Expand", ratio_range=(1, 2)),
    dict(type="PhotoMetricDistortion"),
    dict(type="Corrupt", corruption="defocus_blur"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=False),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

metainfo = {"classes": CLASSES}
train_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file="train.json",
                 data_prefix=dict(img=""), pipeline=train_pipeline))
val_dataloader = dict(
    dataset=dict(data_root=data_root, metainfo=metainfo, ann_file="val.json",
                 data_prefix=dict(img=""), pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + "val.json")
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=total_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=5e-5), clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=total_epochs, by_epoch=True,
         milestones=[2 * total_epochs // 3, total_epochs - 2], gamma=0.1),
]
# To continue from the released weights instead of the COCO-pretrained base:
# load_from = "rcnn-res101-5-10.pth"
