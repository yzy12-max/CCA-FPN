_base_ = [
    # '../_base_/models/retinanet_r50_fpn.py',
    # '../_base_/datasets/cocoT_detection.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]
# fp16 = dict(loss_scale=512.)
# model settings:从AugFPN复制了CORE1,roi_head,box_head,看__init__可以看出
model = dict(
    type='AugSingleStageDetector',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='AugFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        train_with_auxiliary=True,  #
        num_outs=5),
    bbox_head=dict(
        type='AugRetinaHead',
        num_classes=4, #bg in还要去修改,在检测器里BUILD的头
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        #norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        smoothl1_beta=0.11,
        gamma=2.0,
        alpha=0.25,
        allowed_border=-1,
        pos_weight=-1,
        use_consistent_supervision=True,
        debug=False,
        auxiliary=dict(
            fpn_type=True,
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            proposal=dict(
                nms_pre=2000,
                nms_post=2000,
                max_num=2000,
                score_thr=0.01,
                min_bbox_size=0,
                nms=dict(type='nms', iou_thr=0.7),
                max_per_img=1000),
            rcnn=dict(
                pos_weight=-1,
                debug=False)
        )),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# optimizer   因为只有两个GPU，修改学习率 0.01/4=0.0025
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = './work_dirs/underwater/retinanet_r50_augfpn_1x_coco/'
# resume_from = '/data1/yzycode/mmdetection/work_dirs/retinanet_r50_augfpn_1x_coco/epoch_5.pth'

