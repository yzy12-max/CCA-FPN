_base_ = [
     '../_base_/datasets/coco_detection.py',
    #'../_base_/datasets/cocoT_detection.py',
    # '../_base_/datasets/underwater_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# fp16 = dict(loss_scale='dynamic') # 测试加速用的,也能夠減少大概一半训练时占的显存
# fp16 = dict(loss_scale=512.)
model = dict(
    type='AugSingleStageDetector',
    # type='ATSS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        # strides=(1, 2, 2, 1),
        # dilations=(1, 1, 1, 3),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        # init_cfg=dict(type='Pretrained',checkpoint='torchvision://resnet101')),
    # backbone=dict(
    #         type='MobileNetV2',
    #         out_indices=(1, 2, 4, 7),
    #         act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    #         init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='AugFPN',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,  #
        add_extra_convs='on_output',
        train_with_auxiliary=True,  # xiugai
        num_outs=5),
    # neck=[dict(
    #         type='SIFPN',
    #         in_channels=[256, 512, 1024, 2048],
    #         out_channels=256,
    #         start_level=1,
    #         add_extra_convs='on_output',
    #         num_outs=5),
    #         dict(
    #             type='BFP',
    #             in_channels=256,
    #             num_levels=5,
    #             refine_level=2,     #原论文代码
    #             refine_type='non_local')],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=4, #
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),  # [8, 16, 32, 64, 128]
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
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
                        debug=False))
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
work_dir = './work_dirs/underwater/atss_r50_augfpn_1x_coco/'
# resume_from = 'work_dirs/atss_r50_fpn_1x_coco/epoch_7.pth'