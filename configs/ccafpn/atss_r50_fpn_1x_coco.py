_base_ = [
    # '../_base_/datasets/coco_detection.py',
    '../_base_/datasets/cocoT_detection.py',  # 真正的COCO
    # '../_base_/datasets/underwater_detection.py',
    # '../_base_/datasets/watervoice_detection.py',
    # '../_base_/schedules/schedule_2x.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
# fp16 = dict(loss_scale='dynamic') # 测试加速用的,也能夠減少大概一半训练时占的显存
# fp16 = dict(loss_scale=512.)
model = dict(
    # type='AugSingleStageDetector',
    type='ATSS',
    backbone=dict(
        type='ResNet',
        depth=50,  ## 修改网络时,深度也要跟着改
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    #     init_cfg=dict(type='Pretrained',checkpoint='torchvision://resnet101')),
    # backbone=dict(
    #             type='AResNet',
    #             frozen_stages=1,
    #             filter_size=4,
    #             norm_eval=True,
    #             init_cfg=dict(type='Pretrained', checkpoint='/home/yzy/.cache/torch/hub/checkpoints/resnet50_lpf4_finetune-cad66808.pth')),
                #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # backbone=dict(
    #     type='ResNeXt',
    #     depth=101,
    #     groups=32,
    #     base_width=4,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     norm_eval=True,
    #     style='pytorch',
    #     init_cfg=dict(
    #         type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
    # backbone=dict(
    #         type='MobileNetV2',
    #         out_indices=(2, 4, 6, 7),   # 32,96,320,1280
    #         act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    #         init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='CFPN',    #
        # in_channels=[32, 96, 320, 1280],
        # out_channels=96,
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,  #
        add_extra_convs='on_output',
        # train_with_auxiliary=True,  # xiugai
        num_outs=5),
    # neck=[dict(
    #         type='SIFPN',
    #         in_channels=[256, 512, 1024, 2048],
    #         out_channels=256,
    #         start_level=1,
    #         add_extra_convs='on_output',
    #         num_outs=5),
    #         dict(
    #             type='CBFP',
    #             in_channels=256,
    #             num_levels=5,
    #             refine_level=2,     #原论文代码
    #             refine_type='non_local')],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=80, # 类别数量
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
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# learning policy
# lr_config = dict(step=[16, 22])
# runner = dict(type='EpochBasedRunner', max_epochs=24)
# optimizer
checkpoint_config = dict(interval=1,max_keep_ckpts=3)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)   #
# work_dir = './work_dirs/underwater/atss_r50_fpn_2x_coco/'
# resume_from = 'work_dirs/atss_r50_fpn_1x_coco/epoch_9.pth'