_base_ = [
    '../_base_/datasets/cocoT_detection.py',
    # '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/watervoice_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]    #修改过
model = dict(
    type='GFL',
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
    #     backbone=dict(
    #                         type='PASAResNet',
    #                         frozen_stages=2,
    #                         filter_size=3,
    #                         groups=1,
    #                         pasa_group=2,
    #                         norm_eval=True,
    #                         init_cfg=dict(type='Pretrained', checkpoint='/home/yzy/.cache/torch/hub/checkpoints/resnet50_lpf4_finetune-cad66808.pth')),
    #                         #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
#     backbone=dict(
    #                         type='AResNet',
    #                         frozen_stages=1,
    #                         filter_size=4,
    #                         norm_eval=True,
    #                         init_cfg=dict(type='Pretrained', checkpoint='/home/yzy/.cache/torch/hub/checkpoints/resnet50_lpf4_finetune-cad66808.pth')),
    #                         #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
            type='SIFPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
    # neck=[dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     start_level=0,  #
    #     add_extra_convs='on_output',
    #     num_outs=5),
    #     dict(
    #                 type='BFP',
    #                 in_channels=256,
    #                 num_levels=5,   #跟上面匹配
    #                 refine_level=2, #这个很占显存，原来8600-8200
    #                 refine_type='non_local')], #加了100M的样子
    bbox_head=dict(
        type='GFLHead',
        num_classes=80,  #
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),  #[8,16,32,64,128]
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,       # 不同与GFV1
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        # reg_topk=4,     #
        # reg_channels=64,
        # add_mean=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# dataset settings
# dataset_type = 'UnderwaterDataset'
# #data_root = 'data/coco/'
# #data_root = '/data/yzycode/datasets/VOCdevkit/'
# data_root = '/data/datasets/train/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# # albu_train_transforms = [
# #     # dict(
# #     #     type='ShiftScaleRotate',
# #     #     shift_limit=0.0625,
# #     #     scale_limit=0.0,
# #     #     rotate_limit=0,
# #     #     interpolation=1,
# #     #     p=0.5),
# #     dict(type="CLAHE", clip_limit=4.0, tile_grid_size=(8, 8),  p=0.5),
# #     dict(type="IAAEmboss", alpha=(0.2,0.5),strength=(0.2,0.7),  p=0.5),
# #     dict(type="IAASharpen", alpha=(0.2,0.5),lightness=(0.5, 1.0),  p=0.5),
# #     dict(
# #         type='RandomBrightnessContrast',
# #         brightness_limit=0.2,
# #         contrast_limit=0.2,
# #         brightness_by_max=True,
# #         always_apply=False,
# #         p=0.5,)
# #     # dict(
# #     #     type='OneOf',
# #     #     transforms=[
# #     #         dict(
# #     #             type='RGBShift',
# #     #             r_shift_limit=10,
# #     #             g_shift_limit=10,
# #     #             b_shift_limit=10,
# #     #             p=1.0),
# #     #         dict(
# #     #             type='HueSaturationValue',
# #     #             hue_shift_limit=20,
# #     #             sat_shift_limit=30,
# #     #             val_shift_limit=20,
# #     #             p=1.0)
# #     #     ],
# #     #     p=0.1),
# #     # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
# #     # dict(type='ChannelShuffle', p=0.1),
# #     # dict(
# #     #     type='OneOf',
# #     #     transforms=[
# #     #         #dict(type='Blur', blur_limit=7, p=0.5),
# #     #         dict(type='MedianBlur', blur_limit=3, p=0.5),
# #     #         dict(type="MotionBlur", blur_limit=3,p=0.5)  #其中blur_limit是卷积核大小的范围,卷积核越大，模糊效果越明显；p是进行运动模糊操作概率。
# #     #     ],
# #     #     p=0.5),
# # ]
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     # dict(
#     #         type='Resize',
#     #         img_scale=[(2000, 1216),
#     #                    (2000, 704)],
#     #         multiscale_mode='range',
#     #         keep_ratio=True),  #多尺度訓練
#     dict(type='RandomFlip', flip_ratio=0.5),
#     #dict(type='RandomFlip', direction=['horizontal'], flip_ratio=0.5), #随机左右翻转
#     #dict(type='MotionBlur', p=0.3),  #随即运动模糊
#     #dict(type='AutoAugment', autoaug_type='v1'),
#     dict(type='Pad', size_divisor=32),      #Pad本来是在Normalize后的
#     # dict(
#     #             type='Albu',
#     #             transforms=albu_train_transforms,
#     #             bbox_params=dict(
#     #                 type='BboxParams',
#     #                 format='pascal_voc',
#     #                 label_fields=['gt_labels'],
#     #                 min_visibility=0.0,
#     #                 filter_lost_elements=True),
#     #             keymap={
#     #                 'img': 'image',
#     #                 #'gt_masks': 'masks',
#     #                 'gt_bboxes': 'bboxes'
#     #             },
#     #             update_pad_shape=False,
#     #             skip_img_without_anno=True),
#     dict(type='Normalize', **img_norm_cfg),
#
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         # type='MultiScaleFlipAug',
#         #         # 0.568
#         # img_scale=[(2000, 704), (2000, 960), (2000, 1216)],
#         # flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         #ann_file=data_root + 'annotations/instances_train2017.json',
#         #img_prefix=data_root + 'train2017/' ,
#         # ann_file=data_root + 'voc07_trainval.json',
#         # img_prefix=data_root ,
#         ann_file=data_root + 'annotations/train.json',
#         img_prefix=data_root +'train-image/' ,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         #ann_file=data_root + 'annotations/instances_val2017.json',
#         #img_prefix=data_root + 'val2017/' ,
#         # ann_file=data_root + 'voc07_test.json',
#         # img_prefix=data_root  ,
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root +'val-image/' ,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         # ann_file=data_root + 'annotations/instances_val2017.json',
#         # img_prefix=data_root + 'val2017/',
#         # ann_file=data_root + 'voc07_test.json',
#         # img_prefix=data_root ,
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root +'val-image/' ,
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict(_delete_=True,grad_clip=dict(max_norm=35, norm_type=2))  #进行梯度裁减

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# resume_from = '/data1/yzycode/mmdetection/work_dirs/gfl_r50_fpn_1x_coco/epoch_11.pth'
# load_from = '/data1/yzycode/mmdetection/work_dirs/gfl_r50_fpn_1x_coco/epoch_12_SIFPNCOCO.pth'