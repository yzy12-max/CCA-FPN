_base_ = [
    '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/cocoT_detection.py',
    # '../_base_/datasets/underwater_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]    #修改过，解析：https://zhuanlan.zhihu.com/p/358056615
model = dict(
    type='AugSingleStageDetector',
    # type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        #depth=101,
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
                add_extra_convs='on_output',
                train_with_auxiliary=True,
                num_outs=5),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=4, #
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
        # loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),   #
        loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
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
        nms=dict(type='nms', iou_threshold=0.6),    # 0.5
        max_per_img=100))

# img_norm_cfg = dict(   #不同于BASE，这些都是以_base_里的为准,这个是caffe格式的图片均值和方差
#     mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
# optimizer
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer 学习率转换https://blog.csdn.net/qq_17403617/article/details/117263557
optimizer = dict( #修改前0.01
    lr=0.0025, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(     #将优化参数设置为3组，卷积 bias、BN 和卷积权重
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2)) # 梯度均衡参数
optimizer_config = dict(_delete_=True, grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,  # 起始的学习率
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir = './work_dirs/underwater/fcos_r50_augfpn_1x_coco/'
# resume_from = '/home/yzy/data/yzycode/mmdetection/work_dirs/fcos_r50_fpn_1x_coco/epoch_5.pth'