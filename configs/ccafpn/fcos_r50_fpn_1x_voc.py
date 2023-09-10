_base_ = [
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]    #修改过，解析：https://zhuanlan.zhihu.com/p/358056615
model = dict(    #4 个卷积层不是采用 BN，而是 GN，实验表明在 FCOS 中，GN 效果远远好于 BN
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True), #resnet网络的2,3,4层加入可变形卷集
        norm_cfg=dict(type='BN', requires_grad=True), #pytorch设置维TRUE
        norm_eval=True,
        style='pytorch', #'torchvision://resnet50'下载连接在/home/yzy/.conda/envs/open-mmlab2/lib/python3.7/site-packages/torchvision/models/resnt
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        # init_cfg=dict(type='Pretrained', checkpoint='/home/yzy/data/yzycode/premodels/resnet50_msra-5891d200.pth')),
    neck=dict(
        type='SIFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,#[256,512,256], #自己修改了输出，FPN代码里的输出都要进行修改进行匹配。本来只有256
        num_outs=5,
        start_level=1, #使用C2层，要匹配上面的输出层数num_outs=6，也要进行下面步长strides和限制范围的修改。
        add_extra_convs='on_output',
        relu_before_extra_convs=True, #3X3卷集后是否RELU
        ),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=4,  #
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        center_sampling=True,
        center_sample_radius=1.5,
        centerness_on_reg=True,
        # dcn_on_last_conv=True, #顶层加入可变形卷集
        # conv_bias=True,
        strides=[8, 16, 32, 64, 128], #根据特征图大小进行修改，原来是[8, 16, 32, 64,128]
        loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        #loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    # train_cfg=dict(
    #     assigner=dict(type='ATSSAssigner', topk=9),
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    train_cfg=dict(     #这个在FCOS有什么用,感觉去掉也可以
        assigner=dict(
            type='MaxIoUAssigner',   #正样本定义
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1, #
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
# optimizer 学习率转换https://blog.csdn.net/qq_17403617/article/details/117263557
#計算：基基礎--0.00125XGPU數量X每張GPU放的图片数量
optimizer = dict( #修改前0.01
    lr=0.0025, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(     #将优化参数设置为3组，卷积 bias、BN 和卷积权重
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2)) # 梯度均衡参数
    #_delete_=True, grad_clip=None)    #没裁减梯度，训练不太稳定，南收敛,可以减小学习率试试
# learning policy
lr_config = dict(
    policy='step',
    #warmup='constant',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# 我们采用 COCO 预训练过的 Cascade Mask R-CNN R50 模型权重作为初始化权重，可以得到更加稳定的性能
#load_from = '/home/yzy/data/yzycode/mmdetection/work_dirs/fcos_r50_fpn_1x_voc/epoch_15.pth'
#resume_from = '/home/yzy/data/yzycode/mmdetection/work_dirs/fcos_r50_fpn_1x_voc/epoch_5.pth'

