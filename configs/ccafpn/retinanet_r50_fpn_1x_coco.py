_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/cocoT_detection.py',
    '../_base_/default_runtime.py'
]
# fp16 settings
# fp16 = dict(loss_scale=512.)
model = dict(neck=dict(type='SIFPN'),bbox_head=dict(num_classes=4))
# optimizer   因为只有两个GPU，修改学习率 0.01/4=0.0025, batch = 16
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
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
work_dir = './work_dirs/underwater/retinanet_r50_fpn_1x_coco/'
# load_from = '/data1/weight/dete'  # 为什么YOLOV5效果要好,使用的COCO数据集预训练的权重
# resume_from = '/data1/yzycode/mmdetection/work_dirs/retinanet_r50_fpn_1x_coco/epoch_11.pth'