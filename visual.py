
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.backbones.antialiased_resnet50 import AResNet
from mmdet.models.necks.sie_fpn import SFPN
from mmdet.models.necks.sie1_fpn import SIFPN
from mmdet.models.necks.fpn import FPN
from mmdet.models.necks.pafpn import PAFPN
from mmdet.models.necks.high_fpn import HFPN
from mmdet.models.necks.myfpn import MFPN
from mmdet.models.necks.ct_resnet_neck import CTResNetNeck
import torch
# import mmcv
import matplotlib.pyplot as plt

# import netron
# netron.start('work_dirs/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco.onnx')
#训练
# from mmcv import Config
# from mmdet.models import build_detector
# cfg = Config.fromfile('configs/pascal_voc/fcos_r50_fpn_1x_voc.py')
# train_cfg = build_detector(cfg.model,train_cfg=cfg.get('train_cfg'),
#         test_cfg=cfg.get('test_cfg'))
# from mmdet.datasets import build_dataset
# datasets = [build_dataset(cfg.data.train)]  #里面的列表竟然可以计算数据集个类别的数目。
# b = datasets[0]
# backbone = AResNet(
#     init_cfg=dict(type='Pretrained', checkpoint='/home/yzy/.cache/torch/hub/checkpoints/resnet50_lpf4_finetune-cad66808.pth')
# )
from mmdet.models.backbones.pasa_res50 import PASAResNet
# backbone=PASAResNet( frozen_stages=1,
#                     filter_size=3,
#                     norm_eval=True,
#                     #init_cfg=dict(type='Pretrained', checkpoint='/home/yzy/.cache/torch/hub/checkpoints/resnet50_lpf4_finetune-cad66808.pth')),
#                     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
from mmdet.models.backbones.mobilenet_v2 import MobileNetV2
# backbone = MobileNetV2(
#             out_indices=(2, 4, 6),
#             act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
#             init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2'))

backbone = ResNet(
        depth=50,
        num_stages=4,
        # strides=(1, 2, 2, 2),
        # dilations=(1, 1, 1, 1),
        # dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, False, False, True),
        out_indices=(0, 1, 2, 3),
        # out_indices=(2,3,4,5),
        frozen_stages=1,
        # dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        # plugins=[
        #     dict(
        #         cfg=dict(type='ContextBlock', ratio=1. / 4),
        #         stages=(False, True, True, True),
        #         position='after_conv3')
        # ]
                  )
from mmdet.models.backbones.darknet import Darknet
# backbone = Darknet(
#         depth=53,
#         out_indices=(3, 4, 5),
#         init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53'))
# from mmdet.models.backbones.efficientnet import EfficientNet
# backbone = EfficientNet(arch='b0',drop_path_rate=0.2,out_indices=(3,4,5))
imgpath = '/data/yzycode/mmdetection/work_dirs/000009.jpg'
# img = plt.imread(imgpath)
# imgtensor = torch.Tensor(img).unsqueeze(0).permute(0,3,1,2)
# a = train_cfg.backbone(imgtensor)  #train_cfg就相当于FCOS这个大类，找到fcos.py文件，看那些属性可以调用。
# out0 = train_cfg.neck(a)
#数据预处理
# from mmcv import Config
# import numpy as np
from mmdet.datasets.pipelines.compose import Compose
# cfg = Config.fromfile('configs/pascal_voc/atss_r50_fpn_1x_coco.py')
# if isinstance(img, np.ndarray): #图像的DEMO里可以
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
# test_pipeline = Compose(cfg.data.test.pipeline)
test_pipeline = Compose(train_pipeline)
# datas = []
# for img in [imgpath]:
#     # prepare data
#     if isinstance(img, np.ndarray):
#         # directly add img
#         data = dict(img=img)
#     else:
#         # add information into dict
#         data = dict(img_info=dict(filename=img), img_prefix=None)
#     # build the data pipeline
#     data = test_pipeline(data)
#     datas.append(data)
data = dict(img_info=dict(filename=imgpath), img_prefix=None)
data = test_pipeline(data)
# datas.append(data)
# dataimg = data.data[0]
# dataimg = dataimg.to('cuda:0')
# 单张图片不用，这个是用多张图片的
# from mmcv.parallel import collate
# data = collate(datas, samples_per_gpu=1)
# just get the actual data from DataContainer
img_metas = data['img_metas'].data
img = data['img'].data
imgtensor = img.unsqueeze(0).cuda()
backbone.cuda()
inputs = backbone(imgtensor)
in_channels = [256,512,1024,2048]
from mmdet.models.necks.CAM_fpn import CFPN
from mmdet.models.necks.cnn_fpn import CNFPN
from mmdet.models.necks.yolo_neck import YOLOV3Neck
from mmdet.models.necks.pacfpn import PACFPN
neck =MFPN(in_channels=in_channels,out_channels=256,num_outs=5,start_level=1,add_extra_convs='on_output'#,relu_before_extra_convs=True
           )
out = neck(inputs,img_metas)






