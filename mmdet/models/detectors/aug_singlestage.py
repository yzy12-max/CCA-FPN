# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
#
from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        bbox2roi, multi_apply, reduce_mean, unmap)

from mmdet.core1 import (weighted_cross_entropy, weighted_smoothl1)
from .. import builder


@DETECTORS.register_module()
class AugSingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 use_consistent_supervision=True,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(AugSingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        #
        self.use_consistent_supervision = use_consistent_supervision
        self.num_classes = 4      # 记得修改
        # self.point = True
        # from mmdet.core.anchor.point_generator import MlvlPointGenerator
        # self.prior_generator = MlvlPointGenerator((8, 16, 32))
        if self.use_consistent_supervision:
            bbox_roi_extractor = dict(
                type='AuxAllLevelRoIExtractor',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                out_channels=256,
                featmap_strides=[8, 16, 32])  # only apply to feature map belonging to FPN

            self.auxiliary_bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            # bbox_head = dict(
            #     type='Shared2FCBBoxHead',   # 修改
            #     num_shared_fcs=2,
            #     in_channels=256,
            #     fc_out_channels=1024,
            #     roi_feat_size=7,
            #     num_classes=4,
            #     target_means=[0., 0., 0., 0.],
            #     target_stds=[0.1, 0.1, 0.2, 0.2],
            #     reg_class_agnostic=False)
            auxiliary_bbox_head = dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=self.num_classes,   #还有上面的self.num_classes = 80,记得修改
                bbox_coder=dict(          #
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                # bbox_coder=dict(type='DistancePointBBoxCoder'),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))

            self.auxiliary_bbox_head = builder.build_head(auxiliary_bbox_head)



    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(AugSingleStageDetector, self).forward_train(img, img_metas)
        if self.use_consistent_supervision:  # 使用
            x, y = self.extract_feat(img)
            gt_bboxes_auxiliary = [gt.clone() for gt in gt_bboxes]  # 复制GT_BOX:[N,4]
            gt_labels_auxiliary = [label.clone() for label in gt_labels]
        else:
            x = self.extract_feat(img)
        outs = self.bbox_head(x)  # pre_cls,box
        loss_inputs = outs + (
        gt_bboxes, gt_labels, img_metas, self.train_cfg)  # 预测输出+(gt_bboxes, gt_labels, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs)
        if self.use_consistent_supervision:
            proposal_cfg = self.train_cfg.auxiliary.proposal  # 这样调用配置文件的配置属性
            if len(outs)==2:
                proposal_inputs = outs + (None,img_metas, proposal_cfg)  # 回归两个预测值的
            else:
                proposal_inputs = outs + (img_metas, proposal_cfg)
            proposal_list= self.bbox_head.get_bboxes(*proposal_inputs)  # 后处理 [1000,5]预测的TOPK分数还原成BOX，作为侯选框，1000个
            # proposal_list = [proposal_list[0][0],proposal_list[1][0]]   # 这里自己为了适应mmdet2.19,要返回列表的第1个值,不用返回Label
            proposals = []
            for proposal in proposal_list:
                proposals.append(proposal[0])
            proposal_list = proposals
            bbox_assigner = build_assigner(self.train_cfg.auxiliary.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.auxiliary.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(  # 预测框与GT框最大IOU分配，返回1000个总的正负样本
                    proposal_list[i], gt_bboxes_auxiliary[i], gt_bboxes_ignore[i],
                    gt_labels_auxiliary[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,      #
                    proposal_list[i],  #
                    gt_bboxes_auxiliary[i],
                    gt_labels_auxiliary[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            rois = bbox2roi([res.bboxes for res in sampling_results])  # y:表示1X1卷集后的三个卷基层
            bbox_feats_raw = self.auxiliary_bbox_roi_extractor(y[:self.auxiliary_bbox_roi_extractor.num_inputs],
                                                               rois)  # ROI
            cls_score_auxiliary, bbox_pred_auxiliary = self.auxiliary_bbox_head(bbox_feats_raw)  # 全链接
            # super(当前类的名字Animal,self).父类的方法名()
            # if self.point:
            #     # assert len(cls_score_auxiliary) == len(bbox_pred_auxiliary) == len(y)
            #     featmap_sizes = [featmap.size()[-2:] for featmap in y]
            #     all_level_points = self.prior_generator.grid_priors(  # 点的生成 self.prior_generator = MlvlPointGenerator(strides)
            #         featmap_sizes,
            #         dtype=bbox_pred_auxiliary[0].dtype,
            #         device=bbox_pred_auxiliary[0].device)
            bbox_targets = self.auxiliary_bbox_head.get_targets(
                sampling_results, gt_bboxes, gt_labels,     # gt_bboxes, gt_labels传进去,没什么用
                self.train_cfg.auxiliary.rcnn)  # 返回【label,label_weight,box,box_weight】
            # [512,],[512,],[512,4*81],[512,4*81]
            # loss_bbox_auxiliary = self.auxiliary_bbox_head.loss(cls_score_auxiliary, bbox_pred_auxiliary,rois,
            #                                                     *bbox_targets)  #
            num_level = len(y)
            losses_roi = dict()
            for i in range(num_level):
                labels = bbox_targets[0]
                label_weight = bbox_targets[1]
                box_target = bbox_targets[2]
                box_weight = bbox_targets[3]
                cls_score_level_i = cls_score_auxiliary[i::num_level, :]  # 每层的:[512,]
                bbox_pred_level_i = bbox_pred_auxiliary[i::num_level, :]  #
                losses_roi['loss_cls_level%d' % i] = weighted_cross_entropy(cls_score_level_i, labels,label_weight,
                                                                        reduce=True) * 0.25
                bbox_pred = bbox_pred_level_i
                if bbox_pred_auxiliary is not None:
                    bg_class_ind = self.num_classes  # 背景类别号
                    # 0~self.num_classes-1 are FG, self.num_classes is BG
                    pos_inds = (labels >= 0) & (labels < bg_class_ind)
                    # do not perform bounding box regression for BG anymore.
                    if pos_inds.any():
                        pos_bbox_pred = bbox_pred.view(  # [1024,4*num_class]-[1024,4,4]-[p,4]
                            bbox_pred.size(0), -1,
                            4)[pos_inds.type(torch.bool),  # [1024,4,4][p,label]-[p,4]
                               labels[pos_inds.type(torch.bool)]]
                losses_roi['loss_reg_level%d' % i] = weighted_smoothl1(pos_bbox_pred, box_target[pos_inds.type(torch.bool)],
                                             box_weight[pos_inds.type(torch.bool)],avg_factor=bbox_targets[2].size(0)) * 0.25


            losses.update(losses_roi)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # feat = self.extract_feat(img)   #FPN的输出
        # results_list = self.bbox_head.simple_test(
        #     feat, img_metas, rescale=rescale) #
        # bbox_results = [
        #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
        #     for det_bboxes, det_labels in results_list  #就是将LABEL值与预测的det_bboxes进行匹配，返回0-CLASSES排好顺序的BOX
        # ]
        # return bbox_results

        if self.use_consistent_supervision:
            x, y = self.extract_feat(img)
        else:
            x = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            x, img_metas, rescale=rescale)  #
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list  # 就是将LABEL值与预测的det_bboxes进行匹配，返回0-CLASSES排好顺序的BOX
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.
        #测试时增强,这里会为原始图像造出多个不同版本，包括不同区域裁剪和更改缩放程度等，并将它们输入到模型中；然后对多个版本进行计算得到平均输出，
        作为图像的最终输出分数.可将准确率提高若干个百分点.
        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels

