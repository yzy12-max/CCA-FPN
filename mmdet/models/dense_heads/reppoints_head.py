# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d

from mmdet.core import (build_assigner, build_sampler, images_to_levels,
                        multi_apply, unmap)
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import filter_scores_and_topk
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class RepPointsHead(AnchorFreeHead):
    """RepPoint head.

    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 use_grid_points=False,
                 center_init=True,
                 transform_method='moment',
                 moment_mul=0.01,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.num_points = num_points   #9
        self.point_feat_channels = point_feat_channels  #256
        self.use_grid_points = use_grid_points  #
        self.center_init = center_init          #

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))  #3
        self.dcn_pad = int((self.dcn_kernel - 1) / 2) #pad = 1
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,   ##可变形卷积的初始化x,y偏移量
                             self.dcn_pad + 1).astype(np.float64)  #[-1.,0.,1.]，为什么要np.float64
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)         #[-1.,0.,1.]复制三次
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)             #[-1.,0.,1.]复制三次
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))  #
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)  #展平

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            init_cfg=init_cfg,
            **kwargs)

        self.gradient_mul = gradient_mul    #0.1
        self.point_base_scale = point_base_scale  #4
        self.point_strides = point_strides  #[8, 16, 32, 64, 128]
        self.prior_generator = MlvlPointGenerator(
            self.point_strides, offset=0.5)    # 这里初始化偏移量怎么给的0阿  正常应该 self.point_strides, offset=0.5


        self.sampling = loss_cls['type'] not in ['FocalLoss']  #F
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(
                self.train_cfg.refine.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.transform_method = transform_method    #'moment'
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)   #自定义学习参数
            self.moment_mul = moment_mul

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)  #
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1  #为什么这样阿
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

    def _init_layers(self):   #和FCOS不同，看他的结构图
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pts_out_dim = 4 if self.use_grid_points else 2 * self.num_points
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,       #256
                                               self.point_feat_channels, #256
                                               self.dcn_kernel, 1,      #k = 3， stride = 1
                                               self.dcn_pad)            #1
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,  #【256，256】
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)  #18
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def points2bbox(self, pts, y_first=True):
        """Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_first=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])  #空的就没有【N，9，2】
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]  #取出Y值
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]  #取出X值
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)        #对9个点的Y值求平均，变为一个平均值
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)  #求方差
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))   # #自定义self.moment_transfer参数，可以用来学习的
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                             dim=1)  #转换成矩形左上角和右下角的形式
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.

        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)  #center_init = TRUE 将float64_>float32
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2         #没用到
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale,
                                      scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))  #得到的返回值是[b,18,h,w]可变形卷集的偏移坐标
        if self.use_grid_points:                                #每个点都回归9个语义点的xy坐标
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(     #
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init    #[b,18,h,w] + 0
        # refine and classify reppoints  (1-0.1)*pts_out_init(不回传梯度)+0.1*pts_out_init
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(  #区别在那？
        ) + self.gradient_mul * pts_out_init    #初始pts_out_init分支梯度乘上系数，目的是希望降低该分支的梯度权重，使更好收敛
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset  #【b,18,h,w】利用offset预测值，减掉每个特征图上kernel所对应的9个点base坐标([-1,-1],[0,-1],...),作为dcn的offset输入值
        #应用dcn对分类分支和refine回归分支进行特征自适应，得到新的特征图，然后经过两个1x1卷积得到最终输出
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))    #256,[b,18,h,w] 可变形卷集确实是两个输入，X，offset
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))  #256，256
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:   #refine加上初始预测就可以得到refine后的输出9个点坐标   #在一阶段的基础上，再进行偏移预测，预测偏移很小，所以二阶段相加就是整个偏移梁
            pts_out_refine = pts_out_refine + pts_out_init.detach() #detach()分离出来不进行梯度更新
            #pts_out_refine = pts_out_init.detach()         #不要再生成pts_out_init的时候反响更新时，影响到pts_out_refine

        if self.training:
            return cls_out, pts_out_init, pts_out_refine
        else:
            return cls_out, self.points2bbox(pts_out_refine)  #test,返回分类分数和预测的BOX

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points.

        Only used in :class:`MaxIoUAssigner`.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)): #5
            pts_lvl = []
            for i_img in range(len(center_list)): #2
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(  #
                    1, self.num_points)    #在每层特征层上扩展成9个中心样本点（X，Y）
                pts_shift = pred_list[i_lvl][i_img]   #每张图片每层的9个偏移坐标（）
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)    #展平成【H，W，18】-【HW，18】相当于【YX，18】
                y_pts_shift = yx_pts_shift[..., 0::2]  #取坐标Y的值
                x_pts_shift = yx_pts_shift[..., 1::2]   #取坐标X的值
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1) #本来是（Y，X），变成（X，Y），因为是数组形式，行列取值
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1) #【16800，9，2】-【16800，18】
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center  #偏移*S+中心（X，Y）
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)   #在0维前面扩展一维，堆叠
            pts_list.append(pts_lvl)            #5层
        return pts_list

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample proposals #第一阶段的是【样本点（X，Y），步长（S，S）】
        proposals = flat_proposals[inside_flags, :]  #第二阶段是【转换成的BOX左上角和右下角坐标】

        if stage == 'init':
            assigner = self.init_assigner  #构造 'PointAssigner'
            pos_weight = self.train_cfg.init.pos_weight #-1
        else:  #分两个阶段进行，这个是refine阶段
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
        assign_result = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore,  #封装成一个类
                                        None if self.sampling else gt_labels)   #self.sampling=FLASE
        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes) #都有，assign_result，所有样本点，所有gt_bboxes
                                            #去除重复的样本点，封装成一个类
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long) #80为背景
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)
        #初始化，好方便正样本处有值，抚养本处对应为0或相匹配的值
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0   #抚养本权重为1，计算LOSS可能会甬道

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)

        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
                - proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
                - num_total_pos (int): Number of positive samples in all images.  # noqa: E501
                - num_total_neg (int): Number of negative samples in all images.  # noqa: E501
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels 每层的样本点
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor 合并所有层
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None: #不设置忽略样本
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             stage=stage,
             unmap_outputs=unmap_outputs)
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images 总的数据
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        proposal_weights_list = images_to_levels(all_proposal_weights,
                                                 num_level_proposals)
        return (labels_list, label_weights_list, bbox_gt_list, proposals_list,
                proposal_weights_list, num_total_pos, num_total_neg)

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, labels,
                    label_weights, bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine, stride,
                    num_total_samples_init, num_total_samples_refine):
        # classification loss  #两张图片上的数据进行汇总
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        cls_score = cls_score.contiguous()  #.contiguous()改变了存储顺序，使数字连续，通常在permute后结合使用
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_refine)
        # points loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(  #pts_pred_init是点偏移转换成点坐标的，所以不用加中心点坐标了
            pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)  #将点坐标转换成BOX
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)                  #注意现在 y_first=False
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)        #因为
        bbox_pred_refine = self.points2bbox(
            pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride  # 4*stride
        loss_pts_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=num_total_samples_init)  #smooth_l1_loss是左上角和右下角两个点的坐标
        loss_pts_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)
        return loss_cls, loss_pts_init, loss_pts_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1   #80
        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas, device)  #center_list就是像素点：(coord_x, coord_y, stride_w, stride_h)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,   #将九个点，从点偏移到点坐标的变化
                                                       pts_preds_init)  #
        if self.train_cfg.init.assigner['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list
        cls_reg_targets_init = self.get_targets(    #基本上每个HEAD都有的
            candidate_list,  #
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init #第一阶段没有LABEL targets。
        num_total_samples_init = (
            num_total_pos_init +
            num_total_neg_init if self.sampling else num_total_pos_init)  #没有使用从抚养本中采样些作为正样本的方法

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas, device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)  #将九个点的偏移转换成坐标
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):  #5ceng【2，18，H，W】
                bbox_preds_init = self.points2bbox(
                    pts_preds_init[i_lvl].detach())  #将一阶段九个点的偏移按要求转换成左上角和右下角的形式BOX，[2,4,H,W]
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]  #还是[2,4,H,W]
                bbox_center = torch.cat(
                    [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)  #一个样本坐标CAT成两个[N,4]
                bbox.append(bbox_center +   #
                            bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4)) #坐标中心+偏移，转换成左上角和右下角的形式的anchor。
            bbox_list.append(bbox)  #相当于是生成的anchor，用来匹配GT框的
        cls_reg_targets_refine = self.get_targets(
            bbox_list,   #预测的左上角和右下角
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='refine',
            label_channels=label_channels)
        (labels_list, label_weights_list, bbox_gt_list_refine,
         candidate_list_refine, bbox_weights_list_refine, num_total_pos_refine,
         num_total_neg_refine) = cls_reg_targets_refine  #第二阶段给labels_list
        num_total_samples_refine = (
            num_total_pos_refine +
            num_total_neg_refine if self.sampling else num_total_pos_refine)

        # compute loss
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_single,
            cls_scores,
            pts_coordinate_preds_init,   #九个坐标值（消除了偏移的）坐标值就是（X，Y的），初始的是（HW）-（YX）
            pts_coordinate_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,       #就是GT
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_refine)
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine
        }
        return loss_dict_all

    # Same as base_dense_head/_get_bboxes_single except self._bbox_decode
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, mlvl_priors)): #预测分类分数，BOX和生成的样本点
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:  #后处理要用到
                scores = cls_score.sigmoid()  #预测结果要经过sigmoid激活函数，因为输出没用，计算损失的时候用到了
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(  #NMS
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors)) #预测结果和样本点整合在一起更好索引
            scores, labels, _, filtered_results = results
            #
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self._bbox_decode(priors, bbox_pred,
                                       self.point_strides[level_idx],
                                       img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(
            mlvl_scores,
            mlvl_labels,
            mlvl_bboxes,
            img_meta['scale_factor'],
            cfg,
            rescale=rescale,
            with_nms=with_nms)

    def _bbox_decode(self, points, bbox_pred, stride, max_shape):
        bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
        bboxes = bbox_pred * stride + bbox_pos_center   #自己的解码方式，偏移+中心点得到真正的BOX
        x1 = bboxes[:, 0].clamp(min=0, max=max_shape[1])
        y1 = bboxes[:, 1].clamp(min=0, max=max_shape[0])
        x2 = bboxes[:, 2].clamp(min=0, max=max_shape[1])
        y2 = bboxes[:, 3].clamp(min=0, max=max_shape[0])
        decoded_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return decoded_bboxes
