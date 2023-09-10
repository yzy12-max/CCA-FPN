# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
# from .single_stage import SingleStageDetector
from .singlestage_heatmap import SingleStageDetectorHp

@DETECTORS.register_module()
class ATSShp(SingleStageDetectorHp):  #
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ATSShp, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)