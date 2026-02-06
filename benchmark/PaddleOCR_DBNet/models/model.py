# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:57
# @Author  : zhoujun
from addict import Dict
from paddle import nn
import paddle.nn.functional as F

from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head


class Model(nn.Layer):
    def __init__(self, model_config: dict):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop("type")
        neck_type = model_config.neck.pop("type")
        head_type = model_config.head.pop("type")
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(
            neck_type, in_channels=self.backbone.out_channels, **model_config.neck
        )
        self.head = build_head(
            head_type, in_channels=self.neck.out_channels, **model_config.head
        )
        self.name = f"{backbone_type}_{neck_type}_{head_type}"

    def forward(self, x):
        _, _, H, W = x.shape
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=True)
        return y
