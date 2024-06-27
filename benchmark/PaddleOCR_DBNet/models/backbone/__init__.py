# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun

from .resnet import *

__all__ = ["build_backbone"]

support_backbone = [
    "resnet18",
    "deformable_resnet18",
    "deformable_resnet50",
    "resnet50",
    "resnet34",
    "resnet101",
    "resnet152",
]


def build_backbone(backbone_name, **kwargs):
    assert (
        backbone_name in support_backbone
    ), f"all support backbone is {support_backbone}"
    backbone = eval(backbone_name)(**kwargs)
    return backbone
