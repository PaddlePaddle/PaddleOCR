# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 15:17
# @Author  : zhoujun

from .seg_detector_representer import SegDetectorRepresenter


def get_post_processing(config):
    try:
        cls = eval(config["type"])(**config["args"])
        return cls
    except:
        return None
