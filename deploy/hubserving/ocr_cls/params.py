# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(object):
    pass


def read_params():
    cfg = Config()

    #params for text classifier
    cfg.cls_model_dir = "./inference/ch_ppocr_mobile_v2.0_cls_infer/"
    cfg.cls_image_shape = "3, 48, 192"
    cfg.label_list = ['0', '180']
    cfg.cls_batch_num = 30
    cfg.cls_thresh = 0.9

    cfg.use_pdserving = False
    cfg.use_tensorrt = False

    return cfg
