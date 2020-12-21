# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(object):
    pass


def read_params():
    cfg = Config()

    #params for text detector
    cfg.det_algorithm = "DB"
    cfg.det_model_dir = "./inference/ch_ppocr_mobile_v2.0_det_infer/"
    cfg.det_limit_side_len = 960
    cfg.det_limit_type = 'max'

    #DB parmas
    cfg.det_db_thresh = 0.3
    cfg.det_db_box_thresh = 0.5
    cfg.det_db_unclip_ratio = 2.0

    # #EAST parmas
    # cfg.det_east_score_thresh = 0.8
    # cfg.det_east_cover_thresh = 0.1
    # cfg.det_east_nms_thresh = 0.2

    cfg.use_zero_copy_run = False
    cfg.use_pdserving = False

    return cfg
