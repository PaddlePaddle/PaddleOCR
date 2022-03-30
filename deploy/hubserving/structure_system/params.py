# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deploy.hubserving.structure_table.params import read_params as table_read_params


def read_params():
    cfg = table_read_params()

    # params for layout parser model
    cfg.layout_path_model = 'lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config'
    cfg.layout_label_map = None

    cfg.mode = 'structure'
    cfg.output = './output'
    return cfg
