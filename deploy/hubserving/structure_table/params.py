# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deploy.hubserving.ocr_system.params import read_params as pp_ocr_read_params


def read_params():
    cfg = pp_ocr_read_params()

    # params for table structure model
    cfg.table_max_len = 488
    cfg.table_model_dir = './inference/en_ppocr_mobile_v2.0_table_structure_infer/'
    cfg.table_char_type = 'en'
    cfg.table_char_dict_path = './ppocr/utils/dict/table_structure_dict.txt'
    cfg.show_log = False
    return cfg
