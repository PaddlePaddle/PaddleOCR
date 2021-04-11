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
    cfg.det_db_unclip_ratio = 1.6
    cfg.use_dilation = False

    #EAST parmas
    cfg.det_east_score_thresh = 0.8
    cfg.det_east_cover_thresh = 0.1
    cfg.det_east_nms_thresh = 0.2

    #params for text recognizer
    cfg.rec_algorithm = "CRNN"
    cfg.rec_model_dir = "./inference/ch_ppocr_mobile_v2.0_rec_infer/"

    cfg.rec_image_shape = "3, 32, 320"
    cfg.rec_char_type = 'ch'
    cfg.rec_batch_num = 30
    cfg.max_text_length = 25

    cfg.rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
    cfg.use_space_char = True

    #params for text classifier
    cfg.use_angle_cls = True
    cfg.cls_model_dir = "./inference/ch_ppocr_mobile_v2.0_cls_infer/"
    cfg.cls_image_shape = "3, 48, 192"
    cfg.label_list = ['0', '180']
    cfg.cls_batch_num = 30
    cfg.cls_thresh = 0.9

    cfg.use_pdserving = False
    cfg.use_tensorrt = False
    cfg.drop_score = 0.5

    return cfg
