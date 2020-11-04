# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Config(object):
    pass

def read_params():
    cfg = Config()
    #use gpu
    cfg.use_gpu = False
    cfg.use_pdserving = True

    #params for text detector
    cfg.det_algorithm = "DB"
    cfg.det_server_dir = "../../inference/ch_ppocr_mobile_v1.1_det_infer/serving_server_dir"
    cfg.det_client_dir = "../../inference/ch_ppocr_mobile_v1.1_det_infer/serving_client_dir"
    cfg.det_max_side_len = 960

    #DB parmas
    cfg.det_db_thresh =0.3
    cfg.det_db_box_thresh =0.5
    cfg.det_db_unclip_ratio =2.0

    #EAST parmas
    cfg.det_east_score_thresh = 0.8
    cfg.det_east_cover_thresh = 0.1
    cfg.det_east_nms_thresh = 0.2

    #params for text recognizer
    cfg.rec_algorithm = "CRNN"
    cfg.rec_server_dir = "../../inference/ch_ppocr_mobile_v1.1_rec_infer/serving_server_dir"
    cfg.rec_client_dir = "../../inference/ch_ppocr_mobile_v1.1_rec_infer/serving_client_dir"

    cfg.rec_image_shape = "3, 32, 320"
    cfg.rec_char_type = 'ch'
    cfg.rec_batch_num = 30
    cfg.max_text_length = 25

    cfg.rec_char_dict_path = "../../ppocr/utils/ppocr_keys_v1.txt"
    cfg.use_space_char = True

    #params for text classifier
    cfg.use_angle_cls = True
    cfg.cls_server_dir = "../../inference/ch_ppocr_mobile_v1.1_cls_infer/serving_server_dir"
    cfg.cls_client_dir = "../../inference/ch_ppocr_mobile_v1.1_cls_infer/serving_client_dir"
    cfg.cls_image_shape = "3, 48, 192"
    cfg.label_list = ['0', '180']
    cfg.cls_batch_num = 30
    cfg.cls_thresh = 0.9

    return cfg
