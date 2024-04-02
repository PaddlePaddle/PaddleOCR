# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Config(object):
    pass


def read_params():
    cfg = Config()

    #params for text detector
    cfg.det_algorithm = "DB"
    cfg.det_model_dir = "./inference/ch_PP-OCRv3_det_infer/"
    cfg.det_limit_side_len = 960
    cfg.det_limit_type = 'max'

    #DB parmas
    cfg.det_db_thresh = 0.3
    cfg.det_db_box_thresh = 0.6
    cfg.det_db_unclip_ratio = 1.5
    cfg.use_dilation = False
    cfg.det_db_score_mode = "fast"

    # #EAST parmas
    # cfg.det_east_score_thresh = 0.8
    # cfg.det_east_cover_thresh = 0.1
    # cfg.det_east_nms_thresh = 0.2

    cfg.use_pdserving = False
    cfg.use_tensorrt = False

    return cfg
