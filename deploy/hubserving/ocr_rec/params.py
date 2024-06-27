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

    # params for text recognizer
    cfg.rec_algorithm = "CRNN"
    cfg.rec_model_dir = "./inference/ch_PP-OCRv3_rec_infer/"

    cfg.rec_image_shape = "3, 48, 320"
    cfg.rec_batch_num = 6
    cfg.max_text_length = 25

    cfg.rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
    cfg.use_space_char = True

    cfg.use_pdserving = False
    cfg.use_tensorrt = False

    return cfg
