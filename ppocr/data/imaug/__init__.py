# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .iaa_augment import IaaAugment
from .make_border_map import MakeBorderMap
from .make_shrink_map import MakeShrinkMap
from .random_crop_data import EastRandomCropData, RandomCropImgMask
from .make_pse_gt import MakePseGt


from .rec_img_aug import (
    BaseDataAugmentation,
    RecAug,
    RecConAug,
    RecResizeImg,
    ClsResizeImg,
    SRNRecResizeImg,
    GrayRecResizeImg,
    SARRecResizeImg,
    PRENResizeImg,
    ABINetRecResizeImg,
    SVTRRecResizeImg,
    ABINetRecAug,
    VLRecResizeImg,
    SPINRecResizeImg,
    RobustScannerRecResizeImg,
    RFLRecResizeImg,
    SVTRRecAug,
    ParseQRecAug,
)
from .ssl_img_aug import SSLRotateResize
from .randaugment import RandAugment
from .copy_paste import CopyPaste
from .ColorJitter import ColorJitter
from .operators import *
from .label_ops import *

from .east_process import *
from .sast_process import *
from .pg_process import *
from .table_ops import *

from .vqa import *

from .fce_aug import *
from .fce_targets import FCENetTargets
from .ct_process import *
from .drrg_targets import DRRGTargets
from .latex_ocr_aug import *


def transform(data, ops=None):
    """transform"""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), "operator config should be a list"
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops
