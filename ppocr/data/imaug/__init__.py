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
from .random_crop_data import EastRandomCropData, PSERandomCrop

from .rec_img_aug import RecAug, RecResizeImg, ClsResizeImg, SRNRecResizeImg
from .randaugment import RandAugment
from .operators import *
from .label_ops import *

from .east_process import *
from .sast_process import *
from .pg_process import *

from .det_aug import *

def transform(data, ops=None):
    """ transform """
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
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


class OneOf(object):
    def __init__(self, ops, **kwargs):
        self.ops = []
        ext_data_num = 0
        for op in ops:
            if isinstance(op, dict):
                op = [op]
            op = create_operators(op, kwargs)
            for opop in op:
                if hasattr(opop, 'ext_data_num'):
                    ext_data_num = getattr(opop, 'ext_data_num')
                    break
            self.ops.append(op)
        self.ext_data_num = ext_data_num
        self._small_area_text_contrib = 0

    @property
    def small_area_text_contrib(self,):
        return self._small_area_text_contrib

    @small_area_text_contrib.setter
    def small_area_text_contrib(self,value):
        for op in self.ops:
            for o in op:
                if hasattr(o, 'small_area_text_contrib'):
                    setattr(o, 'small_area_text_contrib', value)

    def __call__(self, data):
        op = np.random.choice(self.ops)
        return transform(data, op)
