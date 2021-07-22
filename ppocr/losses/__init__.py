# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import paddle
import paddle.nn as nn

# det loss
from .det_db_loss import DBLoss
from .det_east_loss import EASTLoss
from .det_sast_loss import SASTLoss

# rec loss
from .rec_ctc_loss import CTCLoss
from .rec_att_loss import AttentionLoss
from .rec_srn_loss import SRNLoss

# cls loss
from .cls_loss import ClsLoss

# e2e loss
from .e2e_pg_loss import PGLoss

# basic loss function
from .basic_loss import DistanceLoss

# combined loss function
from .combined_loss import CombinedLoss

# table loss
from .table_att_loss import TableAttentionLoss

from .rec_aster_loss import AsterLoss


def build_loss(config):
    support_dict = [
        'DBLoss', 'EASTLoss', 'SASTLoss', 'CTCLoss', 'ClsLoss', 'AttentionLoss',
        'SRNLoss', 'PGLoss', 'CombinedLoss', 'TableAttentionLoss', 'AsterLoss'
    ]
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
