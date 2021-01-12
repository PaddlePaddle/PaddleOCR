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


def build_loss(config):
    # det loss
    from .det_db_loss import DBLoss
    from .det_east_loss import EASTLoss
    from .det_sast_loss import SASTLoss

    # rec loss
    from .rec_ctc_loss import CTCLoss

    # cls loss
    from .cls_loss import ClsLoss

    support_dict = ['DBLoss', 'EASTLoss', 'SASTLoss', 'CTCLoss', 'ClsLoss']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
