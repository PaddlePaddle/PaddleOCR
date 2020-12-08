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

__all__ = ['build_head']


def build_head(config):
    # det head
    from .det_db_head import DBHead
    from .det_east_head import EASTHead
    from .det_sast_head import SASTHead

    # rec head
    from .rec_ctc_head import CTCHead

    # cls head
    from .cls_head import ClsHead
    support_dict = ['DBHead', 'EASTHead', 'SASTHead', 'CTCHead', 'ClsHead']

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
