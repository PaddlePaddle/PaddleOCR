# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import os
import paddle

__all__ = ['load_dygraph_pretrain']


def load_dygraph_pretrain(model, logger, path=None, load_static_weights=False):
    if not os.path.exists(path + '.pdparams'):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + '.pdparams')
    model.set_state_dict(param_state_dict)
    logger.info("load pretrained model from {}".format(path))
    return
