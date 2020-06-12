#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from ppocr.utils.utility import initial_logger

logger = initial_logger()


def AdamDecay(params, parameter_list=None):
    """
    define optimizer function
    args:
        params(dict): the super parameters
        parameter_list (list): list of Variable names to update to minimize loss
    return:
    """
    base_lr = params['base_lr']
    beta1 = params['beta1']
    beta2 = params['beta2']
    if 'decay' in params:
        params = params['decay']
        decay_mode = params['function']
        step_each_epoch = params['step_each_epoch']
        total_epoch = params['total_epoch']
        if decay_mode == "cosine_decay":
            base_lr = fluid.layers.cosine_decay(
                learning_rate=base_lr,
                step_each_epoch=step_each_epoch,
                epochs=total_epoch)
        else:
            logger.info("Only support Cosine decay currently")
    optimizer = fluid.optimizer.Adam(
        learning_rate=base_lr,
        beta1=beta1,
        beta2=beta2,
        parameter_list=parameter_list)
    return optimizer
