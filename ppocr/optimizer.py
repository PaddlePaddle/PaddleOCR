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
from paddle.fluid.regularizer import L2Decay

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
    l2_decay = params.get("l2_decay", 0.0)

    if 'decay' in params:
        supported_decay_mode = ["cosine_decay", "piecewise_decay"]
        params = params['decay']
        decay_mode = params['function']
        assert decay_mode in supported_decay_mode, "Supported decay mode is {}, but got {}".format(
            supported_decay_mode, decay_mode)

        if decay_mode == "cosine_decay":
            step_each_epoch = params['step_each_epoch']
            total_epoch = params['total_epoch']
            base_lr = fluid.layers.cosine_decay(
                learning_rate=base_lr,
                step_each_epoch=step_each_epoch,
                epochs=total_epoch)
        elif decay_mode == "piecewise_decay":
            boundaries = params["boundaries"]
            decay_rate = params["decay_rate"]
            values = [
                base_lr * decay_rate**idx
                for idx in range(len(boundaries) + 1)
            ]
            base_lr = fluid.layers.piecewise_decay(boundaries, values)

    optimizer = fluid.optimizer.Adam(
        learning_rate=base_lr,
        beta1=beta1,
        beta2=beta2,
        regularization=L2Decay(regularization_coeff=l2_decay),
        parameter_list=parameter_list)
    return optimizer


def RMSProp(params, parameter_list=None):
    """
    define optimizer function
    args:
        params(dict): the super parameters
        parameter_list (list): list of Variable names to update to minimize loss
    return:
    """
    base_lr = params.get("base_lr", 0.001)
    l2_decay = params.get("l2_decay", 0.00005)

    if 'decay' in params:
        supported_decay_mode = ["cosine_decay", "piecewise_decay"]
        params = params['decay']
        decay_mode = params['function']
        assert decay_mode in supported_decay_mode, "Supported decay mode is {}, but got {}".format(
            supported_decay_mode, decay_mode)

        if decay_mode == "cosine_decay":
            step_each_epoch = params['step_each_epoch']
            total_epoch = params['total_epoch']
            base_lr = fluid.layers.cosine_decay(
                learning_rate=base_lr,
                step_each_epoch=step_each_epoch,
                epochs=total_epoch)
        elif decay_mode == "piecewise_decay":
            boundaries = params["boundaries"]
            decay_rate = params["decay_rate"]
            values = [
                base_lr * decay_rate**idx
                for idx in range(len(boundaries) + 1)
            ]
            base_lr = fluid.layers.piecewise_decay(boundaries, values)

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=base_lr,
        regularization=fluid.regularizer.L2Decay(regularization_coeff=l2_decay))
        
    return optimizer