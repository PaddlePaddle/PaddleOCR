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
import math
import paddle.fluid as fluid
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import paddle.fluid.layers.ops as ops

from ppocr.utils.utility import initial_logger

logger = initial_logger()


def cosine_decay_with_warmup(learning_rate,
                             step_each_epoch,
                             epochs=500,
                             warmup_minibatch=1000):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    decrease lr for every mini-batch and start with warmup.
    """
    global_step = _decay_step_counter()
    lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    warmup_minibatch = fluid.layers.fill_constant(
        shape=[1],
        dtype='float32',
        value=float(warmup_minibatch),
        force_cpu=True)

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step < warmup_minibatch):
            decayed_lr = learning_rate * (1.0 * global_step / warmup_minibatch)
            fluid.layers.tensor.assign(input=decayed_lr, output=lr)
        with switch.default():
            decayed_lr = learning_rate * \
                (ops.cos((global_step - warmup_minibatch) * (math.pi / (epochs * step_each_epoch))) + 1)/2
            fluid.layers.tensor.assign(input=decayed_lr, output=lr)
    return lr


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
        supported_decay_mode = [
            "cosine_decay", "cosine_decay_warmup", "piecewise_decay"
        ]
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
        elif decay_mode == "cosine_decay_warmup":
            step_each_epoch = params['step_each_epoch']
            total_epoch = params['total_epoch']
            warmup_minibatch = params.get("warmup_minibatch", 1000)
            base_lr = cosine_decay_with_warmup(
                learning_rate=base_lr,
                step_each_epoch=step_each_epoch,
                epochs=total_epoch,
                warmup_minibatch=warmup_minibatch)
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
