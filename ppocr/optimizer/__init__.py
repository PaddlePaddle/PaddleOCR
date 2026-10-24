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
import copy
import math
import paddle

__all__ = ["build_optimizer"]


class CosineWeightDecayScheduler(object):
    """Cosine-anneal the optimizer's weight decay each step.

    wd(t) = end + 0.5 * (start - end) * (1 + cos(pi * t / T))

    During warmup the coefficient is held at `start_factor`.
    """

    def __init__(
        self, optimizer, start_factor, end_factor, total_steps, warmup_steps=0
    ):
        self.optimizer = optimizer
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            wd = self.start_factor
        else:
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            wd = self.end_factor + 0.5 * (self.start_factor - self.end_factor) * (
                1 + math.cos(math.pi * progress)
            )
        self.optimizer.regularization._coeff = wd

    def get_wd(self):
        return self.optimizer.regularization._coeff


def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    from . import learning_rate

    lr_config.update({"epochs": epochs, "step_each_epoch": step_each_epoch})
    lr_name = lr_config.pop("name", "Const")
    lr = getattr(learning_rate, lr_name)(**lr_config)()
    return lr


def build_optimizer(config, epochs, step_each_epoch, model):
    from . import regularizer, optimizer

    config = copy.deepcopy(config)
    # step1 build lr
    lr = build_lr_scheduler(config.pop("lr"), epochs, step_each_epoch)

    # step2 build regularization
    wd_scheduler = None
    if "regularizer" in config and config["regularizer"] is not None:
        reg_config = config.pop("regularizer")
        reg_name = reg_config.pop("name")
        if not hasattr(regularizer, reg_name):
            reg_name += "Decay"
        reg_obj = getattr(regularizer, reg_name)(**reg_config)
        reg = reg_obj()

        # Build weight decay scheduler for CosineL2Decay
        if isinstance(reg_obj, regularizer.CosineL2Decay):
            warmup_epoch = reg_obj.warmup_epoch
            warmup_steps = round(warmup_epoch * step_each_epoch)
            total_steps = step_each_epoch * epochs
            wd_scheduler = {
                "start_factor": reg_obj.start_factor,
                "end_factor": reg_obj.end_factor,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            }
    elif "weight_decay" in config:
        reg = config.pop("weight_decay")
    else:
        reg = None

    # step3 build optimizer
    optim_name = config.pop("name")
    if "clip_norm" in config:
        clip_norm = config.pop("clip_norm")
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
    elif "clip_norm_global" in config:
        clip_norm = config.pop("clip_norm_global")
        grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=clip_norm)
    else:
        grad_clip = None
    optim = getattr(optimizer, optim_name)(
        learning_rate=lr, weight_decay=reg, grad_clip=grad_clip, **config
    )
    built_optim = optim(model)

    # Instantiate the scheduler now that we have the real optimizer
    if wd_scheduler is not None:
        wd_scheduler = CosineWeightDecayScheduler(built_optim, **wd_scheduler)

    return built_optim, lr, wd_scheduler
