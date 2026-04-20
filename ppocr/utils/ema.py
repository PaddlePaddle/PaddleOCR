# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import math

import paddle


class ModelEMA:
    """Exponential Moving Average for model parameters.

    Maintains shadow copies of model parameters and updates them with:
        ema_param = decay * ema_param + (1 - decay) * cur_param

    Reference: PaddleDetection ppdet/optimizer/ema.py

    Args:
        model (nn.Layer): The model whose parameters will be averaged.
        decay (float): EMA decay coefficient. Default: 0.9998.
        gamma (int): Warmup parameter for threshold/exponential decay.
            Default: 2000.
        ema_decay_type (str): Decay schedule type, one of
            'threshold' (default), 'exponential', 'normal'.
        ema_filter_no_grad (bool): If True, parameters with
            stop_gradient=True (e.g. frozen Teacher in distillation)
            are excluded from EMA and pass through unchanged.
            BN running stats are kept even if no-grad. Default: False.
    """

    def __init__(
        self,
        model,
        decay=0.9998,
        gamma=2000,
        ema_decay_type="threshold",
        ema_filter_no_grad=False,
    ):
        self.decay = decay
        self.gamma = gamma
        self.ema_decay_type = ema_decay_type
        self.step = 0
        self._decay = decay

        # Build black list: frozen params (excluding BN running stats)
        self.ema_black_list = set()
        if ema_filter_no_grad:
            bn_state_names = set()
            for name, layer in model.named_sublayers():
                if isinstance(layer, (paddle.nn.BatchNorm2D, paddle.nn.BatchNorm1D)):
                    prefix = name + "." if name else ""
                    bn_state_names.add(prefix + "_mean")
                    bn_state_names.add(prefix + "_variance")
            for n, p in model.named_parameters():
                if p.stop_gradient and n not in bn_state_names:
                    self.ema_black_list.add(n)

        # Initialize shadow weights
        self.state_dict = {}
        for k, v in model.state_dict().items():
            if k in self.ema_black_list:
                self.state_dict[k] = v.clone()
            else:
                self.state_dict[k] = paddle.zeros_like(v).astype("float32")

    def _get_decay(self):
        if self.ema_decay_type == "threshold":
            return min(self.decay, (1 + self.step) / (10 + self.step))
        elif self.ema_decay_type == "exponential":
            return self.decay * (1 - math.exp(-(self.step + 1) / self.gamma))
        else:  # normal
            return self.decay

    def update(self, model):
        """Update shadow weights with current model parameters."""
        decay = self._get_decay()
        self._decay = decay
        model_dict = model.state_dict()
        for k, v in self.state_dict.items():
            if k not in self.ema_black_list and k in model_dict:
                v = decay * v + (1 - decay) * model_dict[k].astype("float32")
                v.stop_gradient = True
                self.state_dict[k] = v
        self.step += 1

    def apply(self):
        """Return bias-corrected EMA state dict for eval/save.

        Does NOT modify internal state.
        """
        if self.step == 0:
            return {k: v.clone() for k, v in self.state_dict.items()}
        state = {}
        for k, v in self.state_dict.items():
            if k in self.ema_black_list:
                state[k] = v
            else:
                if self.ema_decay_type != "exponential":
                    # threshold / normal need bias-correction
                    v = v / (1 - self._decay**self.step)
                v = v.clone()
                v.stop_gradient = True
                state[k] = v
        return state

    def state_dict_for_save(self):
        """Return serializable dict for checkpoint."""
        return {"ema_state": self.state_dict, "step": self.step}

    def set_state_dict(self, d):
        """Restore from checkpoint."""
        self.state_dict = d["ema_state"]
        self.step = d["step"]
