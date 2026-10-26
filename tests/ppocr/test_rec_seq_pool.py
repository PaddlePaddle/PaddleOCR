# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import pytest

from ppocr.modeling.backbones.rec_hgnet import PPHGNet_small
from ppocr.modeling.backbones.rec_lcnetv3 import PPLCNetV3
from ppocr.modeling.backbones.rec_lcnetv4 import PPLCNetV4
from ppocr.modeling.backbones.rec_pphgnetv2 import PPHGNetV2_B4
from ppocr.modeling.backbones.rec_seq_pool import rec_seq_pool

np.random.seed(42)


def _feat(h, w, c=8):
    return paddle.to_tensor(np.random.rand(1, c, h, w).astype("float32"))


class TestRecSeqPool:
    def test_train_default_is_fixed_40(self):
        assert rec_seq_pool(_feat(3, 240), 40, training=True).shape == [1, 8, 1, 40]

    def test_train_proportional_matches_inference(self):
        assert rec_seq_pool(_feat(3, 240), -1, training=True).shape == [1, 8, 1, 120]

    @pytest.mark.parametrize("h", [2, 3, 4])
    def test_train_proportional_any_height(self, h):
        assert rec_seq_pool(_feat(h, 160), -1, training=True).shape == [1, 8, 1, 80]

    def test_train_proportional_narrow_input(self):
        assert rec_seq_pool(_feat(3, 1), -1, training=True).shape == [1, 8, 1, 1]

    def test_eval_halves_width(self):
        assert rec_seq_pool(_feat(3, 240), 40, training=False).shape == [1, 8, 1, 120]

    def test_eval_height_check(self):
        with pytest.raises(AssertionError, match="Feature height"):
            rec_seq_pool(_feat(2, 240), -1, training=False, check_height=True)


def _seq_len(model, h=48, w=960, training=True):
    model.train() if training else model.eval()
    x = paddle.to_tensor(np.random.rand(1, 3, h, w).astype("float32"))
    with paddle.no_grad():
        out = model(x)
    assert out.shape[2] == 1
    return out.shape[3]


BACKBONES = [
    pytest.param(lambda **kw: PPLCNetV4(model_size="small", **kw), id="v4_small"),
    pytest.param(lambda **kw: PPLCNetV4(model_size="tiny", **kw), id="v4_tiny"),
    pytest.param(lambda **kw: PPLCNetV3(scale=0.95, **kw), id="v3"),
    pytest.param(
        lambda **kw: PPHGNetV2_B4(text_rec=True, **kw),
        id="hgnetv2_b4",
        marks=pytest.mark.resource_intensive,
    ),
    pytest.param(
        lambda **kw: PPHGNet_small(**kw),
        id="hgnet_small",
        marks=pytest.mark.resource_intensive,
    ),
]


@pytest.mark.parametrize("make", BACKBONES)
class TestBackboneTrainSeqLen:
    def test_default_train_is_40(self, make):
        assert _seq_len(make(), training=True) == 40

    def test_proportional_train(self, make):
        assert _seq_len(make(train_seq_len=-1), training=True) == 120

    def test_eval_unchanged(self, make):
        assert _seq_len(make(), training=False) == 120
        assert _seq_len(make(train_seq_len=-1), training=False) == 120

    @pytest.mark.parametrize("h", [32, 64])
    def test_proportional_other_heights(self, make, h):
        assert _seq_len(make(train_seq_len=-1), h=h, w=640, training=True) == 80

    @pytest.mark.parametrize("bad", [0, -2])
    def test_invalid_train_seq_len(self, make, bad):
        with pytest.raises(AssertionError, match="train_seq_len"):
            make(train_seq_len=bad)
