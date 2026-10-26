# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.nn.functional as F

__all__ = ["rec_seq_pool", "check_train_seq_len"]


def check_train_seq_len(train_seq_len):
    assert (
        train_seq_len == -1 or train_seq_len > 0
    ), "train_seq_len must be -1 (proportional to width) or > 0, got {}".format(
        train_seq_len
    )


def rec_seq_pool(x, train_seq_len=40, training=True, check_height=False):
    """Pool [B, C, H, W] recognition features to the [B, C, 1, T] CTC sequence.

    Inference: avg_pool [3, 2], so T = W // 2 (H must be 3, i.e. 48 px input).
    Training:
        train_seq_len > 0  -> fixed T = train_seq_len (legacy behaviour, 40);
        train_seq_len == -1 -> T = W // 2, the same as inference, so long text
                               lines still fit into CTC (it needs T >= label length).
    """
    if not training:
        if check_height:
            assert x.shape[2] >= 3, f"Feature height {x.shape[2]} < pool kernel 3."
        return F.avg_pool2d(x, [3, 2])
    if train_seq_len == -1:
        width = x.shape[3]
        assert width > 0, (
            "train_seq_len=-1 needs a known feature width at train time, got {}; "
            "it is not supported with Global.to_static".format(width)
        )
        return F.adaptive_avg_pool2d(x, [1, max(1, width // 2)])
    return F.adaptive_avg_pool2d(x, [1, train_seq_len])
