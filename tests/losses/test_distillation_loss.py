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

"""``KLCTCLogits._kldiv`` must be a batch mean, like the copy in ``basic_loss``.

``ppocr/losses/basic_loss.py`` implements the same helper and reduces once::

    # batch mean loss
    loss = paddle.sum(loss) / loss.shape[0]

The distillation copy used to reduce twice -
``paddle.sum(paddle.mean(loss, axis=1)) / loss.shape[0]`` - which is smaller by
the size of axis 1. The helper was introduced by PR #5643, which validated the
batch-mean form against ``F.kl_div(..., reduction="batchmean")``, and the
shipped default ``mode="mean"`` already takes the correct branch, so the
non-default modes disagreed with the default by that factor.
"""

import paddle
import pytest

from ppocr.losses.distillation_loss import KLCTCLogits


def _term(x, target, eps=1.0e-10):
    """The unreduced KL term both copies start from."""
    return target * (paddle.log(target + eps) - x)


def _batch_mean(x, target):
    term = _term(x, target)
    return float(paddle.sum(term) / term.shape[0])


def _class_mean(x, target):
    """The double reduction this test guards against."""
    term = _term(x, target)
    return float(paddle.sum(paddle.mean(term, axis=1)) / term.shape[0])


def test_kldiv_reduces_once():
    op = KLCTCLogits(mode="log")
    target = paddle.to_tensor(
        [[0.2, 0.3, 0.5, 0.1, 0.15, 0.25, 0.2, 0.3]], dtype="float32"
    )
    x = paddle.log(target) + paddle.to_tensor(
        [[0.4, -0.2, 0.1, 0.3, -0.1, 0.2, -0.3, 0.05]], dtype="float32"
    )

    batch_mean = _batch_mean(x, target)
    class_mean = _class_mean(x, target)

    # The two reductions differ by exactly the size of axis 1 ...
    assert batch_mean == pytest.approx(class_mean * target.shape[1], rel=1e-5)
    # ... so they are only distinguishable when they actually differ.
    assert batch_mean != pytest.approx(class_mean, rel=1e-3)

    assert float(op._kldiv(x, target)) == pytest.approx(batch_mean, rel=1e-6)


def test_forward_log_uses_the_batch_mean_reduction():
    """The public entry point must not be scaled down by the class axis."""
    op = KLCTCLogits(mode="log")
    paddle.seed(0)
    out1 = paddle.rand([2, 8, 16])
    out2 = paddle.rand([2, 8, 16])

    # Rebuild what forward_log feeds to _kldiv, then reduce both ways.
    s1 = op.act(out1) + 1e-10
    s2 = op.act(out2) + 1e-10
    l1 = paddle.log(s1)
    l2 = paddle.log(s2)

    batch_mean = (_batch_mean(l1, s2) + _batch_mean(l2, s1)) / 2.0
    class_mean = (_class_mean(l1, s2) + _class_mean(l2, s1)) / 2.0

    got = float(op.forward_log(out1, out2))

    assert got == pytest.approx(batch_mean, rel=1e-6)
    assert got != pytest.approx(class_mean, rel=1e-3)
    assert batch_mean == pytest.approx(class_mean * s2.shape[1], rel=1e-4)


def test_forward_log_returns_a_zero_dim_tensor():
    """The reduction change must not alter the result's shape."""
    op = KLCTCLogits(mode="log")
    paddle.seed(1)
    out = float(op.forward_log(paddle.rand([2, 4, 8]), paddle.rand([2, 4, 8])))
    assert isinstance(out, float)
