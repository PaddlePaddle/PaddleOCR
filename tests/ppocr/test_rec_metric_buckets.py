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

import pytest

from ppocr.metrics.rec_metric import (
    RecMetric,
    length_bucket_index,
    length_bucket_names,
)


def _batch(pairs):
    """[(pred, target), ...] -> (preds, labels) as produced by the post-process."""
    return [(p, 1.0) for p, _ in pairs], [(t, None) for _, t in pairs]


@pytest.fixture
def pairs():
    return [
        ("a" * 5, "a" * 5),  # 0-15, correct
        ("b" * 30, "b" * 30),  # 16-40, correct
        ("c" * 49, "c" * 50),  # 41-60, wrong (1 char missing)
        ("d" * 80, "d" * 80),  # 61+, correct
        ("x" * 70, "e" * 80),  # 61+, wrong
    ]


class TestBucketHelpers:
    def test_names(self):
        assert length_bucket_names([15, 40, 60]) == [
            "0_15",
            "16_40",
            "41_60",
            "61_inf",
        ]

    @pytest.mark.parametrize(
        "length,expected", [(0, 0), (15, 0), (16, 1), (40, 1), (60, 2), (61, 3)]
    )
    def test_upper_bound_inclusive(self, length, expected):
        assert length_bucket_index(length, [15, 40, 60]) == expected

    @pytest.mark.parametrize("bad", [[], [40, 15], [15, 15]])
    def test_invalid(self, bad):
        with pytest.raises(AssertionError, match="length_buckets"):
            length_bucket_names(bad)


class TestRecMetricBuckets:
    def test_default_metric_unchanged(self, pairs):
        metric = RecMetric(ignore_space=False)
        metric(_batch(pairs))
        result = metric.get_metric()
        assert set(result) == {"acc", "norm_edit_dis"}
        assert result["acc"] == pytest.approx(3 / 5, abs=1e-4)

    def test_bucket_values(self, pairs):
        metric = RecMetric(ignore_space=False, length_buckets=[15, 40, 60])
        metric(_batch(pairs[:2]))
        metric(_batch(pairs[2:]))
        result = metric.get_metric()
        assert result["acc"] == pytest.approx(3 / 5, abs=1e-4)
        assert result["acc_len_0_15"] == 1.0
        assert result["acc_len_16_40"] == 1.0
        assert result["acc_len_41_60"] == 0.0
        assert result["acc_len_61_inf"] == 0.5
        assert result["n_len_61_inf"] == 2.0
        assert result["ned_len_41_60"] == pytest.approx(1 - 1 / 50)
        assert result["ned_len_61_inf"] == pytest.approx(0.5)
        assert all(isinstance(v, float) for v in result.values())

    def test_empty_buckets_hidden(self, pairs):
        metric = RecMetric(ignore_space=False, length_buckets=[15, 40, 60])
        metric(_batch(pairs[:1]))
        keys = {k for k in metric.get_metric() if "_len_" in k}
        assert keys == {"acc_len_0_15", "ned_len_0_15", "n_len_0_15"}

    def test_get_metric_resets_buckets(self, pairs):
        metric = RecMetric(ignore_space=False, length_buckets=[15, 40, 60])
        metric(_batch(pairs))
        metric.get_metric()
        metric(_batch(pairs[:1]))
        result = metric.get_metric()
        assert result["n_len_0_15"] == 1.0
        assert "n_len_61_inf" not in result

    def test_length_counted_after_ignore_space(self):
        metric = RecMetric(ignore_space=True, length_buckets=[15])
        metric(_batch([("a b c d e f g h", "a b c d e f g h i")]))
        # 17 chars with spaces, 9 without -> first bucket
        assert "n_len_0_15" in metric.get_metric()
