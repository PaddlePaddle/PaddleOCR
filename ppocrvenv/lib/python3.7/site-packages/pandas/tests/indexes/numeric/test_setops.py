from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas import (
    Float64Index,
    Index,
    Int64Index,
    RangeIndex,
    UInt64Index,
)
import pandas._testing as tm


@pytest.fixture
def index_large():
    # large values used in TestUInt64Index where no compat needed with Int64/Float64
    large = [2 ** 63, 2 ** 63 + 10, 2 ** 63 + 15, 2 ** 63 + 20, 2 ** 63 + 25]
    return UInt64Index(large)


class TestSetOps:
    @pytest.mark.parametrize("dtype", ["f8", "u8", "i8"])
    def test_union_non_numeric(self, dtype):
        # corner case, non-numeric
        index = Index(np.arange(5, dtype=dtype), dtype=dtype)
        assert index.dtype == dtype

        other = Index([datetime.now() + timedelta(i) for i in range(4)], dtype=object)
        result = index.union(other)
        expected = Index(np.concatenate((index, other)))
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        expected = Index(np.concatenate((other, index)))
        tm.assert_index_equal(result, expected)

    def test_intersection(self):
        index = Int64Index(range(5))

        other = Index([1, 2, 3, 4, 5])
        result = index.intersection(other)
        expected = Index(np.sort(np.intersect1d(index.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("dtype", ["int64", "uint64"])
    def test_int_float_union_dtype(self, dtype):
        # https://github.com/pandas-dev/pandas/issues/26778
        # [u]int | float -> float
        index = Index([0, 2, 3], dtype=dtype)
        other = Float64Index([0.5, 1.5])
        expected = Float64Index([0.0, 0.5, 1.5, 2.0, 3.0])
        result = index.union(other)
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_range_float_union_dtype(self):
        # https://github.com/pandas-dev/pandas/issues/26778
        index = RangeIndex(start=0, stop=3)
        other = Float64Index([0.5, 1.5])
        result = index.union(other)
        expected = Float64Index([0.0, 0.5, 1, 1.5, 2.0])
        tm.assert_index_equal(result, expected)

        result = other.union(index)
        tm.assert_index_equal(result, expected)

    def test_float64_index_difference(self):
        # https://github.com/pandas-dev/pandas/issues/35217
        float_index = Index([1.0, 2, 3])
        string_index = Index(["1", "2", "3"])

        result = float_index.difference(string_index)
        tm.assert_index_equal(result, float_index)

        result = string_index.difference(float_index)
        tm.assert_index_equal(result, string_index)

    def test_intersection_uint64_outside_int64_range(self, index_large):
        other = Index([2 ** 63, 2 ** 63 + 5, 2 ** 63 + 10, 2 ** 63 + 15, 2 ** 63 + 20])
        result = index_large.intersection(other)
        expected = Index(np.sort(np.intersect1d(index_large.values, other.values)))
        tm.assert_index_equal(result, expected)

        result = other.intersection(index_large)
        expected = Index(
            np.sort(np.asarray(np.intersect1d(index_large.values, other.values)))
        )
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "index2,keeps_name",
        [
            (Index([4, 7, 6, 5, 3], name="index"), True),
            (Index([4, 7, 6, 5, 3], name="other"), False),
        ],
    )
    def test_intersection_monotonic(self, index2, keeps_name, sort):
        index1 = Index([5, 3, 2, 4, 1], name="index")
        expected = Index([5, 3, 4])

        if keeps_name:
            expected.name = "index"

        result = index1.intersection(index2, sort=sort)
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

    def test_symmetric_difference(self, sort):
        # smoke
        index1 = Index([5, 2, 3, 4], name="index1")
        index2 = Index([2, 3, 4, 1])
        result = index1.symmetric_difference(index2, sort=sort)
        expected = Index([5, 1])
        assert tm.equalContents(result, expected)
        assert result.name is None
        if sort is None:
            expected = expected.sort_values()
        tm.assert_index_equal(result, expected)

        # __xor__ syntax
        with tm.assert_produces_warning(FutureWarning):
            expected = index1 ^ index2
        assert tm.equalContents(result, expected)
        assert result.name is None


class TestSetOpsSort:
    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_other_special(self, slice_):
        # https://github.com/pandas-dev/pandas/issues/24959

        idx = Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]
        tm.assert_index_equal(idx.union(other), idx)
        tm.assert_index_equal(other.union(idx), idx)

        # sort=False
        tm.assert_index_equal(idx.union(other, sort=False), idx)

    @pytest.mark.xfail(reason="Not implemented")
    @pytest.mark.parametrize("slice_", [slice(None), slice(0)])
    def test_union_sort_special_true(self, slice_):
        # TODO: decide on True behaviour
        # sort=True
        idx = Index([1, 0, 2])
        # default, sort=None
        other = idx[slice_]

        result = idx.union(other, sort=True)
        expected = Index([0, 1, 2])
        tm.assert_index_equal(result, expected)
