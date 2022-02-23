import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil


class TestComparisonOps(BaseOpsUtil):
    def _compare_other(self, data, op_name, other):
        op = self.get_op_from_name(op_name)

        # array
        result = pd.Series(op(data, other))
        expected = pd.Series(op(data._data, other), dtype="boolean")

        # fill the nan locations
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)

        # series
        s = pd.Series(data)
        result = op(s, other)

        expected = op(pd.Series(data._data), other)

        # fill the nan locations
        expected[data._mask] = pd.NA
        expected = expected.astype("boolean")

        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("other", [True, False, pd.NA, -1.0, 0.0, 1])
    def test_scalar(self, other, all_compare_operators):
        op = self.get_op_from_name(all_compare_operators)
        a = pd.array([1.0, 0.0, None], dtype="Float64")

        result = op(a, other)

        if other is pd.NA:
            expected = pd.array([None, None, None], dtype="boolean")
        else:
            values = op(a._data, other)
            expected = pd.arrays.BooleanArray(values, a._mask, copy=True)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(a, pd.array([1.0, 0.0, None], dtype="Float64"))

    def test_array(self, all_compare_operators):
        op = self.get_op_from_name(all_compare_operators)
        a = pd.array([0, 1, 2, None, None, None], dtype="Float64")
        b = pd.array([0, 1, None, 0, 1, None], dtype="Float64")

        result = op(a, b)
        values = op(a._data, b._data)
        mask = a._mask | b._mask

        expected = pd.arrays.BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = pd.NA
        tm.assert_extension_array_equal(
            a, pd.array([0, 1, 2, None, None, None], dtype="Float64")
        )
        tm.assert_extension_array_equal(
            b, pd.array([0, 1, None, 0, 1, None], dtype="Float64")
        )

    def test_compare_with_booleanarray(self, all_compare_operators):
        op = self.get_op_from_name(all_compare_operators)
        a = pd.array([True, False, None] * 3, dtype="boolean")
        b = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype="Float64")
        other = pd.array([False] * 3 + [True] * 3 + [None] * 3, dtype="boolean")
        expected = op(a, other)
        result = op(a, b)
        tm.assert_extension_array_equal(result, expected)
        expected = op(other, a)
        result = op(b, a)
        tm.assert_extension_array_equal(result, expected)

    def test_compare_with_integerarray(self, all_compare_operators):
        op = self.get_op_from_name(all_compare_operators)
        a = pd.array([0, 1, None] * 3, dtype="Int64")
        b = pd.array([0] * 3 + [1] * 3 + [None] * 3, dtype="Float64")
        other = b.astype("Int64")
        expected = op(a, other)
        result = op(a, b)
        tm.assert_extension_array_equal(result, expected)
        expected = op(other, a)
        result = op(b, a)
        tm.assert_extension_array_equal(result, expected)

    def test_no_shared_mask(self, data):
        result = data + 1
        assert np.shares_memory(result._mask, data._mask) is False

    def test_compare_to_string(self, dtype):
        # GH 28930
        s = pd.Series([1, None], dtype=dtype)
        result = s == "a"
        expected = pd.Series([False, pd.NA], dtype="boolean")

        self.assert_series_equal(result, expected)


def test_equals():
    # GH-30652
    # equals is generally tested in /tests/extension/base/methods, but this
    # specifically tests that two arrays of the same class but different dtype
    # do not evaluate equal
    a1 = pd.array([1, 2, None], dtype="Float64")
    a2 = pd.array([1, 2, None], dtype="Float32")
    assert a1.equals(a2) is False
