import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.tests.extension.base import BaseOpsUtil


@pytest.fixture
def data():
    return pd.array(
        [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False],
        dtype="boolean",
    )


class TestComparisonOps(BaseOpsUtil):
    def _compare_other(self, data, op_name, other):
        op = self.get_op_from_name(op_name)

        # array
        result = pd.Series(op(data, other))
        expected = pd.Series(op(data._data, other), dtype="boolean")
        # propagate NAs
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)

        # series
        s = pd.Series(data)
        result = op(s, other)

        expected = pd.Series(data._data)
        expected = op(expected, other)
        expected = expected.astype("boolean")
        # propagate NAs
        expected[data._mask] = pd.NA

        tm.assert_series_equal(result, expected)

    def test_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        self._compare_other(data, op_name, True)

    def test_compare_array(self, data, all_compare_operators):
        op_name = all_compare_operators
        other = pd.array([True] * len(data), dtype="boolean")
        self._compare_other(data, op_name, other)
        other = np.array([True] * len(data))
        self._compare_other(data, op_name, other)
        other = pd.Series([True] * len(data))
        self._compare_other(data, op_name, other)

    @pytest.mark.parametrize("other", [True, False, pd.NA])
    def test_scalar(self, other, all_compare_operators):
        op = self.get_op_from_name(all_compare_operators)
        a = pd.array([True, False, None], dtype="boolean")

        result = op(a, other)

        if other is pd.NA:
            expected = pd.array([None, None, None], dtype="boolean")
        else:
            values = op(a._data, other)
            expected = BooleanArray(values, a._mask, copy=True)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = None
        tm.assert_extension_array_equal(
            a, pd.array([True, False, None], dtype="boolean")
        )

    def test_array(self, all_compare_operators):
        op = self.get_op_from_name(all_compare_operators)
        a = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        b = pd.array([True, False, None] * 3, dtype="boolean")

        result = op(a, b)

        values = op(a._data, b._data)
        mask = a._mask | b._mask
        expected = BooleanArray(values, mask)
        tm.assert_extension_array_equal(result, expected)

        # ensure we haven't mutated anything inplace
        result[0] = None
        tm.assert_extension_array_equal(
            a, pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
        )
        tm.assert_extension_array_equal(
            b, pd.array([True, False, None] * 3, dtype="boolean")
        )
