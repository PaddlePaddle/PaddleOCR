import numpy as np

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,
    Timestamp,
)
import pandas._testing as tm


class TestToNumpy:
    def test_to_numpy(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})
        expected = np.array([[1, 3], [2, 4.5]])
        result = df.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    def test_to_numpy_dtype(self):
        df = DataFrame({"A": [1, 2], "B": [3, 4.5]})
        expected = np.array([[1, 3], [2, 4]], dtype="int64")
        result = df.to_numpy(dtype="int64")
        tm.assert_numpy_array_equal(result, expected)

    @td.skip_array_manager_invalid_test
    def test_to_numpy_copy(self):
        arr = np.random.randn(4, 3)
        df = DataFrame(arr)
        assert df.values.base is arr
        assert df.to_numpy(copy=False).base is arr
        assert df.to_numpy(copy=True).base is not arr

    def test_to_numpy_mixed_dtype_to_str(self):
        # https://github.com/pandas-dev/pandas/issues/35455
        df = DataFrame([[Timestamp("2020-01-01 00:00:00"), 100.0]])
        result = df.to_numpy(dtype=str)
        expected = np.array([["2020-01-01 00:00:00", "100.0"]], dtype=str)
        tm.assert_numpy_array_equal(result, expected)
