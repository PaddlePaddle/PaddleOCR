import numpy as np

from pandas import (
    Categorical,
    Series,
)
import pandas._testing as tm


class TestUnique:
    def test_unique_data_ownership(self):
        # it works! GH#1807
        Series(Series(["a", "c", "b"]).unique()).sort_values()

    def test_unique(self):
        # GH#714 also, dtype=float
        ser = Series([1.2345] * 100)
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

        # explicit f4 dtype
        ser = Series([1.2345] * 100, dtype="f4")
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

    def test_unique_nan_object_dtype(self):
        # NAs in object arrays GH#714
        ser = Series(["foo"] * 100, dtype="O")
        ser[::2] = np.nan
        result = ser.unique()
        assert len(result) == 2

    def test_unique_none(self):
        # decision about None
        ser = Series([1, 2, 3, None, None, None], dtype=object)
        result = ser.unique()
        expected = np.array([1, 2, 3, None], dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    def test_unique_categorical(self):
        # GH#18051
        cat = Categorical([])
        ser = Series(cat)
        result = ser.unique()
        tm.assert_categorical_equal(result, cat)

        cat = Categorical([np.nan])
        ser = Series(cat)
        result = ser.unique()
        tm.assert_categorical_equal(result, cat)
