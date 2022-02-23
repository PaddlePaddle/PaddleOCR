import numpy as np

from pandas import (
    DataFrame,
    Series,
    period_range,
)


def test_iat(float_frame):

    for i, row in enumerate(float_frame.index):
        for j, col in enumerate(float_frame.columns):
            result = float_frame.iat[i, j]
            expected = float_frame.at[row, col]
            assert result == expected


def test_iat_duplicate_columns():
    # https://github.com/pandas-dev/pandas/issues/11754
    df = DataFrame([[1, 2]], columns=["x", "x"])
    assert df.iat[0, 0] == 1


def test_iat_getitem_series_with_period_index():
    # GH#4390, iat incorrectly indexing
    index = period_range("1/1/2001", periods=10)
    ser = Series(np.random.randn(10), index=index)
    expected = ser[index[0]]
    result = ser.iat[0]
    assert expected == result
