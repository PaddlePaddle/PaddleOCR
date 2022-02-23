import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
)
import pandas._testing as tm


class TestDataFrameUpdate:
    def test_update_nan(self):
        # #15593 #15617
        # test 1
        df1 = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
        df2 = DataFrame({"A": [None, 2, 3]})
        expected = df1.copy()
        df1.update(df2, overwrite=False)

        tm.assert_frame_equal(df1, expected)

        # test 2
        df1 = DataFrame({"A": [1.0, None, 3], "B": date_range("2000", periods=3)})
        df2 = DataFrame({"A": [None, 2, 3]})
        expected = DataFrame({"A": [1.0, 2, 3], "B": date_range("2000", periods=3)})
        df1.update(df2, overwrite=False)

        tm.assert_frame_equal(df1, expected)

    def test_update(self):
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other)

        expected = DataFrame(
            [[1.5, np.nan, 3], [3.6, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
        )
        tm.assert_frame_equal(df, expected)

    def test_update_dtypes(self):

        # gh 3016
        df = DataFrame(
            [[1.0, 2.0, False, True], [4.0, 5.0, True, False]],
            columns=["A", "B", "bool1", "bool2"],
        )

        other = DataFrame([[45, 45]], index=[0], columns=["A", "B"])
        df.update(other)

        expected = DataFrame(
            [[45.0, 45.0, False, True], [4.0, 5.0, True, False]],
            columns=["A", "B", "bool1", "bool2"],
        )
        tm.assert_frame_equal(df, expected)

    def test_update_nooverwrite(self):
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other, overwrite=False)

        expected = DataFrame(
            [[1.5, np.nan, 3], [1.5, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 3.0]]
        )
        tm.assert_frame_equal(df, expected)

    def test_update_filtered(self):
        df = DataFrame(
            [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

        df.update(other, filter_func=lambda x: x > 2)

        expected = DataFrame(
            [[1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
        )
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "bad_kwarg, exception, msg",
        [
            # errors must be 'ignore' or 'raise'
            ({"errors": "something"}, ValueError, "The parameter errors must.*"),
            ({"join": "inner"}, NotImplementedError, "Only left join is supported"),
        ],
    )
    def test_update_raise_bad_parameter(self, bad_kwarg, exception, msg):
        df = DataFrame([[1.5, 1, 3.0]])
        with pytest.raises(exception, match=msg):
            df.update(df, **bad_kwarg)

    def test_update_raise_on_overlap(self):
        df = DataFrame(
            [[1.5, 1, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
        )

        other = DataFrame([[2.0, np.nan], [np.nan, 7]], index=[1, 3], columns=[1, 2])
        with pytest.raises(ValueError, match="Data overlaps"):
            df.update(other, errors="raise")

    def test_update_from_non_df(self):
        d = {"a": Series([1, 2, 3, 4]), "b": Series([5, 6, 7, 8])}
        df = DataFrame(d)

        d["a"] = Series([5, 6, 7, 8])
        df.update(d)

        expected = DataFrame(d)

        tm.assert_frame_equal(df, expected)

        d = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
        df = DataFrame(d)

        d["a"] = [5, 6, 7, 8]
        df.update(d)

        expected = DataFrame(d)

        tm.assert_frame_equal(df, expected)

    def test_update_datetime_tz(self):
        # GH 25807
        result = DataFrame([pd.Timestamp("2019", tz="UTC")])
        result.update(result)
        expected = DataFrame([pd.Timestamp("2019", tz="UTC")])
        tm.assert_frame_equal(result, expected)

    def test_update_with_different_dtype(self):
        # GH#3217
        df = DataFrame({"a": [1, 3], "b": [np.nan, 2]})
        df["c"] = np.nan
        df["c"].update(Series(["foo"], index=[0]))

        expected = DataFrame({"a": [1, 3], "b": [np.nan, 2], "c": ["foo", np.nan]})
        tm.assert_frame_equal(df, expected)
