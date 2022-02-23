""" test scalar indexing, including at and iat """
from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.indexing.common import Base


class TestScalar(Base):
    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_at_and_iat_get(self, kind):
        def _check(f, func, values=False):

            if f is not None:
                indices = self.generate_indices(f, values)
                for i in indices:
                    result = getattr(f, func)[i]
                    expected = self.get_value(func, f, i, values)
                    tm.assert_almost_equal(result, expected)

        d = getattr(self, kind)

        # iat
        for f in [d["ints"], d["uints"]]:
            _check(f, "iat", values=True)

        for f in [d["labels"], d["ts"], d["floats"]]:
            if f is not None:
                msg = "iAt based indexing can only have integer indexers"
                with pytest.raises(ValueError, match=msg):
                    self.check_values(f, "iat")

        # at
        for f in [d["ints"], d["uints"], d["labels"], d["ts"], d["floats"]]:
            _check(f, "at")

    @pytest.mark.parametrize("kind", ["series", "frame"])
    def test_at_and_iat_set(self, kind):
        def _check(f, func, values=False):

            if f is not None:
                indices = self.generate_indices(f, values)
                for i in indices:
                    getattr(f, func)[i] = 1
                    expected = self.get_value(func, f, i, values)
                    tm.assert_almost_equal(expected, 1)

        d = getattr(self, kind)

        # iat
        for f in [d["ints"], d["uints"]]:
            _check(f, "iat", values=True)

        for f in [d["labels"], d["ts"], d["floats"]]:
            if f is not None:
                msg = "iAt based indexing can only have integer indexers"
                with pytest.raises(ValueError, match=msg):
                    _check(f, "iat")

        # at
        for f in [d["ints"], d["uints"], d["labels"], d["ts"], d["floats"]]:
            _check(f, "at")


class TestScalar2:
    # TODO: Better name, just separating things that dont need Base class

    def test_at_iat_coercion(self):

        # as timestamp is not a tuple!
        dates = date_range("1/1/2000", periods=8)
        df = DataFrame(np.random.randn(8, 4), index=dates, columns=["A", "B", "C", "D"])
        s = df["A"]

        result = s.at[dates[5]]
        xp = s.values[5]
        assert result == xp

        # GH 7729
        # make sure we are boxing the returns
        s = Series(["2014-01-01", "2014-02-02"], dtype="datetime64[ns]")
        expected = Timestamp("2014-02-02")

        for r in [lambda: s.iat[1], lambda: s.iloc[1]]:
            result = r()
            assert result == expected

        s = Series(["1 days", "2 days"], dtype="timedelta64[ns]")
        expected = Timedelta("2 days")

        for r in [lambda: s.iat[1], lambda: s.iloc[1]]:
            result = r()
            assert result == expected

    def test_iat_invalid_args(self):
        pass

    def test_imethods_with_dups(self):

        # GH6493
        # iat/iloc with dups

        s = Series(range(5), index=[1, 1, 2, 2, 3], dtype="int64")
        result = s.iloc[2]
        assert result == 2
        result = s.iat[2]
        assert result == 2

        msg = "index 10 is out of bounds for axis 0 with size 5"
        with pytest.raises(IndexError, match=msg):
            s.iat[10]
        msg = "index -10 is out of bounds for axis 0 with size 5"
        with pytest.raises(IndexError, match=msg):
            s.iat[-10]

        result = s.iloc[[2, 3]]
        expected = Series([2, 3], [2, 2], dtype="int64")
        tm.assert_series_equal(result, expected)

        df = s.to_frame()
        result = df.iloc[2]
        expected = Series(2, index=[0], name=2)
        tm.assert_series_equal(result, expected)

        result = df.iat[2, 0]
        assert result == 2

    def test_frame_at_with_duplicate_axes(self):
        # GH#33041
        arr = np.random.randn(6).reshape(3, 2)
        df = DataFrame(arr, columns=["A", "A"])

        result = df.at[0, "A"]
        expected = df.iloc[0]

        tm.assert_series_equal(result, expected)

        result = df.T.at["A", 0]
        tm.assert_series_equal(result, expected)

        # setter
        df.at[1, "A"] = 2
        expected = Series([2.0, 2.0], index=["A", "A"], name=1)
        tm.assert_series_equal(df.iloc[1], expected)

    def test_at_getitem_dt64tz_values(self):
        # gh-15822
        df = DataFrame(
            {
                "name": ["John", "Anderson"],
                "date": [
                    Timestamp(2017, 3, 13, 13, 32, 56),
                    Timestamp(2017, 2, 16, 12, 10, 3),
                ],
            }
        )
        df["date"] = df["date"].dt.tz_localize("Asia/Shanghai")

        expected = Timestamp("2017-03-13 13:32:56+0800", tz="Asia/Shanghai")

        result = df.loc[0, "date"]
        assert result == expected

        result = df.at[0, "date"]
        assert result == expected

    def test_mixed_index_at_iat_loc_iloc_series(self):
        # GH 19860
        s = Series([1, 2, 3, 4, 5], index=["a", "b", "c", 1, 2])
        for el, item in s.items():
            assert s.at[el] == s.loc[el] == item
        for i in range(len(s)):
            assert s.iat[i] == s.iloc[i] == i + 1

        with pytest.raises(KeyError, match="^4$"):
            s.at[4]
        with pytest.raises(KeyError, match="^4$"):
            s.loc[4]

    def test_mixed_index_at_iat_loc_iloc_dataframe(self):
        # GH 19860
        df = DataFrame(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], columns=["a", "b", "c", 1, 2]
        )
        for rowIdx, row in df.iterrows():
            for el, item in row.items():
                assert df.at[rowIdx, el] == df.loc[rowIdx, el] == item

        for row in range(2):
            for i in range(5):
                assert df.iat[row, i] == df.iloc[row, i] == row * 5 + i

        with pytest.raises(KeyError, match="^3$"):
            df.at[0, 3]
        with pytest.raises(KeyError, match="^3$"):
            df.loc[0, 3]

    def test_iat_setter_incompatible_assignment(self):
        # GH 23236
        result = DataFrame({"a": [0, 1], "b": [4, 5]})
        result.iat[0, 0] = None
        expected = DataFrame({"a": [None, 1], "b": [4, 5]})
        tm.assert_frame_equal(result, expected)

    def test_getitem_zerodim_np_array(self):
        # GH24924
        # dataframe __getitem__
        df = DataFrame([[1, 2], [3, 4]])
        result = df[np.array(0)]
        expected = Series([1, 3], name=0)
        tm.assert_series_equal(result, expected)

        # series __getitem__
        s = Series([1, 2])
        result = s[np.array(0)]
        assert result == 1


def test_iat_dont_wrap_object_datetimelike():
    # GH#32809 .iat calls go through DataFrame._get_value, should not
    #  call maybe_box_datetimelike
    dti = date_range("2016-01-01", periods=3)
    tdi = dti - dti
    ser = Series(dti.to_pydatetime(), dtype=object)
    ser2 = Series(tdi.to_pytimedelta(), dtype=object)
    df = DataFrame({"A": ser, "B": ser2})
    assert (df.dtypes == object).all()

    for result in [df.at[0, "A"], df.iat[0, 0], df.loc[0, "A"], df.iloc[0, 0]]:
        assert result is ser[0]
        assert isinstance(result, datetime)
        assert not isinstance(result, Timestamp)

    for result in [df.at[1, "B"], df.iat[1, 1], df.loc[1, "B"], df.iloc[1, 1]]:
        assert result is ser2[1]
        assert isinstance(result, timedelta)
        assert not isinstance(result, Timedelta)


def test_at_with_tuple_index_get():
    # GH 26989
    # DataFrame.at getter works with Index of tuples
    df = DataFrame({"a": [1, 2]}, index=[(1, 2), (3, 4)])
    assert df.index.nlevels == 1
    assert df.at[(1, 2), "a"] == 1

    # Series.at getter works with Index of tuples
    series = df["a"]
    assert series.index.nlevels == 1
    assert series.at[(1, 2)] == 1


def test_at_with_tuple_index_set():
    # GH 26989
    # DataFrame.at setter works with Index of tuples
    df = DataFrame({"a": [1, 2]}, index=[(1, 2), (3, 4)])
    assert df.index.nlevels == 1
    df.at[(1, 2), "a"] = 2
    assert df.at[(1, 2), "a"] == 2

    # Series.at setter works with Index of tuples
    series = df["a"]
    assert series.index.nlevels == 1
    series.at[1, 2] = 3
    assert series.at[1, 2] == 3


class TestMultiIndexScalar:
    def test_multiindex_at_get(self):
        # GH 26989
        # DataFrame.at and DataFrame.loc getter works with MultiIndex
        df = DataFrame({"a": [1, 2]}, index=[[1, 2], [3, 4]])
        assert df.index.nlevels == 2
        assert df.at[(1, 3), "a"] == 1
        assert df.loc[(1, 3), "a"] == 1

        # Series.at and Series.loc getter works with MultiIndex
        series = df["a"]
        assert series.index.nlevels == 2
        assert series.at[1, 3] == 1
        assert series.loc[1, 3] == 1

    def test_multiindex_at_set(self):
        # GH 26989
        # DataFrame.at and DataFrame.loc setter works with MultiIndex
        df = DataFrame({"a": [1, 2]}, index=[[1, 2], [3, 4]])
        assert df.index.nlevels == 2
        df.at[(1, 3), "a"] = 3
        assert df.at[(1, 3), "a"] == 3
        df.loc[(1, 3), "a"] = 4
        assert df.loc[(1, 3), "a"] == 4

        # Series.at and Series.loc setter works with MultiIndex
        series = df["a"]
        assert series.index.nlevels == 2
        series.at[1, 3] = 5
        assert series.at[1, 3] == 5
        series.loc[1, 3] = 6
        assert series.loc[1, 3] == 6

    def test_multiindex_at_get_one_level(self):
        # GH#38053
        s2 = Series((0, 1), index=[[False, True]])
        result = s2.at[False]
        assert result == 0
