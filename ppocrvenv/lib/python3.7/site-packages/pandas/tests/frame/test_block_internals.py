from datetime import (
    datetime,
    timedelta,
)
from io import StringIO
import itertools

import numpy as np
import pytest

from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    Series,
    Timestamp,
    compat,
    date_range,
    option_context,
)
import pandas._testing as tm
from pandas.core.internals import (
    NumericBlock,
    ObjectBlock,
)

# Segregated collection of methods that require the BlockManager internal data
# structure


# TODO(ArrayManager) check which of those tests need to be rewritten to test the
# equivalent for ArrayManager
pytestmark = td.skip_array_manager_invalid_test


class TestDataFrameBlockInternals:
    def test_setitem_invalidates_datetime_index_freq(self):
        # GH#24096 altering a datetime64tz column inplace invalidates the
        #  `freq` attribute on the underlying DatetimeIndex

        dti = date_range("20130101", periods=3, tz="US/Eastern")
        ts = dti[1]

        df = DataFrame({"B": dti})
        assert df["B"]._values.freq is None

        df.iloc[1, 0] = pd.NaT
        assert df["B"]._values.freq is None

        # check that the DatetimeIndex was not altered in place
        assert dti.freq == "D"
        assert dti[1] == ts

    def test_cast_internals(self, float_frame):
        casted = DataFrame(float_frame._mgr, dtype=int)
        expected = DataFrame(float_frame._series, dtype=int)
        tm.assert_frame_equal(casted, expected)

        casted = DataFrame(float_frame._mgr, dtype=np.int32)
        expected = DataFrame(float_frame._series, dtype=np.int32)
        tm.assert_frame_equal(casted, expected)

    def test_consolidate(self, float_frame):
        float_frame["E"] = 7.0
        consolidated = float_frame._consolidate()
        assert len(consolidated._mgr.blocks) == 1

        # Ensure copy, do I want this?
        recons = consolidated._consolidate()
        assert recons is not consolidated
        tm.assert_frame_equal(recons, consolidated)

        float_frame["F"] = 8.0
        assert len(float_frame._mgr.blocks) == 3

        return_value = float_frame._consolidate_inplace()
        assert return_value is None
        assert len(float_frame._mgr.blocks) == 1

    def test_consolidate_inplace(self, float_frame):
        frame = float_frame.copy()  # noqa

        # triggers in-place consolidation
        for letter in range(ord("A"), ord("Z")):
            float_frame[chr(letter)] = chr(letter)

    def test_values_consolidate(self, float_frame):
        float_frame["E"] = 7.0
        assert not float_frame._mgr.is_consolidated()
        _ = float_frame.values
        assert float_frame._mgr.is_consolidated()

    def test_modify_values(self, float_frame):
        float_frame.values[5] = 5
        assert (float_frame.values[5] == 5).all()

        # unconsolidated
        float_frame["E"] = 7.0
        col = float_frame["E"]
        float_frame.values[6] = 6
        assert (float_frame.values[6] == 6).all()

        # check that item_cache was cleared
        assert float_frame["E"] is not col
        assert (col == 7).all()

    def test_boolean_set_uncons(self, float_frame):
        float_frame["E"] = 7.0

        expected = float_frame.values.copy()
        expected[expected > 1] = 2

        float_frame[float_frame > 1] = 2
        tm.assert_almost_equal(expected, float_frame.values)

    def test_constructor_with_convert(self):
        # this is actually mostly a test of lib.maybe_convert_objects
        # #2845
        df = DataFrame({"A": [2 ** 63 - 1]})
        result = df["A"]
        expected = Series(np.asarray([2 ** 63 - 1], np.int64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2 ** 63]})
        result = df["A"]
        expected = Series(np.asarray([2 ** 63], np.uint64), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [datetime(2005, 1, 1), True]})
        result = df["A"]
        expected = Series(
            np.asarray([datetime(2005, 1, 1), True], np.object_), name="A"
        )
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [None, 1]})
        result = df["A"]
        expected = Series(np.asarray([np.nan, 1], np.float_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0, 2]})
        result = df["A"]
        expected = Series(np.asarray([1.0, 2], np.float_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, 3]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, 3], np.complex_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, 3.0]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, 3.0], np.complex_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, True]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, True], np.object_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0, None]})
        result = df["A"]
        expected = Series(np.asarray([1.0, np.nan], np.float_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [1.0 + 2.0j, None]})
        result = df["A"]
        expected = Series(np.asarray([1.0 + 2.0j, np.nan], np.complex_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2.0, 1, True, None]})
        result = df["A"]
        expected = Series(np.asarray([2.0, 1, True, None], np.object_), name="A")
        tm.assert_series_equal(result, expected)

        df = DataFrame({"A": [2.0, 1, datetime(2006, 1, 1), None]})
        result = df["A"]
        expected = Series(
            np.asarray([2.0, 1, datetime(2006, 1, 1), None], np.object_), name="A"
        )
        tm.assert_series_equal(result, expected)

    def test_construction_with_mixed(self, float_string_frame):
        # test construction edge cases with mixed types

        # f7u12, this does not work without extensive workaround
        data = [
            [datetime(2001, 1, 5), np.nan, datetime(2001, 1, 2)],
            [datetime(2000, 1, 2), datetime(2000, 1, 3), datetime(2000, 1, 1)],
        ]
        df = DataFrame(data)

        # check dtypes
        result = df.dtypes
        expected = Series({"datetime64[ns]": 3})

        # mixed-type frames
        float_string_frame["datetime"] = datetime.now()
        float_string_frame["timedelta"] = timedelta(days=1, seconds=1)
        assert float_string_frame["datetime"].dtype == "M8[ns]"
        assert float_string_frame["timedelta"].dtype == "m8[ns]"
        result = float_string_frame.dtypes
        expected = Series(
            [np.dtype("float64")] * 4
            + [
                np.dtype("object"),
                np.dtype("datetime64[ns]"),
                np.dtype("timedelta64[ns]"),
            ],
            index=list("ABCD") + ["foo", "datetime", "timedelta"],
        )
        tm.assert_series_equal(result, expected)

    def test_construction_with_conversions(self):

        # convert from a numpy array of non-ns timedelta64
        arr = np.array([1, 2, 3], dtype="timedelta64[s]")
        df = DataFrame(index=range(3))
        df["A"] = arr
        expected = DataFrame(
            {"A": pd.timedelta_range("00:00:01", periods=3, freq="s")}, index=range(3)
        )
        tm.assert_frame_equal(df, expected)

        expected = DataFrame(
            {
                "dt1": Timestamp("20130101"),
                "dt2": date_range("20130101", periods=3),
                # 'dt3' : date_range('20130101 00:00:01',periods=3,freq='s'),
            },
            index=range(3),
        )

        df = DataFrame(index=range(3))
        df["dt1"] = np.datetime64("2013-01-01")
        df["dt2"] = np.array(
            ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
        )

        # df['dt3'] = np.array(['2013-01-01 00:00:01','2013-01-01
        # 00:00:02','2013-01-01 00:00:03'],dtype='datetime64[s]')

        tm.assert_frame_equal(df, expected)

    def test_constructor_compound_dtypes(self):
        # GH 5191
        # compound dtypes should raise not-implementederror

        def f(dtype):
            data = list(itertools.repeat((datetime(2001, 1, 1), "aa", 20), 9))
            return DataFrame(data=data, columns=["A", "B", "C"], dtype=dtype)

        msg = "compound dtypes are not implemented in the DataFrame constructor"
        with pytest.raises(NotImplementedError, match=msg):
            f([("A", "datetime64[h]"), ("B", "str"), ("C", "int32")])

        # these work (though results may be unexpected)
        depr_msg = "either all columns will be cast to that dtype, or a TypeError will"
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            f("int64")
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            f("float64")

        # 10822
        # invalid error message on dt inference
        if not compat.is_platform_windows():
            f("M8[ns]")

    def test_pickle(self, float_string_frame, timezone_frame):
        empty_frame = DataFrame()

        unpickled = tm.round_trip_pickle(float_string_frame)
        tm.assert_frame_equal(float_string_frame, unpickled)

        # buglet
        float_string_frame._mgr.ndim

        # empty
        unpickled = tm.round_trip_pickle(empty_frame)
        repr(unpickled)

        # tz frame
        unpickled = tm.round_trip_pickle(timezone_frame)
        tm.assert_frame_equal(timezone_frame, unpickled)

    def test_consolidate_datetime64(self):
        # numpy vstack bug

        data = (
            "starting,ending,measure\n"
            "2012-06-21 00:00,2012-06-23 07:00,77\n"
            "2012-06-23 07:00,2012-06-23 16:30,65\n"
            "2012-06-23 16:30,2012-06-25 08:00,77\n"
            "2012-06-25 08:00,2012-06-26 12:00,0\n"
            "2012-06-26 12:00,2012-06-27 08:00,77\n"
        )
        df = pd.read_csv(StringIO(data), parse_dates=[0, 1])

        ser_starting = df.starting
        ser_starting.index = ser_starting.values
        ser_starting = ser_starting.tz_localize("US/Eastern")
        ser_starting = ser_starting.tz_convert("UTC")
        ser_starting.index.name = "starting"

        ser_ending = df.ending
        ser_ending.index = ser_ending.values
        ser_ending = ser_ending.tz_localize("US/Eastern")
        ser_ending = ser_ending.tz_convert("UTC")
        ser_ending.index.name = "ending"

        df.starting = ser_starting.index
        df.ending = ser_ending.index

        tm.assert_index_equal(pd.DatetimeIndex(df.starting), ser_starting.index)
        tm.assert_index_equal(pd.DatetimeIndex(df.ending), ser_ending.index)

    def test_is_mixed_type(self, float_frame, float_string_frame):
        assert not float_frame._is_mixed_type
        assert float_string_frame._is_mixed_type

    def test_stale_cached_series_bug_473(self):

        # this is chained, but ok
        with option_context("chained_assignment", None):
            Y = DataFrame(
                np.random.random((4, 4)),
                index=("a", "b", "c", "d"),
                columns=("e", "f", "g", "h"),
            )
            repr(Y)
            Y["e"] = Y["e"].astype("object")
            Y["g"]["c"] = np.NaN
            repr(Y)
            result = Y.sum()  # noqa
            exp = Y["g"].sum()  # noqa
            assert pd.isna(Y["g"]["c"])

    def test_strange_column_corruption_issue(self):
        # FIXME: dont leave commented-out
        # (wesm) Unclear how exactly this is related to internal matters
        df = DataFrame(index=[0, 1])
        df[0] = np.nan
        wasCol = {}

        with tm.assert_produces_warning(PerformanceWarning):
            for i, dt in enumerate(df.index):
                for col in range(100, 200):
                    if col not in wasCol:
                        wasCol[col] = 1
                        df[col] = np.nan
                    df[col][dt] = i

        myid = 100

        first = len(df.loc[pd.isna(df[myid]), [myid]])
        second = len(df.loc[pd.isna(df[myid]), [myid]])
        assert first == second == 0

    def test_constructor_no_pandas_array(self):
        # Ensure that PandasArray isn't allowed inside Series
        # See https://github.com/pandas-dev/pandas/issues/23995 for more.
        arr = Series([1, 2, 3]).array
        result = DataFrame({"A": arr})
        expected = DataFrame({"A": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)
        assert isinstance(result._mgr.blocks[0], NumericBlock)

    def test_add_column_with_pandas_array(self):
        # GH 26390
        df = DataFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
        df["c"] = pd.arrays.PandasArray(np.array([1, 2, None, 3], dtype=object))
        df2 = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": ["a", "b", "c", "d"],
                "c": pd.arrays.PandasArray(np.array([1, 2, None, 3], dtype=object)),
            }
        )
        assert type(df["c"]._mgr.blocks[0]) == ObjectBlock
        assert type(df2["c"]._mgr.blocks[0]) == ObjectBlock
        tm.assert_frame_equal(df, df2)


def test_update_inplace_sets_valid_block_values():
    # https://github.com/pandas-dev/pandas/issues/33457
    df = DataFrame({"a": Series([1, 2, None], dtype="category")})

    # inplace update of a single column
    df["a"].fillna(1, inplace=True)

    # check we haven't put a Series into any block.values
    assert isinstance(df._mgr.blocks[0].values, Categorical)

    # smoketest for OP bug from GH#35731
    assert df.isnull().sum().sum() == 0


def test_nonconsolidated_item_cache_take():
    # https://github.com/pandas-dev/pandas/issues/35521

    # create non-consolidated dataframe with object dtype columns
    df = DataFrame()
    df["col1"] = Series(["a"], dtype=object)
    df["col2"] = Series([0], dtype=object)

    # access column (item cache)
    df["col1"] == "A"
    # take operation
    # (regression was that this consolidated but didn't reset item cache,
    # resulting in an invalid cache and the .at operation not working properly)
    df[df["col2"] == 0]

    # now setting value should update actual dataframe
    df.at[0, "col1"] = "A"

    expected = DataFrame({"col1": ["A"], "col2": [0]}, dtype=object)
    tm.assert_frame_equal(df, expected)
    assert df.at[0, "col1"] == "A"
