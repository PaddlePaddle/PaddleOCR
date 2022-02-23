import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    NaT,
    PeriodIndex,
    Series,
    TimedeltaIndex,
    Timestamp,
    date_range,
)
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float


class TestFillNA:
    def test_fillna_datetime(self, datetime_frame):
        tf = datetime_frame
        tf.loc[tf.index[:5], "A"] = np.nan
        tf.loc[tf.index[-5:], "A"] = np.nan

        zero_filled = datetime_frame.fillna(0)
        assert (zero_filled.loc[zero_filled.index[:5], "A"] == 0).all()

        padded = datetime_frame.fillna(method="pad")
        assert np.isnan(padded.loc[padded.index[:5], "A"]).all()
        assert (
            padded.loc[padded.index[-5:], "A"] == padded.loc[padded.index[-5], "A"]
        ).all()

        msg = "Must specify a fill 'value' or 'method'"
        with pytest.raises(ValueError, match=msg):
            datetime_frame.fillna()
        msg = "Cannot specify both 'value' and 'method'"
        with pytest.raises(ValueError, match=msg):
            datetime_frame.fillna(5, method="ffill")

    def test_fillna_mixed_type(self, float_string_frame):

        mf = float_string_frame
        mf.loc[mf.index[5:20], "foo"] = np.nan
        mf.loc[mf.index[-10:], "A"] = np.nan
        # TODO: make stronger assertion here, GH 25640
        mf.fillna(value=0)
        mf.fillna(method="pad")

    def test_fillna_mixed_float(self, mixed_float_frame):

        # mixed numeric (but no float16)
        mf = mixed_float_frame.reindex(columns=["A", "B", "D"])
        mf.loc[mf.index[-10:], "A"] = np.nan
        result = mf.fillna(value=0)
        _check_mixed_float(result, dtype={"C": None})

        result = mf.fillna(method="pad")
        _check_mixed_float(result, dtype={"C": None})

    def test_fillna_empty(self):
        # empty frame (GH#2778)
        df = DataFrame(columns=["x"])
        for m in ["pad", "backfill"]:
            df.x.fillna(method=m, inplace=True)
            df.x.fillna(method=m)

    def test_fillna_different_dtype(self):
        # with different dtype (GH#3386)
        df = DataFrame(
            [["a", "a", np.nan, "a"], ["b", "b", np.nan, "b"], ["c", "c", np.nan, "c"]]
        )

        result = df.fillna({2: "foo"})
        expected = DataFrame(
            [["a", "a", "foo", "a"], ["b", "b", "foo", "b"], ["c", "c", "foo", "c"]]
        )
        tm.assert_frame_equal(result, expected)

        return_value = df.fillna({2: "foo"}, inplace=True)
        tm.assert_frame_equal(df, expected)
        assert return_value is None

    def test_fillna_limit_and_value(self):
        # limit and value
        df = DataFrame(np.random.randn(10, 3))
        df.iloc[2:7, 0] = np.nan
        df.iloc[3:5, 2] = np.nan

        expected = df.copy()
        expected.iloc[2, 0] = 999
        expected.iloc[3, 2] = 999
        result = df.fillna(999, limit=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_datelike(self):
        # with datelike
        # GH#6344
        df = DataFrame(
            {
                "Date": [NaT, Timestamp("2014-1-1")],
                "Date2": [Timestamp("2013-1-1"), NaT],
            }
        )

        expected = df.copy()
        expected["Date"] = expected["Date"].fillna(df.loc[df.index[0], "Date2"])
        result = df.fillna(value={"Date": df["Date2"]})
        tm.assert_frame_equal(result, expected)

    def test_fillna_tzaware(self):
        # with timezone
        # GH#15855
        df = DataFrame({"A": [Timestamp("2012-11-11 00:00:00+01:00"), NaT]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        tm.assert_frame_equal(df.fillna(method="pad"), exp)

        df = DataFrame({"A": [NaT, Timestamp("2012-11-11 00:00:00+01:00")]})
        exp = DataFrame(
            {
                "A": [
                    Timestamp("2012-11-11 00:00:00+01:00"),
                    Timestamp("2012-11-11 00:00:00+01:00"),
                ]
            }
        )
        tm.assert_frame_equal(df.fillna(method="bfill"), exp)

    def test_fillna_tzaware_different_column(self):
        # with timezone in another column
        # GH#15522
        df = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1, 2, np.nan, np.nan],
            }
        )
        result = df.fillna(method="pad")
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=4, tz="US/Eastern"),
                "B": [1.0, 2.0, 2.0, 2.0],
            }
        )
        tm.assert_frame_equal(result, expected)

    def test_na_actions_categorical(self):

        cat = Categorical([1, 2, 3, np.nan], categories=[1, 2, 3])
        vals = ["a", "b", np.nan, "d"]
        df = DataFrame({"cats": cat, "vals": vals})
        cat2 = Categorical([1, 2, 3, 3], categories=[1, 2, 3])
        vals2 = ["a", "b", "b", "d"]
        df_exp_fill = DataFrame({"cats": cat2, "vals": vals2})
        cat3 = Categorical([1, 2, 3], categories=[1, 2, 3])
        vals3 = ["a", "b", np.nan]
        df_exp_drop_cats = DataFrame({"cats": cat3, "vals": vals3})
        cat4 = Categorical([1, 2], categories=[1, 2, 3])
        vals4 = ["a", "b"]
        df_exp_drop_all = DataFrame({"cats": cat4, "vals": vals4})

        # fillna
        res = df.fillna(value={"cats": 3, "vals": "b"})
        tm.assert_frame_equal(res, df_exp_fill)

        msg = "Cannot setitem on a Categorical with a new category"
        with pytest.raises(ValueError, match=msg):
            df.fillna(value={"cats": 4, "vals": "c"})

        res = df.fillna(method="pad")
        tm.assert_frame_equal(res, df_exp_fill)

        # dropna
        res = df.dropna(subset=["cats"])
        tm.assert_frame_equal(res, df_exp_drop_cats)

        res = df.dropna()
        tm.assert_frame_equal(res, df_exp_drop_all)

        # make sure that fillna takes missing values into account
        c = Categorical([np.nan, "b", np.nan], categories=["a", "b"])
        df = DataFrame({"cats": c, "vals": [1, 2, 3]})

        cat_exp = Categorical(["a", "b", "a"], categories=["a", "b"])
        df_exp = DataFrame({"cats": cat_exp, "vals": [1, 2, 3]})

        res = df.fillna("a")
        tm.assert_frame_equal(res, df_exp)

    def test_fillna_categorical_nan(self):
        # GH#14021
        # np.nan should always be a valid filler
        cat = Categorical([np.nan, 2, np.nan])
        val = Categorical([np.nan, np.nan, np.nan])
        df = DataFrame({"cats": cat, "vals": val})

        # GH#32950 df.median() is poorly behaved because there is no
        #  Categorical.median
        median = Series({"cats": 2.0, "vals": np.nan})

        res = df.fillna(median)
        v_exp = [np.nan, np.nan, np.nan]
        df_exp = DataFrame({"cats": [2, 2, 2], "vals": v_exp}, dtype="category")
        tm.assert_frame_equal(res, df_exp)

        result = df.cats.fillna(np.nan)
        tm.assert_series_equal(result, df.cats)

        result = df.vals.fillna(np.nan)
        tm.assert_series_equal(result, df.vals)

        idx = DatetimeIndex(
            ["2011-01-01 09:00", "2016-01-01 23:45", "2011-01-01 09:00", NaT, NaT]
        )
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        idx = PeriodIndex(["2011-01", "2011-01", "2011-01", NaT, NaT], freq="M")
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

        idx = TimedeltaIndex(["1 days", "2 days", "1 days", NaT, NaT])
        df = DataFrame({"a": Categorical(idx)})
        tm.assert_frame_equal(df.fillna(value=NaT), df)

    @td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) implement downcast
    def test_fillna_downcast(self):
        # GH#15277
        # infer int64 from float64
        df = DataFrame({"a": [1.0, np.nan]})
        result = df.fillna(0, downcast="infer")
        expected = DataFrame({"a": [1, 0]})
        tm.assert_frame_equal(result, expected)

        # infer int64 from float64 when fillna value is a dict
        df = DataFrame({"a": [1.0, np.nan]})
        result = df.fillna({"a": 0}, downcast="infer")
        expected = DataFrame({"a": [1, 0]})
        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) object upcasting
    def test_fillna_dtype_conversion(self):
        # make sure that fillna on an empty frame works
        df = DataFrame(index=["A", "B", "C"], columns=[1, 2, 3, 4, 5])
        result = df.dtypes
        expected = Series([np.dtype("object")] * 5, index=[1, 2, 3, 4, 5])
        tm.assert_series_equal(result, expected)

        result = df.fillna(1)
        expected = DataFrame(1, index=["A", "B", "C"], columns=[1, 2, 3, 4, 5])
        tm.assert_frame_equal(result, expected)

        # empty block
        df = DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
        result = df.fillna("nan")
        expected = DataFrame("nan", index=range(3), columns=["A", "B"])
        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_not_yet_implemented  # TODO(ArrayManager) object upcasting
    @pytest.mark.parametrize("val", ["", 1, np.nan, 1.0])
    def test_fillna_dtype_conversion_equiv_replace(self, val):
        df = DataFrame({"A": [1, np.nan], "B": [1.0, 2.0]})
        expected = df.replace(np.nan, val)
        result = df.fillna(val)
        tm.assert_frame_equal(result, expected)

    @td.skip_array_manager_invalid_test
    def test_fillna_datetime_columns(self):
        # GH#7095
        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        result = df.fillna("?")
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": date_range("20130101", periods=3),
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        tm.assert_frame_equal(result, expected)

        df = DataFrame(
            {
                "A": [-1, -2, np.nan],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), NaT],
                "C": ["foo", "bar", None],
                "D": ["foo2", "bar2", None],
            },
            index=date_range("20130110", periods=3),
        )
        result = df.fillna("?")
        expected = DataFrame(
            {
                "A": [-1, -2, "?"],
                "B": [Timestamp("2013-01-01"), Timestamp("2013-01-02"), "?"],
                "C": ["foo", "bar", "?"],
                "D": ["foo2", "bar2", "?"],
            },
            index=date_range("20130110", periods=3),
        )
        tm.assert_frame_equal(result, expected)

    def test_ffill(self, datetime_frame):
        datetime_frame["A"][:5] = np.nan
        datetime_frame["A"][-5:] = np.nan

        tm.assert_frame_equal(
            datetime_frame.ffill(), datetime_frame.fillna(method="ffill")
        )

    def test_ffill_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        df = DataFrame({"a": [1, 2, 3]})
        msg = (
            r"In a future version of pandas all arguments of DataFrame.ffill "
            r"will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.ffill(0)
        expected = DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_bfill(self, datetime_frame):
        datetime_frame["A"][:5] = np.nan
        datetime_frame["A"][-5:] = np.nan

        tm.assert_frame_equal(
            datetime_frame.bfill(), datetime_frame.fillna(method="bfill")
        )

    def test_bfill_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        df = DataFrame({"a": [1, 2, 3]})
        msg = (
            r"In a future version of pandas all arguments of DataFrame.bfill "
            r"will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.bfill(0)
        expected = DataFrame({"a": [1, 2, 3]})
        tm.assert_frame_equal(result, expected)

    def test_frame_pad_backfill_limit(self):
        index = np.arange(10)
        df = DataFrame(np.random.randn(10, 4), index=index)

        result = df[:2].reindex(index, method="pad", limit=5)

        expected = df[:2].reindex(index).fillna(method="pad")
        expected.iloc[-3:] = np.nan
        tm.assert_frame_equal(result, expected)

        result = df[-2:].reindex(index, method="backfill", limit=5)

        expected = df[-2:].reindex(index).fillna(method="backfill")
        expected.iloc[:3] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_frame_fillna_limit(self):
        index = np.arange(10)
        df = DataFrame(np.random.randn(10, 4), index=index)

        result = df[:2].reindex(index)
        result = result.fillna(method="pad", limit=5)

        expected = df[:2].reindex(index).fillna(method="pad")
        expected.iloc[-3:] = np.nan
        tm.assert_frame_equal(result, expected)

        result = df[-2:].reindex(index)
        result = result.fillna(method="backfill", limit=5)

        expected = df[-2:].reindex(index).fillna(method="backfill")
        expected.iloc[:3] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_fillna_skip_certain_blocks(self):
        # don't try to fill boolean, int blocks

        df = DataFrame(np.random.randn(10, 4).astype(int))

        # it works!
        df.fillna(np.nan)

    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_positive_limit(self, type):
        df = DataFrame(np.random.randn(10, 4)).astype(type)

        msg = "Limit must be greater than 0"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=-5)

    @pytest.mark.parametrize("type", [int, float])
    def test_fillna_integer_limit(self, type):
        df = DataFrame(np.random.randn(10, 4)).astype(type)

        msg = "Limit must be an integer"
        with pytest.raises(ValueError, match=msg):
            df.fillna(0, limit=0.5)

    def test_fillna_inplace(self):
        df = DataFrame(np.random.randn(10, 4))
        df[1][:4] = np.nan
        df[3][-4:] = np.nan

        expected = df.fillna(value=0)
        assert expected is not df

        df.fillna(value=0, inplace=True)
        tm.assert_frame_equal(df, expected)

        expected = df.fillna(value={0: 0}, inplace=True)
        assert expected is None

        df[1][:4] = np.nan
        df[3][-4:] = np.nan
        expected = df.fillna(method="ffill")
        assert expected is not df

        df.fillna(method="ffill", inplace=True)
        tm.assert_frame_equal(df, expected)

    def test_fillna_dict_series(self):
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            }
        )

        result = df.fillna({"a": 0, "b": 5})

        expected = df.copy()
        expected["a"] = expected["a"].fillna(0)
        expected["b"] = expected["b"].fillna(5)
        tm.assert_frame_equal(result, expected)

        # it works
        result = df.fillna({"a": 0, "b": 5, "d": 7})

        # Series treated same as dict
        result = df.fillna(df.max())
        expected = df.fillna(df.max().to_dict())
        tm.assert_frame_equal(result, expected)

        # disable this for now
        with pytest.raises(NotImplementedError, match="column by column"):
            df.fillna(df.max(1), axis=1)

    def test_fillna_dataframe(self):
        # GH#8377
        df = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, np.nan],
                "b": [1, 2, 3, np.nan, np.nan],
                "c": [np.nan, 1, 2, 3, 4],
            },
            index=list("VWXYZ"),
        )

        # df2 may have different index and columns
        df2 = DataFrame(
            {
                "a": [np.nan, 10, 20, 30, 40],
                "b": [50, 60, 70, 80, 90],
                "foo": ["bar"] * 5,
            },
            index=list("VWXuZ"),
        )

        result = df.fillna(df2)

        # only those columns and indices which are shared get filled
        expected = DataFrame(
            {
                "a": [np.nan, 1, 2, np.nan, 40],
                "b": [1, 2, 3, np.nan, 90],
                "c": [np.nan, 1, 2, 3, 4],
            },
            index=list("VWXYZ"),
        )

        tm.assert_frame_equal(result, expected)

    def test_fillna_columns(self):
        df = DataFrame(np.random.randn(10, 10))
        df.values[:, ::2] = np.nan

        result = df.fillna(method="ffill", axis=1)
        expected = df.T.fillna(method="pad").T
        tm.assert_frame_equal(result, expected)

        df.insert(6, "foo", 5)
        result = df.fillna(method="ffill", axis=1)
        expected = df.astype(float).fillna(method="ffill", axis=1)
        tm.assert_frame_equal(result, expected)

    def test_fillna_invalid_method(self, float_frame):
        with pytest.raises(ValueError, match="ffil"):
            float_frame.fillna(method="ffil")

    def test_fillna_invalid_value(self, float_frame):
        # list
        msg = '"value" parameter must be a scalar or dict, but you passed a "{}"'
        with pytest.raises(TypeError, match=msg.format("list")):
            float_frame.fillna([1, 2])
        # tuple
        with pytest.raises(TypeError, match=msg.format("tuple")):
            float_frame.fillna((1, 2))
        # frame with series
        msg = (
            '"value" parameter must be a scalar, dict or Series, but you '
            'passed a "DataFrame"'
        )
        with pytest.raises(TypeError, match=msg):
            float_frame.iloc[:, 0].fillna(float_frame)

    def test_fillna_col_reordering(self):
        cols = ["COL." + str(i) for i in range(5, 0, -1)]
        data = np.random.rand(20, 5)
        df = DataFrame(index=range(20), columns=cols, data=data)
        filled = df.fillna(method="ffill")
        assert df.columns.tolist() == filled.columns.tolist()

    def test_fill_corner(self, float_frame, float_string_frame):
        mf = float_string_frame
        mf.loc[mf.index[5:20], "foo"] = np.nan
        mf.loc[mf.index[-10:], "A"] = np.nan

        filled = float_string_frame.fillna(value=0)
        assert (filled.loc[filled.index[5:20], "foo"] == 0).all()
        del float_string_frame["foo"]

        empty_float = float_frame.reindex(columns=[])

        # TODO(wesm): unused?
        result = empty_float.fillna(value=0)  # noqa

    def test_fillna_downcast_dict(self):
        # GH#40809
        df = DataFrame({"col1": [1, np.nan]})
        result = df.fillna({"col1": 2}, downcast={"col1": "int64"})
        expected = DataFrame({"col1": [1, 2]})
        tm.assert_frame_equal(result, expected)

    def test_fillna_pos_args_deprecation(self):
        # https://github.com/pandas-dev/pandas/issues/41485
        df = DataFrame({"a": [1, 2, 3, np.nan]}, dtype=float)
        msg = (
            r"In a future version of pandas all arguments of DataFrame.fillna "
            r"except for the argument 'value' will be keyword-only"
        )
        with tm.assert_produces_warning(FutureWarning, match=msg):
            result = df.fillna(0, None, None)
        expected = DataFrame({"a": [1, 2, 3, 0]}, dtype=float)
        tm.assert_frame_equal(result, expected)


def test_fillna_nonconsolidated_frame():
    # https://github.com/pandas-dev/pandas/issues/36495
    df = DataFrame(
        [
            [1, 1, 1, 1.0],
            [2, 2, 2, 2.0],
            [3, 3, 3, 3.0],
        ],
        columns=["i1", "i2", "i3", "f1"],
    )
    df_nonconsol = df.pivot("i1", "i2")
    result = df_nonconsol.fillna(0)
    assert result.isna().sum().sum() == 0
