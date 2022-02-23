from datetime import datetime

import numpy as np
import pytest

from pandas.core.dtypes.common import is_scalar

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    StringDtype,
    Timestamp,
    date_range,
    isna,
)
import pandas._testing as tm


@pytest.fixture(params=["default", "float_string", "mixed_float", "mixed_int"])
def where_frame(request, float_string_frame, mixed_float_frame, mixed_int_frame):
    if request.param == "default":
        return DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
    if request.param == "float_string":
        return float_string_frame
    if request.param == "mixed_float":
        return mixed_float_frame
    if request.param == "mixed_int":
        return mixed_int_frame


def _safe_add(df):
    # only add to the numeric items
    def is_ok(s):
        return (
            issubclass(s.dtype.type, (np.integer, np.floating)) and s.dtype != "uint8"
        )

    return DataFrame(dict((c, s + 1) if is_ok(s) else (c, s) for c, s in df.items()))


class TestDataFrameIndexingWhere:
    def test_where_get(self, where_frame, float_string_frame):
        def _check_get(df, cond, check_dtypes=True):
            other1 = _safe_add(df)
            rs = df.where(cond, other1)
            rs2 = df.where(cond.values, other1)
            for k, v in rs.items():
                exp = Series(np.where(cond[k], df[k], other1[k]), index=v.index)
                tm.assert_series_equal(v, exp, check_names=False)
            tm.assert_frame_equal(rs, rs2)

            # dtypes
            if check_dtypes:
                assert (rs.dtypes == df.dtypes).all()

        # check getting
        df = where_frame
        if df is float_string_frame:
            msg = "'>' not supported between instances of 'str' and 'int'"
            with pytest.raises(TypeError, match=msg):
                df > 0
            return
        cond = df > 0
        _check_get(df, cond)

    def test_where_upcasting(self):
        # upcasting case (GH # 2794)
        df = DataFrame(
            {
                c: Series([1] * 3, dtype=c)
                for c in ["float32", "float64", "int32", "int64"]
            }
        )
        df.iloc[1, :] = 0
        result = df.dtypes
        expected = Series(
            [
                np.dtype("float32"),
                np.dtype("float64"),
                np.dtype("int32"),
                np.dtype("int64"),
            ],
            index=["float32", "float64", "int32", "int64"],
        )

        # when we don't preserve boolean casts
        #
        # expected = Series({ 'float32' : 1, 'float64' : 3 })

        tm.assert_series_equal(result, expected)

    def test_where_alignment(self, where_frame, float_string_frame):
        # aligning
        def _check_align(df, cond, other, check_dtypes=True):
            rs = df.where(cond, other)
            for i, k in enumerate(rs.columns):
                result = rs[k]
                d = df[k].values
                c = cond[k].reindex(df[k].index).fillna(False).values

                if is_scalar(other):
                    o = other
                else:
                    if isinstance(other, np.ndarray):
                        o = Series(other[:, i], index=result.index).values
                    else:
                        o = other[k].values

                new_values = d if c.all() else np.where(c, d, o)
                expected = Series(new_values, index=result.index, name=k)

                # since we can't always have the correct numpy dtype
                # as numpy doesn't know how to downcast, don't check
                tm.assert_series_equal(result, expected, check_dtype=False)

            # dtypes
            # can't check dtype when other is an ndarray

            if check_dtypes and not isinstance(other, np.ndarray):
                assert (rs.dtypes == df.dtypes).all()

        df = where_frame
        if df is float_string_frame:
            msg = "'>' not supported between instances of 'str' and 'int'"
            with pytest.raises(TypeError, match=msg):
                df > 0
            return

        # other is a frame
        cond = (df > 0)[1:]
        _check_align(df, cond, _safe_add(df))

        # check other is ndarray
        cond = df > 0
        _check_align(df, cond, (_safe_add(df).values))

        # integers are upcast, so don't check the dtypes
        cond = df > 0
        check_dtypes = all(not issubclass(s.type, np.integer) for s in df.dtypes)
        _check_align(df, cond, np.nan, check_dtypes=check_dtypes)

    def test_where_invalid(self):
        # invalid conditions
        df = DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
        cond = df > 0

        err1 = (df + 1).values[0:2, :]
        msg = "other must be the same shape as self when an ndarray"
        with pytest.raises(ValueError, match=msg):
            df.where(cond, err1)

        err2 = cond.iloc[:2, :].values
        other1 = _safe_add(df)
        msg = "Array conditional must be same shape as self"
        with pytest.raises(ValueError, match=msg):
            df.where(err2, other1)

        with pytest.raises(ValueError, match=msg):
            df.mask(True)
        with pytest.raises(ValueError, match=msg):
            df.mask(0)

    def test_where_set(self, where_frame, float_string_frame):
        # where inplace

        def _check_set(df, cond, check_dtypes=True):
            dfi = df.copy()
            econd = cond.reindex_like(df).fillna(True)
            expected = dfi.mask(~econd)

            return_value = dfi.where(cond, np.nan, inplace=True)
            assert return_value is None
            tm.assert_frame_equal(dfi, expected)

            # dtypes (and confirm upcasts)x
            if check_dtypes:
                for k, v in df.dtypes.items():
                    if issubclass(v.type, np.integer) and not cond[k].all():
                        v = np.dtype("float64")
                    assert dfi[k].dtype == v

        df = where_frame
        if df is float_string_frame:
            msg = "'>' not supported between instances of 'str' and 'int'"
            with pytest.raises(TypeError, match=msg):
                df > 0
            return

        cond = df > 0
        _check_set(df, cond)

        cond = df >= 0
        _check_set(df, cond)

        # aligning
        cond = (df >= 0)[1:]
        _check_set(df, cond)

    def test_where_series_slicing(self):
        # GH 10218
        # test DataFrame.where with Series slicing
        df = DataFrame({"a": range(3), "b": range(4, 7)})
        result = df.where(df["a"] == 1)
        expected = df[df["a"] == 1].reindex(df.index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("klass", [list, tuple, np.array])
    def test_where_array_like(self, klass):
        # see gh-15414
        df = DataFrame({"a": [1, 2, 3]})
        cond = [[False], [True], [True]]
        expected = DataFrame({"a": [np.nan, 2, 3]})

        result = df.where(klass(cond))
        tm.assert_frame_equal(result, expected)

        df["b"] = 2
        expected["b"] = [2, np.nan, 2]
        cond = [[False, True], [True, False], [True, True]]

        result = df.where(klass(cond))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "cond",
        [
            [[1], [0], [1]],
            Series([[2], [5], [7]]),
            DataFrame({"a": [2, 5, 7]}),
            [["True"], ["False"], ["True"]],
            [[Timestamp("2017-01-01")], [pd.NaT], [Timestamp("2017-01-02")]],
        ],
    )
    def test_where_invalid_input_single(self, cond):
        # see gh-15414: only boolean arrays accepted
        df = DataFrame({"a": [1, 2, 3]})
        msg = "Boolean array expected for the condition"

        with pytest.raises(ValueError, match=msg):
            df.where(cond)

    @pytest.mark.parametrize(
        "cond",
        [
            [[0, 1], [1, 0], [1, 1]],
            Series([[0, 2], [5, 0], [4, 7]]),
            [["False", "True"], ["True", "False"], ["True", "True"]],
            DataFrame({"a": [2, 5, 7], "b": [4, 8, 9]}),
            [
                [pd.NaT, Timestamp("2017-01-01")],
                [Timestamp("2017-01-02"), pd.NaT],
                [Timestamp("2017-01-03"), Timestamp("2017-01-03")],
            ],
        ],
    )
    def test_where_invalid_input_multiple(self, cond):
        # see gh-15414: only boolean arrays accepted
        df = DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})
        msg = "Boolean array expected for the condition"

        with pytest.raises(ValueError, match=msg):
            df.where(cond)

    def test_where_dataframe_col_match(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        cond = DataFrame([[True, False, True], [False, False, True]])

        result = df.where(cond)
        expected = DataFrame([[1.0, np.nan, 3], [np.nan, np.nan, 6]])
        tm.assert_frame_equal(result, expected)

        # this *does* align, though has no matching columns
        cond.columns = ["a", "b", "c"]
        result = df.where(cond)
        expected = DataFrame(np.nan, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_where_ndframe_align(self):
        msg = "Array conditional must be same shape as self"
        df = DataFrame([[1, 2, 3], [4, 5, 6]])

        cond = [True]
        with pytest.raises(ValueError, match=msg):
            df.where(cond)

        expected = DataFrame([[1, 2, 3], [np.nan, np.nan, np.nan]])

        out = df.where(Series(cond))
        tm.assert_frame_equal(out, expected)

        cond = np.array([False, True, False, True])
        with pytest.raises(ValueError, match=msg):
            df.where(cond)

        expected = DataFrame([[np.nan, np.nan, np.nan], [4, 5, 6]])

        out = df.where(Series(cond))
        tm.assert_frame_equal(out, expected)

    def test_where_bug(self):
        # see gh-2793
        df = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}, dtype="float64"
        )
        expected = DataFrame(
            {"a": [np.nan, np.nan, 3.0, 4.0], "b": [4.0, 3.0, np.nan, np.nan]},
            dtype="float64",
        )
        result = df.where(df > 2, np.nan)
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(result > 2, np.nan, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_where_bug_mixed(self, sint_dtype):
        # see gh-2793
        df = DataFrame(
            {
                "a": np.array([1, 2, 3, 4], dtype=sint_dtype),
                "b": np.array([4.0, 3.0, 2.0, 1.0], dtype="float64"),
            }
        )

        expected = DataFrame(
            {"a": [np.nan, np.nan, 3.0, 4.0], "b": [4.0, 3.0, np.nan, np.nan]},
            dtype="float64",
        )

        result = df.where(df > 2, np.nan)
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(result > 2, np.nan, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_where_bug_transposition(self):
        # see gh-7506
        a = DataFrame({0: [1, 2], 1: [3, 4], 2: [5, 6]})
        b = DataFrame({0: [np.nan, 8], 1: [9, np.nan], 2: [np.nan, np.nan]})
        do_not_replace = b.isna() | (a > b)

        expected = a.copy()
        expected[~do_not_replace] = b

        result = a.where(do_not_replace, b)
        tm.assert_frame_equal(result, expected)

        a = DataFrame({0: [4, 6], 1: [1, 0]})
        b = DataFrame({0: [np.nan, 3], 1: [3, np.nan]})
        do_not_replace = b.isna() | (a > b)

        expected = a.copy()
        expected[~do_not_replace] = b

        result = a.where(do_not_replace, b)
        tm.assert_frame_equal(result, expected)

    def test_where_datetime(self):

        # GH 3311
        df = DataFrame(
            {
                "A": date_range("20130102", periods=5),
                "B": date_range("20130104", periods=5),
                "C": np.random.randn(5),
            }
        )

        stamp = datetime(2013, 1, 3)
        msg = "'>' not supported between instances of 'float' and 'datetime.datetime'"
        with pytest.raises(TypeError, match=msg):
            df > stamp

        result = df[df.iloc[:, :-1] > stamp]

        expected = df.copy()
        expected.loc[[0, 1], "A"] = np.nan
        expected.loc[:, "C"] = np.nan
        tm.assert_frame_equal(result, expected)

    def test_where_none(self):
        # GH 4667
        # setting with None changes dtype
        df = DataFrame({"series": Series(range(10))}).astype(float)
        df[df > 7] = None
        expected = DataFrame(
            {"series": Series([0, 1, 2, 3, 4, 5, 6, 7, np.nan, np.nan])}
        )
        tm.assert_frame_equal(df, expected)

        # GH 7656
        df = DataFrame(
            [
                {"A": 1, "B": np.nan, "C": "Test"},
                {"A": np.nan, "B": "Test", "C": np.nan},
            ]
        )
        msg = "boolean setting on mixed-type"

        with pytest.raises(TypeError, match=msg):
            df.where(~isna(df), None, inplace=True)

    def test_where_empty_df_and_empty_cond_having_non_bool_dtypes(self):
        # see gh-21947
        df = DataFrame(columns=["a"])
        cond = df
        assert (cond.dtypes == object).all()

        result = df.where(cond)
        tm.assert_frame_equal(result, df)

    def test_where_align(self):
        def create():
            df = DataFrame(np.random.randn(10, 3))
            df.iloc[3:5, 0] = np.nan
            df.iloc[4:6, 1] = np.nan
            df.iloc[5:8, 2] = np.nan
            return df

        # series
        df = create()
        expected = df.fillna(df.mean())
        result = df.where(pd.notna(df), df.mean(), axis="columns")
        tm.assert_frame_equal(result, expected)

        return_value = df.where(pd.notna(df), df.mean(), inplace=True, axis="columns")
        assert return_value is None
        tm.assert_frame_equal(df, expected)

        df = create().fillna(0)
        expected = df.apply(lambda x, y: x.where(x > 0, y), y=df[0])
        result = df.where(df > 0, df[0], axis="index")
        tm.assert_frame_equal(result, expected)
        result = df.where(df > 0, df[0], axis="rows")
        tm.assert_frame_equal(result, expected)

        # frame
        df = create()
        expected = df.fillna(1)
        result = df.where(
            pd.notna(df), DataFrame(1, index=df.index, columns=df.columns)
        )
        tm.assert_frame_equal(result, expected)

    def test_where_complex(self):
        # GH 6345
        expected = DataFrame([[1 + 1j, 2], [np.nan, 4 + 1j]], columns=["a", "b"])
        df = DataFrame([[1 + 1j, 2], [5 + 1j, 4 + 1j]], columns=["a", "b"])
        df[df.abs() >= 5] = np.nan
        tm.assert_frame_equal(df, expected)

    def test_where_axis(self):
        # GH 9736
        df = DataFrame(np.random.randn(2, 2))
        mask = DataFrame([[False, False], [False, False]])
        s = Series([0, 1])

        expected = DataFrame([[0, 0], [1, 1]], dtype="float64")
        result = df.where(mask, s, axis="index")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(mask, s, axis="index", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[0, 1], [0, 1]], dtype="float64")
        result = df.where(mask, s, axis="columns")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(mask, s, axis="columns", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

        # Upcast needed
        df = DataFrame([[1, 2], [3, 4]], dtype="int64")
        mask = DataFrame([[False, False], [False, False]])
        s = Series([0, np.nan])

        expected = DataFrame([[0, 0], [np.nan, np.nan]], dtype="float64")
        result = df.where(mask, s, axis="index")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(mask, s, axis="index", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[0, np.nan], [0, np.nan]])
        result = df.where(mask, s, axis="columns")
        tm.assert_frame_equal(result, expected)

        expected = DataFrame(
            {
                0: np.array([0, 0], dtype="int64"),
                1: np.array([np.nan, np.nan], dtype="float64"),
            }
        )
        result = df.copy()
        return_value = result.where(mask, s, axis="columns", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_where_axis_multiple_dtypes(self):
        # Multiple dtypes (=> multiple Blocks)
        df = pd.concat(
            [
                DataFrame(np.random.randn(10, 2)),
                DataFrame(np.random.randint(0, 10, size=(10, 2)), dtype="int64"),
            ],
            ignore_index=True,
            axis=1,
        )
        mask = DataFrame(False, columns=df.columns, index=df.index)
        s1 = Series(1, index=df.columns)
        s2 = Series(2, index=df.index)

        result = df.where(mask, s1, axis="columns")
        expected = DataFrame(1.0, columns=df.columns, index=df.index)
        expected[2] = expected[2].astype("int64")
        expected[3] = expected[3].astype("int64")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(mask, s1, axis="columns", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

        result = df.where(mask, s2, axis="index")
        expected = DataFrame(2.0, columns=df.columns, index=df.index)
        expected[2] = expected[2].astype("int64")
        expected[3] = expected[3].astype("int64")
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(mask, s2, axis="index", inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)

        # DataFrame vs DataFrame
        d1 = df.copy().drop(1, axis=0)
        expected = df.copy()
        expected.loc[1, :] = np.nan

        result = df.where(mask, d1)
        tm.assert_frame_equal(result, expected)
        result = df.where(mask, d1, axis="index")
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        return_value = result.where(mask, d1, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        return_value = result.where(mask, d1, inplace=True, axis="index")
        assert return_value is None
        tm.assert_frame_equal(result, expected)

        d2 = df.copy().drop(1, axis=1)
        expected = df.copy()
        expected.loc[:, 1] = np.nan

        result = df.where(mask, d2)
        tm.assert_frame_equal(result, expected)
        result = df.where(mask, d2, axis="columns")
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        return_value = result.where(mask, d2, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)
        result = df.copy()
        return_value = result.where(mask, d2, inplace=True, axis="columns")
        assert return_value is None
        tm.assert_frame_equal(result, expected)

    def test_where_callable(self):
        # GH 12533
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = df.where(lambda x: x > 4, lambda x: x + 1)
        exp = DataFrame([[2, 3, 4], [5, 5, 6], [7, 8, 9]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.where(df > 4, df + 1))

        # return ndarray and scalar
        result = df.where(lambda x: (x % 2 == 0).values, lambda x: 99)
        exp = DataFrame([[99, 2, 99], [4, 99, 6], [99, 8, 99]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, df.where(df % 2 == 0, 99))

        # chain
        result = (df + 2).where(lambda x: x > 8, lambda x: x + 10)
        exp = DataFrame([[13, 14, 15], [16, 17, 18], [9, 10, 11]])
        tm.assert_frame_equal(result, exp)
        tm.assert_frame_equal(result, (df + 2).where((df + 2) > 8, (df + 2) + 10))

    def test_where_tz_values(self, tz_naive_fixture):
        df1 = DataFrame(
            DatetimeIndex(["20150101", "20150102", "20150103"], tz=tz_naive_fixture),
            columns=["date"],
        )
        df2 = DataFrame(
            DatetimeIndex(["20150103", "20150104", "20150105"], tz=tz_naive_fixture),
            columns=["date"],
        )
        mask = DataFrame([True, True, False], columns=["date"])
        exp = DataFrame(
            DatetimeIndex(["20150101", "20150102", "20150105"], tz=tz_naive_fixture),
            columns=["date"],
        )
        result = df1.where(mask, df2)
        tm.assert_frame_equal(exp, result)

    def test_df_where_change_dtype(self):
        # GH#16979
        df = DataFrame(np.arange(2 * 3).reshape(2, 3), columns=list("ABC"))
        mask = np.array([[True, False, False], [False, False, True]])

        result = df.where(mask)
        expected = DataFrame(
            [[0, np.nan, np.nan], [np.nan, np.nan, 5]], columns=list("ABC")
        )

        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("kwargs", [{}, {"other": None}])
    def test_df_where_with_category(self, kwargs):
        # GH#16979
        df = DataFrame(np.arange(2 * 3).reshape(2, 3), columns=list("ABC"))
        mask = np.array([[True, False, False], [False, False, True]])

        # change type to category
        df.A = df.A.astype("category")
        df.B = df.B.astype("category")
        df.C = df.C.astype("category")

        result = df.where(mask, **kwargs)
        A = pd.Categorical([0, np.nan], categories=[0, 3])
        B = pd.Categorical([np.nan, np.nan], categories=[1, 4])
        C = pd.Categorical([np.nan, 5], categories=[2, 5])
        expected = DataFrame({"A": A, "B": B, "C": C})

        tm.assert_frame_equal(result, expected)

        # Check Series.where while we're here
        result = df.A.where(mask[:, 0], **kwargs)
        expected = Series(A, name="A")

        tm.assert_series_equal(result, expected)

    def test_where_categorical_filtering(self):
        # GH#22609 Verify filtering operations on DataFrames with categorical Series
        df = DataFrame(data=[[0, 0], [1, 1]], columns=["a", "b"])
        df["b"] = df["b"].astype("category")

        result = df.where(df["a"] > 0)
        expected = df.copy()
        expected.loc[0, :] = np.nan

        tm.assert_equal(result, expected)

    def test_where_ea_other(self):
        # GH#38729/GH#38742
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        arr = pd.array([7, pd.NA, 9])
        ser = Series(arr)
        mask = np.ones(df.shape, dtype=bool)
        mask[1, :] = False

        # TODO: ideally we would get Int64 instead of object
        result = df.where(mask, ser, axis=0)
        expected = DataFrame({"A": [1, pd.NA, 3], "B": [4, pd.NA, 6]}).astype(object)
        tm.assert_frame_equal(result, expected)

        ser2 = Series(arr[:2], index=["A", "B"])
        expected = DataFrame({"A": [1, 7, 3], "B": [4, pd.NA, 6]})
        expected["B"] = expected["B"].astype(object)
        result = df.where(mask, ser2, axis=1)
        tm.assert_frame_equal(result, expected)


def test_where_try_cast_deprecated(frame_or_series):
    obj = DataFrame(np.random.randn(4, 3))
    if frame_or_series is not DataFrame:
        obj = obj[0]

    mask = obj > 0

    with tm.assert_produces_warning(FutureWarning):
        # try_cast keyword deprecated
        obj.where(mask, -1, try_cast=False)


def test_where_copies_with_noop(frame_or_series):
    # GH-39595
    result = frame_or_series([1, 2, 3, 4])
    expected = result.copy()
    col = result[0] if frame_or_series is DataFrame else result

    where_res = result.where(col < 5)
    where_res *= 2

    tm.assert_equal(result, expected)

    where_res = result.where(col > 5, [1, 2, 3, 4])
    where_res *= 2

    tm.assert_equal(result, expected)


def test_where_string_dtype(frame_or_series):
    # GH40824
    obj = frame_or_series(
        ["a", "b", "c", "d"], index=["id1", "id2", "id3", "id4"], dtype=StringDtype()
    )
    filtered_obj = frame_or_series(
        ["b", "c"], index=["id2", "id3"], dtype=StringDtype()
    )
    filter_ser = Series([False, True, True, False])

    result = obj.where(filter_ser, filtered_obj)
    expected = frame_or_series(
        [pd.NA, "b", "c", pd.NA],
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    tm.assert_equal(result, expected)


def test_where_bool_comparison():
    # GH 10336
    df_mask = DataFrame(
        {"AAA": [True] * 4, "BBB": [False] * 4, "CCC": [True, False, True, False]}
    )
    result = df_mask.where(df_mask == False)  # noqa:E712
    expected = DataFrame(
        {
            "AAA": np.array([np.nan] * 4, dtype=object),
            "BBB": [False] * 4,
            "CCC": [np.nan, False, np.nan, False],
        }
    )
    tm.assert_frame_equal(result, expected)


def test_where_none_nan_coerce():
    # GH 15613
    expected = DataFrame(
        {
            "A": [Timestamp("20130101"), pd.NaT, Timestamp("20130103")],
            "B": [1, 2, np.nan],
        }
    )
    result = expected.where(expected.notnull(), None)
    tm.assert_frame_equal(result, expected)


def test_where_non_keyword_deprecation():
    # GH 41485
    s = DataFrame(range(5))
    msg = (
        "In a future version of pandas all arguments of "
        "DataFrame.where except for the arguments 'cond' "
        "and 'other' will be keyword-only"
    )
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = s.where(s > 1, 10, False)
    expected = DataFrame([10, 10, 2, 3, 4])
    tm.assert_frame_equal(expected, result)
