import warnings

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
from pandas import (
    DataFrame,
    Series,
    isna,
)
import pandas._testing as tm


class TestDataFrameCov:
    def test_cov(self, float_frame, float_string_frame):
        # min_periods no NAs (corner case)
        expected = float_frame.cov()
        result = float_frame.cov(min_periods=len(float_frame))

        tm.assert_frame_equal(expected, result)

        result = float_frame.cov(min_periods=len(float_frame) + 1)
        assert isna(result.values).all()

        # with NAs
        frame = float_frame.copy()
        frame["A"][:5] = np.nan
        frame["B"][5:10] = np.nan
        result = frame.cov(min_periods=len(frame) - 8)
        expected = frame.cov()
        expected.loc["A", "B"] = np.nan
        expected.loc["B", "A"] = np.nan
        tm.assert_frame_equal(result, expected)

        # regular
        result = frame.cov()
        expected = frame["A"].cov(frame["C"])
        tm.assert_almost_equal(result["A"]["C"], expected)

        # exclude non-numeric types
        result = float_string_frame.cov()
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].cov()
        tm.assert_frame_equal(result, expected)

        # Single column frame
        df = DataFrame(np.linspace(0.0, 1.0, 10))
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values.T).reshape((1, 1)), index=df.columns, columns=df.columns
        )
        tm.assert_frame_equal(result, expected)
        df.loc[0] = np.nan
        result = df.cov()
        expected = DataFrame(
            np.cov(df.values[1:].T).reshape((1, 1)),
            index=df.columns,
            columns=df.columns,
        )
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("test_ddof", [None, 0, 1, 2, 3])
    def test_cov_ddof(self, test_ddof):
        # GH#34611
        np_array1 = np.random.rand(10)
        np_array2 = np.random.rand(10)
        df = DataFrame({0: np_array1, 1: np_array2})
        result = df.cov(ddof=test_ddof)
        expected_np = np.cov(np_array1, np_array2, ddof=test_ddof)
        expected = DataFrame(expected_np)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "other_column", [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])]
    )
    def test_cov_nullable_integer(self, other_column):
        # https://github.com/pandas-dev/pandas/issues/33803
        data = DataFrame({"a": pd.array([1, 2, None]), "b": other_column})
        result = data.cov()
        arr = np.array([[0.5, 0.5], [0.5, 1.0]])
        expected = DataFrame(arr, columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)


class TestDataFrameCorr:
    # DataFrame.corr(), as opposed to DataFrame.corrwith

    @pytest.mark.parametrize("method", ["pearson", "kendall", "spearman"])
    @td.skip_if_no_scipy
    def test_corr_scipy_method(self, float_frame, method):
        float_frame["A"][:5] = np.nan
        float_frame["B"][5:10] = np.nan
        float_frame["A"][:10] = float_frame["A"][10:20]

        correls = float_frame.corr(method=method)
        expected = float_frame["A"].corr(float_frame["C"], method=method)
        tm.assert_almost_equal(correls["A"]["C"], expected)

    # ---------------------------------------------------------------------

    def test_corr_non_numeric(self, float_string_frame):
        # exclude non-numeric types
        result = float_string_frame.corr()
        expected = float_string_frame.loc[:, ["A", "B", "C", "D"]].corr()
        tm.assert_frame_equal(result, expected)

    @td.skip_if_no_scipy
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_nooverlap(self, meth):
        # nothing in common
        df = DataFrame(
            {
                "A": [1, 1.5, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1.5, 1],
                "C": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.loc["A", "B"])
        assert isna(rs.loc["B", "A"])
        assert rs.loc["A", "A"] == 1
        assert rs.loc["B", "B"] == 1
        assert isna(rs.loc["C", "C"])

    @pytest.mark.parametrize("meth", ["pearson", "spearman"])
    def test_corr_constant(self, meth):
        # constant --> all NA
        df = DataFrame(
            {
                "A": [1, 1, 1, np.nan, np.nan, np.nan],
                "B": [np.nan, np.nan, np.nan, 1, 1, 1],
            }
        )
        rs = df.corr(meth)
        assert isna(rs.values).all()

    @td.skip_if_no_scipy
    @pytest.mark.parametrize("meth", ["pearson", "kendall", "spearman"])
    def test_corr_int_and_boolean(self, meth):
        # when dtypes of pandas series are different
        # then ndarray will have dtype=object,
        # so it need to be properly handled
        df = DataFrame({"a": [True, False], "b": [1, 0]})

        expected = DataFrame(np.ones((2, 2)), index=["a", "b"], columns=["a", "b"])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", RuntimeWarning)
            result = df.corr(meth)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("method", ["cov", "corr"])
    def test_corr_cov_independent_index_column(self, method):
        # GH#14617
        df = DataFrame(np.random.randn(4 * 10).reshape(10, 4), columns=list("abcd"))
        result = getattr(df, method)()
        assert result.index is not result.columns
        assert result.index.equals(result.columns)

    def test_corr_invalid_method(self):
        # GH#22298
        df = DataFrame(np.random.normal(size=(10, 2)))
        msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
        with pytest.raises(ValueError, match=msg):
            df.corr(method="____")

    def test_corr_int(self):
        # dtypes other than float64 GH#1761
        df = DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

        df.cov()
        df.corr()

    @td.skip_if_no_scipy
    @pytest.mark.parametrize(
        "nullable_column", [pd.array([1, 2, 3]), pd.array([1, 2, None])]
    )
    @pytest.mark.parametrize(
        "other_column",
        [pd.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, np.nan])],
    )
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_nullable_integer(self, nullable_column, other_column, method):
        # https://github.com/pandas-dev/pandas/issues/33803
        data = DataFrame({"a": nullable_column, "b": other_column})
        result = data.corr(method=method)
        expected = DataFrame(np.ones((2, 2)), columns=["a", "b"], index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_corr_item_cache(self, using_array_manager):
        # Check that corr does not lead to incorrect entries in item_cache

        df = DataFrame({"A": range(10)})
        df["B"] = range(10)[::-1]

        ser = df["A"]  # populate item_cache
        if not using_array_manager:
            assert len(df._mgr.blocks) == 2

        _ = df.corr()

        # Check that the corr didn't break link between ser and df
        ser.values[0] = 99
        assert df.loc[0, "A"] == 99
        assert df["A"] is ser
        assert df.values[0, 0] == 99

    @pytest.mark.parametrize("length", [2, 20, 200, 2000])
    def test_corr_for_constant_columns(self, length):
        # GH: 37448
        df = DataFrame(length * [[0.4, 0.1]], columns=["A", "B"])
        result = df.corr()
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)

    def test_calc_corr_small_numbers(self):
        # GH: 37452
        df = DataFrame(
            {"A": [1.0e-20, 2.0e-20, 3.0e-20], "B": [1.0e-20, 2.0e-20, 3.0e-20]}
        )
        result = df.corr()
        expected = DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]}, index=["A", "B"])
        tm.assert_frame_equal(result, expected)

    @td.skip_if_no_scipy
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_corr_min_periods_greater_than_length(self, method):
        df = DataFrame({"A": [1, 2], "B": [1, 2]})
        result = df.corr(method=method, min_periods=3)
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]}, index=["A", "B"]
        )
        tm.assert_frame_equal(result, expected)


class TestDataFrameCorrWith:
    def test_corrwith(self, datetime_frame):
        a = datetime_frame
        noise = Series(np.random.randn(len(a)), index=a.index)

        b = datetime_frame.add(noise, axis=0)

        # make sure order does not matter
        b = b.reindex(columns=b.columns[::-1], index=b.index[::-1][10:])
        del b["B"]

        colcorr = a.corrwith(b, axis=0)
        tm.assert_almost_equal(colcorr["A"], a["A"].corr(b["A"]))

        rowcorr = a.corrwith(b, axis=1)
        tm.assert_series_equal(rowcorr, a.T.corrwith(b.T, axis=0))

        dropped = a.corrwith(b, axis=0, drop=True)
        tm.assert_almost_equal(dropped["A"], a["A"].corr(b["A"]))
        assert "B" not in dropped

        dropped = a.corrwith(b, axis=1, drop=True)
        assert a.index[-1] not in dropped.index

        # non time-series data
        index = ["a", "b", "c", "d", "e"]
        columns = ["one", "two", "three", "four"]
        df1 = DataFrame(np.random.randn(5, 4), index=index, columns=columns)
        df2 = DataFrame(np.random.randn(4, 4), index=index[:4], columns=columns)
        correls = df1.corrwith(df2, axis=1)
        for row in index[:4]:
            tm.assert_almost_equal(correls[row], df1.loc[row].corr(df2.loc[row]))

    def test_corrwith_with_objects(self):
        df1 = tm.makeTimeDataFrame()
        df2 = tm.makeTimeDataFrame()
        cols = ["A", "B", "C", "D"]

        df1["obj"] = "foo"
        df2["obj"] = "bar"

        result = df1.corrwith(df2)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols])
        tm.assert_series_equal(result, expected)

        result = df1.corrwith(df2, axis=1)
        expected = df1.loc[:, cols].corrwith(df2.loc[:, cols], axis=1)
        tm.assert_series_equal(result, expected)

    def test_corrwith_series(self, datetime_frame):
        result = datetime_frame.corrwith(datetime_frame["A"])
        expected = datetime_frame.apply(datetime_frame["A"].corr)

        tm.assert_series_equal(result, expected)

    def test_corrwith_matches_corrcoef(self):
        df1 = DataFrame(np.arange(10000), columns=["a"])
        df2 = DataFrame(np.arange(10000) ** 2, columns=["a"])
        c1 = df1.corrwith(df2)["a"]
        c2 = np.corrcoef(df1["a"], df2["a"])[0][1]

        tm.assert_almost_equal(c1, c2)
        assert c1 < 1

    def test_corrwith_mixed_dtypes(self):
        # GH#18570
        df = DataFrame(
            {"a": [1, 4, 3, 2], "b": [4, 6, 7, 3], "c": ["a", "b", "c", "d"]}
        )
        s = Series([0, 6, 7, 3])
        result = df.corrwith(s)
        corrs = [df["a"].corr(s), df["b"].corr(s)]
        expected = Series(data=corrs, index=["a", "b"])
        tm.assert_series_equal(result, expected)

    def test_corrwith_index_intersection(self):
        df1 = DataFrame(np.random.random(size=(10, 2)), columns=["a", "b"])
        df2 = DataFrame(np.random.random(size=(10, 3)), columns=["a", "b", "c"])

        result = df1.corrwith(df2, drop=True).index.sort_values()
        expected = df1.columns.intersection(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_index_union(self):
        df1 = DataFrame(np.random.random(size=(10, 2)), columns=["a", "b"])
        df2 = DataFrame(np.random.random(size=(10, 3)), columns=["a", "b", "c"])

        result = df1.corrwith(df2, drop=False).index.sort_values()
        expected = df1.columns.union(df2.columns).sort_values()
        tm.assert_index_equal(result, expected)

    def test_corrwith_dup_cols(self):
        # GH#21925
        df1 = DataFrame(np.vstack([np.arange(10)] * 3).T)
        df2 = df1.copy()
        df2 = pd.concat((df2, df2[0]), axis=1)

        result = df1.corrwith(df2)
        expected = Series(np.ones(4), index=[0, 0, 1, 2])
        tm.assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_corrwith_spearman(self):
        # GH#21925
        df = DataFrame(np.random.random(size=(100, 3)))
        result = df.corrwith(df ** 2, method="spearman")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_corrwith_kendall(self):
        # GH#21925
        df = DataFrame(np.random.random(size=(100, 3)))
        result = df.corrwith(df ** 2, method="kendall")
        expected = Series(np.ones(len(result)))
        tm.assert_series_equal(result, expected)
