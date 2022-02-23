import numpy as np
import pytest

from pandas.compat.numpy import np_percentile_argname

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestDataFrameQuantile:
    @pytest.mark.parametrize(
        "df,expected",
        [
            [
                DataFrame(
                    {
                        0: Series(pd.arrays.SparseArray([1, 2])),
                        1: Series(pd.arrays.SparseArray([3, 4])),
                    }
                ),
                Series([1.5, 3.5], name=0.5),
            ],
            [
                DataFrame(Series([0.0, None, 1.0, 2.0], dtype="Sparse[float]")),
                Series([1.0], name=0.5),
            ],
        ],
    )
    def test_quantile_sparse(self, df, expected):
        # GH#17198
        # GH#24600
        result = df.quantile()

        tm.assert_series_equal(result, expected)

    def test_quantile(self, datetime_frame):
        from numpy import percentile

        df = datetime_frame
        q = df.quantile(0.1, axis=0)
        assert q["A"] == percentile(df["A"], 10)
        tm.assert_index_equal(q.index, df.columns)

        q = df.quantile(0.9, axis=1)
        assert q["2000-01-17"] == percentile(df.loc["2000-01-17"], 90)
        tm.assert_index_equal(q.index, df.index)

        # test degenerate case
        q = DataFrame({"x": [], "y": []}).quantile(0.1, axis=0)
        assert np.isnan(q["x"]) and np.isnan(q["y"])

        # non-numeric exclusion
        df = DataFrame({"col1": ["A", "A", "B", "B"], "col2": [1, 2, 3, 4]})
        rs = df.quantile(0.5)
        with tm.assert_produces_warning(FutureWarning, match="Select only valid"):
            xp = df.median().rename(0.5)
        tm.assert_series_equal(rs, xp)

        # axis
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1)
        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)

        result = df.quantile([0.5, 0.75], axis=1)
        expected = DataFrame(
            {1: [1.5, 1.75], 2: [2.5, 2.75], 3: [3.5, 3.75]}, index=[0.5, 0.75]
        )
        tm.assert_frame_equal(result, expected, check_index_type=True)

        # We may want to break API in the future to change this
        # so that we exclude non-numeric along the same axis
        # See GH #7312
        df = DataFrame([[1, 2, 3], ["a", "b", 4]])
        result = df.quantile(0.5, axis=1)
        expected = Series([3.0, 4.0], index=[0, 1], name=0.5)
        tm.assert_series_equal(result, expected)

    def test_quantile_date_range(self):
        # GH 2460

        dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
        ser = Series(dti)
        df = DataFrame(ser)

        result = df.quantile(numeric_only=False)
        expected = Series(
            ["2016-01-02 00:00:00"], name=0.5, dtype="datetime64[ns, US/Pacific]"
        )

        tm.assert_series_equal(result, expected)

    def test_quantile_axis_mixed(self):

        # mixed on axis=1
        df = DataFrame(
            {
                "A": [1, 2, 3],
                "B": [2.0, 3.0, 4.0],
                "C": pd.date_range("20130101", periods=3),
                "D": ["foo", "bar", "baz"],
            }
        )
        result = df.quantile(0.5, axis=1)
        expected = Series([1.5, 2.5, 3.5], name=0.5)
        tm.assert_series_equal(result, expected)

        # must raise
        msg = "'<' not supported between instances of 'Timestamp' and 'float'"
        with pytest.raises(TypeError, match=msg):
            df.quantile(0.5, axis=1, numeric_only=False)

    def test_quantile_axis_parameter(self):
        # GH 9543/9544

        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])

        result = df.quantile(0.5, axis=0)

        expected = Series([2.0, 3.0], index=["A", "B"], name=0.5)
        tm.assert_series_equal(result, expected)

        expected = df.quantile(0.5, axis="index")
        tm.assert_series_equal(result, expected)

        result = df.quantile(0.5, axis=1)

        expected = Series([1.5, 2.5, 3.5], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)

        result = df.quantile(0.5, axis="columns")
        tm.assert_series_equal(result, expected)

        msg = "No axis named -1 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis=-1)
        msg = "No axis named column for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            df.quantile(0.1, axis="column")

    def test_quantile_interpolation(self):
        # see gh-10174

        # interpolation method other than default linear
        df = DataFrame({"A": [1, 2, 3], "B": [2, 3, 4]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1, 2, 3], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)

        # cross-check interpolation=nearest results in original dtype
        exp = np.percentile(
            np.array([[1, 2, 3], [2, 3, 4]]),
            0.5,
            axis=0,
            **{np_percentile_argname: "nearest"},
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="int64")
        tm.assert_series_equal(result, expected)

        # float
        df = DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]}, index=[1, 2, 3])
        result = df.quantile(0.5, axis=1, interpolation="nearest")
        expected = Series([1.0, 2.0, 3.0], index=[1, 2, 3], name=0.5)
        tm.assert_series_equal(result, expected)
        exp = np.percentile(
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            0.5,
            axis=0,
            **{np_percentile_argname: "nearest"},
        )
        expected = Series(exp, index=[1, 2, 3], name=0.5, dtype="float64")
        tm.assert_series_equal(result, expected)

        # axis
        result = df.quantile([0.5, 0.75], axis=1, interpolation="lower")
        expected = DataFrame(
            {1: [1.0, 1.0], 2: [2.0, 2.0], 3: [3.0, 3.0]}, index=[0.5, 0.75]
        )
        tm.assert_frame_equal(result, expected)

        # test degenerate case
        df = DataFrame({"x": [], "y": []})
        q = df.quantile(0.1, axis=0, interpolation="higher")
        assert np.isnan(q["x"]) and np.isnan(q["y"])

        # multi
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile([0.25, 0.5], interpolation="midpoint")

        # https://github.com/numpy/numpy/issues/7163
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        tm.assert_frame_equal(result, expected)

    def test_quantile_interpolation_datetime(self, datetime_frame):
        # see gh-10174

        # interpolation = linear (default case)
        df = datetime_frame
        q = df.quantile(0.1, axis=0, interpolation="linear")
        assert q["A"] == np.percentile(df["A"], 10)

    def test_quantile_interpolation_int(self, int_frame):
        # see gh-10174

        df = int_frame
        # interpolation = linear (default case)
        q = df.quantile(0.1)
        assert q["A"] == np.percentile(df["A"], 10)

        # test with and without interpolation keyword
        q1 = df.quantile(0.1, axis=0, interpolation="linear")
        assert q1["A"] == np.percentile(df["A"], 10)
        tm.assert_series_equal(q, q1)

    def test_quantile_multi(self):
        df = DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], columns=["a", "b", "c"])
        result = df.quantile([0.25, 0.5])
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]],
            index=[0.25, 0.5],
            columns=["a", "b", "c"],
        )
        tm.assert_frame_equal(result, expected)

        # axis = 1
        result = df.quantile([0.25, 0.5], axis=1)
        expected = DataFrame(
            [[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]], index=[0.25, 0.5], columns=[0, 1, 2]
        )

        # empty
        result = DataFrame({"x": [], "y": []}).quantile([0.1, 0.9], axis=0)
        expected = DataFrame(
            {"x": [np.nan, np.nan], "y": [np.nan, np.nan]}, index=[0.1, 0.9]
        )
        tm.assert_frame_equal(result, expected)

    def test_quantile_datetime(self):
        df = DataFrame({"a": pd.to_datetime(["2010", "2011"]), "b": [0, 5]})

        # exclude datetime
        result = df.quantile(0.5)
        expected = Series([2.5], index=["b"])

        # datetime
        result = df.quantile(0.5, numeric_only=False)
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), 2.5], index=["a", "b"], name=0.5
        )
        tm.assert_series_equal(result, expected)

        # datetime w/ multi
        result = df.quantile([0.5], numeric_only=False)
        expected = DataFrame(
            [[Timestamp("2010-07-02 12:00:00"), 2.5]], index=[0.5], columns=["a", "b"]
        )
        tm.assert_frame_equal(result, expected)

        # axis = 1
        df["c"] = pd.to_datetime(["2011", "2012"])
        result = df[["a", "c"]].quantile(0.5, axis=1, numeric_only=False)
        expected = Series(
            [Timestamp("2010-07-02 12:00:00"), Timestamp("2011-07-02 12:00:00")],
            index=[0, 1],
            name=0.5,
        )
        tm.assert_series_equal(result, expected)

        result = df[["a", "c"]].quantile([0.5], axis=1, numeric_only=False)
        expected = DataFrame(
            [[Timestamp("2010-07-02 12:00:00"), Timestamp("2011-07-02 12:00:00")]],
            index=[0.5],
            columns=[0, 1],
        )
        tm.assert_frame_equal(result, expected)

        # empty when numeric_only=True
        # FIXME (gives empty frame in 0.18.1, broken in 0.19.0)
        # result = df[['a', 'c']].quantile(.5)
        # result = df[['a', 'c']].quantile([.5])

    def test_quantile_invalid(self, datetime_frame):
        msg = "percentiles should all be in the interval \\[0, 1\\]"
        for invalid in [-1, 2, [0.5, -1], [0.5, 2]]:
            with pytest.raises(ValueError, match=msg):
                datetime_frame.quantile(invalid)

    def test_quantile_box(self):
        df = DataFrame(
            {
                "A": [
                    Timestamp("2011-01-01"),
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-03"),
                ],
                "B": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "C": [
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                ],
            }
        )

        res = df.quantile(0.5, numeric_only=False)

        exp = Series(
            [
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timedelta("2 days"),
            ],
            name=0.5,
            index=["A", "B", "C"],
        )
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5], numeric_only=False)
        exp = DataFrame(
            [
                [
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.Timedelta("2 days"),
                ]
            ],
            index=[0.5],
            columns=["A", "B", "C"],
        )
        tm.assert_frame_equal(res, exp)

        # DatetimeLikeBlock may be consolidated and contain NaT in different loc
        df = DataFrame(
            {
                "A": [
                    Timestamp("2011-01-01"),
                    pd.NaT,
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-03"),
                ],
                "a": [
                    Timestamp("2011-01-01"),
                    Timestamp("2011-01-02"),
                    pd.NaT,
                    Timestamp("2011-01-03"),
                ],
                "B": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "b": [
                    Timestamp("2011-01-01", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-01-03", tz="US/Eastern"),
                ],
                "C": [
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                    pd.NaT,
                ],
                "c": [
                    pd.NaT,
                    pd.Timedelta("1 days"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("3 days"),
                ],
            },
            columns=list("AaBbCc"),
        )

        res = df.quantile(0.5, numeric_only=False)
        exp = Series(
            [
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                Timestamp("2011-01-02", tz="US/Eastern"),
                pd.Timedelta("2 days"),
                pd.Timedelta("2 days"),
            ],
            name=0.5,
            index=list("AaBbCc"),
        )
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5], numeric_only=False)
        exp = DataFrame(
            [
                [
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    Timestamp("2011-01-02", tz="US/Eastern"),
                    pd.Timedelta("2 days"),
                    pd.Timedelta("2 days"),
                ]
            ],
            index=[0.5],
            columns=list("AaBbCc"),
        )
        tm.assert_frame_equal(res, exp)

    def test_quantile_nan(self):

        # GH 14357 - float block where some cols have missing values
        df = DataFrame({"a": np.arange(1, 6.0), "b": np.arange(1, 6.0)})
        df.iloc[-1, 1] = np.nan

        res = df.quantile(0.5)
        exp = Series([3.0, 2.5], index=["a", "b"], name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5, 0.75])
        exp = DataFrame({"a": [3.0, 4.0], "b": [2.5, 3.25]}, index=[0.5, 0.75])
        tm.assert_frame_equal(res, exp)

        res = df.quantile(0.5, axis=1)
        exp = Series(np.arange(1.0, 6.0), name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5, 0.75], axis=1)
        exp = DataFrame([np.arange(1.0, 6.0)] * 2, index=[0.5, 0.75])
        tm.assert_frame_equal(res, exp)

        # full-nan column
        df["b"] = np.nan

        res = df.quantile(0.5)
        exp = Series([3.0, np.nan], index=["a", "b"], name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5, 0.75])
        exp = DataFrame({"a": [3.0, 4.0], "b": [np.nan, np.nan]}, index=[0.5, 0.75])
        tm.assert_frame_equal(res, exp)

    def test_quantile_nat(self):

        # full NaT column
        df = DataFrame({"a": [pd.NaT, pd.NaT, pd.NaT]})

        res = df.quantile(0.5, numeric_only=False)
        exp = Series([pd.NaT], index=["a"], name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5], numeric_only=False)
        exp = DataFrame({"a": [pd.NaT]}, index=[0.5])
        tm.assert_frame_equal(res, exp)

        # mixed non-null / full null column
        df = DataFrame(
            {
                "a": [
                    Timestamp("2012-01-01"),
                    Timestamp("2012-01-02"),
                    Timestamp("2012-01-03"),
                ],
                "b": [pd.NaT, pd.NaT, pd.NaT],
            }
        )

        res = df.quantile(0.5, numeric_only=False)
        exp = Series([Timestamp("2012-01-02"), pd.NaT], index=["a", "b"], name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5], numeric_only=False)
        exp = DataFrame(
            [[Timestamp("2012-01-02"), pd.NaT]], index=[0.5], columns=["a", "b"]
        )
        tm.assert_frame_equal(res, exp)

    def test_quantile_empty_no_rows(self):

        # floats
        df = DataFrame(columns=["a", "b"], dtype="float64")

        res = df.quantile(0.5)
        exp = Series([np.nan, np.nan], index=["a", "b"], name=0.5)
        tm.assert_series_equal(res, exp)

        res = df.quantile([0.5])
        exp = DataFrame([[np.nan, np.nan]], columns=["a", "b"], index=[0.5])
        tm.assert_frame_equal(res, exp)

        # FIXME (gives empty frame in 0.18.1, broken in 0.19.0)
        # res = df.quantile(0.5, axis=1)
        # res = df.quantile([0.5], axis=1)

        # ints
        df = DataFrame(columns=["a", "b"], dtype="int64")

        # FIXME (gives empty frame in 0.18.1, broken in 0.19.0)
        # res = df.quantile(0.5)

        # datetimes
        df = DataFrame(columns=["a", "b"], dtype="datetime64[ns]")

        # FIXME (gives NaNs instead of NaT in 0.18.1 or 0.19.0)
        # res = df.quantile(0.5, numeric_only=False)

    def test_quantile_empty_no_columns(self):
        # GH#23925 _get_numeric_data may drop all columns
        df = DataFrame(pd.date_range("1/1/18", periods=5))
        df.columns.name = "captain tightpants"
        result = df.quantile(0.5)
        expected = Series([], index=[], name=0.5, dtype=np.float64)
        expected.index.name = "captain tightpants"
        tm.assert_series_equal(result, expected)

        result = df.quantile([0.5])
        expected = DataFrame([], index=[0.5], columns=[])
        expected.columns.name = "captain tightpants"
        tm.assert_frame_equal(result, expected)

    def test_quantile_item_cache(self, using_array_manager):
        # previous behavior incorrect retained an invalid _item_cache entry
        df = DataFrame(np.random.randn(4, 3), columns=["A", "B", "C"])
        df["D"] = df["A"] * 2
        ser = df["A"]
        if not using_array_manager:
            assert len(df._mgr.blocks) == 2

        df.quantile(numeric_only=False)
        ser.values[0] = 99

        assert df.iloc[0, 0] == df["A"][0]


class TestQuantileExtensionDtype:
    # TODO: tests for axis=1?
    # TODO: empty case?  might as well do dt64 and td64 here too

    @pytest.fixture(
        params=[
            pytest.param(
                pd.IntervalIndex.from_breaks(range(10)),
                marks=pytest.mark.xfail(reason="raises when trying to add Intervals"),
            ),
            pd.period_range("2016-01-01", periods=9, freq="D"),
            pd.date_range("2016-01-01", periods=9, tz="US/Pacific"),
            pd.array(np.arange(9), dtype="Int64"),
            pd.array(np.arange(9), dtype="Float64"),
        ],
        ids=lambda x: str(x.dtype),
    )
    def index(self, request):
        # NB: not actually an Index object
        idx = request.param
        idx.name = "A"
        return idx

    @pytest.fixture
    def obj(self, index, frame_or_series):
        # bc index is not always an Index (yet), we need to re-patch .name
        obj = frame_or_series(index).copy()

        if frame_or_series is Series:
            obj.name = "A"
        else:
            obj.columns = ["A"]
        return obj

    def compute_quantile(self, obj, qs):
        if isinstance(obj, Series):
            result = obj.quantile(qs)
        else:
            result = obj.quantile(qs, numeric_only=False)
        return result

    def test_quantile_ea(self, obj, index):

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)

        # expected here assumes len(index) == 9
        expected = Series(
            [index[4], index[0], index[-1]], dtype=index.dtype, index=qs, name="A"
        )
        expected = type(obj)(expected)

        tm.assert_equal(result, expected)

    def test_quantile_ea_with_na(self, obj, index):

        obj.iloc[0] = index._na_value
        obj.iloc[-1] = index._na_value

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)

        # expected here assumes len(index) == 9
        expected = Series(
            [index[4], index[1], index[-2]], dtype=index.dtype, index=qs, name="A"
        )
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    # TODO: filtering can be removed after GH#39763 is fixed
    @pytest.mark.filterwarnings("ignore:Using .astype to convert:FutureWarning")
    def test_quantile_ea_all_na(self, obj, index, frame_or_series):

        obj.iloc[:] = index._na_value

        # TODO(ArrayManager): this casting should be unnecessary after GH#39763 is fixed
        obj[:] = obj.astype(index.dtype)
        assert np.all(obj.dtypes == index.dtype)

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = [0.5, 0, 1]
        result = self.compute_quantile(obj, qs)

        expected = index.take([-1, -1, -1], allow_fill=True, fill_value=index._na_value)
        expected = Series(expected, index=qs, name="A")
        expected = type(obj)(expected)
        tm.assert_equal(result, expected)

    def test_quantile_ea_scalar(self, obj, index):
        # scalar qs

        # result should be invariant to shuffling
        indexer = np.arange(len(index), dtype=np.intp)
        np.random.shuffle(indexer)
        obj = obj.iloc[indexer]

        qs = 0.5
        result = self.compute_quantile(obj, qs)

        expected = Series({"A": index[4]}, dtype=index.dtype, name=0.5)
        if isinstance(obj, Series):
            expected = expected["A"]
            assert result == expected
        else:
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, expected_data, expected_index, axis",
        [
            ["float64", [], [], 1],
            ["int64", [], [], 1],
            ["float64", [np.nan, np.nan], ["a", "b"], 0],
            ["int64", [np.nan, np.nan], ["a", "b"], 0],
        ],
    )
    def test_empty_numeric(self, dtype, expected_data, expected_index, axis):
        # GH 14564
        df = DataFrame(columns=["a", "b"], dtype=dtype)
        result = df.quantile(0.5, axis=axis)
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype="float64"
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype, expected_data, expected_index, axis, expected_dtype",
        [
            pytest.param(
                "datetime64[ns]",
                [],
                [],
                1,
                "datetime64[ns]",
                marks=pytest.mark.xfail(reason="#GH 41544"),
            ),
            ["datetime64[ns]", [pd.NaT, pd.NaT], ["a", "b"], 0, "datetime64[ns]"],
        ],
    )
    def test_empty_datelike(
        self, dtype, expected_data, expected_index, axis, expected_dtype
    ):
        # GH 14564
        df = DataFrame(columns=["a", "b"], dtype=dtype)
        result = df.quantile(0.5, axis=axis, numeric_only=False)
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype=expected_dtype
        )
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "expected_data, expected_index, axis",
        [
            [[np.nan, np.nan], range(2), 1],
            [[], [], 0],
        ],
    )
    def test_datelike_numeric_only(self, expected_data, expected_index, axis):
        # GH 14564
        df = DataFrame(
            {
                "a": pd.to_datetime(["2010", "2011"]),
                "b": [0, 5],
                "c": pd.to_datetime(["2011", "2012"]),
            }
        )
        result = df[["a", "c"]].quantile(0.5, axis=axis)
        expected = Series(
            expected_data, name=0.5, index=Index(expected_index), dtype=np.float64
        )
        tm.assert_series_equal(result, expected)
