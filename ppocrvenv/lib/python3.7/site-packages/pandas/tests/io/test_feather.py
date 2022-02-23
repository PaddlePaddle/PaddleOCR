""" test feather-format compat """
import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm

from pandas.io.feather_format import read_feather, to_feather  # isort:skip

pyarrow = pytest.importorskip("pyarrow")


filter_sparse = pytest.mark.filterwarnings("ignore:The Sparse")


@filter_sparse
@pytest.mark.single
@pytest.mark.filterwarnings("ignore:CategoricalBlock is deprecated:DeprecationWarning")
class TestFeather:
    def check_error_on_write(self, df, exc, err_msg):
        # check that we are raising the exception
        # on writing

        with pytest.raises(exc, match=err_msg):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_external_error_on_write(self, df):
        # check that we are raising the exception
        # on writing

        with tm.external_error_raised(Exception):
            with tm.ensure_clean() as path:
                to_feather(df, path)

    def check_round_trip(self, df, expected=None, write_kwargs={}, **read_kwargs):

        if expected is None:
            expected = df

        with tm.ensure_clean() as path:
            to_feather(df, path, **write_kwargs)

            result = read_feather(path, **read_kwargs)
            tm.assert_frame_equal(result, expected)

    def test_error(self):

        msg = "feather only support IO with DataFrames"
        for obj in [
            pd.Series([1, 2, 3]),
            1,
            "foo",
            pd.Timestamp("20130101"),
            np.array([1, 2, 3]),
        ]:
            self.check_error_on_write(obj, ValueError, msg)

    def test_basic(self):

        df = pd.DataFrame(
            {
                "string": list("abc"),
                "int": list(range(1, 4)),
                "uint": np.arange(3, 6).astype("u1"),
                "float": np.arange(4.0, 7.0, dtype="float64"),
                "float_with_null": [1.0, np.nan, 3],
                "bool": [True, False, True],
                "bool_with_null": [True, np.nan, False],
                "cat": pd.Categorical(list("abc")),
                "dt": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3)), freq=None
                ),
                "dttz": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, tz="US/Eastern")),
                    freq=None,
                ),
                "dt_with_null": [
                    pd.Timestamp("20130101"),
                    pd.NaT,
                    pd.Timestamp("20130103"),
                ],
                "dtns": pd.DatetimeIndex(
                    list(pd.date_range("20130101", periods=3, freq="ns")), freq=None
                ),
            }
        )
        df["periods"] = pd.period_range("2013", freq="M", periods=3)
        df["timedeltas"] = pd.timedelta_range("1 day", periods=3)
        # TODO temporary disable due to regression in pyarrow 0.17.1
        # https://github.com/pandas-dev/pandas/issues/34255
        # df["intervals"] = pd.interval_range(0, 3, 3)

        assert df.dttz.dtype.tz.zone == "US/Eastern"
        self.check_round_trip(df)

    def test_duplicate_columns(self):

        # https://github.com/wesm/feather/issues/53
        # not currently able to handle duplicate columns
        df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list("aaa")).copy()
        self.check_external_error_on_write(df)

    def test_stringify_columns(self):

        df = pd.DataFrame(np.arange(12).reshape(4, 3)).copy()
        msg = "feather must have string column names"
        self.check_error_on_write(df, ValueError, msg)

    def test_read_columns(self):
        # GH 24025
        df = pd.DataFrame(
            {
                "col1": list("abc"),
                "col2": list(range(1, 4)),
                "col3": list("xyz"),
                "col4": list(range(4, 7)),
            }
        )
        columns = ["col1", "col3"]
        self.check_round_trip(df, expected=df[columns], columns=columns)

    @td.skip_if_no("pyarrow", min_version="0.17.1")
    def read_columns_different_order(self):
        # GH 33878
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"], "C": [True, False]})
        self.check_round_trip(df, columns=["B", "A"])

    def test_unsupported_other(self):

        # mixed python objects
        df = pd.DataFrame({"a": ["a", 1, 2.0]})
        self.check_external_error_on_write(df)

    def test_rw_use_threads(self):
        df = pd.DataFrame({"A": np.arange(100000)})
        self.check_round_trip(df, use_threads=True)
        self.check_round_trip(df, use_threads=False)

    def test_write_with_index(self):

        df = pd.DataFrame({"A": [1, 2, 3]})
        self.check_round_trip(df)

        msg = (
            r"feather does not support serializing .* for the index; "
            r"you can \.reset_index\(\) to make the index into column\(s\)"
        )
        # non-default index
        for index in [
            [2, 3, 4],
            pd.date_range("20130101", periods=3),
            list("abc"),
            [1, 3, 4],
            pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)]),
        ]:

            df.index = index
            self.check_error_on_write(df, ValueError, msg)

        # index with meta-data
        df.index = [0, 1, 2]
        df.index.name = "foo"
        msg = "feather does not serialize index meta-data on a default index"
        self.check_error_on_write(df, ValueError, msg)

        # column multi-index
        df.index = [0, 1, 2]
        df.columns = pd.MultiIndex.from_tuples([("a", 1)])
        msg = "feather must have string column names"
        self.check_error_on_write(df, ValueError, msg)

    def test_path_pathlib(self):
        df = tm.makeDataFrame().reset_index()
        result = tm.round_trip_pathlib(df.to_feather, read_feather)
        tm.assert_frame_equal(df, result)

    def test_path_localpath(self):
        df = tm.makeDataFrame().reset_index()
        result = tm.round_trip_localpath(df.to_feather, read_feather)
        tm.assert_frame_equal(df, result)

    @td.skip_if_no("pyarrow", min_version="0.17.0")
    def test_passthrough_keywords(self):
        df = tm.makeDataFrame().reset_index()
        self.check_round_trip(df, write_kwargs={"version": 1})

    @td.skip_if_no("pyarrow")
    @tm.network
    def test_http_path(self, feather_file):
        # GH 29055
        url = (
            "https://raw.githubusercontent.com/pandas-dev/pandas/master/"
            "pandas/tests/io/data/feather/feather-0_3_1.feather"
        )
        expected = read_feather(feather_file)
        res = read_feather(url)
        tm.assert_frame_equal(expected, res)
