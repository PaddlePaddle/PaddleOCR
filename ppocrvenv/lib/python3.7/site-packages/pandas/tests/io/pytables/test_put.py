import datetime
import re
from warnings import (
    catch_warnings,
    simplefilter,
)

import numpy as np
import pytest

from pandas._libs.tslibs import Timestamp

import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    Int64Index,
    MultiIndex,
    RangeIndex,
    Series,
    _testing as tm,
    concat,
)
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_path,
    ensure_clean_store,
)
from pandas.util import _test_decorators as td

pytestmark = pytest.mark.single


def test_format_type(setup_path):
    df = DataFrame({"A": [1, 2]})
    with ensure_clean_path(setup_path) as path:
        with HDFStore(path) as store:
            store.put("a", df, format="fixed")
            store.put("b", df, format="table")

            assert store.get_storer("a").format_type == "fixed"
            assert store.get_storer("b").format_type == "table"


def test_format_kwarg_in_constructor(setup_path):
    # GH 13291

    msg = "format is not a defined argument for HDFStore"

    with tm.ensure_clean(setup_path) as path:
        with pytest.raises(ValueError, match=msg):
            HDFStore(path, format="table")


def test_api_default_format(setup_path):

    # default_format option
    with ensure_clean_store(setup_path) as store:
        df = tm.makeDataFrame()

        pd.set_option("io.hdf.default_format", "fixed")
        _maybe_remove(store, "df")
        store.put("df", df)
        assert not store.get_storer("df").is_table

        msg = "Can only append to Tables"

        with pytest.raises(ValueError, match=msg):
            store.append("df2", df)

        pd.set_option("io.hdf.default_format", "table")
        _maybe_remove(store, "df")
        store.put("df", df)
        assert store.get_storer("df").is_table
        _maybe_remove(store, "df2")
        store.append("df2", df)
        assert store.get_storer("df").is_table

        pd.set_option("io.hdf.default_format", None)

    with ensure_clean_path(setup_path) as path:

        df = tm.makeDataFrame()

        pd.set_option("io.hdf.default_format", "fixed")
        df.to_hdf(path, "df")
        with HDFStore(path) as store:
            assert not store.get_storer("df").is_table
        with pytest.raises(ValueError, match=msg):
            df.to_hdf(path, "df2", append=True)

        pd.set_option("io.hdf.default_format", "table")
        df.to_hdf(path, "df3")
        with HDFStore(path) as store:
            assert store.get_storer("df3").is_table
        df.to_hdf(path, "df4", append=True)
        with HDFStore(path) as store:
            assert store.get_storer("df4").is_table

        pd.set_option("io.hdf.default_format", None)


def test_put(setup_path):

    with ensure_clean_store(setup_path) as store:

        ts = tm.makeTimeSeries()
        df = tm.makeTimeDataFrame()
        store["a"] = ts
        store["b"] = df[:10]
        store["foo/bar/bah"] = df[:10]
        store["foo"] = df[:10]
        store["/foo"] = df[:10]
        store.put("c", df[:10], format="table")

        # not OK, not a table
        msg = "Can only append to Tables"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df[10:], append=True)

        # node does not currently exist, test _is_table_type returns False
        # in this case
        _maybe_remove(store, "f")
        with pytest.raises(ValueError, match=msg):
            store.put("f", df[10:], append=True)

        # can't put to a table (use append instead)
        with pytest.raises(ValueError, match=msg):
            store.put("c", df[10:], append=True)

        # overwrite table
        store.put("c", df[:10], format="table", append=False)
        tm.assert_frame_equal(df[:10], store["c"])


def test_put_string_index(setup_path):

    with ensure_clean_store(setup_path) as store:

        index = Index([f"I am a very long string index: {i}" for i in range(20)])
        s = Series(np.arange(20), index=index)
        df = DataFrame({"A": s, "B": s})

        store["a"] = s
        tm.assert_series_equal(store["a"], s)

        store["b"] = df
        tm.assert_frame_equal(store["b"], df)

        # mixed length
        index = Index(
            ["abcdefghijklmnopqrstuvwxyz1234567890"]
            + [f"I am a very long string index: {i}" for i in range(20)]
        )
        s = Series(np.arange(21), index=index)
        df = DataFrame({"A": s, "B": s})
        store["a"] = s
        tm.assert_series_equal(store["a"], s)

        store["b"] = df
        tm.assert_frame_equal(store["b"], df)


def test_put_compression(setup_path):

    with ensure_clean_store(setup_path) as store:
        df = tm.makeTimeDataFrame()

        store.put("c", df, format="table", complib="zlib")
        tm.assert_frame_equal(store["c"], df)

        # can't compress if format='fixed'
        msg = "Compression not supported on Fixed format stores"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df, format="fixed", complib="zlib")


@td.skip_if_windows_python_3
def test_put_compression_blosc(setup_path):
    df = tm.makeTimeDataFrame()

    with ensure_clean_store(setup_path) as store:

        # can't compress if format='fixed'
        msg = "Compression not supported on Fixed format stores"
        with pytest.raises(ValueError, match=msg):
            store.put("b", df, format="fixed", complib="blosc")

        store.put("c", df, format="table", complib="blosc")
        tm.assert_frame_equal(store["c"], df)


def test_put_mixed_type(setup_path):
    df = tm.makeTimeDataFrame()
    df["obj1"] = "foo"
    df["obj2"] = "bar"
    df["bool1"] = df["A"] > 0
    df["bool2"] = df["B"] > 0
    df["bool3"] = True
    df["int1"] = 1
    df["int2"] = 2
    df["timestamp1"] = Timestamp("20010102")
    df["timestamp2"] = Timestamp("20010103")
    df["datetime1"] = datetime.datetime(2001, 1, 2, 0, 0)
    df["datetime2"] = datetime.datetime(2001, 1, 3, 0, 0)
    df.loc[df.index[3:6], ["obj1"]] = np.nan
    df = df._consolidate()._convert(datetime=True)

    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, "df")

        # PerformanceWarning
        with catch_warnings(record=True):
            simplefilter("ignore", pd.errors.PerformanceWarning)
            store.put("df", df)

        expected = store.get("df")
        tm.assert_frame_equal(expected, df)


def test_store_index_types(setup_path):
    # GH5386
    # test storing various index types

    with ensure_clean_store(setup_path) as store:

        def check(format, index):
            df = DataFrame(np.random.randn(10, 2), columns=list("AB"))
            df.index = index(len(df))

            _maybe_remove(store, "df")
            store.put("df", df, format=format)
            tm.assert_frame_equal(df, store["df"])

        for index in [
            tm.makeFloatIndex,
            tm.makeStringIndex,
            tm.makeIntIndex,
            tm.makeDateIndex,
        ]:

            check("table", index)
            check("fixed", index)

        # period index currently broken for table
        # seee GH7796 FIXME
        check("fixed", tm.makePeriodIndex)
        # check('table',tm.makePeriodIndex)

        # unicode
        index = tm.makeUnicodeIndex
        check("table", index)
        check("fixed", index)


def test_column_multiindex(setup_path):
    # GH 4710
    # recreate multi-indexes properly

    index = MultiIndex.from_tuples(
        [("A", "a"), ("A", "b"), ("B", "a"), ("B", "b")], names=["first", "second"]
    )
    df = DataFrame(np.arange(12).reshape(3, 4), columns=index)
    expected = df.copy()
    if isinstance(expected.index, RangeIndex):
        expected.index = Int64Index(expected.index)

    with ensure_clean_store(setup_path) as store:

        store.put("df", df)
        tm.assert_frame_equal(
            store["df"], expected, check_index_type=True, check_column_type=True
        )

        store.put("df1", df, format="table")
        tm.assert_frame_equal(
            store["df1"], expected, check_index_type=True, check_column_type=True
        )

        msg = re.escape("cannot use a multi-index on axis [1] with data_columns ['A']")
        with pytest.raises(ValueError, match=msg):
            store.put("df2", df, format="table", data_columns=["A"])
        msg = re.escape("cannot use a multi-index on axis [1] with data_columns True")
        with pytest.raises(ValueError, match=msg):
            store.put("df3", df, format="table", data_columns=True)

    # appending multi-column on existing table (see GH 6167)
    with ensure_clean_store(setup_path) as store:
        store.append("df2", df)
        store.append("df2", df)

        tm.assert_frame_equal(store["df2"], concat((df, df)))

    # non_index_axes name
    df = DataFrame(np.arange(12).reshape(3, 4), columns=Index(list("ABCD"), name="foo"))
    expected = df.copy()
    if isinstance(expected.index, RangeIndex):
        expected.index = Int64Index(expected.index)

    with ensure_clean_store(setup_path) as store:

        store.put("df1", df, format="table")
        tm.assert_frame_equal(
            store["df1"], expected, check_index_type=True, check_column_type=True
        )


def test_store_multiindex(setup_path):

    # validate multi-index names
    # GH 5527
    with ensure_clean_store(setup_path) as store:

        def make_index(names=None):
            return MultiIndex.from_tuples(
                [
                    (datetime.datetime(2013, 12, d), s, t)
                    for d in range(1, 3)
                    for s in range(2)
                    for t in range(3)
                ],
                names=names,
            )

        # no names
        _maybe_remove(store, "df")
        df = DataFrame(np.zeros((12, 2)), columns=["a", "b"], index=make_index())
        store.append("df", df)
        tm.assert_frame_equal(store.select("df"), df)

        # partial names
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", None, None]),
        )
        store.append("df", df)
        tm.assert_frame_equal(store.select("df"), df)

        # series
        _maybe_remove(store, "s")
        s = Series(np.zeros(12), index=make_index(["date", None, None]))
        store.append("s", s)
        xp = Series(np.zeros(12), index=make_index(["date", "level_1", "level_2"]))
        tm.assert_series_equal(store.select("s"), xp)

        # dup with column
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "a", "t"]),
        )
        msg = "duplicate names/columns in the multi-index when storing as a table"
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # dup within level
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "date", "date"]),
        )
        with pytest.raises(ValueError, match=msg):
            store.append("df", df)

        # fully names
        _maybe_remove(store, "df")
        df = DataFrame(
            np.zeros((12, 2)),
            columns=["a", "b"],
            index=make_index(["date", "s", "t"]),
        )
        store.append("df", df)
        tm.assert_frame_equal(store.select("df"), df)
