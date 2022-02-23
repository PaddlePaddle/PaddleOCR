import inspect
import pydoc

import numpy as np
import pytest

from pandas.util._test_decorators import skip_if_no

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
)
import pandas._testing as tm


class TestSeriesMisc:
    def test_tab_completion(self):
        # GH 9910
        s = Series(list("abcd"))
        # Series of str values should have .str but not .dt/.cat in __dir__
        assert "str" in dir(s)
        assert "dt" not in dir(s)
        assert "cat" not in dir(s)

        # similarly for .dt
        s = Series(date_range("1/1/2015", periods=5))
        assert "dt" in dir(s)
        assert "str" not in dir(s)
        assert "cat" not in dir(s)

        # Similarly for .cat, but with the twist that str and dt should be
        # there if the categories are of that type first cat and str.
        s = Series(list("abbcd"), dtype="category")
        assert "cat" in dir(s)
        assert "str" in dir(s)  # as it is a string categorical
        assert "dt" not in dir(s)

        # similar to cat and str
        s = Series(date_range("1/1/2015", periods=5)).astype("category")
        assert "cat" in dir(s)
        assert "str" not in dir(s)
        assert "dt" in dir(s)  # as it is a datetime categorical

    def test_tab_completion_with_categorical(self):
        # test the tab completion display
        ok_for_cat = [
            "categories",
            "codes",
            "ordered",
            "set_categories",
            "add_categories",
            "remove_categories",
            "rename_categories",
            "reorder_categories",
            "remove_unused_categories",
            "as_ordered",
            "as_unordered",
        ]

        def get_dir(s):
            results = [r for r in s.cat.__dir__() if not r.startswith("_")]
            return sorted(set(results))

        s = Series(list("aabbcde")).astype("category")
        results = get_dir(s)
        tm.assert_almost_equal(results, sorted(set(ok_for_cat)))

    @pytest.mark.parametrize(
        "index",
        [
            tm.makeUnicodeIndex(10),
            tm.makeStringIndex(10),
            tm.makeCategoricalIndex(10),
            Index(["foo", "bar", "baz"] * 2),
            tm.makeDateIndex(10),
            tm.makePeriodIndex(10),
            tm.makeTimedeltaIndex(10),
            tm.makeIntIndex(10),
            tm.makeUIntIndex(10),
            tm.makeIntIndex(10),
            tm.makeFloatIndex(10),
            Index([True, False]),
            Index([f"a{i}" for i in range(101)]),
            pd.MultiIndex.from_tuples(zip("ABCD", "EFGH")),
            pd.MultiIndex.from_tuples(zip([0, 1, 2, 3], "EFGH")),
        ],
    )
    def test_index_tab_completion(self, index):
        # dir contains string-like values of the Index.
        s = Series(index=index, dtype=object)
        dir_s = dir(s)
        for i, x in enumerate(s.index.unique(level=0)):
            if i < 100:
                assert not isinstance(x, str) or not x.isidentifier() or x in dir_s
            else:
                assert x not in dir_s

    def test_not_hashable(self):
        s_empty = Series(dtype=object)
        s = Series([1])
        msg = "unhashable type: 'Series'"
        with pytest.raises(TypeError, match=msg):
            hash(s_empty)
        with pytest.raises(TypeError, match=msg):
            hash(s)

    def test_contains(self, datetime_series):
        tm.assert_contains_all(datetime_series.index, datetime_series)

    def test_raise_on_info(self):
        s = Series(np.random.randn(10))
        msg = "'Series' object has no attribute 'info'"
        with pytest.raises(AttributeError, match=msg):
            s.info()

    def test_axis_alias(self):
        s = Series([1, 2, np.nan])
        tm.assert_series_equal(s.dropna(axis="rows"), s.dropna(axis="index"))
        assert s.dropna().sum("rows") == 3
        assert s._get_axis_number("rows") == 0
        assert s._get_axis_name("rows") == "index"

    def test_class_axis(self):
        # https://github.com/pandas-dev/pandas/issues/18147
        # no exception and no empty docstring
        assert pydoc.getdoc(Series.index)

    def test_ndarray_compat(self):

        # test numpy compat with Series as sub-class of NDFrame
        tsdf = DataFrame(
            np.random.randn(1000, 3),
            columns=["A", "B", "C"],
            index=date_range("1/1/2000", periods=1000),
        )

        def f(x):
            return x[x.idxmax()]

        result = tsdf.apply(f)
        expected = tsdf.max()
        tm.assert_series_equal(result, expected)

        # using an ndarray like function
        s = Series(np.random.randn(10))
        result = Series(np.ones_like(s))
        expected = Series(1, index=range(10), dtype="float64")
        tm.assert_series_equal(result, expected)

        # ravel
        s = Series(np.random.randn(10))
        tm.assert_almost_equal(s.ravel(order="F"), s.values.ravel(order="F"))

    def test_empty_method(self):
        s_empty = Series(dtype=object)
        assert s_empty.empty

        s2 = Series(index=[1], dtype=object)
        for full_series in [Series([1]), s2]:
            assert not full_series.empty

    def test_integer_series_size(self):
        # GH 25580
        s = Series(range(9))
        assert s.size == 9
        s = Series(range(9), dtype="Int64")
        assert s.size == 9

    def test_attrs(self):
        s = Series([0, 1], name="abc")
        assert s.attrs == {}
        s.attrs["version"] = 1
        result = s + 1
        assert result.attrs == {"version": 1}

    @skip_if_no("jinja2")
    def test_inspect_getmembers(self):
        # GH38782
        ser = Series(dtype=object)
        with tm.assert_produces_warning(None):
            inspect.getmembers(ser)
