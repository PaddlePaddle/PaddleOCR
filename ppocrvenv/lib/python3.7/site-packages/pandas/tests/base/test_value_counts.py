import collections
from datetime import timedelta
from io import StringIO

import numpy as np
import pytest

from pandas._libs import iNaT
from pandas.compat import np_array_datetime64_compat

from pandas.core.dtypes.common import needs_i8_conversion

import pandas as pd
from pandas import (
    DatetimeIndex,
    Index,
    Interval,
    IntervalIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
)
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops


def test_value_counts(index_or_series_obj):
    obj = index_or_series_obj
    obj = np.repeat(obj, range(1, len(obj) + 1))
    result = obj.value_counts()

    counter = collections.Counter(obj)
    expected = Series(dict(counter.most_common()), dtype=np.int64, name=obj.name)
    expected.index = expected.index.astype(obj.dtype)
    if isinstance(obj, pd.MultiIndex):
        expected.index = Index(expected.index)

    # TODO: Order of entries with the same count is inconsistent on CI (gh-32449)
    if obj.duplicated().any():
        result = result.sort_index()
        expected = expected.sort_index()
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_value_counts_null(null_obj, index_or_series_obj):
    orig = index_or_series_obj
    obj = orig.copy()

    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    elif isinstance(orig, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    values = obj.values
    if needs_i8_conversion(obj.dtype):
        values[0:2] = iNaT
    else:
        values[0:2] = null_obj

    klass = type(obj)
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    obj = klass(repeated_values, dtype=obj.dtype)

    # because np.nan == np.nan is False, but None == None is True
    # np.nan would be duplicated, whereas None wouldn't
    counter = collections.Counter(obj.dropna())
    expected = Series(dict(counter.most_common()), dtype=np.int64)
    expected.index = expected.index.astype(obj.dtype)

    result = obj.value_counts()
    if obj.duplicated().any():
        # TODO:
        #  Order of entries with the same count is inconsistent on CI (gh-32449)
        expected = expected.sort_index()
        result = result.sort_index()
    tm.assert_series_equal(result, expected)

    # can't use expected[null_obj] = 3 as
    # IntervalIndex doesn't allow assignment
    new_entry = Series({np.nan: 3}, dtype=np.int64)
    expected = expected.append(new_entry)

    result = obj.value_counts(dropna=False)
    if obj.duplicated().any():
        # TODO:
        #  Order of entries with the same count is inconsistent on CI (gh-32449)
        expected = expected.sort_index()
        result = result.sort_index()
    tm.assert_series_equal(result, expected)


def test_value_counts_inferred(index_or_series):
    klass = index_or_series
    s_values = ["a", "b", "b", "b", "b", "c", "d", "d", "a", "a"]
    s = klass(s_values)
    expected = Series([4, 3, 2, 1], index=["b", "a", "d", "c"])
    tm.assert_series_equal(s.value_counts(), expected)

    if isinstance(s, Index):
        exp = Index(np.unique(np.array(s_values, dtype=np.object_)))
        tm.assert_index_equal(s.unique(), exp)
    else:
        exp = np.unique(np.array(s_values, dtype=np.object_))
        tm.assert_numpy_array_equal(s.unique(), exp)

    assert s.nunique() == 4
    # don't sort, have to sort after the fact as not sorting is
    # platform-dep
    hist = s.value_counts(sort=False).sort_values()
    expected = Series([3, 1, 4, 2], index=list("acbd")).sort_values()
    tm.assert_series_equal(hist, expected)

    # sort ascending
    hist = s.value_counts(ascending=True)
    expected = Series([1, 2, 3, 4], index=list("cdab"))
    tm.assert_series_equal(hist, expected)

    # relative histogram.
    hist = s.value_counts(normalize=True)
    expected = Series([0.4, 0.3, 0.2, 0.1], index=["b", "a", "d", "c"])
    tm.assert_series_equal(hist, expected)


def test_value_counts_bins(index_or_series):
    klass = index_or_series
    s_values = ["a", "b", "b", "b", "b", "c", "d", "d", "a", "a"]
    s = klass(s_values)

    # bins
    msg = "bins argument only works with numeric data"
    with pytest.raises(TypeError, match=msg):
        s.value_counts(bins=1)

    s1 = Series([1, 1, 2, 3])
    res1 = s1.value_counts(bins=1)
    exp1 = Series({Interval(0.997, 3.0): 4})
    tm.assert_series_equal(res1, exp1)
    res1n = s1.value_counts(bins=1, normalize=True)
    exp1n = Series({Interval(0.997, 3.0): 1.0})
    tm.assert_series_equal(res1n, exp1n)

    if isinstance(s1, Index):
        tm.assert_index_equal(s1.unique(), Index([1, 2, 3]))
    else:
        exp = np.array([1, 2, 3], dtype=np.int64)
        tm.assert_numpy_array_equal(s1.unique(), exp)

    assert s1.nunique() == 3

    # these return the same
    res4 = s1.value_counts(bins=4, dropna=True)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]))
    tm.assert_series_equal(res4, exp4)

    res4 = s1.value_counts(bins=4, dropna=False)
    intervals = IntervalIndex.from_breaks([0.997, 1.5, 2.0, 2.5, 3.0])
    exp4 = Series([2, 1, 1, 0], index=intervals.take([0, 1, 3, 2]))
    tm.assert_series_equal(res4, exp4)

    res4n = s1.value_counts(bins=4, normalize=True)
    exp4n = Series([0.5, 0.25, 0.25, 0], index=intervals.take([0, 1, 3, 2]))
    tm.assert_series_equal(res4n, exp4n)

    # handle NA's properly
    s_values = ["a", "b", "b", "b", np.nan, np.nan, "d", "d", "a", "a", "b"]
    s = klass(s_values)
    expected = Series([4, 3, 2], index=["b", "a", "d"])
    tm.assert_series_equal(s.value_counts(), expected)

    if isinstance(s, Index):
        exp = Index(["a", "b", np.nan, "d"])
        tm.assert_index_equal(s.unique(), exp)
    else:
        exp = np.array(["a", "b", np.nan, "d"], dtype=object)
        tm.assert_numpy_array_equal(s.unique(), exp)
    assert s.nunique() == 3

    s = klass({}) if klass is dict else klass({}, dtype=object)
    expected = Series([], dtype=np.int64)
    tm.assert_series_equal(s.value_counts(), expected, check_index_type=False)
    # returned dtype differs depending on original
    if isinstance(s, Index):
        tm.assert_index_equal(s.unique(), Index([]), exact=False)
    else:
        tm.assert_numpy_array_equal(s.unique(), np.array([]), check_dtype=False)

    assert s.nunique() == 0


def test_value_counts_datetime64(index_or_series):
    klass = index_or_series

    # GH 3002, datetime64[ns]
    # don't test names though
    txt = "\n".join(
        [
            "xxyyzz20100101PIE",
            "xxyyzz20100101GUM",
            "xxyyzz20100101EGG",
            "xxyyww20090101EGG",
            "foofoo20080909PIE",
            "foofoo20080909GUM",
        ]
    )
    f = StringIO(txt)
    df = pd.read_fwf(
        f, widths=[6, 8, 3], names=["person_id", "dt", "food"], parse_dates=["dt"]
    )

    s = klass(df["dt"].copy())
    s.name = None
    idx = pd.to_datetime(
        ["2010-01-01 00:00:00", "2008-09-09 00:00:00", "2009-01-01 00:00:00"]
    )
    expected_s = Series([3, 2, 1], index=idx)
    tm.assert_series_equal(s.value_counts(), expected_s)

    expected = np_array_datetime64_compat(
        ["2010-01-01 00:00:00", "2009-01-01 00:00:00", "2008-09-09 00:00:00"],
        dtype="datetime64[ns]",
    )
    if isinstance(s, Index):
        tm.assert_index_equal(s.unique(), DatetimeIndex(expected))
    else:
        tm.assert_numpy_array_equal(s.unique(), expected)

    assert s.nunique() == 3

    # with NaT
    s = df["dt"].copy()
    s = klass(list(s.values) + [pd.NaT] * 4)

    result = s.value_counts()
    assert result.index.dtype == "datetime64[ns]"
    tm.assert_series_equal(result, expected_s)

    result = s.value_counts(dropna=False)
    expected_s = pd.concat([Series([4], index=DatetimeIndex([pd.NaT])), expected_s])
    tm.assert_series_equal(result, expected_s)

    assert s.dtype == "datetime64[ns]"
    unique = s.unique()
    assert unique.dtype == "datetime64[ns]"

    # numpy_array_equal cannot compare pd.NaT
    if isinstance(s, Index):
        exp_idx = DatetimeIndex(expected.tolist() + [pd.NaT])
        tm.assert_index_equal(unique, exp_idx)
    else:
        tm.assert_numpy_array_equal(unique[:3], expected)
        assert pd.isna(unique[3])

    assert s.nunique() == 3
    assert s.nunique(dropna=False) == 4

    # timedelta64[ns]
    td = df.dt - df.dt + timedelta(1)
    td = klass(td, name="dt")

    result = td.value_counts()
    expected_s = Series([6], index=[Timedelta("1day")], name="dt")
    tm.assert_series_equal(result, expected_s)

    expected = TimedeltaIndex(["1 days"], name="dt")
    if isinstance(td, Index):
        tm.assert_index_equal(td.unique(), expected)
    else:
        tm.assert_numpy_array_equal(td.unique(), expected.values)

    td2 = timedelta(1) + (df.dt - df.dt)
    td2 = klass(td2, name="dt")
    result2 = td2.value_counts()
    tm.assert_series_equal(result2, expected_s)


@pytest.mark.parametrize("dropna", [True, False])
def test_value_counts_with_nan(dropna, index_or_series):
    # GH31944
    klass = index_or_series
    values = [True, pd.NA, np.nan]
    s = klass(values)
    res = s.value_counts(dropna=dropna)
    if dropna is True:
        expected = Series([1], index=[True])
    else:
        expected = Series([2, 1], index=[pd.NA, True])
    tm.assert_series_equal(res, expected)
