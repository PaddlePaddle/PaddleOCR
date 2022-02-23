import numpy as np
import pytest

from pandas import (
    Categorical,
    Index,
    MultiIndex,
    NaT,
    Period,
    PeriodIndex,
    Series,
    date_range,
    isna,
)
import pandas._testing as tm


def test_reindex(datetime_series, string_series):
    identity = string_series.reindex(string_series.index)

    # __array_interface__ is not defined for older numpies
    # and on some pythons
    try:
        assert np.may_share_memory(string_series.index, identity.index)
    except AttributeError:
        pass

    assert identity.index.is_(string_series.index)
    assert identity.index.identical(string_series.index)

    subIndex = string_series.index[10:20]
    subSeries = string_series.reindex(subIndex)

    for idx, val in subSeries.items():
        assert val == string_series[idx]

    subIndex2 = datetime_series.index[10:20]
    subTS = datetime_series.reindex(subIndex2)

    for idx, val in subTS.items():
        assert val == datetime_series[idx]
    stuffSeries = datetime_series.reindex(subIndex)

    assert np.isnan(stuffSeries).all()

    # This is extremely important for the Cython code to not screw up
    nonContigIndex = datetime_series.index[::2]
    subNonContig = datetime_series.reindex(nonContigIndex)
    for idx, val in subNonContig.items():
        assert val == datetime_series[idx]

    # return a copy the same index here
    result = datetime_series.reindex()
    assert not (result is datetime_series)


def test_reindex_nan():
    ts = Series([2, 3, 5, 7], index=[1, 4, np.nan, 8])

    i, j = [np.nan, 1, np.nan, 8, 4, np.nan], [2, 0, 2, 3, 1, 2]
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j])

    ts.index = ts.index.astype("object")

    # reindex coerces index.dtype to float, loc/iloc doesn't
    tm.assert_series_equal(ts.reindex(i), ts.iloc[j], check_index_type=False)


def test_reindex_series_add_nat():
    rng = date_range("1/1/2000 00:00:00", periods=10, freq="10s")
    series = Series(rng)

    result = series.reindex(range(15))
    assert np.issubdtype(result.dtype, np.dtype("M8[ns]"))

    mask = result.isna()
    assert mask[-5:].all()
    assert not mask[:-5].any()


def test_reindex_with_datetimes():
    rng = date_range("1/1/2000", periods=20)
    ts = Series(np.random.randn(20), index=rng)

    result = ts.reindex(list(ts.index[5:10]))
    expected = ts[5:10]
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)

    result = ts[list(ts.index[5:10])]
    tm.assert_series_equal(result, expected)


def test_reindex_corner(datetime_series):
    # (don't forget to fix this) I think it's fixed
    empty = Series(dtype=object)
    empty.reindex(datetime_series.index, method="pad")  # it works

    # corner case: pad empty series
    reindexed = empty.reindex(datetime_series.index, method="pad")

    # pass non-Index
    reindexed = datetime_series.reindex(list(datetime_series.index))
    datetime_series.index = datetime_series.index._with_freq(None)
    tm.assert_series_equal(datetime_series, reindexed)

    # bad fill method
    ts = datetime_series[::2]
    msg = (
        r"Invalid fill method\. Expecting pad \(ffill\), backfill "
        r"\(bfill\) or nearest\. Got foo"
    )
    with pytest.raises(ValueError, match=msg):
        ts.reindex(datetime_series.index, method="foo")


def test_reindex_pad():
    s = Series(np.arange(10), dtype="int64")
    s2 = s[::2]

    reindexed = s2.reindex(s.index, method="pad")
    reindexed2 = s2.reindex(s.index, method="ffill")
    tm.assert_series_equal(reindexed, reindexed2)

    expected = Series([0, 0, 2, 2, 4, 4, 6, 6, 8, 8], index=np.arange(10))
    tm.assert_series_equal(reindexed, expected)

    # GH4604
    s = Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    new_index = ["a", "g", "c", "f"]
    expected = Series([1, 1, 3, 3], index=new_index)

    # this changes dtype because the ffill happens after
    result = s.reindex(new_index).ffill()
    tm.assert_series_equal(result, expected.astype("float64"))

    result = s.reindex(new_index).ffill(downcast="infer")
    tm.assert_series_equal(result, expected)

    expected = Series([1, 5, 3, 5], index=new_index)
    result = s.reindex(new_index, method="ffill")
    tm.assert_series_equal(result, expected)

    # inference of new dtype
    s = Series([True, False, False, True], index=list("abcd"))
    new_index = "agc"
    result = s.reindex(list(new_index)).ffill()
    expected = Series([True, True, False], index=list(new_index))
    tm.assert_series_equal(result, expected)

    # GH4618 shifted series downcasting
    s = Series(False, index=range(0, 5))
    result = s.shift(1).fillna(method="bfill")
    expected = Series(False, index=range(0, 5))
    tm.assert_series_equal(result, expected)


def test_reindex_nearest():
    s = Series(np.arange(10, dtype="int64"))
    target = [0.1, 0.9, 1.5, 2.0]
    result = s.reindex(target, method="nearest")
    expected = Series(np.around(target).astype("int64"), target)
    tm.assert_series_equal(expected, result)

    result = s.reindex(target, method="nearest", tolerance=0.2)
    expected = Series([0, 1, np.nan, 2], target)
    tm.assert_series_equal(expected, result)

    result = s.reindex(target, method="nearest", tolerance=[0.3, 0.01, 0.4, 3])
    expected = Series([0, np.nan, np.nan, 2], target)
    tm.assert_series_equal(expected, result)


def test_reindex_backfill():
    pass


def test_reindex_int(datetime_series):
    ts = datetime_series[::2]
    int_ts = Series(np.zeros(len(ts), dtype=int), index=ts.index)

    # this should work fine
    reindexed_int = int_ts.reindex(datetime_series.index)

    # if NaNs introduced
    assert reindexed_int.dtype == np.float_

    # NO NaNs introduced
    reindexed_int = int_ts.reindex(int_ts.index[::2])
    assert reindexed_int.dtype == np.int_


def test_reindex_bool(datetime_series):
    # A series other than float, int, string, or object
    ts = datetime_series[::2]
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)

    # this should work fine
    reindexed_bool = bool_ts.reindex(datetime_series.index)

    # if NaNs introduced
    assert reindexed_bool.dtype == np.object_

    # NO NaNs introduced
    reindexed_bool = bool_ts.reindex(bool_ts.index[::2])
    assert reindexed_bool.dtype == np.bool_


def test_reindex_bool_pad(datetime_series):
    # fail
    ts = datetime_series[5:]
    bool_ts = Series(np.zeros(len(ts), dtype=bool), index=ts.index)
    filled_bool = bool_ts.reindex(datetime_series.index, method="pad")
    assert isna(filled_bool[:5]).all()


def test_reindex_categorical():
    index = date_range("20000101", periods=3)

    # reindexing to an invalid Categorical
    s = Series(["a", "b", "c"], dtype="category")
    result = s.reindex(index)
    expected = Series(
        Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
    )
    expected.index = index
    tm.assert_series_equal(result, expected)

    # partial reindexing
    expected = Series(Categorical(values=["b", "c"], categories=["a", "b", "c"]))
    expected.index = [1, 2]
    result = s.reindex([1, 2])
    tm.assert_series_equal(result, expected)

    expected = Series(Categorical(values=["c", np.nan], categories=["a", "b", "c"]))
    expected.index = [2, 3]
    result = s.reindex([2, 3])
    tm.assert_series_equal(result, expected)


def test_reindex_astype_order_consistency():
    # GH#17444
    ser = Series([1, 2, 3], index=[2, 0, 1])
    new_index = [0, 1, 2]
    temp_dtype = "category"
    new_dtype = str
    result = ser.reindex(new_index).astype(temp_dtype).astype(new_dtype)
    expected = ser.astype(temp_dtype).reindex(new_index).astype(new_dtype)
    tm.assert_series_equal(result, expected)


def test_reindex_fill_value():
    # -----------------------------------------------------------
    # floats
    floats = Series([1.0, 2.0, 3.0])
    result = floats.reindex([1, 2, 3])
    expected = Series([2.0, 3.0, np.nan], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)

    result = floats.reindex([1, 2, 3], fill_value=0)
    expected = Series([2.0, 3.0, 0], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # ints
    ints = Series([1, 2, 3])

    result = ints.reindex([1, 2, 3])
    expected = Series([2.0, 3.0, np.nan], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)

    # don't upcast
    result = ints.reindex([1, 2, 3], fill_value=0)
    expected = Series([2, 3, 0], index=[1, 2, 3])
    assert issubclass(result.dtype.type, np.integer)
    tm.assert_series_equal(result, expected)

    # -----------------------------------------------------------
    # objects
    objects = Series([1, 2, 3], dtype=object)

    result = objects.reindex([1, 2, 3])
    expected = Series([2, 3, np.nan], index=[1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)

    result = objects.reindex([1, 2, 3], fill_value="foo")
    expected = Series([2, 3, "foo"], index=[1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)

    # ------------------------------------------------------------
    # bools
    bools = Series([True, False, True])

    result = bools.reindex([1, 2, 3])
    expected = Series([False, True, np.nan], index=[1, 2, 3], dtype=object)
    tm.assert_series_equal(result, expected)

    result = bools.reindex([1, 2, 3], fill_value=False)
    expected = Series([False, True, False], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)


def test_reindex_datetimeindexes_tz_naive_and_aware():
    # GH 8306
    idx = date_range("20131101", tz="America/Chicago", periods=7)
    newidx = date_range("20131103", periods=10, freq="H")
    s = Series(range(7), index=idx)
    msg = (
        r"Cannot compare dtypes datetime64\[ns, America/Chicago\] "
        r"and datetime64\[ns\]"
    )
    with pytest.raises(TypeError, match=msg):
        s.reindex(newidx, method="ffill")


def test_reindex_empty_series_tz_dtype():
    # GH 20869
    result = Series(dtype="datetime64[ns, UTC]").reindex([0, 1])
    expected = Series([NaT] * 2, dtype="datetime64[ns, UTC]")
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "p_values, o_values, values, expected_values",
    [
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC"), "All"],
            [1.0, 1.0],
            [1.0, 1.0, np.nan],
        ),
        (
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [Period("2019Q1", "Q-DEC"), Period("2019Q2", "Q-DEC")],
            [1.0, 1.0],
            [1.0, 1.0],
        ),
    ],
)
def test_reindex_periodindex_with_object(p_values, o_values, values, expected_values):
    # GH#28337
    period_index = PeriodIndex(p_values)
    object_index = Index(o_values)

    ser = Series(values, index=period_index)
    result = ser.reindex(object_index)
    expected = Series(expected_values, index=object_index)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
def test_reindex_empty_with_level(values):
    # GH41170
    ser = Series(
        range(len(values[0])), index=MultiIndex.from_arrays(values), dtype="object"
    )
    result = ser.reindex(np.array(["b"]), level=0)
    expected = Series(
        index=MultiIndex(levels=[["b"], values[1]], codes=[[], []]), dtype="object"
    )
    tm.assert_series_equal(result, expected)
