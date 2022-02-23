""" test get/set & misc """
from datetime import timedelta
import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    IndexSlice,
    MultiIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)
import pandas._testing as tm


def test_basic_indexing():
    s = Series(np.random.randn(5), index=["a", "b", "a", "a", "b"])

    msg = "index 5 is out of bounds for axis 0 with size 5"
    with pytest.raises(IndexError, match=msg):
        s[5]
    with pytest.raises(IndexError, match=msg):
        s[5] = 0

    with pytest.raises(KeyError, match=r"^'c'$"):
        s["c"]

    s = s.sort_index()

    with pytest.raises(IndexError, match=msg):
        s[5]
    msg = r"index 5 is out of bounds for axis (0|1) with size 5|^5$"
    with pytest.raises(IndexError, match=msg):
        s[5] = 0


def test_basic_getitem_with_labels(datetime_series):
    indices = datetime_series.index[[5, 10, 15]]

    result = datetime_series[indices]
    expected = datetime_series.reindex(indices)
    tm.assert_series_equal(result, expected)

    result = datetime_series[indices[0] : indices[2]]
    expected = datetime_series.loc[indices[0] : indices[2]]
    tm.assert_series_equal(result, expected)


def test_basic_getitem_dt64tz_values():

    # GH12089
    # with tz for values
    ser = Series(
        date_range("2011-01-01", periods=3, tz="US/Eastern"), index=["a", "b", "c"]
    )
    expected = Timestamp("2011-01-01", tz="US/Eastern")
    result = ser.loc["a"]
    assert result == expected
    result = ser.iloc[0]
    assert result == expected
    result = ser["a"]
    assert result == expected


def test_getitem_setitem_ellipsis():
    s = Series(np.random.randn(10))

    np.fix(s)

    result = s[...]
    tm.assert_series_equal(result, s)

    s[...] = 5
    assert (result == 5).all()


@pytest.mark.parametrize(
    "result_1, duplicate_item, expected_1",
    [
        [
            Series({1: 12, 2: [1, 2, 2, 3]}),
            Series({1: 313}),
            Series({1: 12}, dtype=object),
        ],
        [
            Series({1: [1, 2, 3], 2: [1, 2, 2, 3]}),
            Series({1: [1, 2, 3]}),
            Series({1: [1, 2, 3]}),
        ],
    ],
)
def test_getitem_with_duplicates_indices(result_1, duplicate_item, expected_1):
    # GH 17610
    result = result_1.append(duplicate_item)
    expected = expected_1.append(duplicate_item)
    tm.assert_series_equal(result[1], expected)
    assert result[2] == result_1[2]


def test_getitem_setitem_integers():
    # caused bug without test
    s = Series([1, 2, 3], ["a", "b", "c"])

    assert s.iloc[0] == s["a"]
    s.iloc[0] = 5
    tm.assert_almost_equal(s["a"], 5)


def test_series_box_timestamp():
    rng = date_range("20090415", "20090519", freq="B")
    ser = Series(rng)
    assert isinstance(ser[0], Timestamp)
    assert isinstance(ser.at[1], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[3], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)

    ser = Series(rng, index=rng)
    assert isinstance(ser[0], Timestamp)
    assert isinstance(ser.at[rng[1]], Timestamp)
    assert isinstance(ser.iat[2], Timestamp)
    assert isinstance(ser.loc[rng[3]], Timestamp)
    assert isinstance(ser.iloc[4], Timestamp)


def test_series_box_timedelta():
    rng = timedelta_range("1 day 1 s", periods=5, freq="h")
    ser = Series(rng)
    assert isinstance(ser[0], Timedelta)
    assert isinstance(ser.at[1], Timedelta)
    assert isinstance(ser.iat[2], Timedelta)
    assert isinstance(ser.loc[3], Timedelta)
    assert isinstance(ser.iloc[4], Timedelta)


def test_getitem_ambiguous_keyerror(indexer_sl):
    ser = Series(range(10), index=list(range(0, 20, 2)))
    with pytest.raises(KeyError, match=r"^1$"):
        indexer_sl(ser)[1]


def test_getitem_dups_with_missing(indexer_sl):
    # breaks reindex, so need to use .loc internally
    # GH 4246
    ser = Series([1, 2, 3, 4], ["foo", "bar", "foo", "bah"])
    with pytest.raises(KeyError, match=re.escape("['bam'] not in index")):
        indexer_sl(ser)[["foo", "bar", "bah", "bam"]]


def test_setitem_ambiguous_keyerror(indexer_sl):
    s = Series(range(10), index=list(range(0, 20, 2)))

    # equivalent of an append
    s2 = s.copy()
    indexer_sl(s2)[1] = 5
    expected = s.append(Series([5], index=[1]))
    tm.assert_series_equal(s2, expected)


def test_setitem(datetime_series, string_series):
    datetime_series[datetime_series.index[5]] = np.NaN
    datetime_series[[1, 2, 17]] = np.NaN
    datetime_series[6] = np.NaN
    assert np.isnan(datetime_series[6])
    assert np.isnan(datetime_series[2])
    datetime_series[np.isnan(datetime_series)] = 5
    assert not np.isnan(datetime_series[2])


def test_setslice(datetime_series):
    sl = datetime_series[5:20]
    assert len(sl) == len(sl.index)
    assert sl.index.is_unique is True


# FutureWarning from NumPy about [slice(None, 5).
@pytest.mark.filterwarnings("ignore:Using a non-tuple:FutureWarning")
def test_basic_getitem_setitem_corner(datetime_series):
    # invalid tuples, e.g. td.ts[:, None] vs. td.ts[:, 2]
    msg = "key of type tuple not found and not a MultiIndex"
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2]
    with pytest.raises(KeyError, match=msg):
        datetime_series[:, 2] = 2

    # weird lists. [slice(0, 5)] will work but not two slices
    with tm.assert_produces_warning(FutureWarning):
        # GH#31299
        result = datetime_series[[slice(None, 5)]]
    expected = datetime_series[:5]
    tm.assert_series_equal(result, expected)

    # OK
    msg = r"unhashable type(: 'slice')?"
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, slice(None, None)]]
    with pytest.raises(TypeError, match=msg):
        datetime_series[[5, slice(None, None)]] = 2


def test_slice(string_series, object_series):
    numSlice = string_series[10:20]
    numSliceEnd = string_series[-10:]
    objSlice = object_series[10:20]

    assert string_series.index[9] not in numSlice.index
    assert object_series.index[9] not in objSlice.index

    assert len(numSlice) == len(numSlice.index)
    assert string_series[numSlice.index[0]] == numSlice[numSlice.index[0]]

    assert numSlice.index[1] == string_series.index[11]
    assert tm.equalContents(numSliceEnd, np.array(string_series)[-10:])

    # Test return view.
    sl = string_series[10:20]
    sl[:] = 0

    assert (string_series[10:20] == 0).all()


def test_timedelta_assignment():
    # GH 8209
    s = Series([], dtype=object)
    s.loc["B"] = timedelta(1)
    tm.assert_series_equal(s, Series(Timedelta("1 days"), index=["B"]))

    s = s.reindex(s.index.insert(0, "A"))
    tm.assert_series_equal(s, Series([np.nan, Timedelta("1 days")], index=["A", "B"]))

    s.loc["A"] = timedelta(1)
    expected = Series(Timedelta("1 days"), index=["A", "B"])
    tm.assert_series_equal(s, expected)


def test_underlying_data_conversion():
    # GH 4080
    df = DataFrame({c: [1, 2, 3] for c in ["a", "b", "c"]})
    return_value = df.set_index(["a", "b", "c"], inplace=True)
    assert return_value is None
    s = Series([1], index=[(2, 2, 2)])
    df["val"] = 0
    df
    df["val"].update(s)

    expected = DataFrame(
        {"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3], "val": [0, 1, 0]}
    )
    return_value = expected.set_index(["a", "b", "c"], inplace=True)
    assert return_value is None
    tm.assert_frame_equal(df, expected)


def test_preserve_refs(datetime_series):
    seq = datetime_series[[5, 10, 15]]
    seq[1] = np.NaN
    assert not np.isnan(datetime_series[10])


def test_cast_on_putmask():
    # GH 2746

    # need to upcast
    s = Series([1, 2], index=[1, 2], dtype="int64")
    s[[True, False]] = Series([0], index=[1], dtype="int64")
    expected = Series([0, 2], index=[1, 2], dtype="int64")

    tm.assert_series_equal(s, expected)


def test_type_promote_putmask():
    # GH8387: test that changing types does not break alignment
    ts = Series(np.random.randn(100), index=np.arange(100, 0, -1)).round(5)
    left, mask = ts.copy(), ts > 0
    right = ts[mask].copy().map(str)
    left[mask] = right
    tm.assert_series_equal(left, ts.map(lambda t: str(t) if t > 0 else t))


def test_setitem_mask_promote_strs():

    ser = Series([0, 1, 2, 0])
    mask = ser > 0
    ser2 = ser[mask].map(str)
    ser[mask] = ser2

    expected = Series([0, "1", "2", 0])
    tm.assert_series_equal(ser, expected)


def test_setitem_mask_promote():

    ser = Series([0, "foo", "bar", 0])
    mask = Series([False, True, True, False])
    ser2 = ser[mask]
    ser[mask] = ser2

    expected = Series([0, "foo", "bar", 0])
    tm.assert_series_equal(ser, expected)


def test_multilevel_preserve_name(indexer_sl):
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    ser = Series(np.random.randn(len(index)), index=index, name="sth")

    result = indexer_sl(ser)["foo"]
    assert result.name == ser.name


"""
miscellaneous methods
"""


def test_slice_with_zero_step_raises(index, frame_or_series, indexer_sli):
    ts = frame_or_series(np.arange(len(index)), index=index)

    with pytest.raises(ValueError, match="slice step cannot be zero"):
        indexer_sli(ts)[::0]


@pytest.mark.parametrize(
    "index",
    [
        date_range("2014-01-01", periods=20, freq="MS"),
        period_range("2014-01", periods=20, freq="M"),
        timedelta_range("0", periods=20, freq="H"),
    ],
)
def test_slice_with_negative_step(index):
    def assert_slices_equivalent(l_slc, i_slc):
        expected = ts.iloc[i_slc]

        tm.assert_series_equal(ts[l_slc], expected)
        tm.assert_series_equal(ts.loc[l_slc], expected)

    keystr1 = str(index[9])
    keystr2 = str(index[13])
    box = type(index[0])

    ts = Series(np.arange(20), index)
    SLC = IndexSlice

    for key in [keystr1, box(keystr1)]:
        assert_slices_equivalent(SLC[key::-1], SLC[9::-1])
        assert_slices_equivalent(SLC[:key:-1], SLC[:8:-1])

        for key2 in [keystr2, box(keystr2)]:
            assert_slices_equivalent(SLC[key2:key:-1], SLC[13:8:-1])
            assert_slices_equivalent(SLC[key:key2:-1], SLC[0:0:-1])


def test_tuple_index():
    # GH 35534 - Selecting values when a Series has an Index of tuples
    s = Series([1, 2], index=[("a",), ("b",)])
    assert s[("a",)] == 1
    assert s[("b",)] == 2
    s[("b",)] = 3
    assert s[("b",)] == 3


def test_frozenset_index():
    # GH35747 - Selecting values when a Series has an Index of frozenset
    idx0, idx1 = frozenset("a"), frozenset("b")
    s = Series([1, 2], index=[idx0, idx1])
    assert s[idx0] == 1
    assert s[idx1] == 2
    s[idx1] = 3
    assert s[idx1] == 3
