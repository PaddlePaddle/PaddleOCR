import numpy as np
import pytest
import pytz

import pandas as pd
from pandas import (
    Series,
    date_range,
    period_range,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "first_slice,second_slice",
    [
        [[2, None], [None, -5]],
        [[None, 0], [None, -5]],
        [[None, -5], [None, 0]],
        [[None, 0], [None, 0]],
    ],
)
@pytest.mark.parametrize("fill", [None, -1])
def test_align(datetime_series, first_slice, second_slice, join_type, fill):
    a = datetime_series[slice(*first_slice)]
    b = datetime_series[slice(*second_slice)]

    aa, ab = a.align(b, join=join_type, fill_value=fill)

    join_index = a.index.join(b.index, how=join_type)
    if fill is not None:
        diff_a = aa.index.difference(join_index)
        diff_b = ab.index.difference(join_index)
        if len(diff_a) > 0:
            assert (aa.reindex(diff_a) == fill).all()
        if len(diff_b) > 0:
            assert (ab.reindex(diff_b) == fill).all()

    ea = a.reindex(join_index)
    eb = b.reindex(join_index)

    if fill is not None:
        ea = ea.fillna(fill)
        eb = eb.fillna(fill)

    tm.assert_series_equal(aa, ea)
    tm.assert_series_equal(ab, eb)
    assert aa.name == "ts"
    assert ea.name == "ts"
    assert ab.name == "ts"
    assert eb.name == "ts"


@pytest.mark.parametrize(
    "first_slice,second_slice",
    [
        [[2, None], [None, -5]],
        [[None, 0], [None, -5]],
        [[None, -5], [None, 0]],
        [[None, 0], [None, 0]],
    ],
)
@pytest.mark.parametrize("method", ["pad", "bfill"])
@pytest.mark.parametrize("limit", [None, 1])
def test_align_fill_method(
    datetime_series, first_slice, second_slice, join_type, method, limit
):
    a = datetime_series[slice(*first_slice)]
    b = datetime_series[slice(*second_slice)]

    aa, ab = a.align(b, join=join_type, method=method, limit=limit)

    join_index = a.index.join(b.index, how=join_type)
    ea = a.reindex(join_index)
    eb = b.reindex(join_index)

    ea = ea.fillna(method=method, limit=limit)
    eb = eb.fillna(method=method, limit=limit)

    tm.assert_series_equal(aa, ea)
    tm.assert_series_equal(ab, eb)


def test_align_nocopy(datetime_series):
    b = datetime_series[:5].copy()

    # do copy
    a = datetime_series.copy()
    ra, _ = a.align(b, join="left")
    ra[:5] = 5
    assert not (a[:5] == 5).any()

    # do not copy
    a = datetime_series.copy()
    ra, _ = a.align(b, join="left", copy=False)
    ra[:5] = 5
    assert (a[:5] == 5).all()

    # do copy
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join="right")
    rb[:3] = 5
    assert not (b[:3] == 5).any()

    # do not copy
    a = datetime_series.copy()
    b = datetime_series[:5].copy()
    _, rb = a.align(b, join="right", copy=False)
    rb[:2] = 5
    assert (b[:2] == 5).all()


def test_align_same_index(datetime_series):
    a, b = datetime_series.align(datetime_series, copy=False)
    assert a.index is datetime_series.index
    assert b.index is datetime_series.index

    a, b = datetime_series.align(datetime_series, copy=True)
    assert a.index is not datetime_series.index
    assert b.index is not datetime_series.index


def test_align_multiindex():
    # GH 10665

    midx = pd.MultiIndex.from_product(
        [range(2), range(3), range(2)], names=("a", "b", "c")
    )
    idx = pd.Index(range(2), name="b")
    s1 = Series(np.arange(12, dtype="int64"), index=midx)
    s2 = Series(np.arange(2, dtype="int64"), index=idx)

    # these must be the same results (but flipped)
    res1l, res1r = s1.align(s2, join="left")
    res2l, res2r = s2.align(s1, join="right")

    expl = s1
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr = Series([0, 0, 1, 1, np.nan, np.nan] * 2, index=midx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)

    res1l, res1r = s1.align(s2, join="right")
    res2l, res2r = s2.align(s1, join="left")

    exp_idx = pd.MultiIndex.from_product(
        [range(2), range(2), range(2)], names=("a", "b", "c")
    )
    expl = Series([0, 1, 2, 3, 6, 7, 8, 9], index=exp_idx)
    tm.assert_series_equal(expl, res1l)
    tm.assert_series_equal(expl, res2r)
    expr = Series([0, 0, 1, 1] * 2, index=exp_idx)
    tm.assert_series_equal(expr, res1r)
    tm.assert_series_equal(expr, res2l)


@pytest.mark.parametrize("method", ["backfill", "bfill", "pad", "ffill", None])
def test_align_with_dataframe_method(method):
    # GH31788
    ser = Series(range(3), index=range(3))
    df = pd.DataFrame(0.0, index=range(3), columns=range(3))

    result_ser, result_df = ser.align(df, method=method)
    tm.assert_series_equal(result_ser, ser)
    tm.assert_frame_equal(result_df, df)


def test_align_dt64tzindex_mismatched_tzs():
    idx1 = date_range("2001", periods=5, freq="H", tz="US/Eastern")
    ser = Series(np.random.randn(len(idx1)), index=idx1)
    ser_central = ser.tz_convert("US/Central")
    # different timezones convert to UTC

    new1, new2 = ser.align(ser_central)
    assert new1.index.tz == pytz.UTC
    assert new2.index.tz == pytz.UTC


def test_align_periodindex(join_type):
    rng = period_range("1/1/2000", "1/1/2010", freq="A")
    ts = Series(np.random.randn(len(rng)), index=rng)

    # TODO: assert something?
    ts.align(ts[::2], join=join_type)
