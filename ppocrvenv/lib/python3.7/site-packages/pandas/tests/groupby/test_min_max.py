import numpy as np
import pytest

from pandas._libs.tslibs import iNaT

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


def test_max_min_non_numeric():
    # #2700
    aa = DataFrame({"nn": [11, 11, 22, 22], "ii": [1, 2, 3, 4], "ss": 4 * ["mama"]})

    result = aa.groupby("nn").max()
    assert "ss" in result

    result = aa.groupby("nn").max(numeric_only=False)
    assert "ss" in result

    result = aa.groupby("nn").min()
    assert "ss" in result

    result = aa.groupby("nn").min(numeric_only=False)
    assert "ss" in result


def test_max_min_object_multiple_columns(using_array_manager):
    # GH#41111 case where the aggregation is valid for some columns but not
    # others; we split object blocks column-wise, consistent with
    # DataFrame._reduce

    df = DataFrame(
        {
            "A": [1, 1, 2, 2, 3],
            "B": [1, "foo", 2, "bar", False],
            "C": ["a", "b", "c", "d", "e"],
        }
    )
    df._consolidate_inplace()  # should already be consolidate, but double-check
    if not using_array_manager:
        assert len(df._mgr.blocks) == 2

    gb = df.groupby("A")

    with tm.assert_produces_warning(FutureWarning, match="Dropping invalid"):
        result = gb.max(numeric_only=False)
    # "max" is valid for column "C" but not for "B"
    ei = Index([1, 2, 3], name="A")
    expected = DataFrame({"C": ["b", "d", "e"]}, index=ei)
    tm.assert_frame_equal(result, expected)

    with tm.assert_produces_warning(FutureWarning, match="Dropping invalid"):
        result = gb.min(numeric_only=False)
    # "min" is valid for column "C" but not for "B"
    ei = Index([1, 2, 3], name="A")
    expected = DataFrame({"C": ["a", "c", "e"]}, index=ei)
    tm.assert_frame_equal(result, expected)


def test_min_date_with_nans():
    # GH26321
    dates = pd.to_datetime(
        Series(["2019-05-09", "2019-05-09", "2019-05-09"]), format="%Y-%m-%d"
    ).dt.date
    df = DataFrame({"a": [np.nan, "1", np.nan], "b": [0, 1, 1], "c": dates})

    result = df.groupby("b", as_index=False)["c"].min()["c"]
    expected = pd.to_datetime(
        Series(["2019-05-09", "2019-05-09"], name="c"), format="%Y-%m-%d"
    ).dt.date
    tm.assert_series_equal(result, expected)

    result = df.groupby("b")["c"].min()
    expected.index.name = "b"
    tm.assert_series_equal(result, expected)


def test_max_inat():
    # GH#40767 dont interpret iNaT as NaN
    ser = Series([1, iNaT])
    gb = ser.groupby([1, 1])

    result = gb.max(min_count=2)
    expected = Series({1: 1}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)

    result = gb.min(min_count=2)
    expected = Series({1: iNaT}, dtype=np.int64)
    tm.assert_series_equal(result, expected, check_exact=True)

    # not enough entries -> gets masked to NaN
    result = gb.min(min_count=3)
    expected = Series({1: np.nan})
    tm.assert_series_equal(result, expected, check_exact=True)


def test_max_inat_not_all_na():
    # GH#40767 dont interpret iNaT as NaN

    # make sure we dont round iNaT+1 to iNaT
    ser = Series([1, iNaT, 2, iNaT + 1])
    gb = ser.groupby([1, 2, 3, 3])
    result = gb.min(min_count=2)

    # Note: in converting to float64, the iNaT + 1 maps to iNaT, i.e. is lossy
    expected = Series({1: np.nan, 2: np.nan, 3: iNaT + 1})
    tm.assert_series_equal(result, expected, check_exact=True)


@pytest.mark.parametrize("func", ["min", "max"])
def test_groupby_aggregate_period_column(func):
    # GH 31471
    groups = [1, 2]
    periods = pd.period_range("2020", periods=2, freq="Y")
    df = DataFrame({"a": groups, "b": periods})

    result = getattr(df.groupby("a")["b"], func)()
    idx = pd.Int64Index([1, 2], name="a")
    expected = Series(periods, index=idx, name="b")

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_groupby_aggregate_period_frame(func):
    # GH 31471
    groups = [1, 2]
    periods = pd.period_range("2020", periods=2, freq="Y")
    df = DataFrame({"a": groups, "b": periods})

    result = getattr(df.groupby("a"), func)()
    idx = pd.Int64Index([1, 2], name="a")
    expected = DataFrame({"b": periods}, index=idx)

    tm.assert_frame_equal(result, expected)


def test_aggregate_numeric_object_dtype():
    # https://github.com/pandas-dev/pandas/issues/39329
    # simplified case: multiple object columns where one is all-NaN
    # -> gets split as the all-NaN is inferred as float
    df = DataFrame(
        {"key": ["A", "A", "B", "B"], "col1": list("abcd"), "col2": [np.nan] * 4},
    ).astype(object)
    result = df.groupby("key").min()
    expected = DataFrame(
        {"key": ["A", "B"], "col1": ["a", "c"], "col2": [np.nan, np.nan]}
    ).set_index("key")
    tm.assert_frame_equal(result, expected)

    # same but with numbers
    df = DataFrame(
        {"key": ["A", "A", "B", "B"], "col1": list("abcd"), "col2": range(4)},
    ).astype(object)
    result = df.groupby("key").min()
    expected = DataFrame(
        {"key": ["A", "B"], "col1": ["a", "c"], "col2": [0, 2]}
    ).set_index("key")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", ["min", "max"])
def test_aggregate_categorical_lost_index(func: str):
    # GH: 28641 groupby drops index, when grouping over categorical column with min/max
    ds = Series(["b"], dtype="category").cat.as_ordered()
    df = DataFrame({"A": [1997], "B": ds})
    result = df.groupby("A").agg({"B": func})
    expected = DataFrame({"B": ["b"]}, index=Index([1997], name="A"))

    # ordered categorical dtype should be preserved
    expected["B"] = expected["B"].astype(ds.dtype)

    tm.assert_frame_equal(result, expected)
