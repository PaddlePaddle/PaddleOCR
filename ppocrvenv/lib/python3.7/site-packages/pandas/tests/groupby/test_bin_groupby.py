import numpy as np
import pytest

from pandas._libs import (
    lib,
    reduction as libreduction,
)
import pandas.util._test_decorators as td

import pandas as pd
from pandas import Series
import pandas._testing as tm


def test_series_grouper():
    obj = Series(np.random.randn(10))

    labels = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 1], dtype=np.intp)

    grouper = libreduction.SeriesGrouper(obj, np.mean, labels, 2)
    result, counts = grouper.get_result()

    expected = np.array([obj[3:6].mean(), obj[6:].mean()], dtype=object)
    tm.assert_almost_equal(result, expected)

    exp_counts = np.array([3, 4], dtype=np.int64)
    tm.assert_almost_equal(counts, exp_counts)


def test_series_grouper_result_length_difference():
    # GH 40014
    obj = Series(np.random.randn(10), dtype="float64")
    obj.index = obj.index.astype("O")
    labels = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 1], dtype=np.intp)

    grouper = libreduction.SeriesGrouper(obj, lambda x: all(x > 0), labels, 2)
    result, counts = grouper.get_result()

    expected = np.array([all(obj[3:6] > 0), all(obj[6:] > 0)], dtype=object)
    tm.assert_equal(result, expected)

    exp_counts = np.array([3, 4], dtype=np.int64)
    tm.assert_equal(counts, exp_counts)


def test_series_grouper_requires_nonempty_raises():
    # GH#29500
    obj = Series(np.random.randn(10))
    dummy = obj.iloc[:0]
    labels = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1, 1], dtype=np.intp)

    with pytest.raises(ValueError, match="SeriesGrouper requires non-empty `series`"):
        libreduction.SeriesGrouper(dummy, np.mean, labels, 2)


def test_series_bin_grouper():
    obj = Series(np.random.randn(10))

    bins = np.array([3, 6], dtype=np.int64)

    grouper = libreduction.SeriesBinGrouper(obj, np.mean, bins)
    result, counts = grouper.get_result()

    expected = np.array([obj[:3].mean(), obj[3:6].mean(), obj[6:].mean()], dtype=object)
    tm.assert_almost_equal(result, expected)

    exp_counts = np.array([3, 3, 4], dtype=np.int64)
    tm.assert_almost_equal(counts, exp_counts)


def assert_block_lengths(x):
    assert len(x) == len(x._mgr.blocks[0].mgr_locs)
    return 0


def cumsum_max(x):
    x.cumsum().max()
    return 0


@pytest.mark.parametrize(
    "func",
    [
        cumsum_max,
        pytest.param(assert_block_lengths, marks=td.skip_array_manager_invalid_test),
    ],
)
def test_mgr_locs_updated(func):
    # https://github.com/pandas-dev/pandas/issues/31802
    # Some operations may require creating new blocks, which requires
    # valid mgr_locs
    df = pd.DataFrame({"A": ["a", "a", "a"], "B": ["a", "b", "b"], "C": [1, 1, 1]})
    result = df.groupby(["A", "B"]).agg(func)
    expected = pd.DataFrame(
        {"C": [0, 0]},
        index=pd.MultiIndex.from_product([["a"], ["a", "b"]], names=["A", "B"]),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "binner,closed,expected",
    [
        (
            np.array([0, 3, 6, 9], dtype=np.int64),
            "left",
            np.array([2, 5, 6], dtype=np.int64),
        ),
        (
            np.array([0, 3, 6, 9], dtype=np.int64),
            "right",
            np.array([3, 6, 6], dtype=np.int64),
        ),
        (np.array([0, 3, 6], dtype=np.int64), "left", np.array([2, 5], dtype=np.int64)),
        (
            np.array([0, 3, 6], dtype=np.int64),
            "right",
            np.array([3, 6], dtype=np.int64),
        ),
    ],
)
def test_generate_bins(binner, closed, expected):
    values = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = lib.generate_bins_dt64(values, binner, closed=closed)
    tm.assert_numpy_array_equal(result, expected)


class TestMoments:
    pass
