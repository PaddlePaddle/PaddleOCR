import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm


def test_unstack():
    index = MultiIndex(
        levels=[["bar", "foo"], ["one", "three", "two"]],
        codes=[[1, 1, 0, 0], [0, 1, 0, 2]],
    )

    s = Series(np.arange(4.0), index=index)
    unstacked = s.unstack()

    expected = DataFrame(
        [[2.0, np.nan, 3.0], [0.0, 1.0, np.nan]],
        index=["bar", "foo"],
        columns=["one", "three", "two"],
    )

    tm.assert_frame_equal(unstacked, expected)

    unstacked = s.unstack(level=0)
    tm.assert_frame_equal(unstacked, expected.T)

    index = MultiIndex(
        levels=[["bar"], ["one", "two", "three"], [0, 1]],
        codes=[[0, 0, 0, 0, 0, 0], [0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    s = Series(np.random.randn(6), index=index)
    exp_index = MultiIndex(
        levels=[["one", "two", "three"], [0, 1]],
        codes=[[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 0, 1]],
    )
    expected = DataFrame({"bar": s.values}, index=exp_index).sort_index(level=0)
    unstacked = s.unstack(0).sort_index()
    tm.assert_frame_equal(unstacked, expected)

    # GH5873
    idx = MultiIndex.from_arrays([[101, 102], [3.5, np.nan]])
    ts = Series([1, 2], index=idx)
    left = ts.unstack()
    right = DataFrame(
        [[np.nan, 1], [2, np.nan]], index=[101, 102], columns=[np.nan, 3.5]
    )
    tm.assert_frame_equal(left, right)

    idx = MultiIndex.from_arrays(
        [
            ["cat", "cat", "cat", "dog", "dog"],
            ["a", "a", "b", "a", "b"],
            [1, 2, 1, 1, np.nan],
        ]
    )
    ts = Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
    right = DataFrame(
        [[1.0, 1.3], [1.1, np.nan], [np.nan, 1.4], [1.2, np.nan]],
        columns=["cat", "dog"],
    )
    tpls = [("a", 1), ("a", 2), ("b", np.nan), ("b", 1)]
    right.index = MultiIndex.from_tuples(tpls)
    tm.assert_frame_equal(ts.unstack(level=0), right)


def test_unstack_tuplename_in_multiindex():
    # GH 19966
    idx = MultiIndex.from_product(
        [["a", "b", "c"], [1, 2, 3]], names=[("A", "a"), ("B", "b")]
    )
    ser = Series(1, index=idx)
    result = ser.unstack(("A", "a"))

    expected = DataFrame(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        columns=MultiIndex.from_tuples([("a",), ("b",), ("c",)], names=[("A", "a")]),
        index=pd.Index([1, 2, 3], name=("B", "b")),
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "unstack_idx, expected_values, expected_index, expected_columns",
    [
        (
            ("A", "a"),
            [[1, 1], [1, 1], [1, 1], [1, 1]],
            MultiIndex.from_tuples([(1, 3), (1, 4), (2, 3), (2, 4)], names=["B", "C"]),
            MultiIndex.from_tuples([("a",), ("b",)], names=[("A", "a")]),
        ),
        (
            (("A", "a"), "B"),
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            pd.Index([3, 4], name="C"),
            MultiIndex.from_tuples(
                [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=[("A", "a"), "B"]
            ),
        ),
    ],
)
def test_unstack_mixed_type_name_in_multiindex(
    unstack_idx, expected_values, expected_index, expected_columns
):
    # GH 19966
    idx = MultiIndex.from_product(
        [["a", "b"], [1, 2], [3, 4]], names=[("A", "a"), "B", "C"]
    )
    ser = Series(1, index=idx)
    result = ser.unstack(unstack_idx)

    expected = DataFrame(
        expected_values, columns=expected_columns, index=expected_index
    )
    tm.assert_frame_equal(result, expected)


def test_unstack_multi_index_categorical_values():

    mi = tm.makeTimeDataFrame().stack().index.rename(["major", "minor"])
    ser = Series(["foo"] * len(mi), index=mi, name="category", dtype="category")

    result = ser.unstack()

    dti = ser.index.levels[0]
    c = pd.Categorical(["foo"] * len(dti))
    expected = DataFrame(
        {"A": c.copy(), "B": c.copy(), "C": c.copy(), "D": c.copy()},
        columns=pd.Index(list("ABCD"), name="minor"),
        index=dti.rename("major"),
    )
    tm.assert_frame_equal(result, expected)
