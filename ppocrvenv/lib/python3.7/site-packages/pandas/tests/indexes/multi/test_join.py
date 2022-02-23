import numpy as np
import pytest

from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


@pytest.mark.parametrize(
    "other", [Index(["three", "one", "two"]), Index(["one"]), Index(["one", "three"])]
)
def test_join_level(idx, other, join_type):
    join_index, lidx, ridx = other.join(
        idx, how=join_type, level="second", return_indexers=True
    )

    exp_level = other.join(idx.levels[1], how=join_type)
    assert join_index.levels[0].equals(idx.levels[0])
    assert join_index.levels[1].equals(exp_level)

    # pare down levels
    mask = np.array([x[1] in exp_level for x in idx], dtype=bool)
    exp_values = idx.values[mask]
    tm.assert_numpy_array_equal(join_index.values, exp_values)

    if join_type in ("outer", "inner"):
        join_index2, ridx2, lidx2 = idx.join(
            other, how=join_type, level="second", return_indexers=True
        )

        assert join_index.equals(join_index2)
        tm.assert_numpy_array_equal(lidx, lidx2)
        tm.assert_numpy_array_equal(ridx, ridx2)
        tm.assert_numpy_array_equal(join_index2.values, exp_values)


def test_join_level_corner_case(idx):
    # some corner cases
    index = Index(["three", "one", "two"])
    result = index.join(idx, level="second")
    assert isinstance(result, MultiIndex)

    with pytest.raises(TypeError, match="Join.*MultiIndex.*ambiguous"):
        idx.join(idx, level=1)


def test_join_self(idx, join_type):
    joined = idx.join(idx, how=join_type)
    tm.assert_index_equal(joined, idx)


def test_join_multi():
    # GH 10665
    midx = MultiIndex.from_product([np.arange(4), np.arange(4)], names=["a", "b"])
    idx = Index([1, 2, 5], name="b")

    # inner
    jidx, lidx, ridx = midx.join(idx, how="inner", return_indexers=True)
    exp_idx = MultiIndex.from_product([np.arange(4), [1, 2]], names=["a", "b"])
    exp_lidx = np.array([1, 2, 5, 6, 9, 10, 13, 14], dtype=np.intp)
    exp_ridx = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.intp)
    tm.assert_index_equal(jidx, exp_idx)
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)
    # flip
    jidx, ridx, lidx = idx.join(midx, how="inner", return_indexers=True)
    tm.assert_index_equal(jidx, exp_idx)
    tm.assert_numpy_array_equal(lidx, exp_lidx)
    tm.assert_numpy_array_equal(ridx, exp_ridx)

    # keep MultiIndex
    jidx, lidx, ridx = midx.join(idx, how="left", return_indexers=True)
    exp_ridx = np.array(
        [-1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1, -1, 0, 1, -1], dtype=np.intp
    )
    tm.assert_index_equal(jidx, midx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)
    # flip
    jidx, ridx, lidx = idx.join(midx, how="right", return_indexers=True)
    tm.assert_index_equal(jidx, midx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_join_self_unique(idx, join_type):
    if idx.is_unique:
        joined = idx.join(idx, how=join_type)
        assert (idx == joined).all()


def test_join_multi_wrong_order():
    # GH 25760
    # GH 28956

    midx1 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["b", "a"])

    join_idx, lidx, ridx = midx1.join(midx2, return_indexers=True)

    exp_ridx = np.array([-1, -1, -1, -1], dtype=np.intp)

    tm.assert_index_equal(midx1, join_idx)
    assert lidx is None
    tm.assert_numpy_array_equal(ridx, exp_ridx)


def test_join_multi_return_indexers():
    # GH 34074

    midx1 = MultiIndex.from_product([[1, 2], [3, 4], [5, 6]], names=["a", "b", "c"])
    midx2 = MultiIndex.from_product([[1, 2], [3, 4]], names=["a", "b"])

    result = midx1.join(midx2, return_indexers=False)
    tm.assert_index_equal(result, midx1)
