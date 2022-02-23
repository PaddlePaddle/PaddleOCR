import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    MultiIndex,
)
import pandas._testing as tm


def test_reindex(idx):
    result, indexer = idx.reindex(list(idx[:4]))
    assert isinstance(result, MultiIndex)
    assert result.names == ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]

    result, indexer = idx.reindex(list(idx))
    assert isinstance(result, MultiIndex)
    assert indexer is None
    assert result.names == ["first", "second"]
    assert [level.name for level in result.levels] == ["first", "second"]


def test_reindex_level(idx):
    index = Index(["one"])

    target, indexer = idx.reindex(index, level="second")
    target2, indexer2 = index.reindex(idx, level="second")

    exp_index = idx.join(index, level="second", how="right")
    exp_index2 = idx.join(index, level="second", how="left")

    assert target.equals(exp_index)
    exp_indexer = np.array([0, 2, 4])
    tm.assert_numpy_array_equal(indexer, exp_indexer, check_dtype=False)

    assert target2.equals(exp_index2)
    exp_indexer2 = np.array([0, -1, 0, -1, 0, -1])
    tm.assert_numpy_array_equal(indexer2, exp_indexer2, check_dtype=False)

    with pytest.raises(TypeError, match="Fill method not supported"):
        idx.reindex(idx, method="pad", level="second")

    with pytest.raises(TypeError, match="Fill method not supported"):
        index.reindex(index, method="bfill", level="first")


def test_reindex_preserves_names_when_target_is_list_or_ndarray(idx):
    # GH6552
    idx = idx.copy()
    target = idx.copy()
    idx.names = target.names = [None, None]

    other_dtype = MultiIndex.from_product([[1, 2], [3, 4]])

    # list & ndarray cases
    assert idx.reindex([])[0].names == [None, None]
    assert idx.reindex(np.array([]))[0].names == [None, None]
    assert idx.reindex(target.tolist())[0].names == [None, None]
    assert idx.reindex(target.values)[0].names == [None, None]
    assert idx.reindex(other_dtype.tolist())[0].names == [None, None]
    assert idx.reindex(other_dtype.values)[0].names == [None, None]

    idx.names = ["foo", "bar"]
    assert idx.reindex([])[0].names == ["foo", "bar"]
    assert idx.reindex(np.array([]))[0].names == ["foo", "bar"]
    assert idx.reindex(target.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(target.values)[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.tolist())[0].names == ["foo", "bar"]
    assert idx.reindex(other_dtype.values)[0].names == ["foo", "bar"]


def test_reindex_lvl_preserves_names_when_target_is_list_or_array():
    # GH7774
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]], names=["foo", "bar"])
    assert idx.reindex([], level=0)[0].names == ["foo", "bar"]
    assert idx.reindex([], level=1)[0].names == ["foo", "bar"]


def test_reindex_lvl_preserves_type_if_target_is_empty_list_or_array():
    # GH7774
    idx = MultiIndex.from_product([[0, 1], ["a", "b"]])
    assert idx.reindex([], level=0)[0].levels[0].dtype.type == np.int64
    assert idx.reindex([], level=1)[0].levels[1].dtype.type == np.object_


def test_reindex_base(idx):
    idx = idx
    expected = np.arange(idx.size, dtype=np.intp)

    actual = idx.get_indexer(idx)
    tm.assert_numpy_array_equal(expected, actual)

    with pytest.raises(ValueError, match="Invalid fill method"):
        idx.get_indexer(idx, method="invalid")


def test_reindex_non_unique():
    idx = MultiIndex.from_tuples([(0, 0), (1, 1), (1, 1), (2, 2)])
    a = pd.Series(np.arange(4), index=idx)
    new_idx = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 2)])

    msg = "cannot handle a non-unique multi-index!"
    with pytest.raises(ValueError, match=msg):
        a.reindex(new_idx)


@pytest.mark.parametrize("values", [[["a"], ["x"]], [[], []]])
def test_reindex_empty_with_level(values):
    # GH41170
    idx = MultiIndex.from_arrays(values)
    result, result_indexer = idx.reindex(np.array(["b"]), level=0)
    expected = MultiIndex(levels=[["b"], values[1]], codes=[[], []])
    expected_indexer = np.array([], dtype=result_indexer.dtype)
    tm.assert_index_equal(result, expected)
    tm.assert_numpy_array_equal(result_indexer, expected_indexer)


def test_reindex_not_all_tuples():
    keys = [("i", "i"), ("i", "j"), ("j", "i"), "j"]
    mi = MultiIndex.from_tuples(keys[:-1])
    idx = Index(keys)
    res, indexer = mi.reindex(idx)

    tm.assert_index_equal(res, idx)
    expected = np.array([0, 1, 2, -1], dtype=np.intp)
    tm.assert_numpy_array_equal(indexer, expected)
