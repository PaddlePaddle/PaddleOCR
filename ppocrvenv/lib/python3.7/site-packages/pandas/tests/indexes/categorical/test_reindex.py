import numpy as np
import pytest

from pandas import (
    Categorical,
    CategoricalIndex,
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm


class TestReindex:
    def test_reindex_dtype(self):
        c = CategoricalIndex(["a", "b", "c", "a"])
        res, indexer = c.reindex(["a", "c"])
        tm.assert_index_equal(res, Index(["a", "a", "c"]), exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 3, 2], dtype=np.intp))

        c = CategoricalIndex(["a", "b", "c", "a"])
        res, indexer = c.reindex(Categorical(["a", "c"]))

        exp = CategoricalIndex(["a", "a", "c"], categories=["a", "c"])
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 3, 2], dtype=np.intp))

        c = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        res, indexer = c.reindex(["a", "c"])
        exp = Index(["a", "a", "c"], dtype="object")
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 3, 2], dtype=np.intp))

        c = CategoricalIndex(["a", "b", "c", "a"], categories=["a", "b", "c", "d"])
        res, indexer = c.reindex(Categorical(["a", "c"]))
        exp = CategoricalIndex(["a", "a", "c"], categories=["a", "c"])
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 3, 2], dtype=np.intp))

    def test_reindex_duplicate_target(self):
        # See GH25459
        cat = CategoricalIndex(["a", "b", "c"], categories=["a", "b", "c", "d"])
        res, indexer = cat.reindex(["a", "c", "c"])
        exp = Index(["a", "c", "c"], dtype="object")
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))

        res, indexer = cat.reindex(
            CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])
        )
        exp = CategoricalIndex(["a", "c", "c"], categories=["a", "b", "c", "d"])
        tm.assert_index_equal(res, exp, exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([0, 2, 2], dtype=np.intp))

    def test_reindex_empty_index(self):
        # See GH16770
        c = CategoricalIndex([])
        res, indexer = c.reindex(["a", "b"])
        tm.assert_index_equal(res, Index(["a", "b"]), exact=True)
        tm.assert_numpy_array_equal(indexer, np.array([-1, -1], dtype=np.intp))

    def test_reindex_missing_category(self):
        # GH: 18185
        ser = Series([1, 2, 3, 1], dtype="category")
        msg = "'fill_value=-1' is not present in this Categorical's categories"
        with pytest.raises(TypeError, match=msg):
            ser.reindex([1, 2, 3, 4, 5], fill_value=-1)

    @pytest.mark.parametrize(
        "index_df,index_res,index_exp",
        [
            (
                CategoricalIndex([], categories=["A"]),
                Index(["A"]),
                Index(["A"]),
            ),
            (
                CategoricalIndex([], categories=["A"]),
                Index(["B"]),
                Index(["B"]),
            ),
            (
                CategoricalIndex([], categories=["A"]),
                CategoricalIndex(["A"]),
                CategoricalIndex(["A"]),
            ),
            (
                CategoricalIndex([], categories=["A"]),
                CategoricalIndex(["B"]),
                CategoricalIndex(["B"]),
            ),
        ],
    )
    def test_reindex_not_category(self, index_df, index_res, index_exp):
        # GH: 28690
        df = DataFrame(index=index_df)
        result = df.reindex(index=index_res)
        expected = DataFrame(index=index_exp)
        tm.assert_frame_equal(result, expected)
