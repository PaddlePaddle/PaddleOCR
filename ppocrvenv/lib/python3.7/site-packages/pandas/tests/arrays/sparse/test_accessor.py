import string

import numpy as np
import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.sparse import (
    SparseArray,
    SparseDtype,
)


class TestSeriesAccessor:
    # TODO: collect other Series accessor tests
    def test_to_dense(self):
        s = pd.Series([0, 1, 0, 10], dtype="Sparse[int64]")
        result = s.sparse.to_dense()
        expected = pd.Series([0, 1, 0, 10])
        tm.assert_series_equal(result, expected)


class TestFrameAccessor:
    def test_accessor_raises(self):
        df = pd.DataFrame({"A": [0, 1]})
        with pytest.raises(AttributeError, match="sparse"):
            df.sparse

    @pytest.mark.parametrize("format", ["csc", "csr", "coo"])
    @pytest.mark.parametrize("labels", [None, list(string.ascii_letters[:10])])
    @pytest.mark.parametrize("dtype", ["float64", "int64"])
    @td.skip_if_no_scipy
    def test_from_spmatrix(self, format, labels, dtype):
        import scipy.sparse

        sp_dtype = SparseDtype(dtype, np.array(0, dtype=dtype).item())

        mat = scipy.sparse.eye(10, format=format, dtype=dtype)
        result = pd.DataFrame.sparse.from_spmatrix(mat, index=labels, columns=labels)
        expected = pd.DataFrame(
            np.eye(10, dtype=dtype), index=labels, columns=labels
        ).astype(sp_dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("format", ["csc", "csr", "coo"])
    @td.skip_if_no_scipy
    def test_from_spmatrix_including_explicit_zero(self, format):
        import scipy.sparse

        mat = scipy.sparse.random(10, 2, density=0.5, format=format)
        mat.data[0] = 0
        result = pd.DataFrame.sparse.from_spmatrix(mat)
        dtype = SparseDtype("float64", 0.0)
        expected = pd.DataFrame(mat.todense()).astype(dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "columns",
        [["a", "b"], pd.MultiIndex.from_product([["A"], ["a", "b"]]), ["a", "a"]],
    )
    @td.skip_if_no_scipy
    def test_from_spmatrix_columns(self, columns):
        import scipy.sparse

        dtype = SparseDtype("float64", 0.0)

        mat = scipy.sparse.random(10, 2, density=0.5)
        result = pd.DataFrame.sparse.from_spmatrix(mat, columns=columns)
        expected = pd.DataFrame(mat.toarray(), columns=columns).astype(dtype)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("colnames", [("A", "B"), (1, 2), (1, pd.NA), (0.1, 0.2)])
    @td.skip_if_no_scipy
    def test_to_coo(self, colnames):
        import scipy.sparse

        df = pd.DataFrame(
            {colnames[0]: [0, 1, 0], colnames[1]: [1, 0, 0]}, dtype="Sparse[int64, 0]"
        )
        result = df.sparse.to_coo()
        expected = scipy.sparse.coo_matrix(np.asarray(df))
        assert (result != expected).nnz == 0

    def test_to_dense(self):
        df = pd.DataFrame(
            {
                "A": SparseArray([1, 0], dtype=SparseDtype("int64", 0)),
                "B": SparseArray([1, 0], dtype=SparseDtype("int64", 1)),
                "C": SparseArray([1.0, 0.0], dtype=SparseDtype("float64", 0.0)),
            },
            index=["b", "a"],
        )
        result = df.sparse.to_dense()
        expected = pd.DataFrame(
            {"A": [1, 0], "B": [1, 0], "C": [1.0, 0.0]}, index=["b", "a"]
        )
        tm.assert_frame_equal(result, expected)

    def test_density(self):
        df = pd.DataFrame(
            {
                "A": SparseArray([1, 0, 2, 1], fill_value=0),
                "B": SparseArray([0, 1, 1, 1], fill_value=0),
            }
        )
        res = df.sparse.density
        expected = 0.75
        assert res == expected

    @pytest.mark.parametrize("dtype", ["int64", "float64"])
    @pytest.mark.parametrize("dense_index", [True, False])
    @td.skip_if_no_scipy
    def test_series_from_coo(self, dtype, dense_index):
        import scipy.sparse

        A = scipy.sparse.eye(3, format="coo", dtype=dtype)
        result = pd.Series.sparse.from_coo(A, dense_index=dense_index)
        index = pd.MultiIndex.from_tuples([(0, 0), (1, 1), (2, 2)])
        expected = pd.Series(SparseArray(np.array([1, 1, 1], dtype=dtype)), index=index)
        if dense_index:
            expected = expected.reindex(pd.MultiIndex.from_product(index.levels))

        tm.assert_series_equal(result, expected)

    @td.skip_if_no_scipy
    def test_series_from_coo_incorrect_format_raises(self):
        # gh-26554
        import scipy.sparse

        m = scipy.sparse.csr_matrix(np.array([[0, 1], [0, 0]]))
        with pytest.raises(
            TypeError, match="Expected coo_matrix. Got csr_matrix instead."
        ):
            pd.Series.sparse.from_coo(m)

    def test_with_column_named_sparse(self):
        # https://github.com/pandas-dev/pandas/issues/30758
        df = pd.DataFrame({"sparse": pd.arrays.SparseArray([1, 2])})
        assert isinstance(df.sparse, pd.core.arrays.sparse.accessor.SparseFrameAccessor)
