# Author: Tom Dupre la Tour
#         Joan Massich <mailsik@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_equal
from sklearn.utils._seq_dataset import (
    ArrayDataset32,
    ArrayDataset64,
    CSRDataset32,
    CSRDataset64,
)

from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_allclose

iris = load_iris()
X64 = iris.data.astype(np.float64)
y64 = iris.target.astype(np.float64)
X_csr64 = sp.csr_matrix(X64)
sample_weight64 = np.arange(y64.size, dtype=np.float64)

X32 = iris.data.astype(np.float32)
y32 = iris.target.astype(np.float32)
X_csr32 = sp.csr_matrix(X32)
sample_weight32 = np.arange(y32.size, dtype=np.float32)


def assert_csr_equal_values(current, expected):
    current.eliminate_zeros()
    expected.eliminate_zeros()
    expected = expected.astype(current.dtype)
    assert current.shape[0] == expected.shape[0]
    assert current.shape[1] == expected.shape[1]
    assert_array_equal(current.data, expected.data)
    assert_array_equal(current.indices, expected.indices)
    assert_array_equal(current.indptr, expected.indptr)


def make_dense_dataset_32():
    return ArrayDataset32(X32, y32, sample_weight32, seed=42)


def make_dense_dataset_64():
    return ArrayDataset64(X64, y64, sample_weight64, seed=42)


def make_sparse_dataset_32():
    return CSRDataset32(
        X_csr32.data, X_csr32.indptr, X_csr32.indices, y32, sample_weight32, seed=42
    )


def make_sparse_dataset_64():
    return CSRDataset64(
        X_csr64.data, X_csr64.indptr, X_csr64.indices, y64, sample_weight64, seed=42
    )


@pytest.mark.parametrize(
    "dataset_constructor",
    [
        make_dense_dataset_32,
        make_dense_dataset_64,
        make_sparse_dataset_32,
        make_sparse_dataset_64,
    ],
)
def test_seq_dataset_basic_iteration(dataset_constructor):
    NUMBER_OF_RUNS = 5
    dataset = dataset_constructor()
    for _ in range(NUMBER_OF_RUNS):
        # next sample
        xi_, yi, swi, idx = dataset._next_py()
        xi = sp.csr_matrix((xi_), shape=(1, X64.shape[1]))

        assert_csr_equal_values(xi, X_csr64[idx])
        assert yi == y64[idx]
        assert swi == sample_weight64[idx]

        # random sample
        xi_, yi, swi, idx = dataset._random_py()
        xi = sp.csr_matrix((xi_), shape=(1, X64.shape[1]))

        assert_csr_equal_values(xi, X_csr64[idx])
        assert yi == y64[idx]
        assert swi == sample_weight64[idx]


@pytest.mark.parametrize(
    "make_dense_dataset,make_sparse_dataset",
    [
        (make_dense_dataset_32, make_sparse_dataset_32),
        (make_dense_dataset_64, make_sparse_dataset_64),
    ],
)
def test_seq_dataset_shuffle(make_dense_dataset, make_sparse_dataset):
    dense_dataset, sparse_dataset = make_dense_dataset(), make_sparse_dataset()
    # not shuffled
    for i in range(5):
        _, _, _, idx1 = dense_dataset._next_py()
        _, _, _, idx2 = sparse_dataset._next_py()
        assert idx1 == i
        assert idx2 == i

    for i in [132, 50, 9, 18, 58]:
        _, _, _, idx1 = dense_dataset._random_py()
        _, _, _, idx2 = sparse_dataset._random_py()
        assert idx1 == i
        assert idx2 == i

    seed = 77
    dense_dataset._shuffle_py(seed)
    sparse_dataset._shuffle_py(seed)

    idx_next = [63, 91, 148, 87, 29]
    idx_shuffle = [137, 125, 56, 121, 127]
    for i, j in zip(idx_next, idx_shuffle):
        _, _, _, idx1 = dense_dataset._next_py()
        _, _, _, idx2 = sparse_dataset._next_py()
        assert idx1 == i
        assert idx2 == i

        _, _, _, idx1 = dense_dataset._random_py()
        _, _, _, idx2 = sparse_dataset._random_py()
        assert idx1 == j
        assert idx2 == j


@pytest.mark.parametrize(
    "make_dataset_32,make_dataset_64",
    [
        (make_dense_dataset_32, make_dense_dataset_64),
        (make_sparse_dataset_32, make_sparse_dataset_64),
    ],
)
def test_fused_types_consistency(make_dataset_32, make_dataset_64):
    dataset_32, dataset_64 = make_dataset_32(), make_dataset_64()
    NUMBER_OF_RUNS = 5
    for _ in range(NUMBER_OF_RUNS):
        # next sample
        (xi_data32, _, _), yi32, _, _ = dataset_32._next_py()
        (xi_data64, _, _), yi64, _, _ = dataset_64._next_py()

        assert xi_data32.dtype == np.float32
        assert xi_data64.dtype == np.float64

        assert_allclose(xi_data64, xi_data32, rtol=1e-5)
        assert_allclose(yi64, yi32, rtol=1e-5)


def test_buffer_dtype_mismatch_error():
    with pytest.raises(ValueError, match="Buffer dtype mismatch"):
        ArrayDataset64(X32, y32, sample_weight32, seed=42),

    with pytest.raises(ValueError, match="Buffer dtype mismatch"):
        ArrayDataset32(X64, y64, sample_weight64, seed=42),

    with pytest.raises(ValueError, match="Buffer dtype mismatch"):
        CSRDataset64(
            X_csr32.data, X_csr32.indptr, X_csr32.indices, y32, sample_weight32, seed=42
        ),

    with pytest.raises(ValueError, match="Buffer dtype mismatch"):
        CSRDataset32(
            X_csr64.data, X_csr64.indptr, X_csr64.indices, y64, sample_weight64, seed=42
        ),
