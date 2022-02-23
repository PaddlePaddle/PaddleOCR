import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

import pytest


def test_csc_getrow():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsc = csc_matrix(X)

    for i in range(N):
        arr_row = X[i:i + 1, :]
        csc_row = Xcsc.getrow(i)

        assert_array_almost_equal(arr_row, csc_row.toarray())
        assert_(type(csc_row) is csr_matrix)


def test_csc_getcol():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsc = csc_matrix(X)

    for i in range(N):
        arr_col = X[:, i:i + 1]
        csc_col = Xcsc.getcol(i)

        assert_array_almost_equal(arr_col, csc_col.toarray())
        assert_(type(csc_col) is csc_matrix)

@pytest.mark.parametrize("matrix_input, axis, expected_shape",
    [(csc_matrix([[1, 0],
                [0, 0],
                [0, 2]]),
      0, (0, 2)),
     (csc_matrix([[1, 0],
                [0, 0],
                [0, 2]]),
      1, (3, 0)),
     (csc_matrix([[1, 0],
                [0, 0],
                [0, 2]]),
      'both', (0, 0)),
     (csc_matrix([[0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 3, 0, 1]]),
      0, (0, 6))])
def test_csc_empty_slices(matrix_input, axis, expected_shape):
    # see gh-11127 for related discussion
    slice_1 = matrix_input.A.shape[0] - 1
    slice_2 = slice_1
    slice_3 = slice_2 - 1

    if axis == 0:
        actual_shape_1 = matrix_input[slice_1:slice_2, :].A.shape
        actual_shape_2 = matrix_input[slice_1:slice_3, :].A.shape
    elif axis == 1:
        actual_shape_1 = matrix_input[:, slice_1:slice_2].A.shape
        actual_shape_2 = matrix_input[:, slice_1:slice_3].A.shape
    elif axis == 'both':
        actual_shape_1 = matrix_input[slice_1:slice_2, slice_1:slice_2].A.shape
        actual_shape_2 = matrix_input[slice_1:slice_3, slice_1:slice_3].A.shape

    assert actual_shape_1 == expected_shape
    assert actual_shape_1 == actual_shape_2


@pytest.mark.parametrize('ax', (-2, -1, 0, 1, None))
def test_argmax_overflow(ax):
    # See gh-13646: Windows integer overflow for large sparse matrices.
    dim = (100000, 100000)
    A = lil_matrix(dim)
    A[-2, -2] = 42
    A[-3, -3] = 0.1234
    A = csc_matrix(A)
    idx = A.argmax(axis=ax)

    if ax is None:
        # idx is a single flattened index
        # that we need to convert to a 2d index pair;
        # can't do this with np.unravel_index because
        # the dimensions are too large
        ii = idx % dim[0]
        jj = idx // dim[0]
    else:
        # idx is an array of size of A.shape[ax];
        # check the max index to make sure no overflows
        # we encountered
        assert np.count_nonzero(idx) == A.nnz
        ii, jj = np.max(idx), np.argmax(idx)

    assert A[ii, jj] == A[-2, -2]
