import numpy as np
from numpy.testing import assert_array_almost_equal, assert_
from scipy.sparse import csr_matrix

import pytest


def _check_csr_rowslice(i, sl, X, Xcsr):
    np_slice = X[i, sl]
    csr_slice = Xcsr[i, sl]
    assert_array_almost_equal(np_slice, csr_slice.toarray()[0])
    assert_(type(csr_slice) is csr_matrix)


def test_csr_rowslice():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    slices = [slice(None, None, None),
              slice(None, None, -1),
              slice(1, -2, 2),
              slice(-2, 1, -2)]

    for i in range(N):
        for sl in slices:
            _check_csr_rowslice(i, sl, X, Xcsr)


def test_csr_getrow():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    for i in range(N):
        arr_row = X[i:i + 1, :]
        csr_row = Xcsr.getrow(i)

        assert_array_almost_equal(arr_row, csr_row.toarray())
        assert_(type(csr_row) is csr_matrix)


def test_csr_getcol():
    N = 10
    np.random.seed(0)
    X = np.random.random((N, N))
    X[X > 0.7] = 0
    Xcsr = csr_matrix(X)

    for i in range(N):
        arr_col = X[:, i:i + 1]
        csr_col = Xcsr.getcol(i)

        assert_array_almost_equal(arr_col, csr_col.toarray())
        assert_(type(csr_col) is csr_matrix)

@pytest.mark.parametrize("matrix_input, axis, expected_shape",
    [(csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      0, (0, 4)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      1, (3, 0)),
     (csr_matrix([[1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 2, 3, 0]]),
      'both', (0, 0)),
     (csr_matrix([[0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 3, 0]]),
      0, (0, 5))])
def test_csr_empty_slices(matrix_input, axis, expected_shape):
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


def test_csr_bool_indexing():
    data = csr_matrix([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    list_indices1 = [False, True, False]
    array_indices1 = np.array(list_indices1)
    list_indices2 = [[False, True, False], [False, True, False], [False, True, False]]
    array_indices2 = np.array(list_indices2)
    list_indices3 = ([False, True, False], [False, True, False])
    array_indices3 = (np.array(list_indices3[0]), np.array(list_indices3[1]))
    slice_list1 = data[list_indices1].toarray()
    slice_array1 = data[array_indices1].toarray()
    slice_list2 = data[list_indices2]
    slice_array2 = data[array_indices2]
    slice_list3 = data[list_indices3]
    slice_array3 = data[array_indices3]
    assert (slice_list1 == slice_array1).all()
    assert (slice_list2 == slice_array2).all()
    assert (slice_list3 == slice_array3).all()

