""" Testing miobase module
"""

import numpy as np

from numpy.testing import assert_equal
from pytest import raises as assert_raises

from scipy.io.matlab.miobase import matdims


def test_matdims():
    # Test matdims dimension finder
    assert_equal(matdims(np.array(1)), (1, 1))  # NumPy scalar
    assert_equal(matdims(np.array([1])), (1, 1))  # 1-D array, 1 element
    assert_equal(matdims(np.array([1,2])), (2, 1))  # 1-D array, 2 elements
    assert_equal(matdims(np.array([[2],[3]])), (2, 1))  # 2-D array, column vector
    assert_equal(matdims(np.array([[2,3]])), (1, 2))  # 2-D array, row vector
    # 3d array, rowish vector
    assert_equal(matdims(np.array([[[2,3]]])), (1, 1, 2))
    assert_equal(matdims(np.array([])), (0, 0))  # empty 1-D array
    assert_equal(matdims(np.array([[]])), (1, 0))  # empty 2-D array
    assert_equal(matdims(np.array([[[]]])), (1, 1, 0))  # empty 3-D array
    assert_equal(matdims(np.empty((1, 0, 1))), (1, 0, 1))  # empty 3-D array
    # Optional argument flips 1-D shape behavior.
    assert_equal(matdims(np.array([1,2]), 'row'), (1, 2))  # 1-D array, 2 elements
    # The argument has to make sense though
    assert_raises(ValueError, matdims, np.array([1,2]), 'bizarre')
    # Check empty sparse matrices get their own shape
    from scipy.sparse import csr_matrix, csc_matrix
    assert_equal(matdims(csr_matrix(np.zeros((3, 3)))), (3, 3))
    assert_equal(matdims(csc_matrix(np.zeros((2, 2)))), (2, 2))
