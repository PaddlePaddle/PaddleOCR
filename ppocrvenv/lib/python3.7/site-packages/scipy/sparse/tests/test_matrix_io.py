import os
import numpy as np
import tempfile

from pytest import raises as assert_raises
from numpy.testing import assert_equal, assert_

from scipy.sparse import (csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
                          coo_matrix, save_npz, load_npz, dok_matrix)


DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _save_and_load(matrix):
    fd, tmpfile = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        save_npz(tmpfile, matrix)
        loaded_matrix = load_npz(tmpfile)
    finally:
        os.remove(tmpfile)
    return loaded_matrix

def _check_save_and_load(dense_matrix):
    for matrix_class in [csc_matrix, csr_matrix, bsr_matrix, dia_matrix, coo_matrix]:
        matrix = matrix_class(dense_matrix)
        loaded_matrix = _save_and_load(matrix)
        assert_(type(loaded_matrix) is matrix_class)
        assert_(loaded_matrix.shape == dense_matrix.shape)
        assert_(loaded_matrix.dtype == dense_matrix.dtype)
        assert_equal(loaded_matrix.toarray(), dense_matrix)

def test_save_and_load_random():
    N = 10
    np.random.seed(0)
    dense_matrix = np.random.random((N, N))
    dense_matrix[dense_matrix > 0.7] = 0
    _check_save_and_load(dense_matrix)

def test_save_and_load_empty():
    dense_matrix = np.zeros((4,6))
    _check_save_and_load(dense_matrix)

def test_save_and_load_one_entry():
    dense_matrix = np.zeros((4,6))
    dense_matrix[1,2] = 1
    _check_save_and_load(dense_matrix)


def test_malicious_load():
    class Executor:
        def __reduce__(self):
            return (assert_, (False, 'unexpected code execution'))

    fd, tmpfile = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        np.savez(tmpfile, format=Executor())

        # Should raise a ValueError, not execute code
        assert_raises(ValueError, load_npz, tmpfile)
    finally:
        os.remove(tmpfile)


def test_py23_compatibility():
    # Try loading files saved on Python 2 and Python 3.  They are not
    # the same, since files saved with SciPy versions < 1.0.0 may
    # contain unicode.

    a = load_npz(os.path.join(DATA_DIR, 'csc_py2.npz'))
    b = load_npz(os.path.join(DATA_DIR, 'csc_py3.npz'))
    c = csc_matrix([[0]])

    assert_equal(a.toarray(), c.toarray())
    assert_equal(b.toarray(), c.toarray())

def test_implemented_error():
    # Attempts to save an unsupported type and checks that an
    # NotImplementedError is raised.

    x = dok_matrix((2,3))
    x[0,1] = 1

    assert_raises(NotImplementedError, save_npz, 'x.npz', x)
