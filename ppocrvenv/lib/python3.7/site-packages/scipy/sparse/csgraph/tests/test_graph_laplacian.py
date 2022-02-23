# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
# License: BSD
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
from pytest import raises as assert_raises
from scipy import sparse

from scipy.sparse import csgraph


def _explicit_laplacian(x, normed=False):
    if sparse.issparse(x):
        x = x.todense()
    x = np.asarray(x)
    y = -1.0 * x
    for j in range(y.shape[0]):
        y[j,j] = x[j,j+1:].sum() + x[j,:j].sum()
    if normed:
        d = np.diag(y).copy()
        d[d == 0] = 1.0
        y /= d[:,None]**.5
        y /= d[None,:]**.5
    return y


def _check_symmetric_graph_laplacian(mat, normed):
    if not hasattr(mat, 'shape'):
        mat = eval(mat, dict(np=np, sparse=sparse))

    if sparse.issparse(mat):
        sp_mat = mat
        mat = sp_mat.todense()
    else:
        sp_mat = sparse.csr_matrix(mat)

    laplacian = csgraph.laplacian(mat, normed=normed)
    n_nodes = mat.shape[0]
    if not normed:
        assert_array_almost_equal(laplacian.sum(axis=0), np.zeros(n_nodes))
    assert_array_almost_equal(laplacian.T, laplacian)
    assert_array_almost_equal(laplacian,
            csgraph.laplacian(sp_mat, normed=normed).todense())

    assert_array_almost_equal(laplacian,
            _explicit_laplacian(mat, normed=normed))


def test_laplacian_value_error():
    for t in int, float, complex:
        for m in ([1, 1],
                  [[[1]]],
                  [[1, 2, 3], [4, 5, 6]],
                  [[1, 2], [3, 4], [5, 5]]):
            A = np.array(m, dtype=t)
            assert_raises(ValueError, csgraph.laplacian, A)


def test_symmetric_graph_laplacian():
    symmetric_mats = ('np.arange(10) * np.arange(10)[:, np.newaxis]',
            'np.ones((7, 7))',
            'np.eye(19)',
            'sparse.diags([1, 1], [-1, 1], shape=(4,4))',
            'sparse.diags([1, 1], [-1, 1], shape=(4,4)).todense()',
            'np.asarray(sparse.diags([1, 1], [-1, 1], shape=(4,4)).todense())',
            'np.vander(np.arange(4)) + np.vander(np.arange(4)).T')
    for mat_str in symmetric_mats:
        for normed in True, False:
            _check_symmetric_graph_laplacian(mat_str, normed)


def _assert_allclose_sparse(a, b, **kwargs):
    # helper function that can deal with sparse matrices
    if sparse.issparse(a):
        a = a.toarray()
    if sparse.issparse(b):
        b = a.toarray()
    assert_allclose(a, b, **kwargs)


def _check_laplacian(A, desired_L, desired_d, normed, use_out_degree):
    for arr_type in np.array, sparse.csr_matrix, sparse.coo_matrix:
        for t in int, float, complex:
            adj = arr_type(A, dtype=t)
            L = csgraph.laplacian(adj, normed=normed, return_diag=False,
                                  use_out_degree=use_out_degree)
            _assert_allclose_sparse(L, desired_L, atol=1e-12)
            L, d = csgraph.laplacian(adj, normed=normed, return_diag=True,
                                  use_out_degree=use_out_degree)
            _assert_allclose_sparse(L, desired_L, atol=1e-12)
            _assert_allclose_sparse(d, desired_d, atol=1e-12)


def test_asymmetric_laplacian():
    # adjacency matrix
    A = [[0, 1, 0],
         [4, 2, 0],
         [0, 0, 0]]

    # Laplacian matrix using out-degree
    L = [[1, -1, 0],
         [-4, 4, 0],
         [0, 0, 0]]
    d = [1, 4, 0]
    _check_laplacian(A, L, d, normed=False, use_out_degree=True)

    # normalized Laplacian matrix using out-degree
    L = [[1, -0.5, 0],
         [-2, 1, 0],
         [0, 0, 0]]
    d = [1, 2, 1]
    _check_laplacian(A, L, d, normed=True, use_out_degree=True)

    # Laplacian matrix using in-degree
    L = [[4, -1, 0],
         [-4, 1, 0],
         [0, 0, 0]]
    d = [4, 1, 0]
    _check_laplacian(A, L, d, normed=False, use_out_degree=False)

    # normalized Laplacian matrix using in-degree
    L = [[1, -0.5, 0],
         [-2, 1, 0],
         [0, 0, 0]]
    d = [2, 1, 1]
    _check_laplacian(A, L, d, normed=True, use_out_degree=False)


def test_sparse_formats():
    for fmt in ('csr', 'csc', 'coo', 'lil', 'dok', 'dia', 'bsr'):
        mat = sparse.diags([1, 1], [-1, 1], shape=(4,4), format=fmt)
        for normed in True, False:
            _check_symmetric_graph_laplacian(mat, normed)

