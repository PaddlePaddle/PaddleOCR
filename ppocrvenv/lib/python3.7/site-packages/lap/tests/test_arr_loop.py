from pytest import approx

import numpy as np
from lap import lapmod, lapjv


def prepare_sparse_cost(shape, cc, ii, jj, cost_limit):
    '''
    Transform the given sparse matrix extending it to a square sparse matrix.

    Parameters
    ==========
    shape: tuple
       - cost matrix shape
    (cc, ii, jj): tuple of floats, ints, ints)
        - cost matrix in COO format, see [1]
    cost_limit: float

    Returns
    =======
    cc, ii, kk
      - extended square cost matrix in CSR format

    1. https://en.wikipedia.org/wiki/Sparse_matrix
    '''
    assert cost_limit < np.inf
    n, m = shape
    cc_ = np.r_[cc, [cost_limit] * n,
                [cost_limit] * m, [0] * len(cc)]
    ii_ = np.r_[ii, np.arange(0, n, dtype=np.uint32),
                np.arange(n, n + m, dtype=np.uint32), n + jj]
    jj_ = np.r_[jj, np.arange(m, n + m, dtype=np.uint32),
                np.arange(0, m, dtype=np.uint32), m + ii]
    order = np.lexsort((jj_, ii_))
    cc_ = cc_[order]
    kk_ = jj_[order]
    ii_ = ii_.astype(np.intp)
    ii_ = np.bincount(ii_, minlength=shape[0]-1)
    ii_ = np.r_[[0], np.cumsum(ii_)]
    ii_ = ii_.astype(np.uint32)
    assert ii_[-1] == 2 * len(cc) + n + m
    return cc_, ii_, kk_


def test_lapjv_arr_loop():
    shape = (7, 3)
    cc = np.array([
        2.593883482138951146e-01, 3.080381437461217620e-01,
        1.976243020727339317e-01, 2.462740976049606068e-01,
        4.203993396282833528e-01, 4.286184525458427985e-01,
        1.706431415909629434e-01, 2.192929371231896185e-01,
        2.117769622802734286e-01, 2.604267578125001315e-01])
    ii = np.array([0, 0, 1, 1, 2, 2, 5, 5, 6, 6])
    jj = np.array([0, 1, 0, 1, 1, 2, 0, 1, 0, 1])
    cost = np.empty(shape)
    cost[:] = 1000.
    cost[ii, jj] = cc
    opt, ind1, ind0 = lapjv(cost, extend_cost=True, return_cost=True)
    assert opt == approx(0.8455356917416, 1e-10)
    assert np.all(ind0 == [5, 1, 2]) or np.all(ind0 == [1, 5, 2])


def test_lapmod_arr_loop():
    shape = (7, 3)
    cc = np.array([
        2.593883482138951146e-01, 3.080381437461217620e-01,
        1.976243020727339317e-01, 2.462740976049606068e-01,
        4.203993396282833528e-01, 4.286184525458427985e-01,
        1.706431415909629434e-01, 2.192929371231896185e-01,
        2.117769622802734286e-01, 2.604267578125001315e-01])
    ii = np.array([0, 0, 1, 1, 2, 2, 5, 5, 6, 6])
    jj = np.array([0, 1, 0, 1, 1, 2, 0, 1, 0, 1])
    cost_limit = 1e3
    cc, ii, kk = prepare_sparse_cost(shape, cc, ii, jj, cost_limit)
    opt, ind1, ind0 = lapmod(len(ii)-1, cc, ii, kk, return_cost=True)
    ind1[ind1 >= shape[1]] = -1
    ind0[ind0 >= shape[0]] = -1
    ind1 = ind1[:shape[0]]
    ind0 = ind0[:shape[1]]
    assert opt == approx(4000.8455356917416, 1e-10)
    assert np.all(ind0 == [5, 1, 2]) or np.all(ind0 == [1, 5, 2])
