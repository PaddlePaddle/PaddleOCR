import numpy as np
import os
from gzip import GzipFile
from scipy.stats import scoreatpercentile


def make_hard(cost, lo, hi):
    hard = cost.copy()
    for row in range(hard.shape[0]):
        hard[row, :] += np.random.randint(lo, hi)
    for col in range(hard.shape[1]):
        hard[:, col] += np.random.randint(lo, hi)
    return hard


def get_dense_8x8_int():
    cost = np.array([[1000, 2, 11, 10, 8, 7, 6, 5],
                     [6, 1000, 1, 8, 8, 4, 6, 7],
                     [5, 12, 1000, 11, 8, 12, 3, 11],
                     [11, 9, 10, 1000, 1, 9, 8, 10],
                     [11, 11, 9, 4, 1000, 2, 10, 9],
                     [12, 8, 5, 2, 11, 1000, 11, 9],
                     [10, 11, 12, 10, 9, 12, 1000, 3],
                     [10, 10, 10, 10, 6, 3, 1, 1000]])
    opt = 17.
    return cost, opt


def get_dense_int(sz, rng, hard=True, seed=1299821):
    np.random.seed(seed)
    cost = np.random.randint(1, rng+1, size=(sz, sz))
    if hard is True:
        cost = make_hard(cost, 0, rng)
    return cost


def get_sparse_int(sz, rng, sparsity, hard=True, seed=1299821):
    np.random.seed(seed)
    cost = np.random.randint(1, rng+1, size=(sz, sz))
    if hard is True:
        cost = make_hard(cost, 0, rng)
    mask = np.random.rand(sz, sz)
    thresh = scoreatpercentile(
            mask.flat, max(0, (sparsity - sz/float(sz*sz)) * 100.))
    mask = mask < thresh
    # Make sure there exists a solution.
    row = np.random.permutation(sz)
    col = np.random.permutation(sz)
    mask[row, col] = True
    return cost, mask


def get_nnz_int(sz, nnz, rng=100, seed=1299821):
    np.random.seed(seed)
    cc = np.random.randint(1, rng+1, size=(sz*nnz,))
    ii = np.empty((sz + 1,), dtype=np.int32)
    ii[0] = 0
    ii[1:] = nnz
    ii = np.cumsum(ii)
    kk = np.empty((sz, nnz), dtype=np.int32)
    # Make sure there exists a solution.
    kk[:, 0] = np.random.permutation(sz)
    for row in range(sz):
        p = np.random.permutation(sz)[:nnz]
        if kk[row, 0] in p:
            kk[row, :] = p
        else:
            kk[row, 1:] = p[:-1]
    # Column indices must be sorted within each row.
    kk = np.sort(kk, axis=1).flatten()
    assert len(cc) == sz * nnz
    assert len(kk)
    assert np.all(kk >= 0)
    assert np.all(kk < sz)
    return cc, ii, kk


def get_dense_100x100_int():
    cost = get_dense_int(100, 100, hard=False, seed=1299821)
    opt = 198.
    return cost, opt


def get_dense_100x100_int_hard():
    cost = get_dense_int(100, 100, hard=True, seed=1299821)
    opt = 11399.
    return cost, opt


def get_sparse_100x100_int():
    cost, mask = get_sparse_int(100, 100, 0.04, seed=1299821)
    opt = 11406
    return cost, np.logical_not(mask), opt


def get_dense_1kx1k_int():
    cost = get_dense_int(1000, 100, hard=False, seed=1299821)
    opt = 1000.
    return cost, opt


def get_dense_1kx1k_int_hard():
    cost = get_dense_int(1000, 100, hard=True, seed=1299821)
    opt = 101078.0
    return cost, opt


def get_sparse_1kx1k_int():
    cost, mask = get_sparse_int(1000, 100, 0.01, seed=1299821)
    opt = 101078
    return cost, np.logical_not(mask), opt


def get_dense_4kx4k_int():
    cost = get_dense_int(4000, 100, hard=False, seed=1299821)
    opt = 1000.
    return cost, opt


def get_sparse_4kx4k_int():
    cost, mask = get_sparse_int(4000, 100, 0.004, seed=1299821)
    opt = 402541
    return cost, np.logical_not(mask), opt


# Thanks to Michael Lewis for providing this cost matrix.
def get_dense_eps():
    from pytest import approx
    datadir = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(datadir, 'cost_eps.csv.gz')
    cost = np.genfromtxt(GzipFile(filename), delimiter=",")
    opt = approx(224.8899507294651, 0.0000000000001)
    return cost, opt


def sparse_from_dense(cost):
    cc = cost.flatten()
    n_rows = cost.shape[0]
    n_columns = cost.shape[1]
    ii = np.empty((n_rows+1,), dtype=int)
    ii[0] = 0
    ii[1:] = n_columns
    ii = np.cumsum(ii)
    kk = np.tile(np.arange(n_columns, dtype=int), n_rows)
    return n_rows, cc, ii, kk


def sparse_from_masked(cost, mask=None):
    if mask is None:
        mask = np.logical_not(np.isinf(cost))
    cc = cost[mask].flatten()
    n_rows = cost.shape[0]
    n_columns = cost.shape[1]
    ii = np.empty((n_rows+1,), dtype=int)
    ii[0] = 0
    ii[1:] = mask.sum(axis=1)
    ii = np.cumsum(ii)
    kk = np.tile(np.arange(n_columns, dtype=int), cost.shape[0])
    kk = kk[mask.flatten()]
    return n_rows, cc, ii, kk


def sparse_from_dense_CS(cost):
    i = np.tile(
            np.atleast_2d(np.arange(cost.shape[0])).T,
            cost.shape[1]).flatten()
    j = np.tile(np.arange(cost.shape[1]), cost.shape[0])
    cc = cost.flatten()
    return i, j, cc


def sparse_from_masked_CS(cost, mask):
    i = np.tile(
            np.atleast_2d(np.arange(cost.shape[0])).T,
            cost.shape[1])[mask]
    j = np.tile(np.arange(cost.shape[1]), cost.shape[0])[mask.flat]
    cc = cost[mask].flatten()
    return i, j, cc


def get_cost_CS(cost, x):
    return cost[np.arange(cost.shape[0]), x].sum()


def get_platform_maxint():
    import struct
    return 2 ** (struct.Struct('i').size * 8 - 1) - 1
