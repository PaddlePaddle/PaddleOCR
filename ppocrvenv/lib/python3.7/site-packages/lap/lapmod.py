import numpy as np
from bisect import bisect_left

# import logging

from ._lapjv import _lapmod, FP_DYNAMIC_ as FP_DYNAMIC, LARGE_ as LARGE


def _pycrrt(n, cc, ii, kk, free_rows, x, y, v):
    # log = logging.getLogger('do_column_reduction_and_reduction_transfer')
    x[:] = -1
    y[:] = -1
    v[:] = LARGE
    for i in range(n):
        ks = slice(ii[i], ii[i+1])
        js = kk[ks]
        ccs = cc[ks]
        mask = ccs < v[js]
        js = js[mask]
        v[js] = ccs[mask]
        y[js] = i
    # log.debug('v = %s', v)
    # for j in range(cost.shape[1]):
    unique = np.empty((n,), dtype=bool)
    unique[:] = True
    for j in range(n-1, -1, -1):
        i = y[j]
        # If row is not taken yet, initialize it with the minimum stored in y.
        if x[i] < 0:
            x[i] = j
        else:
            unique[i] = False
            y[j] = -1
        # log.debug('bw %s %s %s %s', i, j, x, y)
    # log.debug('unique %s', unique)
    n_free_rows = 0
    for i in range(n):
        # Store unassigned row i.
        if x[i] < 0:
            free_rows[n_free_rows] = i
            n_free_rows += 1
        elif unique[i] and ii[i+1] - ii[i] > 1:
            # >1 check prevents choking on rows with a single entry
            # Transfer from an assigned row.
            j = x[i]
            # Find the current 2nd minimum of the reduced column costs:
            # (cost[i,j] - v[j]) for some j.
            ks = slice(ii[i], ii[i+1])
            js = kk[ks]
            minv = np.min(cc[ks][js != j] - v[js][js != j])
            # log.debug("v[%d] = %f - %f", j, v[j], minv)
            v[j] -= minv
    # log.debug('free: %s', free_rows[:n_free_rows])
    # log.debug('%s %s', x, v)
    return n_free_rows


def find_minima(indices, values):
    if len(indices) > 0:
        j1 = indices[0]
        v1 = values[0]
    else:
        j1 = 0
        v1 = LARGE
    j2 = -1
    v2 = LARGE
    # log = logging.getLogger('find_minima')
    # log.debug(sorted(zip(values, indices))[:2])
    for j, h in zip(indices[1:], values[1:]):
        # log.debug('%d = %f %d = %f', j1, v1, j2, v2)
        if h < v2:
            if h >= v1:
                v2 = h
                j2 = j
            else:
                v2 = v1
                v1 = h
                j2 = j1
                j1 = j
    # log.debug('%d = %f %d = %f', j1, v1, j2, v2)
    return j1, v1, j2, v2


def _pyarr(n, cc, ii, kk, n_free_rows, free_rows, x, y, v):
    # log = logging.getLogger('do_augmenting_row_reduction')
    # log.debug('%s %s %s', x, y, v)
    current = 0
    # log.debug('free: %s', free_rows[:n_free_rows])
    new_free_rows = 0
    while current < n_free_rows:
        free_i = free_rows[current]
        # log.debug('current = %d', current)
        current += 1
        ks = slice(ii[free_i], ii[free_i+1])
        js = kk[ks]
        j1, v1, j2, v2 = find_minima(js, cc[ks] - v[js])
        i0 = y[j1]
        v1_new = v[j1] - (v2 - v1)
        v1_lowers = v1_new < v[j1]
        # log.debug(
        #    '%d %d 1=%s,%f 2=%s,%f %f %s',
        #    free_i, i0, j1, v1, j2, v2, v1_new, v1_lowers)
        if v1_lowers:
            v[j1] = v1_new
        elif i0 >= 0 and j2 != -1:  # i0 is assigned, try j2
            j1 = j2
            i0 = y[j2]
        x[free_i] = j1
        y[j1] = free_i
        if i0 >= 0:
            if v1_lowers:
                current -= 1
                # log.debug('continue augmenting path from current %s %s %s')
                free_rows[current] = i0
            else:
                # log.debug('stop the augmenting path and keep for later')
                free_rows[new_free_rows] = i0
                new_free_rows += 1
        # log.debug('free: %s', free_rows[:new_free_rows])
    return new_free_rows


def binary_search(data, key):
    # log = logging.getLogger('binary_search')
    i = bisect_left(data, key)
    # log.debug('Found data[%d]=%d for %d', i, data[i], key)
    if i < len(data) and data[i] == key:
        return i
    else:
        return None


def _find(hi, d, cols, y):
    lo, hi = hi, hi + 1
    minv = d[cols[lo]]
    # XXX: anytime this happens to be NaN, i'm screwed...
    # assert not np.isnan(minv)
    for k in range(hi, len(cols)):
        j = cols[k]
        if d[j] <= minv:
            # New minimum found, trash the new SCAN columns found so far.
            if d[j] < minv:
                hi = lo
                minv = d[j]
            cols[k], cols[hi] = cols[hi], j
            hi += 1
    return minv, hi, cols


def _scan(n, cc, ii, kk, minv, lo, hi, d, cols, pred, y, v):
    # log = logging.getLogger('_scan')
    # Scan all TODO columns.
    while lo != hi:
        j = cols[lo]
        lo += 1
        i = y[j]
        # log.debug('?%d kk[%d:%d]=%s', j, ii[i], ii[i+1], kk[ii[i]:ii[i+1]])
        kj = binary_search(kk[ii[i]:ii[i+1]], j)
        if kj is None:
            continue
        kj = ii[i] + kj
        h = cc[kj] - v[j] - minv
        # log.debug('i=%d j=%d kj=%s h=%f', i, j, kj, h)
        for k in range(hi, n):
            j = cols[k]
            kj = binary_search(kk[ii[i]:ii[i+1]], j)
            if kj is None:
                continue
            kj = ii[i] + kj
            cred_ij = cc[kj] - v[j] - h
            if cred_ij < d[j]:
                d[j] = cred_ij
                pred[j] = i
                if cred_ij == minv:
                    if y[j] < 0:
                        return j, None, None, d, cols, pred
                    cols[k] = cols[hi]
                    cols[hi] = j
                    hi += 1
    return -1, lo, hi, d, cols, pred


def find_path(n, cc, ii, kk, start_i, y, v):
    # log = logging.getLogger('find_path')
    cols = np.arange(n, dtype=int)
    pred = np.empty((n,), dtype=int)
    pred[:] = start_i
    d = np.empty((n,), dtype=float)
    d[:] = LARGE
    ks = slice(ii[start_i], ii[start_i+1])
    js = kk[ks]
    d[js] = cc[ks] - v[js]
    # log.debug('d = %s', d)
    minv = LARGE
    lo, hi = 0, 0
    n_ready = 0
    final_j = -1
    while final_j == -1:
        # No SCAN columns, find new ones.
        if lo == hi:
            # log.debug('%d..%d -> find', lo, hi)
            # log.debug('cols = %s', cols)
            n_ready = lo
            minv, hi, cols = _find(hi, d, cols, y)
            # log.debug('%d..%d -> check', lo, hi)
            # log.debug('cols = %s', cols)
            # log.debug('y = %s', y)
            for h in range(lo, hi):
                # If any of the new SCAN columns is unassigned, use it.
                if y[cols[h]] < 0:
                    final_j = cols[h]
        if final_j == -1:
            # log.debug('%d..%d -> scan', lo, hi)
            final_j, lo, hi, d, cols, pred = _scan(
                    n, cc, ii, kk, minv, lo, hi, d, cols, pred, y, v)
            # log.debug('d = %s', d)
            # log.debug('cols = %s', cols)
            # log.debug('pred = %s', pred)

    # Update prices for READY columns.
    for k in range(n_ready):
        j0 = cols[k]
        v[j0] += d[j0] - minv

    assert final_j >= 0
    assert final_j < n
    return final_j, pred


def _pya(n, cc, ii, kk, n_free_rows, free_rows, x, y, v):
    # log = logging.getLogger('augment')
    for free_i in free_rows[:n_free_rows]:
        # log.debug('looking at free_i=%s', free_i)
        j, pred = find_path(n, cc, ii, kk, free_i, y, v)
        # Augment the path starting from column j and backtracking to free_i.
        i = -1
        while i != free_i:
            # log.debug('augment %s', j)
            # log.debug('pred = %s', pred)
            i = pred[j]
            assert i >= 0
            assert i < n
            # log.debug('y[%d]=%d -> %d', j, y[j], i)
            y[j] = i
            j, x[i] = x[i], j


def check_cost(n, cc, ii, kk):
    if n == 0:
        raise ValueError('Cost matrix has zero rows.')
    if len(kk) == 0:
        raise ValueError('Cost matrix has zero columns.')
    lo = cc.min()
    hi = cc.max()
    if lo < 0:
        raise ValueError('Cost matrix values must be non-negative.')
    if hi >= LARGE:
        raise ValueError(
                'Cost matrix values must be less than %s' % LARGE)


def get_cost(n, cc, ii, kk, x0):
    ret = 0
    for i, j in enumerate(x0):
        kj = binary_search(kk[ii[i]:ii[i+1]], j)
        if kj is None:
            return np.inf
        kj = ii[i] + kj
        ret += cc[kj]
    return ret


def lapmod(n, cc, ii, kk, fast=True, return_cost=True,
           fp_version=FP_DYNAMIC):
    """Solve sparse linear assignment problem using Jonker-Volgenant algorithm.

    n: number of rows of the assignment cost matrix
    cc: 1D array of all finite elements of the assignement cost matrix
    ii: 1D array of indices of the row starts in cc. The following must hold:
            ii[0] = 0 and ii[n+1] = len(cc).
    kk: 1D array of the column indices so that:
            cost[i, kk[ii[i] + k]] == cc[ii[i] + k].
        Indices within one row must be sorted.
    extend_cost: whether or not extend a non-square matrix [default: False]
    cost_limit: an upper limit for a cost of a single assignment
                [default: np.inf]
    return_cost: whether or not to return the assignment cost

    Returns (opt, x, y) where:
      opt: cost of the assignment
      x: vector of columns assigned to rows
      y: vector of rows assigned to columns
    or (x, y) if return_cost is not True.

    When extend_cost and/or cost_limit is set, all unmatched entries will be
    marked by -1 in x/y.
    """
    # log = logging.getLogger('lapmod')

    check_cost(n, cc, ii, kk)

    if fast is True:
        # log.debug('[----CR & RT & ARR & augmentation ----]')
        x, y = _lapmod(n, cc, ii, kk, fp_version=fp_version)
    else:
        cc = np.ascontiguousarray(cc, dtype=np.float64)
        ii = np.ascontiguousarray(ii, dtype=np.int32)
        kk = np.ascontiguousarray(kk, dtype=np.int32)
        x = np.empty((n,), dtype=np.int32)
        y = np.empty((n,), dtype=np.int32)
        v = np.empty((n,), dtype=np.float64)
        free_rows = np.empty((n,), dtype=np.int32)
        # log.debug('[----Column reduction & reduction transfer----]')
        n_free_rows = _pycrrt(n, cc, ii, kk, free_rows, x, y, v)
        # log.debug(
        #     'free, x, y, v: %s %s %s %s', free_rows[:n_free_rows], x, y, v)
        if n_free_rows == 0:
            # log.info('Reduction solved it.')
            if return_cost is True:
                return get_cost(n, cc, ii, kk, x), x, y
            else:
                return x, y
        for it in range(2):
            # log.debug('[---Augmenting row reduction (iteration: %d)---]', it)
            n_free_rows = _pyarr(
                    n, cc, ii, kk, n_free_rows, free_rows, x, y, v)
            # log.debug(
            #   'free, x, y, v: %s %s %s %s', free_rows[:n_free_rows], x, y, v)
            if n_free_rows == 0:
                # log.info('Augmenting row reduction solved it.')
                if return_cost is True:
                    return get_cost(n, cc, ii, kk, x), x, y
                else:
                    return x, y
        # log.info('[----Augmentation----]')
        _pya(n, cc, ii, kk, n_free_rows, free_rows, x, y, v)
        # log.debug('x, y, v: %s %s %s', x, y, v)
    if return_cost is True:
        return get_cost(n, cc, ii, kk, x), x, y
    else:
        return x, y
