import cython
from cython import Py_ssize_t

from cython cimport floating
from libc.stdlib cimport (
    free,
    malloc,
)

import numpy as np

cimport numpy as cnp
from numpy cimport (
    complex64_t,
    complex128_t,
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from numpy.math cimport NAN

cnp.import_array()

from pandas._libs.algos cimport kth_smallest_c
from pandas._libs.util cimport (
    get_nat,
    numeric,
)

from pandas._libs.algos import (
    ensure_platform_int,
    groupsort_indexer,
    rank_1d,
    take_2d_axis1_float64_float64,
)

from pandas._libs.missing cimport checknull


cdef int64_t NPY_NAT = get_nat()
_int64_max = np.iinfo(np.int64).max

cdef float64_t NaN = <float64_t>np.NaN

cdef enum InterpolationEnumType:
    INTERPOLATION_LINEAR,
    INTERPOLATION_LOWER,
    INTERPOLATION_HIGHER,
    INTERPOLATION_NEAREST,
    INTERPOLATION_MIDPOINT


cdef inline float64_t median_linear(float64_t* a, int n) nogil:
    cdef:
        int i, j, na_count = 0
        float64_t result
        float64_t* tmp

    if n == 0:
        return NaN

    # count NAs
    for i in range(n):
        if a[i] != a[i]:
            na_count += 1

    if na_count:
        if na_count == n:
            return NaN

        tmp = <float64_t*>malloc((n - na_count) * sizeof(float64_t))

        j = 0
        for i in range(n):
            if a[i] == a[i]:
                tmp[j] = a[i]
                j += 1

        a = tmp
        n -= na_count

    if n % 2:
        result = kth_smallest_c(a, n // 2, n)
    else:
        result = (kth_smallest_c(a, n // 2, n) +
                  kth_smallest_c(a, n // 2 - 1, n)) / 2

    if na_count:
        free(a)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def group_median_float64(ndarray[float64_t, ndim=2] out,
                         ndarray[int64_t] counts,
                         ndarray[float64_t, ndim=2] values,
                         ndarray[intp_t] labels,
                         Py_ssize_t min_count=-1) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, ngroups, size
        ndarray[intp_t] _counts
        ndarray[float64_t, ndim=2] data
        ndarray[intp_t] indexer
        float64_t* ptr

    assert min_count == -1, "'min_count' only used in add and prod"

    ngroups = len(counts)
    N, K = (<object>values).shape

    indexer, _counts = groupsort_indexer(labels, ngroups)
    counts[:] = _counts[1:]

    data = np.empty((K, N), dtype=np.float64)
    ptr = <float64_t*>cnp.PyArray_DATA(data)

    take_2d_axis1_float64_float64(values.T, indexer, out=data)

    with nogil:

        for i in range(K):
            # exclude NA group
            ptr += _counts[0]
            for j in range(ngroups):
                size = _counts[j + 1]
                out[j, i] = median_linear(ptr, size)
                ptr += size


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cumprod_float64(float64_t[:, ::1] out,
                          const float64_t[:, :] values,
                          const intp_t[:] labels,
                          int ngroups,
                          bint is_datetimelike,
                          bint skipna=True) -> None:
    """
    Cumulative product of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[np.float64, ndim=2]
        Array to store cumprod in.
    values : np.ndarray[np.float64, ndim=2]
        Values to take cumprod of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    ngroups : int
        Number of groups, larger than all entries of `labels`.
    is_datetimelike : bool
        Always false, `values` is never datetime-like.
    skipna : bool
        If true, ignore nans in `values`.

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    """
    cdef:
        Py_ssize_t i, j, N, K, size
        float64_t val
        float64_t[:, ::1] accum
        intp_t lab

    N, K = (<object>values).shape
    accum = np.ones((ngroups, K), dtype=np.float64)

    with nogil:
        for i in range(N):
            lab = labels[i]

            if lab < 0:
                continue
            for j in range(K):
                val = values[i, j]
                if val == val:
                    accum[lab, j] *= val
                    out[i, j] = accum[lab, j]
                else:
                    out[i, j] = NaN
                    if not skipna:
                        accum[lab, j] = NaN
                        break


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cumsum(numeric[:, ::1] out,
                 ndarray[numeric, ndim=2] values,
                 const intp_t[:] labels,
                 int ngroups,
                 is_datetimelike,
                 bint skipna=True) -> None:
    """
    Cumulative sum of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[ndim=2]
        Array to store cumsum in.
    values : np.ndarray[ndim=2]
        Values to take cumsum of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    ngroups : int
        Number of groups, larger than all entries of `labels`.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    skipna : bool
        If true, ignore nans in `values`.

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    """
    cdef:
        Py_ssize_t i, j, N, K, size
        numeric val, y, t
        numeric[:, ::1] accum, compensation
        intp_t lab

    N, K = (<object>values).shape
    accum = np.zeros((ngroups, K), dtype=np.asarray(values).dtype)
    compensation = np.zeros((ngroups, K), dtype=np.asarray(values).dtype)

    with nogil:
        for i in range(N):
            lab = labels[i]

            if lab < 0:
                continue
            for j in range(K):
                val = values[i, j]

                # For floats, use Kahan summation to reduce floating-point
                # error (https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
                if numeric == float32_t or numeric == float64_t:
                    if val == val:
                        y = val - compensation[lab, j]
                        t = accum[lab, j] + y
                        compensation[lab, j] = t - accum[lab, j] - y
                        accum[lab, j] = t
                        out[i, j] = t
                    else:
                        out[i, j] = NaN
                        if not skipna:
                            accum[lab, j] = NaN
                            break
                else:
                    t = val + accum[lab, j]
                    accum[lab, j] = t
                    out[i, j] = t


@cython.boundscheck(False)
@cython.wraparound(False)
def group_shift_indexer(int64_t[::1] out, const intp_t[:] labels,
                        int ngroups, int periods) -> None:
    cdef:
        Py_ssize_t N, i, j, ii, lab
        int offset = 0, sign
        int64_t idxer, idxer_slot
        int64_t[::1] label_seen = np.zeros(ngroups, dtype=np.int64)
        int64_t[:, ::1] label_indexer

    N, = (<object>labels).shape

    if periods < 0:
        periods = -periods
        offset = N - 1
        sign = -1
    elif periods > 0:
        offset = 0
        sign = 1

    if periods == 0:
        with nogil:
            for i in range(N):
                out[i] = i
    else:
        # array of each previous indexer seen
        label_indexer = np.zeros((ngroups, periods), dtype=np.int64)
        with nogil:
            for i in range(N):
                # reverse iterator if shifting backwards
                ii = offset + sign * i
                lab = labels[ii]

                # Skip null keys
                if lab == -1:
                    out[ii] = -1
                    continue

                label_seen[lab] += 1

                idxer_slot = label_seen[lab] % periods
                idxer = label_indexer[lab, idxer_slot]

                if label_seen[lab] > periods:
                    out[ii] = idxer
                else:
                    out[ii] = -1

                label_indexer[lab, idxer_slot] = ii


@cython.wraparound(False)
@cython.boundscheck(False)
def group_fillna_indexer(ndarray[int64_t] out, ndarray[intp_t] labels,
                         ndarray[uint8_t] mask, str direction,
                         int64_t limit, bint dropna) -> None:
    """
    Indexes how to fill values forwards or backwards within a group.

    Parameters
    ----------
    out : np.ndarray[np.int64]
        Values into which this method will write its results.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`.
    values : np.ndarray[np.uint8]
        Containing the truth value of each element.
    mask : np.ndarray[np.uint8]
        Indicating whether a value is na or not.
    direction : {'ffill', 'bfill'}
        Direction for fill to be applied (forwards or backwards, respectively)
    limit : Consecutive values to fill before stopping, or -1 for no limit
    dropna : Flag to indicate if NaN groups should return all NaN values

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        Py_ssize_t i, N, idx
        intp_t[:] sorted_labels
        intp_t curr_fill_idx=-1
        int64_t filled_vals = 0

    N = len(out)

    # Make sure all arrays are the same size
    assert N == len(labels) == len(mask)

    sorted_labels = np.argsort(labels, kind='mergesort').astype(
        np.intp, copy=False)
    if direction == 'bfill':
        sorted_labels = sorted_labels[::-1]

    with nogil:
        for i in range(N):
            idx = sorted_labels[i]
            if dropna and labels[idx] == -1:  # nan-group gets nan-values
                curr_fill_idx = -1
            elif mask[idx] == 1:  # is missing
                # Stop filling once we've hit the limit
                if filled_vals >= limit and limit != -1:
                    curr_fill_idx = -1
                filled_vals += 1
            else:  # reset items when not missing
                filled_vals = 0
                curr_fill_idx = idx

            out[idx] = curr_fill_idx

            # If we move to the next group, reset
            # the fill_idx and counter
            if i == N - 1 or labels[idx] != labels[sorted_labels[i + 1]]:
                curr_fill_idx = -1
                filled_vals = 0


@cython.boundscheck(False)
@cython.wraparound(False)
def group_any_all(int8_t[::1] out,
                  const int8_t[::1] values,
                  const intp_t[:] labels,
                  const uint8_t[::1] mask,
                  str val_test,
                  bint skipna,
                  bint nullable) -> None:
    """
    Aggregated boolean values to show truthfulness of group elements. If the
    input is a nullable type (nullable=True), the result will be computed
    using Kleene logic.

    Parameters
    ----------
    out : np.ndarray[np.int8]
        Values into which this method will write its results.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its
        ordering matching up to the corresponding record in `values`
    values : np.ndarray[np.int8]
        Containing the truth value of each element.
    mask : np.ndarray[np.uint8]
        Indicating whether a value is na or not.
    val_test : {'any', 'all'}
        String object dictating whether to use any or all truth testing
    skipna : bool
        Flag to ignore nan values during truth testing
    nullable : bool
        Whether or not the input is a nullable type. If True, the
        result will be computed using Kleene logic

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object.
    The returned values will either be 0, 1 (False or True, respectively), or
    -1 to signify a masked position in the case of a nullable input.
    """
    cdef:
        Py_ssize_t i, N = len(labels)
        intp_t lab
        int8_t flag_val

    if val_test == 'all':
        # Because the 'all' value of an empty iterable in Python is True we can
        # start with an array full of ones and set to zero when a False value
        # is encountered
        flag_val = 0
    elif val_test == 'any':
        # Because the 'any' value of an empty iterable in Python is False we
        # can start with an array full of zeros and set to one only if any
        # value encountered is True
        flag_val = 1
    else:
        raise ValueError("'bool_func' must be either 'any' or 'all'!")

    out[:] = 1 - flag_val

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0 or (skipna and mask[i]):
                continue

            if nullable and mask[i]:
                # Set the position as masked if `out[lab] != flag_val`, which
                # would indicate True/False has not yet been seen for any/all,
                # so by Kleene logic the result is currently unknown
                if out[lab] != flag_val:
                    out[lab] = -1
                continue

            # If True and 'any' or False and 'all', the result is
            # already determined
            if values[i] == flag_val:
                out[lab] = flag_val


# ----------------------------------------------------------------------
# group_add, group_prod, group_var, group_mean, group_ohlc
# ----------------------------------------------------------------------

ctypedef fused add_t:
    float64_t
    float32_t
    complex64_t
    complex128_t
    object


@cython.wraparound(False)
@cython.boundscheck(False)
def group_add(add_t[:, ::1] out,
              int64_t[::1] counts,
              ndarray[add_t, ndim=2] values,
              const intp_t[:] labels,
              Py_ssize_t min_count=0,
              bint datetimelike=False) -> None:
    """
    Only aggregates on axis=0 using Kahan summation
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        add_t val, t, y
        add_t[:, ::1] sumx, compensation
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    # the below is equivalent to `np.zeros_like(out)` but faster
    sumx = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)
    compensation = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape

    if add_t is object:
        # NB: this does not use 'compensation' like the non-object track does.
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if not checknull(val):
                    nobs[lab, j] += 1

                    if nobs[lab, j] == 1:
                        # i.e. we havent added anything yet; avoid TypeError
                        #  if e.g. val is a str and sumx[lab, j] is 0
                        t = val
                    else:
                        t = sumx[lab, j] + val
                    sumx[lab, j] = t

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = NAN
                else:
                    out[i, j] = sumx[i, j]
    else:
        with nogil:
            for i in range(N):
                lab = labels[i]
                if lab < 0:
                    continue

                counts[lab] += 1
                for j in range(K):
                    val = values[i, j]

                    # not nan
                    # With dt64/td64 values, values have been cast to float64
                    #  instead if int64 for group_add, but the logic
                    #  is otherwise the same as in _treat_as_na
                    if val == val and not (
                        add_t is float64_t
                        and datetimelike
                        and val == <float64_t>NPY_NAT
                    ):
                        nobs[lab, j] += 1
                        y = val - compensation[lab, j]
                        t = sumx[lab, j] + y
                        compensation[lab, j] = t - sumx[lab, j] - y
                        sumx[lab, j] = t

            for i in range(ncounts):
                for j in range(K):
                    if nobs[i, j] < min_count:
                        out[i, j] = NAN
                    else:
                        out[i, j] = sumx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
def group_prod(floating[:, ::1] out,
               int64_t[::1] counts,
               ndarray[floating, ndim=2] values,
               const intp_t[:] labels,
               Py_ssize_t min_count=0) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        floating val, count
        floating[:, ::1] prodx
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    prodx = np.ones((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    prodx[lab, j] *= val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = NAN
                else:
                    out[i, j] = prodx[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def group_var(floating[:, ::1] out,
              int64_t[::1] counts,
              ndarray[floating, ndim=2] values,
              const intp_t[:] labels,
              Py_ssize_t min_count=-1,
              int64_t ddof=1) -> None:
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        floating val, ct, oldmean
        floating[:, ::1] mean
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)

    assert min_count == -1, "'min_count' only used in add and prod"

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    mean = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape

    out[:, :] = 0.0

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1

            for j in range(K):
                val = values[i, j]

                # not nan
                if val == val:
                    nobs[lab, j] += 1
                    oldmean = mean[lab, j]
                    mean[lab, j] += (val - oldmean) / nobs[lab, j]
                    out[lab, j] += (val - mean[lab, j]) * (val - oldmean)

        for i in range(ncounts):
            for j in range(K):
                ct = nobs[i, j]
                if ct <= ddof:
                    out[i, j] = NAN
                else:
                    out[i, j] /= (ct - ddof)


@cython.wraparound(False)
@cython.boundscheck(False)
def group_mean(floating[:, ::1] out,
               int64_t[::1] counts,
               ndarray[floating, ndim=2] values,
               const intp_t[::1] labels,
               Py_ssize_t min_count=-1,
               bint is_datetimelike=False,
               const uint8_t[:, ::1] mask=None,
               uint8_t[:, ::1] result_mask=None
               ) -> None:
    """
    Compute the mean per label given a label assignment for each value.
    NaN values are ignored.

    Parameters
    ----------
    out : np.ndarray[floating]
        Values into which this method will write its results.
    counts : np.ndarray[int64]
        A zeroed array of the same shape as labels,
        populated by group sizes during algorithm.
    values : np.ndarray[floating]
        2-d array of the values to find the mean of.
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its
        ordering matching up to the corresponding record in `values`.
    min_count : Py_ssize_t
        Only used in add and prod. Always -1.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    mask : ndarray[bool, ndim=2], optional
        Not used.
    result_mask : ndarray[bool, ndim=2], optional
        Not used.

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object.
    `counts` is modified to hold group sizes
    """

    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        floating val, count, y, t, nan_val
        floating[:, ::1] sumx, compensation
        int64_t[:, ::1] nobs
        Py_ssize_t len_values = len(values), len_labels = len(labels)

    assert min_count == -1, "'min_count' only used in add and prod"

    if len_values != len_labels:
        raise ValueError("len(index) != len(labels)")

    # the below is equivalent to `np.zeros_like(out)` but faster
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    sumx = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)
    compensation = np.zeros((<object>out).shape, dtype=(<object>out).base.dtype)

    N, K = (<object>values).shape
    nan_val = NPY_NAT if is_datetimelike else NAN

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]
                # not nan
                if val == val and not (is_datetimelike and val == NPY_NAT):
                    nobs[lab, j] += 1
                    y = val - compensation[lab, j]
                    t = sumx[lab, j] + y
                    compensation[lab, j] = t - sumx[lab, j] - y
                    sumx[lab, j] = t

        for i in range(ncounts):
            for j in range(K):
                count = nobs[i, j]
                if nobs[i, j] == 0:
                    out[i, j] = nan_val
                else:
                    out[i, j] = sumx[i, j] / count


@cython.wraparound(False)
@cython.boundscheck(False)
def group_ohlc(floating[:, ::1] out,
               int64_t[::1] counts,
               ndarray[floating, ndim=2] values,
               const intp_t[:] labels,
               Py_ssize_t min_count=-1) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab
        floating val

    assert min_count == -1, "'min_count' only used in add and prod"

    if len(labels) == 0:
        return

    N, K = (<object>values).shape

    if out.shape[1] != 4:
        raise ValueError('Output array must have 4 columns')

    if K > 1:
        raise NotImplementedError("Argument 'values' must have only one dimension")
    out[:] = np.nan

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab == -1:
                continue

            counts[lab] += 1
            val = values[i, 0]
            if val != val:
                continue

            if out[lab, 0] != out[lab, 0]:
                out[lab, 0] = out[lab, 1] = out[lab, 2] = out[lab, 3] = val
            else:
                out[lab, 1] = max(out[lab, 1], val)
                out[lab, 2] = min(out[lab, 2], val)
                out[lab, 3] = val


@cython.boundscheck(False)
@cython.wraparound(False)
def group_quantile(ndarray[float64_t] out,
                   ndarray[numeric, ndim=1] values,
                   ndarray[intp_t] labels,
                   ndarray[uint8_t] mask,
                   float64_t q,
                   str interpolation) -> None:
    """
    Calculate the quantile per group.

    Parameters
    ----------
    out : np.ndarray[np.float64]
        Array of aggregated values that will be written to.
    values : np.ndarray
        Array containing the values to apply the function against.
    labels : ndarray[np.intp]
        Array containing the unique group labels.
    q : float
        The quantile value to search for.
    interpolation : {'linear', 'lower', 'highest', 'nearest', 'midpoint'}

    Notes
    -----
    Rather than explicitly returning a value, this function modifies the
    provided `out` parameter.
    """
    cdef:
        Py_ssize_t i, N=len(labels), ngroups, grp_sz, non_na_sz
        Py_ssize_t grp_start=0, idx=0
        intp_t lab
        uint8_t interp
        float64_t q_idx, frac, val, next_val
        ndarray[int64_t] counts, non_na_counts, sort_arr

    assert values.shape[0] == N

    if not (0 <= q <= 1):
        raise ValueError(f"'q' must be between 0 and 1. Got '{q}' instead")

    inter_methods = {
        'linear': INTERPOLATION_LINEAR,
        'lower': INTERPOLATION_LOWER,
        'higher': INTERPOLATION_HIGHER,
        'nearest': INTERPOLATION_NEAREST,
        'midpoint': INTERPOLATION_MIDPOINT,
    }
    interp = inter_methods[interpolation]

    counts = np.zeros_like(out, dtype=np.int64)
    non_na_counts = np.zeros_like(out, dtype=np.int64)
    ngroups = len(counts)

    # First figure out the size of every group
    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab == -1:  # NA group label
                continue

            counts[lab] += 1
            if not mask[i]:
                non_na_counts[lab] += 1

    # Get an index of values sorted by labels and then values
    if labels.any():
        # Put '-1' (NaN) labels as the last group so it does not interfere
        # with the calculations.
        labels_for_lexsort = np.where(labels == -1, labels.max() + 1, labels)
    else:
        labels_for_lexsort = labels
    order = (values, labels_for_lexsort)
    sort_arr = np.lexsort(order).astype(np.int64, copy=False)

    with nogil:
        for i in range(ngroups):
            # Figure out how many group elements there are
            grp_sz = counts[i]
            non_na_sz = non_na_counts[i]

            if non_na_sz == 0:
                out[i] = NaN
            else:
                # Calculate where to retrieve the desired value
                # Casting to int will intentionally truncate result
                idx = grp_start + <int64_t>(q * <float64_t>(non_na_sz - 1))

                val = values[sort_arr[idx]]
                # If requested quantile falls evenly on a particular index
                # then write that index's value out. Otherwise interpolate
                q_idx = q * (non_na_sz - 1)
                frac = q_idx % 1

                if frac == 0.0 or interp == INTERPOLATION_LOWER:
                    out[i] = val
                else:
                    next_val = values[sort_arr[idx + 1]]
                    if interp == INTERPOLATION_LINEAR:
                        out[i] = val + (next_val - val) * frac
                    elif interp == INTERPOLATION_HIGHER:
                        out[i] = next_val
                    elif interp == INTERPOLATION_MIDPOINT:
                        out[i] = (val + next_val) / 2.0
                    elif interp == INTERPOLATION_NEAREST:
                        if frac > .5 or (frac == .5 and q > .5):  # Always OK?
                            out[i] = next_val
                        else:
                            out[i] = val

            # Increment the index reference in sorted_arr for the next group
            grp_start += grp_sz


# ----------------------------------------------------------------------
# group_nth, group_last, group_rank
# ----------------------------------------------------------------------

ctypedef fused rank_t:
    float64_t
    float32_t
    int64_t
    uint64_t
    object


cdef inline bint _treat_as_na(rank_t val, bint is_datetimelike) nogil:
    if rank_t is object:
        # Should never be used, but we need to avoid the `val != val` below
        #  or else cython will raise about gil acquisition.
        raise NotImplementedError

    elif rank_t is int64_t:
        return is_datetimelike and val == NPY_NAT
    elif rank_t is uint64_t:
        # There is no NA value for uint64
        return False
    else:
        return val != val


# GH#31710 use memorviews once cython 0.30 is released so we can
#  use `const rank_t[:, :] values`
@cython.wraparound(False)
@cython.boundscheck(False)
def group_last(rank_t[:, ::1] out,
               int64_t[::1] counts,
               ndarray[rank_t, ndim=2] values,
               const intp_t[:] labels,
               Py_ssize_t min_count=-1) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        rank_t val
        ndarray[rank_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs
        bint runtime_error = False

    # TODO(cython 3.0):
    # Instead of `labels.shape[0]` use `len(labels)`
    if not len(values) == labels.shape[0]:
        raise AssertionError("len(index) != len(labels)")

    min_count = max(min_count, 1)
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    if rank_t is object:
        resx = np.empty((<object>out).shape, dtype=object)
    else:
        resx = np.empty_like(out)

    N, K = (<object>values).shape

    if rank_t is object:
        # TODO: De-duplicate once conditional-nogil is available
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if not checknull(val):
                    # NB: use _treat_as_na here once
                    #  conditional-nogil is available.
                    nobs[lab, j] += 1
                    resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = None
                else:
                    out[i, j] = resx[i, j]
    else:
        with nogil:
            for i in range(N):
                lab = labels[i]
                if lab < 0:
                    continue

                counts[lab] += 1
                for j in range(K):
                    val = values[i, j]

                    if not _treat_as_na(val, True):
                        # TODO: Sure we always want is_datetimelike=True?
                        nobs[lab, j] += 1
                        resx[lab, j] = val

            for i in range(ncounts):
                for j in range(K):
                    if nobs[i, j] < min_count:
                        if rank_t is int64_t:
                            out[i, j] = NPY_NAT
                        elif rank_t is uint64_t:
                            runtime_error = True
                            break
                        else:
                            out[i, j] = NAN

                    else:
                        out[i, j] = resx[i, j]

    if runtime_error:
        # We cannot raise directly above because that is within a nogil
        #  block.
        raise RuntimeError("empty group with uint64_t")


# GH#31710 use memorviews once cython 0.30 is released so we can
#  use `const rank_t[:, :] values`
@cython.wraparound(False)
@cython.boundscheck(False)
def group_nth(rank_t[:, ::1] out,
              int64_t[::1] counts,
              ndarray[rank_t, ndim=2] values,
              const intp_t[:] labels,
              int64_t min_count=-1,
              int64_t rank=1,
              ) -> None:
    """
    Only aggregates on axis=0
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ncounts = len(counts)
        rank_t val
        ndarray[rank_t, ndim=2] resx
        ndarray[int64_t, ndim=2] nobs
        bint runtime_error = False

    # TODO(cython 3.0):
    # Instead of `labels.shape[0]` use `len(labels)`
    if not len(values) == labels.shape[0]:
        raise AssertionError("len(index) != len(labels)")

    min_count = max(min_count, 1)
    nobs = np.zeros((<object>out).shape, dtype=np.int64)
    if rank_t is object:
        resx = np.empty((<object>out).shape, dtype=object)
    else:
        resx = np.empty_like(out)

    N, K = (<object>values).shape

    if rank_t is object:
        # TODO: De-duplicate once conditional-nogil is available
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if not checknull(val):
                    # NB: use _treat_as_na here once
                    #  conditional-nogil is available.
                    nobs[lab, j] += 1
                    if nobs[lab, j] == rank:
                        resx[lab, j] = val

        for i in range(ncounts):
            for j in range(K):
                if nobs[i, j] < min_count:
                    out[i, j] = None
                else:
                    out[i, j] = resx[i, j]

    else:
        with nogil:
            for i in range(N):
                lab = labels[i]
                if lab < 0:
                    continue

                counts[lab] += 1
                for j in range(K):
                    val = values[i, j]

                    if not _treat_as_na(val, True):
                        # TODO: Sure we always want is_datetimelike=True?
                        nobs[lab, j] += 1
                        if nobs[lab, j] == rank:
                            resx[lab, j] = val

            for i in range(ncounts):
                for j in range(K):
                    if nobs[i, j] < min_count:
                        if rank_t is int64_t:
                            out[i, j] = NPY_NAT
                        elif rank_t is uint64_t:
                            runtime_error = True
                            break
                        else:
                            out[i, j] = NAN
                    else:
                        out[i, j] = resx[i, j]

    if runtime_error:
        # We cannot raise directly above because that is within a nogil
        #  block.
        raise RuntimeError("empty group with uint64_t")


@cython.boundscheck(False)
@cython.wraparound(False)
def group_rank(float64_t[:, ::1] out,
               ndarray[rank_t, ndim=2] values,
               const intp_t[:] labels,
               int ngroups,
               bint is_datetimelike, str ties_method="average",
               bint ascending=True, bint pct=False, str na_option="keep") -> None:
    """
    Provides the rank of values within each group.

    Parameters
    ----------
    out : np.ndarray[np.float64, ndim=2]
        Values to which this method will write its results.
    values : np.ndarray of rank_t values to be ranked
    labels : np.ndarray[np.intp]
        Array containing unique label for each group, with its ordering
        matching up to the corresponding record in `values`
    ngroups : int
        This parameter is not used, is needed to match signatures of other
        groupby functions.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    ties_method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
        * average: average rank of group
        * min: lowest rank in group
        * max: highest rank in group
        * first: ranks assigned in order they appear in the array
        * dense: like 'min', but rank always increases by 1 between groups
    ascending : bool, default True
        False for ranks by high (1) to low (N)
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
    pct : bool, default False
        Compute percentage rank of data within each group
    na_option : {'keep', 'top', 'bottom'}, default 'keep'
        * keep: leave NA values where they are
        * top: smallest rank if ascending
        * bottom: smallest rank if descending

    Notes
    -----
    This method modifies the `out` parameter rather than returning an object
    """
    cdef:
        Py_ssize_t i, k, N
        ndarray[float64_t, ndim=1] result

    N = values.shape[1]

    for k in range(N):
        result = rank_1d(
            values=values[:, k],
            labels=labels,
            is_datetimelike=is_datetimelike,
            ties_method=ties_method,
            ascending=ascending,
            pct=pct,
            na_option=na_option
        )
        for i in range(len(result)):
            # TODO: why cant we do out[:, k] = result?
            out[i, k] = result[i]


# ----------------------------------------------------------------------
# group_min, group_max
# ----------------------------------------------------------------------

# TODO: consider implementing for more dtypes
ctypedef fused groupby_t:
    float64_t
    float32_t
    int64_t
    uint64_t


@cython.wraparound(False)
@cython.boundscheck(False)
cdef group_min_max(groupby_t[:, ::1] out,
                   int64_t[::1] counts,
                   ndarray[groupby_t, ndim=2] values,
                   const intp_t[:] labels,
                   Py_ssize_t min_count=-1,
                   bint is_datetimelike=False,
                   bint compute_max=True):
    """
    Compute minimum/maximum  of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[groupby_t, ndim=2]
        Array to store result in.
    counts : np.ndarray[int64]
        Input as a zeroed array, populated by group sizes during algorithm
    values : array
        Values to find column-wise min/max of.
    labels : np.ndarray[np.intp]
        Labels to group by.
    min_count : Py_ssize_t, default -1
        The minimum number of non-NA group elements, NA result if threshold
        is not met
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    compute_max : bint, default True
        True to compute group-wise max, False to compute min

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    `counts` is modified to hold group sizes
    """
    cdef:
        Py_ssize_t i, j, N, K, lab, ngroups = len(counts)
        groupby_t val, nan_val
        ndarray[groupby_t, ndim=2] group_min_or_max
        bint runtime_error = False
        int64_t[:, ::1] nobs

    # TODO(cython 3.0):
    # Instead of `labels.shape[0]` use `len(labels)`
    if not len(values) == labels.shape[0]:
        raise AssertionError("len(index) != len(labels)")

    min_count = max(min_count, 1)
    nobs = np.zeros((<object>out).shape, dtype=np.int64)

    group_min_or_max = np.empty_like(out)
    if groupby_t is int64_t:
        group_min_or_max[:] = -_int64_max if compute_max else _int64_max
        nan_val = NPY_NAT
    elif groupby_t is uint64_t:
        # NB: We do not define nan_val because there is no such thing
        # for uint64_t.  We carefully avoid having to reference it in this
        # case.
        group_min_or_max[:] = 0 if compute_max else np.iinfo(np.uint64).max
    else:
        group_min_or_max[:] = -np.inf if compute_max else np.inf
        nan_val = NAN

    N, K = (<object>values).shape

    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue

            counts[lab] += 1
            for j in range(K):
                val = values[i, j]

                if not _treat_as_na(val, is_datetimelike):
                    nobs[lab, j] += 1
                    if compute_max:
                        if val > group_min_or_max[lab, j]:
                            group_min_or_max[lab, j] = val
                    else:
                        if val < group_min_or_max[lab, j]:
                            group_min_or_max[lab, j] = val

        for i in range(ngroups):
            for j in range(K):
                if nobs[i, j] < min_count:
                    if groupby_t is uint64_t:
                        runtime_error = True
                        break
                    else:
                        out[i, j] = nan_val
                else:
                    out[i, j] = group_min_or_max[i, j]

    if runtime_error:
        # We cannot raise directly above because that is within a nogil
        #  block.
        raise RuntimeError("empty group with uint64_t")


@cython.wraparound(False)
@cython.boundscheck(False)
def group_max(groupby_t[:, ::1] out,
              int64_t[::1] counts,
              ndarray[groupby_t, ndim=2] values,
              const intp_t[:] labels,
              Py_ssize_t min_count=-1,
              bint is_datetimelike=False) -> None:
    """See group_min_max.__doc__"""
    group_min_max(
        out,
        counts,
        values,
        labels,
        min_count=min_count,
        is_datetimelike=is_datetimelike,
        compute_max=True,
    )


@cython.wraparound(False)
@cython.boundscheck(False)
def group_min(groupby_t[:, ::1] out,
              int64_t[::1] counts,
              ndarray[groupby_t, ndim=2] values,
              const intp_t[:] labels,
              Py_ssize_t min_count=-1,
              bint is_datetimelike=False) -> None:
    """See group_min_max.__doc__"""
    group_min_max(
        out,
        counts,
        values,
        labels,
        min_count=min_count,
        is_datetimelike=is_datetimelike,
        compute_max=False,
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef group_cummin_max(groupby_t[:, ::1] out,
                      ndarray[groupby_t, ndim=2] values,
                      uint8_t[:, ::1] mask,
                      const intp_t[:] labels,
                      int ngroups,
                      bint is_datetimelike,
                      bint compute_max):
    """
    Cumulative minimum/maximum of columns of `values`, in row groups `labels`.

    Parameters
    ----------
    out : np.ndarray[groupby_t, ndim=2]
        Array to store cummin/max in.
    values : np.ndarray[groupby_t, ndim=2]
        Values to take cummin/max of.
    mask : np.ndarray[bool] or None
        If not None, indices represent missing values,
        otherwise the mask will not be used
    labels : np.ndarray[np.intp]
        Labels to group by.
    ngroups : int
        Number of groups, larger than all entries of `labels`.
    is_datetimelike : bool
        True if `values` contains datetime-like entries.
    compute_max : bool
        True if cumulative maximum should be computed, False
        if cumulative minimum should be computed

    Notes
    -----
    This method modifies the `out` parameter, rather than returning an object.
    """
    cdef:
        groupby_t[:, ::1] accum

    accum = np.empty((ngroups, (<object>values).shape[1]), dtype=values.dtype)
    if groupby_t is int64_t:
        accum[:] = -_int64_max if compute_max else _int64_max
    elif groupby_t is uint64_t:
        accum[:] = 0 if compute_max else np.iinfo(np.uint64).max
    else:
        accum[:] = -np.inf if compute_max else np.inf

    if mask is not None:
        masked_cummin_max(out, values, mask, labels, accum, compute_max)
    else:
        cummin_max(out, values, labels, accum, is_datetimelike, compute_max)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cummin_max(groupby_t[:, ::1] out,
                ndarray[groupby_t, ndim=2] values,
                const intp_t[:] labels,
                groupby_t[:, ::1] accum,
                bint is_datetimelike,
                bint compute_max):
    """
    Compute the cumulative minimum/maximum of columns of `values`, in row groups
    `labels`.
    """
    cdef:
        Py_ssize_t i, j, N, K
        groupby_t val, mval
        intp_t lab

    N, K = (<object>values).shape
    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue
            for j in range(K):
                val = values[i, j]
                if not _treat_as_na(val, is_datetimelike):
                    mval = accum[lab, j]
                    if compute_max:
                        if val > mval:
                            accum[lab, j] = mval = val
                    else:
                        if val < mval:
                            accum[lab, j] = mval = val
                    out[i, j] = mval
                else:
                    out[i, j] = val


@cython.boundscheck(False)
@cython.wraparound(False)
cdef masked_cummin_max(groupby_t[:, ::1] out,
                       ndarray[groupby_t, ndim=2] values,
                       uint8_t[:, ::1] mask,
                       const intp_t[:] labels,
                       groupby_t[:, ::1] accum,
                       bint compute_max):
    """
    Compute the cumulative minimum/maximum of columns of `values`, in row groups
    `labels` with a masked algorithm.
    """
    cdef:
        Py_ssize_t i, j, N, K
        groupby_t val, mval
        intp_t lab

    N, K = (<object>values).shape
    with nogil:
        for i in range(N):
            lab = labels[i]
            if lab < 0:
                continue
            for j in range(K):
                if not mask[i, j]:
                    val = values[i, j]
                    mval = accum[lab, j]
                    if compute_max:
                        if val > mval:
                            accum[lab, j] = mval = val
                    else:
                        if val < mval:
                            accum[lab, j] = mval = val
                    out[i, j] = mval


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummin(groupby_t[:, ::1] out,
                 ndarray[groupby_t, ndim=2] values,
                 const intp_t[:] labels,
                 int ngroups,
                 bint is_datetimelike,
                 uint8_t[:, ::1] mask=None) -> None:
    """See group_cummin_max.__doc__"""
    group_cummin_max(
        out,
        values,
        mask,
        labels,
        ngroups,
        is_datetimelike,
        compute_max=False
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def group_cummax(groupby_t[:, ::1] out,
                 ndarray[groupby_t, ndim=2] values,
                 const intp_t[:] labels,
                 int ngroups,
                 bint is_datetimelike,
                 uint8_t[:, ::1] mask=None) -> None:
    """See group_cummin_max.__doc__"""
    group_cummin_max(
        out,
        values,
        mask,
        labels,
        ngroups,
        is_datetimelike,
        compute_max=True
    )
