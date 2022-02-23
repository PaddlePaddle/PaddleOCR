# cython: boundscheck=False, wraparound=False, cdivision=True

import cython

from libc.math cimport round
from libcpp.deque cimport deque

import numpy as np

cimport numpy as cnp
from numpy cimport (
    float32_t,
    float64_t,
    int64_t,
    ndarray,
)

cnp.import_array()


cdef extern from "src/headers/cmath" namespace "std":
    bint isnan(float64_t) nogil
    bint notnan(float64_t) nogil
    int signbit(float64_t) nogil
    float64_t sqrt(float64_t x) nogil

from pandas._libs.algos import is_monotonic

from pandas._libs.util cimport numeric


cdef extern from "../src/skiplist.h":
    ctypedef struct node_t:
        node_t **next
        int *width
        double value
        int is_nil
        int levels
        int ref_count

    ctypedef struct skiplist_t:
        node_t *head
        node_t **tmp_chain
        int *tmp_steps
        int size
        int maxlevels

    skiplist_t* skiplist_init(int) nogil
    void skiplist_destroy(skiplist_t*) nogil
    double skiplist_get(skiplist_t*, int, int*) nogil
    int skiplist_insert(skiplist_t*, double) nogil
    int skiplist_remove(skiplist_t*, double) nogil

cdef:
    float32_t MINfloat32 = np.NINF
    float64_t MINfloat64 = np.NINF

    float32_t MAXfloat32 = np.inf
    float64_t MAXfloat64 = np.inf

    float64_t NaN = <float64_t>np.NaN

cdef bint is_monotonic_increasing_start_end_bounds(
    ndarray[int64_t, ndim=1] start, ndarray[int64_t, ndim=1] end
):
    return is_monotonic(start, False)[0] and is_monotonic(end, False)[0]

# ----------------------------------------------------------------------
# Rolling sum


cdef inline float64_t calc_sum(int64_t minp, int64_t nobs, float64_t sum_x) nogil:
    cdef:
        float64_t result

    if nobs == 0 == minp:
        result = 0
    elif nobs >= minp:
        result = sum_x
    else:
        result = NaN

    return result


cdef inline void add_sum(float64_t val, int64_t *nobs, float64_t *sum_x,
                         float64_t *compensation) nogil:
    """ add a value from the sum calc using Kahan summation """

    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] + 1
        y = val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t


cdef inline void remove_sum(float64_t val, int64_t *nobs, float64_t *sum_x,
                            float64_t *compensation) nogil:
    """ remove a value from the sum calc using Kahan summation """

    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] - 1
        y = - val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t


def roll_sum(const float64_t[:] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j
        float64_t sum_x = 0, compensation_add = 0, compensation_remove = 0
        int64_t s, e
        int64_t nobs = 0, N = len(values)
        ndarray[float64_t] output
        bint is_monotonic_increasing_bounds

    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    output = np.empty(N, dtype=np.float64)

    with nogil:

        for i in range(0, N):
            s = start[i]
            e = end[i]

            if i == 0 or not is_monotonic_increasing_bounds:

                # setup

                for j in range(s, e):
                    add_sum(values[j], &nobs, &sum_x, &compensation_add)

            else:

                # calculate deletes
                for j in range(start[i - 1], s):
                    remove_sum(values[j], &nobs, &sum_x, &compensation_remove)

                # calculate adds
                for j in range(end[i - 1], e):
                    add_sum(values[j], &nobs, &sum_x, &compensation_add)

            output[i] = calc_sum(minp, nobs, sum_x)

            if not is_monotonic_increasing_bounds:
                nobs = 0
                sum_x = 0.0
                compensation_remove = 0.0

    return output


# ----------------------------------------------------------------------
# Rolling mean


cdef inline float64_t calc_mean(int64_t minp, Py_ssize_t nobs,
                                Py_ssize_t neg_ct, float64_t sum_x) nogil:
    cdef:
        float64_t result

    if nobs >= minp and nobs > 0:
        result = sum_x / <float64_t>nobs
        if neg_ct == 0 and result < 0:
            # all positive
            result = 0
        elif neg_ct == nobs and result > 0:
            # all negative
            result = 0
        else:
            pass
    else:
        result = NaN
    return result


cdef inline void add_mean(float64_t val, Py_ssize_t *nobs, float64_t *sum_x,
                          Py_ssize_t *neg_ct, float64_t *compensation) nogil:
    """ add a value from the mean calc using Kahan summation """
    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] + 1
        y = val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t
        if signbit(val):
            neg_ct[0] = neg_ct[0] + 1


cdef inline void remove_mean(float64_t val, Py_ssize_t *nobs, float64_t *sum_x,
                             Py_ssize_t *neg_ct, float64_t *compensation) nogil:
    """ remove a value from the mean calc using Kahan summation """
    cdef:
        float64_t y, t

    if notnan(val):
        nobs[0] = nobs[0] - 1
        y = - val - compensation[0]
        t = sum_x[0] + y
        compensation[0] = t - sum_x[0] - y
        sum_x[0] = t
        if signbit(val):
            neg_ct[0] = neg_ct[0] - 1


def roll_mean(const float64_t[:] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        float64_t val, compensation_add = 0, compensation_remove = 0, sum_x = 0
        int64_t s, e
        Py_ssize_t nobs = 0, i, j, neg_ct = 0, N = len(values)
        ndarray[float64_t] output
        bint is_monotonic_increasing_bounds

    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    output = np.empty(N, dtype=np.float64)

    with nogil:

        for i in range(0, N):
            s = start[i]
            e = end[i]

            if i == 0 or not is_monotonic_increasing_bounds:

                # setup
                for j in range(s, e):
                    val = values[j]
                    add_mean(val, &nobs, &sum_x, &neg_ct, &compensation_add)

            else:

                # calculate deletes
                for j in range(start[i - 1], s):
                    val = values[j]
                    remove_mean(val, &nobs, &sum_x, &neg_ct, &compensation_remove)

                # calculate adds
                for j in range(end[i - 1], e):
                    val = values[j]
                    add_mean(val, &nobs, &sum_x, &neg_ct, &compensation_add)

            output[i] = calc_mean(minp, nobs, neg_ct, sum_x)

            if not is_monotonic_increasing_bounds:
                nobs = 0
                neg_ct = 0
                sum_x = 0.0
                compensation_remove = 0.0
    return output

# ----------------------------------------------------------------------
# Rolling variance


cdef inline float64_t calc_var(int64_t minp, int ddof, float64_t nobs,
                               float64_t ssqdm_x) nogil:
    cdef:
        float64_t result

    # Variance is unchanged if no observation is added or removed
    if (nobs >= minp) and (nobs > ddof):

        # pathological case
        if nobs == 1:
            result = 0
        else:
            result = ssqdm_x / (nobs - <float64_t>ddof)
    else:
        result = NaN

    return result


cdef inline void add_var(float64_t val, float64_t *nobs, float64_t *mean_x,
                         float64_t *ssqdm_x, float64_t *compensation) nogil:
    """ add a value from the var calc """
    cdef:
        float64_t delta, prev_mean, y, t

    # `isnan` instead of equality as fix for GH-21813, msvc 2017 bug
    if isnan(val):
        return

    nobs[0] = nobs[0] + 1
    # Welford's method for the online variance-calculation
    # using Kahan summation
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    prev_mean = mean_x[0] - compensation[0]
    y = val - compensation[0]
    t = y - mean_x[0]
    compensation[0] = t + mean_x[0] - y
    delta = t
    mean_x[0] = mean_x[0] + delta / nobs[0]
    ssqdm_x[0] = ssqdm_x[0] + (val - prev_mean) * (val - mean_x[0])


cdef inline void remove_var(float64_t val, float64_t *nobs, float64_t *mean_x,
                            float64_t *ssqdm_x, float64_t *compensation) nogil:
    """ remove a value from the var calc """
    cdef:
        float64_t delta, prev_mean, y, t
    if notnan(val):
        nobs[0] = nobs[0] - 1
        if nobs[0]:
            # Welford's method for the online variance-calculation
            # using Kahan summation
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            prev_mean = mean_x[0] - compensation[0]
            y = val - compensation[0]
            t = y - mean_x[0]
            compensation[0] = t + mean_x[0] - y
            delta = t
            mean_x[0] = mean_x[0] - delta / nobs[0]
            ssqdm_x[0] = ssqdm_x[0] - (val - prev_mean) * (val - mean_x[0])
        else:
            mean_x[0] = 0
            ssqdm_x[0] = 0


def roll_var(const float64_t[:] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp, int ddof=1) -> np.ndarray:
    """
    Numerically stable implementation using Welford's method.
    """
    cdef:
        float64_t mean_x = 0, ssqdm_x = 0, nobs = 0, compensation_add = 0,
        float64_t compensation_remove = 0,
        float64_t val, prev, delta, mean_x_old
        int64_t s, e
        Py_ssize_t i, j, N = len(values)
        ndarray[float64_t] output
        bint is_monotonic_increasing_bounds

    minp = max(minp, 1)
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    output = np.empty(N, dtype=np.float64)

    with nogil:

        for i in range(0, N):

            s = start[i]
            e = end[i]

            # Over the first window, observations can only be added
            # never removed
            if i == 0 or not is_monotonic_increasing_bounds:

                for j in range(s, e):
                    add_var(values[j], &nobs, &mean_x, &ssqdm_x, &compensation_add)

            else:

                # After the first window, observations can both be added
                # and removed

                # calculate deletes
                for j in range(start[i - 1], s):
                    remove_var(values[j], &nobs, &mean_x, &ssqdm_x,
                               &compensation_remove)

                # calculate adds
                for j in range(end[i - 1], e):
                    add_var(values[j], &nobs, &mean_x, &ssqdm_x, &compensation_add)

            output[i] = calc_var(minp, ddof, nobs, ssqdm_x)

            if not is_monotonic_increasing_bounds:
                nobs = 0.0
                mean_x = 0.0
                ssqdm_x = 0.0
                compensation_remove = 0.0

    return output

# ----------------------------------------------------------------------
# Rolling skewness


cdef inline float64_t calc_skew(int64_t minp, int64_t nobs,
                                float64_t x, float64_t xx,
                                float64_t xxx) nogil:
    cdef:
        float64_t result, dnobs
        float64_t A, B, C, R

    if nobs >= minp:
        dnobs = <float64_t>nobs
        A = x / dnobs
        B = xx / dnobs - A * A
        C = xxx / dnobs - A * A * A - 3 * A * B

        # #18044: with uniform distribution, floating issue will
        #         cause B != 0. and cause the result is a very
        #         large number.
        #
        #         in core/nanops.py nanskew/nankurt call the function
        #         _zero_out_fperr(m2) to fix floating error.
        #         if the variance is less than 1e-14, it could be
        #         treat as zero, here we follow the original
        #         skew/kurt behaviour to check B <= 1e-14
        if B <= 1e-14 or nobs < 3:
            result = NaN
        else:
            R = sqrt(B)
            result = ((sqrt(dnobs * (dnobs - 1.)) * C) /
                      ((dnobs - 2) * R * R * R))
    else:
        result = NaN

    return result


cdef inline void add_skew(float64_t val, int64_t *nobs,
                          float64_t *x, float64_t *xx,
                          float64_t *xxx,
                          float64_t *compensation_x,
                          float64_t *compensation_xx,
                          float64_t *compensation_xxx) nogil:
    """ add a value from the skew calc """
    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] + 1

        y = val - compensation_x[0]
        t = x[0] + y
        compensation_x[0] = t - x[0] - y
        x[0] = t
        y = val * val - compensation_xx[0]
        t = xx[0] + y
        compensation_xx[0] = t - xx[0] - y
        xx[0] = t
        y = val * val * val - compensation_xxx[0]
        t = xxx[0] + y
        compensation_xxx[0] = t - xxx[0] - y
        xxx[0] = t


cdef inline void remove_skew(float64_t val, int64_t *nobs,
                             float64_t *x, float64_t *xx,
                             float64_t *xxx,
                             float64_t *compensation_x,
                             float64_t *compensation_xx,
                             float64_t *compensation_xxx) nogil:
    """ remove a value from the skew calc """
    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] - 1

        y = - val - compensation_x[0]
        t = x[0] + y
        compensation_x[0] = t - x[0] - y
        x[0] = t
        y = - val * val - compensation_xx[0]
        t = xx[0] + y
        compensation_xx[0] = t - xx[0] - y
        xx[0] = t
        y = - val * val * val - compensation_xxx[0]
        t = xxx[0] + y
        compensation_xxx[0] = t - xxx[0] - y
        xxx[0] = t


def roll_skew(ndarray[float64_t] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j
        float64_t val, prev, min_val, mean_val, sum_val = 0
        float64_t compensation_xxx_add = 0, compensation_xxx_remove = 0
        float64_t compensation_xx_add = 0, compensation_xx_remove = 0
        float64_t compensation_x_add = 0, compensation_x_remove = 0
        float64_t x = 0, xx = 0, xxx = 0
        int64_t nobs = 0, N = len(values), nobs_mean = 0
        int64_t s, e
        ndarray[float64_t] output, mean_array, values_copy
        bint is_monotonic_increasing_bounds

    minp = max(minp, 3)
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    output = np.empty(N, dtype=np.float64)
    min_val = np.nanmin(values)
    values_copy = np.copy(values)

    with nogil:
        for i in range(0, N):
            val = values_copy[i]
            if notnan(val):
                nobs_mean += 1
                sum_val += val
        mean_val = sum_val / nobs_mean
        # Other cases would lead to imprecision for smallest values
        if min_val - mean_val > -1e5:
            mean_val = round(mean_val)
            for i in range(0, N):
                values_copy[i] = values_copy[i] - mean_val

        for i in range(0, N):

            s = start[i]
            e = end[i]

            # Over the first window, observations can only be added
            # never removed
            if i == 0 or not is_monotonic_increasing_bounds:

                for j in range(s, e):
                    val = values_copy[j]
                    add_skew(val, &nobs, &x, &xx, &xxx, &compensation_x_add,
                             &compensation_xx_add, &compensation_xxx_add)

            else:

                # After the first window, observations can both be added
                # and removed
                # calculate deletes
                for j in range(start[i - 1], s):
                    val = values_copy[j]
                    remove_skew(val, &nobs, &x, &xx, &xxx, &compensation_x_remove,
                                &compensation_xx_remove, &compensation_xxx_remove)

                # calculate adds
                for j in range(end[i - 1], e):
                    val = values_copy[j]
                    add_skew(val, &nobs, &x, &xx, &xxx, &compensation_x_add,
                             &compensation_xx_add, &compensation_xxx_add)

            output[i] = calc_skew(minp, nobs, x, xx, xxx)

            if not is_monotonic_increasing_bounds:
                nobs = 0
                x = 0.0
                xx = 0.0
                xxx = 0.0

    return output

# ----------------------------------------------------------------------
# Rolling kurtosis


cdef inline float64_t calc_kurt(int64_t minp, int64_t nobs,
                                float64_t x, float64_t xx,
                                float64_t xxx, float64_t xxxx) nogil:
    cdef:
        float64_t result, dnobs
        float64_t A, B, C, D, R, K

    if nobs >= minp:
        dnobs = <float64_t>nobs
        A = x / dnobs
        R = A * A
        B = xx / dnobs - R
        R = R * A
        C = xxx / dnobs - R - 3 * A * B
        R = R * A
        D = xxxx / dnobs - R - 6 * B * A * A - 4 * C * A

        # #18044: with uniform distribution, floating issue will
        #         cause B != 0. and cause the result is a very
        #         large number.
        #
        #         in core/nanops.py nanskew/nankurt call the function
        #         _zero_out_fperr(m2) to fix floating error.
        #         if the variance is less than 1e-14, it could be
        #         treat as zero, here we follow the original
        #         skew/kurt behaviour to check B <= 1e-14
        if B <= 1e-14 or nobs < 4:
            result = NaN
        else:
            K = (dnobs * dnobs - 1.) * D / (B * B) - 3 * ((dnobs - 1.) ** 2)
            result = K / ((dnobs - 2.) * (dnobs - 3.))
    else:
        result = NaN

    return result


cdef inline void add_kurt(float64_t val, int64_t *nobs,
                          float64_t *x, float64_t *xx,
                          float64_t *xxx, float64_t *xxxx,
                          float64_t *compensation_x,
                          float64_t *compensation_xx,
                          float64_t *compensation_xxx,
                          float64_t *compensation_xxxx) nogil:
    """ add a value from the kurotic calc """
    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] + 1

        y = val - compensation_x[0]
        t = x[0] + y
        compensation_x[0] = t - x[0] - y
        x[0] = t
        y = val * val - compensation_xx[0]
        t = xx[0] + y
        compensation_xx[0] = t - xx[0] - y
        xx[0] = t
        y = val * val * val - compensation_xxx[0]
        t = xxx[0] + y
        compensation_xxx[0] = t - xxx[0] - y
        xxx[0] = t
        y = val * val * val * val - compensation_xxxx[0]
        t = xxxx[0] + y
        compensation_xxxx[0] = t - xxxx[0] - y
        xxxx[0] = t


cdef inline void remove_kurt(float64_t val, int64_t *nobs,
                             float64_t *x, float64_t *xx,
                             float64_t *xxx, float64_t *xxxx,
                             float64_t *compensation_x,
                             float64_t *compensation_xx,
                             float64_t *compensation_xxx,
                             float64_t *compensation_xxxx) nogil:
    """ remove a value from the kurotic calc """
    cdef:
        float64_t y, t

    # Not NaN
    if notnan(val):
        nobs[0] = nobs[0] - 1

        y = - val - compensation_x[0]
        t = x[0] + y
        compensation_x[0] = t - x[0] - y
        x[0] = t
        y = - val * val - compensation_xx[0]
        t = xx[0] + y
        compensation_xx[0] = t - xx[0] - y
        xx[0] = t
        y = - val * val * val - compensation_xxx[0]
        t = xxx[0] + y
        compensation_xxx[0] = t - xxx[0] - y
        xxx[0] = t
        y = - val * val * val * val - compensation_xxxx[0]
        t = xxxx[0] + y
        compensation_xxxx[0] = t - xxxx[0] - y
        xxxx[0] = t


def roll_kurt(ndarray[float64_t] values, ndarray[int64_t] start,
              ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j
        float64_t val, prev, mean_val, min_val, sum_val = 0
        float64_t compensation_xxxx_add = 0, compensation_xxxx_remove = 0
        float64_t compensation_xxx_remove = 0, compensation_xxx_add = 0
        float64_t compensation_xx_remove = 0, compensation_xx_add = 0
        float64_t compensation_x_remove = 0, compensation_x_add = 0
        float64_t x = 0, xx = 0, xxx = 0, xxxx = 0
        int64_t nobs = 0, s, e, N = len(values), nobs_mean = 0
        ndarray[float64_t] output, values_copy
        bint is_monotonic_increasing_bounds

    minp = max(minp, 4)
    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    output = np.empty(N, dtype=np.float64)
    values_copy = np.copy(values)
    min_val = np.nanmin(values)

    with nogil:
        for i in range(0, N):
            val = values_copy[i]
            if notnan(val):
                nobs_mean += 1
                sum_val += val
        mean_val = sum_val / nobs_mean
        # Other cases would lead to imprecision for smallest values
        if min_val - mean_val > -1e4:
            mean_val = round(mean_val)
            for i in range(0, N):
                values_copy[i] = values_copy[i] - mean_val

        for i in range(0, N):

            s = start[i]
            e = end[i]

            # Over the first window, observations can only be added
            # never removed
            if i == 0 or not is_monotonic_increasing_bounds:

                for j in range(s, e):
                    add_kurt(values_copy[j], &nobs, &x, &xx, &xxx, &xxxx,
                             &compensation_x_add, &compensation_xx_add,
                             &compensation_xxx_add, &compensation_xxxx_add)

            else:

                # After the first window, observations can both be added
                # and removed
                # calculate deletes
                for j in range(start[i - 1], s):
                    remove_kurt(values_copy[j], &nobs, &x, &xx, &xxx, &xxxx,
                                &compensation_x_remove, &compensation_xx_remove,
                                &compensation_xxx_remove, &compensation_xxxx_remove)

                # calculate adds
                for j in range(end[i - 1], e):
                    add_kurt(values_copy[j], &nobs, &x, &xx, &xxx, &xxxx,
                             &compensation_x_add, &compensation_xx_add,
                             &compensation_xxx_add, &compensation_xxxx_add)

            output[i] = calc_kurt(minp, nobs, x, xx, xxx, xxxx)

            if not is_monotonic_increasing_bounds:
                nobs = 0
                x = 0.0
                xx = 0.0
                xxx = 0.0
                xxxx = 0.0

    return output


# ----------------------------------------------------------------------
# Rolling median, min, max


def roll_median_c(const float64_t[:] values, ndarray[int64_t] start,
                  ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    cdef:
        Py_ssize_t i, j
        bint err = False, is_monotonic_increasing_bounds
        int midpoint, ret = 0
        int64_t nobs = 0, N = len(values), s, e, win
        float64_t val, res, prev
        skiplist_t *sl
        ndarray[float64_t] output

    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )

    # we use the Fixed/Variable Indexer here as the
    # actual skiplist ops outweigh any window computation costs
    output = np.empty(N, dtype=np.float64)

    if (end - start).max() == 0:
        output[:] = NaN
        return output
    win = (end - start).max()
    sl = skiplist_init(<int>win)
    if sl == NULL:
        raise MemoryError("skiplist_init failed")

    with nogil:

        for i in range(0, N):
            s = start[i]
            e = end[i]

            if i == 0 or not is_monotonic_increasing_bounds:

                # setup
                for j in range(s, e):
                    val = values[j]
                    if notnan(val):
                        nobs += 1
                        err = skiplist_insert(sl, val) != 1
                        if err:
                            break

            else:

                # calculate adds
                for j in range(end[i - 1], e):
                    val = values[j]
                    if notnan(val):
                        nobs += 1
                        err = skiplist_insert(sl, val) != 1
                        if err:
                            break

                # calculate deletes
                for j in range(start[i - 1], s):
                    val = values[j]
                    if notnan(val):
                        skiplist_remove(sl, val)
                        nobs -= 1
            if nobs >= minp:
                midpoint = <int>(nobs / 2)
                if nobs % 2:
                    res = skiplist_get(sl, midpoint, &ret)
                else:
                    res = (skiplist_get(sl, midpoint, &ret) +
                           skiplist_get(sl, (midpoint - 1), &ret)) / 2
                if ret == 0:
                    res = NaN
            else:
                res = NaN

            output[i] = res

            if not is_monotonic_increasing_bounds:
                nobs = 0
                skiplist_destroy(sl)
                sl = skiplist_init(<int>win)

    skiplist_destroy(sl)
    if err:
        raise MemoryError("skiplist_insert failed")
    return output


# ----------------------------------------------------------------------

# Moving maximum / minimum code taken from Bottleneck under the terms
# of its Simplified BSD license
# https://github.com/pydata/bottleneck


cdef inline numeric init_mm(numeric ai, Py_ssize_t *nobs, bint is_max) nogil:

    if numeric in cython.floating:
        if ai == ai:
            nobs[0] = nobs[0] + 1
        elif is_max:
            if numeric == cython.float:
                ai = MINfloat32
            else:
                ai = MINfloat64
        else:
            if numeric == cython.float:
                ai = MAXfloat32
            else:
                ai = MAXfloat64

    else:
        nobs[0] = nobs[0] + 1

    return ai


cdef inline void remove_mm(numeric aold, Py_ssize_t *nobs) nogil:
    """ remove a value from the mm calc """
    if numeric in cython.floating and aold == aold:
        nobs[0] = nobs[0] - 1


cdef inline numeric calc_mm(int64_t minp, Py_ssize_t nobs,
                            numeric value) nogil:
    cdef:
        numeric result

    if numeric in cython.floating:
        if nobs >= minp:
            result = value
        else:
            result = NaN
    else:
        result = value

    return result


def roll_max(ndarray[float64_t] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    Moving max of 1d array of any numeric type along axis=0 ignoring NaNs.

    Parameters
    ----------
    values : np.ndarray[np.float64]
    window : int, size of rolling window
    minp : if number of observations in window
          is below this, output a NaN
    index : ndarray, optional
       index for window computation
    closed : 'right', 'left', 'both', 'neither'
            make the interval closed on the right, left,
            both or neither endpoints

    Returns
    -------
    np.ndarray[float]
    """
    return _roll_min_max(values, start, end, minp, is_max=1)


def roll_min(ndarray[float64_t] values, ndarray[int64_t] start,
             ndarray[int64_t] end, int64_t minp) -> np.ndarray:
    """
    Moving min of 1d array of any numeric type along axis=0 ignoring NaNs.

    Parameters
    ----------
    values : np.ndarray[np.float64]
    window : int, size of rolling window
    minp : if number of observations in window
          is below this, output a NaN
    index : ndarray, optional
       index for window computation

    Returns
    -------
    np.ndarray[float]
    """
    return _roll_min_max(values, start, end, minp, is_max=0)


cdef _roll_min_max(ndarray[numeric] values,
                   ndarray[int64_t] starti,
                   ndarray[int64_t] endi,
                   int64_t minp,
                   bint is_max):
    cdef:
        numeric ai
        int64_t curr_win_size, start
        Py_ssize_t i, k, nobs = 0, N = len(values)
        deque Q[int64_t]  # min/max always the front
        deque W[int64_t]  # track the whole window for nobs compute
        ndarray[float64_t, ndim=1] output

    output = np.empty(N, dtype=np.float64)
    Q = deque[int64_t]()
    W = deque[int64_t]()

    with nogil:

        # This is using a modified version of the C++ code in this
        # SO post: http://bit.ly/2nOoHlY
        # The original impl didn't deal with variable window sizes
        # So the code was optimized for that

        # first window's size
        curr_win_size = endi[0] - starti[0]
        # GH 32865
        # Anchor output index to values index to provide custom
        # BaseIndexer support
        for i in range(N):

            curr_win_size = endi[i] - starti[i]
            if i == 0:
                start = starti[i]
            else:
                start = endi[i - 1]

            for k in range(start, endi[i]):
                ai = init_mm(values[k], &nobs, is_max)
                # Discard previous entries if we find new min or max
                if is_max:
                    while not Q.empty() and ((ai >= values[Q.back()]) or
                                             values[Q.back()] != values[Q.back()]):
                        Q.pop_back()
                else:
                    while not Q.empty() and ((ai <= values[Q.back()]) or
                                             values[Q.back()] != values[Q.back()]):
                        Q.pop_back()
                Q.push_back(k)
                W.push_back(k)

            # Discard entries outside and left of current window
            while not Q.empty() and Q.front() <= starti[i] - 1:
                Q.pop_front()
            while not W.empty() and W.front() <= starti[i] - 1:
                remove_mm(values[W.front()], &nobs)
                W.pop_front()

            # Save output based on index in input value array
            if not Q.empty() and curr_win_size > 0:
                output[i] = calc_mm(minp, nobs, values[Q.front()])
            else:
                output[i] = NaN

    return output


cdef enum InterpolationType:
    LINEAR,
    LOWER,
    HIGHER,
    NEAREST,
    MIDPOINT


interpolation_types = {
    'linear': LINEAR,
    'lower': LOWER,
    'higher': HIGHER,
    'nearest': NEAREST,
    'midpoint': MIDPOINT,
}


def roll_quantile(const float64_t[:] values, ndarray[int64_t] start,
                  ndarray[int64_t] end, int64_t minp,
                  float64_t quantile, str interpolation) -> np.ndarray:
    """
    O(N log(window)) implementation using skip list
    """
    cdef:
        Py_ssize_t i, j, s, e, N = len(values), idx
        int ret = 0
        int64_t nobs = 0, win
        float64_t val, prev, midpoint, idx_with_fraction
        float64_t vlow, vhigh
        skiplist_t *skiplist
        InterpolationType interpolation_type
        ndarray[float64_t] output

    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError(f"quantile value {quantile} not in [0, 1]")

    try:
        interpolation_type = interpolation_types[interpolation]
    except KeyError:
        raise ValueError(f"Interpolation '{interpolation}' is not supported")

    is_monotonic_increasing_bounds = is_monotonic_increasing_start_end_bounds(
        start, end
    )
    # we use the Fixed/Variable Indexer here as the
    # actual skiplist ops outweigh any window computation costs
    output = np.empty(N, dtype=np.float64)

    win = (end - start).max()
    if win == 0:
        output[:] = NaN
        return output
    skiplist = skiplist_init(<int>win)
    if skiplist == NULL:
        raise MemoryError("skiplist_init failed")

    with nogil:
        for i in range(0, N):
            s = start[i]
            e = end[i]

            if i == 0 or not is_monotonic_increasing_bounds:
                if not is_monotonic_increasing_bounds:
                    nobs = 0
                    skiplist_destroy(skiplist)
                    skiplist = skiplist_init(<int>win)

                # setup
                for j in range(s, e):
                    val = values[j]
                    if notnan(val):
                        nobs += 1
                        skiplist_insert(skiplist, val)

            else:
                # calculate adds
                for j in range(end[i - 1], e):
                    val = values[j]
                    if notnan(val):
                        nobs += 1
                        skiplist_insert(skiplist, val)

                # calculate deletes
                for j in range(start[i - 1], s):
                    val = values[j]
                    if notnan(val):
                        skiplist_remove(skiplist, val)
                        nobs -= 1
            if nobs >= minp:
                if nobs == 1:
                    # Single value in skip list
                    output[i] = skiplist_get(skiplist, 0, &ret)
                else:
                    idx_with_fraction = quantile * (nobs - 1)
                    idx = <int>idx_with_fraction

                    if idx_with_fraction == idx:
                        # no need to interpolate
                        output[i] = skiplist_get(skiplist, idx, &ret)
                        continue

                    if interpolation_type == LINEAR:
                        vlow = skiplist_get(skiplist, idx, &ret)
                        vhigh = skiplist_get(skiplist, idx + 1, &ret)
                        output[i] = ((vlow + (vhigh - vlow) *
                                      (idx_with_fraction - idx)))
                    elif interpolation_type == LOWER:
                        output[i] = skiplist_get(skiplist, idx, &ret)
                    elif interpolation_type == HIGHER:
                        output[i] = skiplist_get(skiplist, idx + 1, &ret)
                    elif interpolation_type == NEAREST:
                        # the same behaviour as round()
                        if idx_with_fraction - idx == 0.5:
                            if idx % 2 == 0:
                                output[i] = skiplist_get(skiplist, idx, &ret)
                            else:
                                output[i] = skiplist_get(
                                    skiplist, idx + 1, &ret)
                        elif idx_with_fraction - idx < 0.5:
                            output[i] = skiplist_get(skiplist, idx, &ret)
                        else:
                            output[i] = skiplist_get(skiplist, idx + 1, &ret)
                    elif interpolation_type == MIDPOINT:
                        vlow = skiplist_get(skiplist, idx, &ret)
                        vhigh = skiplist_get(skiplist, idx + 1, &ret)
                        output[i] = <float64_t>(vlow + vhigh) / 2

                    if ret == 0:
                        output[i] = NaN
            else:
                output[i] = NaN

    skiplist_destroy(skiplist)

    return output


def roll_apply(object obj,
               ndarray[int64_t] start, ndarray[int64_t] end,
               int64_t minp,
               object function, bint raw,
               tuple args, dict kwargs) -> np.ndarray:
    cdef:
        ndarray[float64_t] output, counts
        ndarray[float64_t, cast=True] arr
        Py_ssize_t i, s, e, N = len(start), n = len(obj)

    if n == 0:
        return obj

    arr = np.asarray(obj)

    # ndarray input
    if raw and not arr.flags.c_contiguous:
        arr = arr.copy('C')

    counts = roll_sum(np.isfinite(arr).astype(float), start, end, minp)

    output = np.empty(N, dtype=np.float64)

    for i in range(N):

        s = start[i]
        e = end[i]

        if counts[i] >= minp:
            if raw:
                output[i] = function(arr[s:e], *args, **kwargs)
            else:
                output[i] = function(obj.iloc[s:e], *args, **kwargs)
        else:
            output[i] = NaN

    return output


# ----------------------------------------------------------------------
# Rolling sum and mean for weighted window


def roll_weighted_sum(
    const float64_t[:] values, const float64_t[:] weights, int minp
) -> np.ndaray:
    return _roll_weighted_sum_mean(values, weights, minp, avg=0)


def roll_weighted_mean(
    const float64_t[:] values, const float64_t[:] weights, int minp
) -> np.ndaray:
    return _roll_weighted_sum_mean(values, weights, minp, avg=1)


cdef float64_t[:] _roll_weighted_sum_mean(const float64_t[:] values,
                                          const float64_t[:] weights,
                                          int minp, bint avg):
    """
    Assume len(weights) << len(values)
    """
    cdef:
        float64_t[:] output, tot_wgt, counts
        Py_ssize_t in_i, win_i, win_n, in_n
        float64_t val_in, val_win, c, w

    in_n = len(values)
    win_n = len(weights)

    output = np.zeros(in_n, dtype=np.float64)
    counts = np.zeros(in_n, dtype=np.float64)
    if avg:
        tot_wgt = np.zeros(in_n, dtype=np.float64)

    elif minp > in_n:
        minp = in_n + 1

    minp = max(minp, 1)

    with nogil:
        if avg:
            for win_i in range(win_n):
                val_win = weights[win_i]
                if val_win != val_win:
                    continue

                for in_i in range(in_n - (win_n - win_i) + 1):
                    val_in = values[in_i]
                    if val_in == val_in:
                        output[in_i + (win_n - win_i) - 1] += val_in * val_win
                        counts[in_i + (win_n - win_i) - 1] += 1
                        tot_wgt[in_i + (win_n - win_i) - 1] += val_win

            for in_i in range(in_n):
                c = counts[in_i]
                if c < minp:
                    output[in_i] = NaN
                else:
                    w = tot_wgt[in_i]
                    if w == 0:
                        output[in_i] = NaN
                    else:
                        output[in_i] /= tot_wgt[in_i]

        else:
            for win_i in range(win_n):
                val_win = weights[win_i]
                if val_win != val_win:
                    continue

                for in_i in range(in_n - (win_n - win_i) + 1):
                    val_in = values[in_i]

                    if val_in == val_in:
                        output[in_i + (win_n - win_i) - 1] += val_in * val_win
                        counts[in_i + (win_n - win_i) - 1] += 1

            for in_i in range(in_n):
                c = counts[in_i]
                if c < minp:
                    output[in_i] = NaN

    return output


# ----------------------------------------------------------------------
# Rolling var for weighted window


cdef inline float64_t calc_weighted_var(float64_t t,
                                        float64_t sum_w,
                                        Py_ssize_t win_n,
                                        unsigned int ddof,
                                        float64_t nobs,
                                        int64_t minp) nogil:
    """
    Calculate weighted variance for a window using West's method.

    Paper: https://dl.acm.org/citation.cfm?id=359153

    Parameters
    ----------
    t: float64_t
        sum of weighted squared differences
    sum_w: float64_t
        sum of weights
    win_n: Py_ssize_t
        window size
    ddof: unsigned int
        delta degrees of freedom
    nobs: float64_t
        number of observations
    minp: int64_t
        minimum number of observations

    Returns
    -------
    result : float64_t
        weighted variance of the window
    """

    cdef:
        float64_t result

    # Variance is unchanged if no observation is added or removed
    if (nobs >= minp) and (nobs > ddof):

        # pathological case
        if nobs == 1:
            result = 0
        else:
            result = t * win_n / ((win_n - ddof) * sum_w)
            if result < 0:
                result = 0
    else:
        result = NaN

    return result


cdef inline void add_weighted_var(float64_t val,
                                  float64_t w,
                                  float64_t *t,
                                  float64_t *sum_w,
                                  float64_t *mean,
                                  float64_t *nobs) nogil:
    """
    Update weighted mean, sum of weights and sum of weighted squared
    differences to include value and weight pair in weighted variance
    calculation using West's method.

    Paper: https://dl.acm.org/citation.cfm?id=359153

    Parameters
    ----------
    val: float64_t
        window values
    w: float64_t
        window weights
    t: float64_t
        sum of weighted squared differences
    sum_w: float64_t
        sum of weights
    mean: float64_t
        weighted mean
    nobs: float64_t
        number of observations
    """

    cdef:
        float64_t temp, q, r

    if isnan(val):
        return

    nobs[0] = nobs[0] + 1

    q = val - mean[0]
    temp = sum_w[0] + w
    r = q * w / temp

    mean[0] = mean[0] + r
    t[0] = t[0] + r * sum_w[0] * q
    sum_w[0] = temp


cdef inline void remove_weighted_var(float64_t val,
                                     float64_t w,
                                     float64_t *t,
                                     float64_t *sum_w,
                                     float64_t *mean,
                                     float64_t *nobs) nogil:
    """
    Update weighted mean, sum of weights and sum of weighted squared
    differences to remove value and weight pair from weighted variance
    calculation using West's method.

    Paper: https://dl.acm.org/citation.cfm?id=359153

    Parameters
    ----------
    val: float64_t
        window values
    w: float64_t
        window weights
    t: float64_t
        sum of weighted squared differences
    sum_w: float64_t
        sum of weights
    mean: float64_t
        weighted mean
    nobs: float64_t
        number of observations
    """

    cdef:
        float64_t temp, q, r

    if notnan(val):
        nobs[0] = nobs[0] - 1

        if nobs[0]:
            q = val - mean[0]
            temp = sum_w[0] - w
            r = q * w / temp

            mean[0] = mean[0] - r
            t[0] = t[0] - r * sum_w[0] * q
            sum_w[0] = temp

        else:
            t[0] = 0
            sum_w[0] = 0
            mean[0] = 0


def roll_weighted_var(const float64_t[:] values, const float64_t[:] weights,
                      int64_t minp, unsigned int ddof):
    """
    Calculates weighted rolling variance using West's online algorithm.

    Paper: https://dl.acm.org/citation.cfm?id=359153

    Parameters
    ----------
    values: float64_t[:]
        values to roll window over
    weights: float64_t[:]
        array of weights whose length is window size
    minp: int64_t
        minimum number of observations to calculate
        variance of a window
    ddof: unsigned int
         the divisor used in variance calculations
         is the window size - ddof

    Returns
    -------
    output: float64_t[:]
        weighted variances of windows
    """

    cdef:
        float64_t t = 0, sum_w = 0, mean = 0, nobs = 0
        float64_t val, pre_val, w, pre_w
        Py_ssize_t i, n, win_n
        float64_t[:] output

    n = len(values)
    win_n = len(weights)
    output = np.empty(n, dtype=np.float64)

    with nogil:

        for i in range(win_n):
            add_weighted_var(values[i], weights[i], &t,
                             &sum_w, &mean, &nobs)

            output[i] = calc_weighted_var(t, sum_w, win_n,
                                          ddof, nobs, minp)

        for i in range(win_n, n):
            val = values[i]
            pre_val = values[i - win_n]

            w = weights[i % win_n]
            pre_w = weights[(i - win_n) % win_n]

            if notnan(val):
                if pre_val == pre_val:
                    remove_weighted_var(pre_val, pre_w, &t,
                                        &sum_w, &mean, &nobs)

                add_weighted_var(val, w, &t, &sum_w, &mean, &nobs)

            elif pre_val == pre_val:
                remove_weighted_var(pre_val, pre_w, &t,
                                    &sum_w, &mean, &nobs)

            output[i] = calc_weighted_var(t, sum_w, win_n,
                                          ddof, nobs, minp)

    return output


# ----------------------------------------------------------------------
# Exponentially weighted moving average

def ewma(const float64_t[:] vals, const int64_t[:] start, const int64_t[:] end,
         int minp, float64_t com, bint adjust, bint ignore_na,
         const float64_t[:] deltas=None) -> np.ndarray:
    """
    Compute exponentially-weighted moving average using center-of-mass.

    Parameters
    ----------
    vals : ndarray (float64 type)
    start: ndarray (int64 type)
    end: ndarray (int64 type)
    minp : int
    com : float64
    adjust : bool
    ignore_na : bool
    deltas : ndarray (float64 type), optional. If None, implicitly assumes equally
             spaced points (used when `times` is not passed)

    Returns
    -------
    np.ndarray[float64_t]
    """

    cdef:
        Py_ssize_t i, j, s, e, nobs, win_size, N = len(vals), M = len(start)
        const float64_t[:] sub_vals
        const float64_t[:] sub_deltas=None
        ndarray[float64_t] sub_output, output = np.empty(N, dtype=np.float64)
        float64_t alpha, old_wt_factor, new_wt, weighted_avg, old_wt, cur
        bint is_observation, use_deltas

    if N == 0:
        return output

    use_deltas = deltas is not None

    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha

    for j in range(M):
        s = start[j]
        e = end[j]
        sub_vals = vals[s:e]
        # note that len(deltas) = len(vals) - 1 and deltas[i] is to be used in
        # conjunction with vals[i+1]
        if use_deltas:
            sub_deltas = deltas[s:e - 1]
        win_size = len(sub_vals)
        sub_output = np.empty(win_size, dtype=np.float64)

        weighted_avg = sub_vals[0]
        is_observation = weighted_avg == weighted_avg
        nobs = int(is_observation)
        sub_output[0] = weighted_avg if nobs >= minp else NaN
        old_wt = 1.

        with nogil:
            for i in range(1, win_size):
                cur = sub_vals[i]
                is_observation = cur == cur
                nobs += is_observation
                if weighted_avg == weighted_avg:

                    if is_observation or not ignore_na:
                        if use_deltas:
                            old_wt *= old_wt_factor ** sub_deltas[i - 1]
                        else:
                            old_wt *= old_wt_factor
                        if is_observation:

                            # avoid numerical errors on constant series
                            if weighted_avg != cur:
                                weighted_avg = ((old_wt * weighted_avg) +
                                                (new_wt * cur)) / (old_wt + new_wt)
                            if adjust:
                                old_wt += new_wt
                            else:
                                old_wt = 1.
                elif is_observation:
                    weighted_avg = cur

                sub_output[i] = weighted_avg if nobs >= minp else NaN

        output[s:e] = sub_output

    return output


# ----------------------------------------------------------------------
# Exponentially weighted moving covariance


def ewmcov(const float64_t[:] input_x, const int64_t[:] start, const int64_t[:] end,
           int minp, const float64_t[:] input_y, float64_t com, bint adjust,
           bint ignore_na, bint bias) -> np.ndarray:
    """
    Compute exponentially-weighted moving variance using center-of-mass.

    Parameters
    ----------
    input_x : ndarray (float64 type)
    start: ndarray (int64 type)
    end: ndarray (int64 type)
    minp : int
    input_y : ndarray (float64 type)
    com : float64
    adjust : bool
    ignore_na : bool
    bias : bool

    Returns
    -------
    np.ndarray[float64_t]
    """

    cdef:
        Py_ssize_t i, j, s, e, win_size, nobs
        Py_ssize_t N = len(input_x), M = len(input_y), L = len(start)
        float64_t alpha, old_wt_factor, new_wt, mean_x, mean_y, cov
        float64_t sum_wt, sum_wt2, old_wt, cur_x, cur_y, old_mean_x, old_mean_y
        float64_t numerator, denominator
        const float64_t[:] sub_x_vals, sub_y_vals
        ndarray[float64_t] sub_out, output = np.empty(N, dtype=np.float64)
        bint is_observation

    if M != N:
        raise ValueError(f"arrays are of different lengths ({N} and {M})")

    if N == 0:
        return output

    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha

    for j in range(L):
        s = start[j]
        e = end[j]
        sub_x_vals = input_x[s:e]
        sub_y_vals = input_y[s:e]
        win_size = len(sub_x_vals)
        sub_out = np.empty(win_size, dtype=np.float64)

        mean_x = sub_x_vals[0]
        mean_y = sub_y_vals[0]
        is_observation = (mean_x == mean_x) and (mean_y == mean_y)
        nobs = int(is_observation)
        if not is_observation:
            mean_x = NaN
            mean_y = NaN
        sub_out[0] = (0. if bias else NaN) if nobs >= minp else NaN
        cov = 0.
        sum_wt = 1.
        sum_wt2 = 1.
        old_wt = 1.

        with nogil:
            for i in range(1, win_size):
                cur_x = sub_x_vals[i]
                cur_y = sub_y_vals[i]
                is_observation = (cur_x == cur_x) and (cur_y == cur_y)
                nobs += is_observation
                if mean_x == mean_x:
                    if is_observation or not ignore_na:
                        sum_wt *= old_wt_factor
                        sum_wt2 *= (old_wt_factor * old_wt_factor)
                        old_wt *= old_wt_factor
                        if is_observation:
                            old_mean_x = mean_x
                            old_mean_y = mean_y

                            # avoid numerical errors on constant series
                            if mean_x != cur_x:
                                mean_x = ((old_wt * old_mean_x) +
                                          (new_wt * cur_x)) / (old_wt + new_wt)

                            # avoid numerical errors on constant series
                            if mean_y != cur_y:
                                mean_y = ((old_wt * old_mean_y) +
                                          (new_wt * cur_y)) / (old_wt + new_wt)
                            cov = ((old_wt * (cov + ((old_mean_x - mean_x) *
                                                     (old_mean_y - mean_y)))) +
                                   (new_wt * ((cur_x - mean_x) *
                                              (cur_y - mean_y)))) / (old_wt + new_wt)
                            sum_wt += new_wt
                            sum_wt2 += (new_wt * new_wt)
                            old_wt += new_wt
                            if not adjust:
                                sum_wt /= old_wt
                                sum_wt2 /= (old_wt * old_wt)
                                old_wt = 1.
                elif is_observation:
                    mean_x = cur_x
                    mean_y = cur_y

                if nobs >= minp:
                    if not bias:
                        numerator = sum_wt * sum_wt
                        denominator = numerator - sum_wt2
                        if denominator > 0:
                            sub_out[i] = (numerator / denominator) * cov
                        else:
                            sub_out[i] = NaN
                    else:
                        sub_out[i] = cov
                else:
                    sub_out[i] = NaN

        output[s:e] = sub_out

    return output
