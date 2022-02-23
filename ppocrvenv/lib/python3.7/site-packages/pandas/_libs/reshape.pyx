import cython
from cython import Py_ssize_t

from numpy cimport (
    float32_t,
    float64_t,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    ndarray,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

import numpy as np

cimport numpy as cnp

cnp.import_array()

from pandas._libs.lib cimport c_is_list_like

ctypedef fused reshape_t:
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    int8_t
    int16_t
    int32_t
    int64_t
    float32_t
    float64_t
    object


@cython.wraparound(False)
@cython.boundscheck(False)
def unstack(reshape_t[:, :] values, const uint8_t[:] mask,
            Py_ssize_t stride, Py_ssize_t length, Py_ssize_t width,
            reshape_t[:, :] new_values, uint8_t[:, :] new_mask) -> None:
    """
    Transform long values to wide new_values.

    Parameters
    ----------
    values : typed ndarray
    mask : np.ndarray[bool]
    stride : int
    length : int
    width : int
    new_values : np.ndarray[bool]
        result array
    new_mask : np.ndarray[bool]
        result mask
    """
    cdef:
        Py_ssize_t i, j, w, nulls, s, offset

    if reshape_t is not object:
        # evaluated at compile-time
        with nogil:
            for i in range(stride):

                nulls = 0
                for j in range(length):

                    for w in range(width):

                        offset = j * width + w

                        if mask[offset]:
                            s = i * width + w
                            new_values[j, s] = values[offset - nulls, i]
                            new_mask[j, s] = 1
                        else:
                            nulls += 1

    else:
        # object-dtype, identical to above but we cannot use nogil
        for i in range(stride):

            nulls = 0
            for j in range(length):

                for w in range(width):

                    offset = j * width + w

                    if mask[offset]:
                        s = i * width + w
                        new_values[j, s] = values[offset - nulls, i]
                        new_mask[j, s] = 1
                    else:
                        nulls += 1


@cython.wraparound(False)
@cython.boundscheck(False)
def explode(ndarray[object] values):
    """
    transform array list-likes to long form
    preserve non-list entries

    Parameters
    ----------
    values : object ndarray

    Returns
    -------
    ndarray[object]
        result
    ndarray[int64_t]
        counts
    """
    cdef:
        Py_ssize_t i, j, count, n
        object v
        ndarray[object] result
        ndarray[int64_t] counts

    # find the resulting len
    n = len(values)
    counts = np.zeros(n, dtype='int64')
    for i in range(n):
        v = values[i]

        if c_is_list_like(v, True):
            if len(v):
                counts[i] += len(v)
            else:
                # empty list-like, use a nan marker
                counts[i] += 1
        else:
            counts[i] += 1

    result = np.empty(counts.sum(), dtype='object')
    count = 0
    for i in range(n):
        v = values[i]

        if c_is_list_like(v, True):
            if len(v):
                v = list(v)
                for j in range(len(v)):
                    result[count] = v[j]
                    count += 1
            else:
                # empty list-like, use a nan marker
                result[count] = np.nan
                count += 1
        else:
            # replace with the existing scalar
            result[count] = v
            count += 1
    return result, counts
