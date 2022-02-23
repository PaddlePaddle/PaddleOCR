""" Utility functions for sparse matrix module
"""

import sys
import operator
import warnings
import numpy as np
from scipy._lib._util import prod

__all__ = ['upcast', 'getdtype', 'getdata', 'isscalarlike', 'isintlike',
           'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype']

supported_dtypes = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc,
                    np.uintc, np.int_, np.uint, np.longlong, np.ulonglong, np.single, np.double,
                    np.longdouble, np.csingle, np.cdouble, np.clongdouble]

_upcast_memo = {}


def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples
    --------

    >>> upcast('int32')
    <type 'numpy.int32'>
    >>> upcast('bool')
    <type 'numpy.bool_'>
    >>> upcast('int32','float32')
    <type 'numpy.float64'>
    >>> upcast('bool',complex,float)
    <type 'numpy.complex128'>

    """

    t = _upcast_memo.get(hash(args))
    if t is not None:
        return t

    upcast = np.find_common_type(args, [])

    for t in supported_dtypes:
        if np.can_cast(upcast, t):
            _upcast_memo[hash(args)] = t
            return t

    raise TypeError('no supported conversion for types: %r' % (args,))


def upcast_char(*args):
    """Same as `upcast` but taking dtype.char as input (faster)."""
    t = _upcast_memo.get(args)
    if t is not None:
        return t
    t = upcast(*map(np.dtype, args))
    _upcast_memo[args] = t
    return t


def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    return (np.array([0], dtype=dtype) * scalar).dtype


def downcast_intp_index(arr):
    """
    Down-cast index array to np.intp dtype if it is of a larger dtype.

    Raise an error if the array contains a value that is too large for
    intp.
    """
    if arr.dtype.itemsize > np.dtype(np.intp).itemsize:
        if arr.size == 0:
            return arr.astype(np.intp)
        maxval = arr.max()
        minval = arr.min()
        if maxval > np.iinfo(np.intp).max or minval < np.iinfo(np.intp).min:
            raise ValueError("Cannot deal with arrays with indices larger "
                             "than the machine maximum address size "
                             "(e.g. 64-bit indices on 32-bit machine).")
        return arr.astype(np.intp)
    return arr


def to_native(A):
    return np.asarray(A, dtype=A.dtype.newbyteorder('native'))


def getdtype(dtype, a=None, default=None):
    """Function used to simplify argument processing. If 'dtype' is not
    specified (is None), returns a.dtype; otherwise returns a np.dtype
    object created from the specified dtype argument. If 'dtype' and 'a'
    are both None, construct a data type out of the 'default' parameter.
    Furthermore, 'dtype' must be in 'allowed' set.
    """
    # TODO is this really what we want?
    if dtype is None:
        try:
            newdtype = a.dtype
        except AttributeError as e:
            if default is not None:
                newdtype = np.dtype(default)
            else:
                raise TypeError("could not interpret data type") from e
    else:
        newdtype = np.dtype(dtype)
        if newdtype == np.object_:
            warnings.warn("object dtype is not supported by sparse matrices")

    return newdtype


def getdata(obj, dtype=None, copy=False):
    """
    This is a wrapper of `np.array(obj, dtype=dtype, copy=copy)`
    that will generate a warning if the result is an object array.
    """
    data = np.array(obj, dtype=dtype, copy=copy)
    # Defer to getdtype for checking that the dtype is OK.
    # This is called for the validation only; we don't need the return value.
    getdtype(data.dtype)
    return data


def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    """

    int32min = np.iinfo(np.int32).min
    int32max = np.iinfo(np.int32).max

    dtype = np.intc
    if maxval is not None:
        if maxval > int32max:
            dtype = np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = np.asarray(arr)
        if not np.can_cast(arr.dtype, np.int32):
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        # a bigger type not needed
                        continue

            dtype = np.int64
            break

    return dtype


def get_sum_dtype(dtype):
    """Mimic numpy's casting for np.sum"""
    if dtype.kind == 'u' and np.can_cast(dtype, np.uint):
        return np.uint
    if np.can_cast(dtype, np.int_):
        return np.int_
    return dtype


def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


def isintlike(x):
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    # Fast-path check to eliminate non-scalar values. operator.index would
    # catch this case too, but the exception catching is slow.
    if np.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)
        except (TypeError, ValueError):
            return False
        if loose_int:
            warnings.warn("Inexact indices into sparse matrices are deprecated",
                          DeprecationWarning)
        return loose_int
    return True


def isshape(x, nonneg=False):
    """Is x a valid 2-tuple of dimensions?

    If nonneg, also checks that the dimensions are non-negative.
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except Exception:
        return False
    else:
        if isintlike(M) and isintlike(N):
            if np.ndim(M) == 0 and np.ndim(N) == 0:
                if not nonneg or (M >= 0 and N >= 0):
                    return True
        return False


def issequence(t):
    return ((isinstance(t, (list, tuple)) and
            (len(t) == 0 or np.isscalar(t[0]))) or
            (isinstance(t, np.ndarray) and (t.ndim == 1)))


def ismatrix(t):
    return ((isinstance(t, (list, tuple)) and
             len(t) > 0 and issequence(t[0])) or
            (isinstance(t, np.ndarray) and t.ndim == 2))


def isdense(x):
    return isinstance(x, np.ndarray)


def validateaxis(axis):
    if axis is not None:
        axis_type = type(axis)

        # In NumPy, you can pass in tuples for 'axis', but they are
        # not very useful for sparse matrices given their limited
        # dimensions, so let's make it explicit that they are not
        # allowed to be passed in
        if axis_type == tuple:
            raise TypeError(("Tuples are not accepted for the 'axis' "
                             "parameter. Please pass in one of the "
                             "following: {-2, -1, 0, 1, None}."))

        # If not a tuple, check that the provided axis is actually
        # an integer and raise a TypeError similar to NumPy's
        if not np.issubdtype(np.dtype(axis_type), np.integer):
            raise TypeError("axis must be an integer, not {name}"
                            .format(name=axis_type.__name__))

        if not (-2 <= axis <= 1):
            raise ValueError("axis out of range")


def check_shape(args, current_shape=None):
    """Imitate numpy.matrix handling of shape arguments"""
    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: "
                        "'shape'")
    elif len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]), )
        else:
            new_shape = tuple(operator.index(arg) for arg in shape_iter)
    else:
        new_shape = tuple(operator.index(arg) for arg in args)

    if current_shape is None:
        if len(new_shape) != 2:
            raise ValueError('shape must be a 2-tuple of positive integers')
        elif new_shape[0] < 0 or new_shape[1] < 0:
            raise ValueError("'shape' elements cannot be negative")

    else:
        # Check the current size only if needed
        current_size = prod(current_shape)

        # Check for negatives
        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if len(negative_indexes) == 0:
            new_size = prod(new_shape)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 .format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = prod(new_shape[0:skip] + new_shape[skip+1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple('newshape' if x < 0 else x for x in new_shape)
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 ''.format(current_size, err_shape))
            new_shape = new_shape[0:skip] + (unspecified,) + new_shape[skip+1:]
        else:
            raise ValueError('can only specify one unknown dimension')

    if len(new_shape) != 2:
        raise ValueError('matrix shape must be two-dimensional')

    return new_shape


def check_reshape_kwargs(kwargs):
    """Unpack keyword arguments for reshape function.

    This is useful because keyword arguments after star arguments are not
    allowed in Python 2, but star keyword arguments are. This function unpacks
    'order' and 'copy' from the star keyword arguments (with defaults) and
    throws an error for any remaining.
    """

    order = kwargs.pop('order', 'C')
    copy = kwargs.pop('copy', False)
    if kwargs:  # Some unused kwargs remain
        raise TypeError('reshape() got unexpected keywords arguments: {}'
                        .format(', '.join(kwargs.keys())))
    return order, copy


def is_pydata_spmatrix(m):
    """
    Check whether object is pydata/sparse matrix, avoiding importing the module.
    """
    base_cls = getattr(sys.modules.get('sparse'), 'SparseArray', None)
    return base_cls is not None and isinstance(m, base_cls)


###############################################################################
# Wrappers for NumPy types that are deprecated

# Numpy versions of these functions raise deprecation warnings, the
# ones below do not.


def matrix(*args, **kwargs):
    return np.array(*args, **kwargs).view(np.matrix)


def asmatrix(data, dtype=None):
    if isinstance(data, np.matrix) and (dtype is None or data.dtype == dtype):
        return data
    return np.asarray(data, dtype=dtype).view(np.matrix)
