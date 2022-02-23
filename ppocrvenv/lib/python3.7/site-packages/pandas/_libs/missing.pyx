from decimal import Decimal
import numbers
from sys import maxsize

import cython
from cython import Py_ssize_t
import numpy as np

cimport numpy as cnp
from numpy cimport (
    float64_t,
    int64_t,
    ndarray,
    uint8_t,
)

cnp.import_array()

from pandas._libs cimport util
from pandas._libs.tslibs.nattype cimport (
    c_NaT as NaT,
    checknull_with_nat,
    is_null_datetimelike,
)
from pandas._libs.tslibs.np_datetime cimport (
    get_datetime64_value,
    get_timedelta64_value,
)

from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

cdef:
    float64_t INF = <float64_t>np.inf
    float64_t NEGINF = -INF

    int64_t NPY_NAT = util.get_nat()

    bint is_32bit = maxsize <= 2 ** 32

    type cDecimal = Decimal  # for faster isinstance checks


cpdef bint is_matching_na(object left, object right, bint nan_matches_none=False):
    """
    Check if two scalars are both NA of matching types.

    Parameters
    ----------
    left : Any
    right : Any
    nan_matches_none : bool, default False
        For backwards compatibility, consider NaN as matching None.

    Returns
    -------
    bool
    """
    if left is None:
        if nan_matches_none and util.is_nan(right):
            return True
        return right is None
    elif left is C_NA:
        return right is C_NA
    elif left is NaT:
        return right is NaT
    elif util.is_float_object(left):
        if nan_matches_none and right is None and util.is_nan(left):
            return True
        return (
            util.is_nan(left)
            and util.is_float_object(right)
            and util.is_nan(right)
        )
    elif util.is_complex_object(left):
        return (
            util.is_nan(left)
            and util.is_complex_object(right)
            and util.is_nan(right)
        )
    elif util.is_datetime64_object(left):
        return (
            get_datetime64_value(left) == NPY_NAT
            and util.is_datetime64_object(right)
            and get_datetime64_value(right) == NPY_NAT
        )
    elif util.is_timedelta64_object(left):
        return (
            get_timedelta64_value(left) == NPY_NAT
            and util.is_timedelta64_object(right)
            and get_timedelta64_value(right) == NPY_NAT
        )
    elif is_decimal_na(left):
        return is_decimal_na(right)
    return False


cpdef bint checknull(object val):
    """
    Return boolean describing of the input is NA-like, defined here as any
    of:
     - None
     - nan
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    val : object

    Returns
    -------
    bool

    Notes
    -----
    The difference between `checknull` and `checknull_old` is that `checknull`
    does *not* consider INF or NEGINF to be NA.
    """
    return (
        val is C_NA
        or is_null_datetimelike(val, inat_is_null=False)
        or is_decimal_na(val)
    )


cdef inline bint is_decimal_na(object val):
    """
    Is this a decimal.Decimal object Decimal("NAN").
    """
    return isinstance(val, cDecimal) and val != val


cpdef bint checknull_old(object val):
    """
    Return boolean describing of the input is NA-like, defined here as any
    of:
     - None
     - nan
     - INF
     - NEGINF
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    val : object

    Returns
    -------
    result : bool

    Notes
    -----
    The difference between `checknull` and `checknull_old` is that `checknull`
    does *not* consider INF or NEGINF to be NA.
    """
    if checknull(val):
        return True
    elif util.is_float_object(val) or util.is_complex_object(val):
        return val == INF or val == NEGINF
    return False


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[uint8_t] isnaobj(ndarray arr):
    """
    Return boolean mask denoting which elements of a 1-D array are na-like,
    according to the criteria defined in `checknull`:
     - None
     - nan
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    result : ndarray (dtype=np.bool_)
    """
    cdef:
        Py_ssize_t i, n
        object val
        ndarray[uint8_t] result

    assert arr.ndim == 1, "'arr' must be 1-D."

    n = len(arr)
    result = np.empty(n, dtype=np.uint8)
    for i in range(n):
        val = arr[i]
        result[i] = checknull(val)
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
def isnaobj_old(arr: ndarray) -> ndarray:
    """
    Return boolean mask denoting which elements of a 1-D array are na-like,
    defined as being any of:
     - None
     - nan
     - INF
     - NEGINF
     - NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    result : ndarray (dtype=np.bool_)
    """
    cdef:
        Py_ssize_t i, n
        object val
        ndarray[uint8_t] result

    assert arr.ndim == 1, "'arr' must be 1-D."

    n = len(arr)
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        val = arr[i]
        result[i] = (
            checknull(val)
            or util.is_float_object(val) and (val == INF or val == NEGINF)
        )
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
def isnaobj2d(arr: ndarray) -> ndarray:
    """
    Return boolean mask denoting which elements of a 2-D array are na-like,
    according to the criteria defined in `checknull`:
     - None
     - nan
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    result : ndarray (dtype=np.bool_)

    Notes
    -----
    The difference between `isnaobj2d` and `isnaobj2d_old` is that `isnaobj2d`
    does *not* consider INF or NEGINF to be NA.
    """
    cdef:
        Py_ssize_t i, j, n, m
        object val
        ndarray[uint8_t, ndim=2] result

    assert arr.ndim == 2, "'arr' must be 2-D."

    n, m = (<object>arr).shape
    result = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            val = arr[i, j]
            if checknull(val):
                result[i, j] = 1
    return result.view(np.bool_)


@cython.wraparound(False)
@cython.boundscheck(False)
def isnaobj2d_old(arr: ndarray) -> ndarray:
    """
    Return boolean mask denoting which elements of a 2-D array are na-like,
    according to the criteria defined in `checknull_old`:
     - None
     - nan
     - INF
     - NEGINF
     - NaT
     - np.datetime64 representation of NaT
     - np.timedelta64 representation of NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    ndarray (dtype=np.bool_)

    Notes
    -----
    The difference between `isnaobj2d` and `isnaobj2d_old` is that `isnaobj2d`
    does *not* consider INF or NEGINF to be NA.
    """
    cdef:
        Py_ssize_t i, j, n, m
        object val
        ndarray[uint8_t, ndim=2] result

    assert arr.ndim == 2, "'arr' must be 2-D."

    n, m = (<object>arr).shape
    result = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            val = arr[i, j]
            if checknull_old(val):
                result[i, j] = 1
    return result.view(np.bool_)


def isposinf_scalar(val: object) -> bool:
    return util.is_float_object(val) and val == INF


def isneginf_scalar(val: object) -> bool:
    return util.is_float_object(val) and val == NEGINF


cdef inline bint is_null_datetime64(v):
    # determine if we have a null for a datetime (or integer versions),
    # excluding np.timedelta64('nat')
    if checknull_with_nat(v):
        return True
    elif util.is_datetime64_object(v):
        return get_datetime64_value(v) == NPY_NAT
    return False


cdef inline bint is_null_timedelta64(v):
    # determine if we have a null for a timedelta (or integer versions),
    # excluding np.datetime64('nat')
    if checknull_with_nat(v):
        return True
    elif util.is_timedelta64_object(v):
        return get_timedelta64_value(v) == NPY_NAT
    return False


cdef bint checknull_with_nat_and_na(object obj):
    # See GH#32214
    return checknull_with_nat(obj) or obj is C_NA


# -----------------------------------------------------------------------------
# Implementation of NA singleton


def _create_binary_propagating_op(name, is_divmod=False):

    def method(self, other):
        if (other is C_NA or isinstance(other, str)
                or isinstance(other, (numbers.Number, np.bool_))
                or isinstance(other, np.ndarray) and not other.shape):
            # Need the other.shape clause to handle NumPy scalars,
            # since we do a setitem on `out` below, which
            # won't work for NumPy scalars.
            if is_divmod:
                return NA, NA
            else:
                return NA

        elif isinstance(other, np.ndarray):
            out = np.empty(other.shape, dtype=object)
            out[:] = NA

            if is_divmod:
                return out, out.copy()
            else:
                return out

        return NotImplemented

    method.__name__ = name
    return method


def _create_unary_propagating_op(name: str):
    def method(self):
        return NA

    method.__name__ = name
    return method


cdef class C_NAType:
    pass


class NAType(C_NAType):
    """
    NA ("not available") missing value indicator.

    .. warning::

       Experimental: the behaviour of NA can still change without warning.

    .. versionadded:: 1.0.0

    The NA singleton is a missing value indicator defined by pandas. It is
    used in certain new extension dtypes (currently the "string" dtype).
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if NAType._instance is None:
            NAType._instance = C_NAType.__new__(cls, *args, **kwargs)
        return NAType._instance

    def __repr__(self) -> str:
        return "<NA>"

    def __format__(self, format_spec) -> str:
        try:
            return self.__repr__().__format__(format_spec)
        except ValueError:
            return self.__repr__()

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")

    def __hash__(self):
        # GH 30013: Ensure hash is large enough to avoid hash collisions with integers
        exponent = 31 if is_32bit else 61
        return 2 ** exponent - 1

    def __reduce__(self):
        return "NA"

    # Binary arithmetic and comparison ops -> propagate

    __add__ = _create_binary_propagating_op("__add__")
    __radd__ = _create_binary_propagating_op("__radd__")
    __sub__ = _create_binary_propagating_op("__sub__")
    __rsub__ = _create_binary_propagating_op("__rsub__")
    __mul__ = _create_binary_propagating_op("__mul__")
    __rmul__ = _create_binary_propagating_op("__rmul__")
    __matmul__ = _create_binary_propagating_op("__matmul__")
    __rmatmul__ = _create_binary_propagating_op("__rmatmul__")
    __truediv__ = _create_binary_propagating_op("__truediv__")
    __rtruediv__ = _create_binary_propagating_op("__rtruediv__")
    __floordiv__ = _create_binary_propagating_op("__floordiv__")
    __rfloordiv__ = _create_binary_propagating_op("__rfloordiv__")
    __mod__ = _create_binary_propagating_op("__mod__")
    __rmod__ = _create_binary_propagating_op("__rmod__")
    __divmod__ = _create_binary_propagating_op("__divmod__", is_divmod=True)
    __rdivmod__ = _create_binary_propagating_op("__rdivmod__", is_divmod=True)
    # __lshift__ and __rshift__ are not implemented

    __eq__ = _create_binary_propagating_op("__eq__")
    __ne__ = _create_binary_propagating_op("__ne__")
    __le__ = _create_binary_propagating_op("__le__")
    __lt__ = _create_binary_propagating_op("__lt__")
    __gt__ = _create_binary_propagating_op("__gt__")
    __ge__ = _create_binary_propagating_op("__ge__")

    # Unary ops

    __neg__ = _create_unary_propagating_op("__neg__")
    __pos__ = _create_unary_propagating_op("__pos__")
    __abs__ = _create_unary_propagating_op("__abs__")
    __invert__ = _create_unary_propagating_op("__invert__")

    # pow has special
    def __pow__(self, other):
        if other is C_NA:
            return NA
        elif isinstance(other, (numbers.Number, np.bool_)):
            if other == 0:
                # returning positive is correct for +/- 0.
                return type(other)(1)
            else:
                return NA
        elif isinstance(other, np.ndarray):
            return np.where(other == 0, other.dtype.type(1), NA)

        return NotImplemented

    def __rpow__(self, other):
        if other is C_NA:
            return NA
        elif isinstance(other, (numbers.Number, np.bool_)):
            if other == 1:
                return other
            else:
                return NA
        elif isinstance(other, np.ndarray):
            return np.where(other == 1, other, NA)
        return NotImplemented

    # Logical ops using Kleene logic

    def __and__(self, other):
        if other is False:
            return False
        elif other is True or other is C_NA:
            return NA
        return NotImplemented

    __rand__ = __and__

    def __or__(self, other):
        if other is True:
            return True
        elif other is False or other is C_NA:
            return NA
        return NotImplemented

    __ror__ = __or__

    def __xor__(self, other):
        if other is False or other is True or other is C_NA:
            return NA
        return NotImplemented

    __rxor__ = __xor__

    __array_priority__ = 1000
    _HANDLED_TYPES = (np.ndarray, numbers.Number, str, np.bool_)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        types = self._HANDLED_TYPES + (NAType,)
        for x in inputs:
            if not isinstance(x, types):
                return NotImplemented

        if method != "__call__":
            raise ValueError(f"ufunc method '{method}' not supported for NA")
        result = maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is NotImplemented:
            # For a NumPy ufunc that's not a binop, like np.logaddexp
            index = [i for i, x in enumerate(inputs) if x is NA][0]
            result = np.broadcast_arrays(*inputs)[index]
            if result.ndim == 0:
                result = result.item()
            if ufunc.nout > 1:
                result = (NA,) * ufunc.nout

        return result


C_NA = NAType()   # C-visible
NA = C_NA         # Python-visible
