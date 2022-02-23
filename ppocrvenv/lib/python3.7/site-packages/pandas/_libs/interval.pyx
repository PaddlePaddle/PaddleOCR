import numbers
from operator import (
    le,
    lt,
)

from cpython.datetime cimport (
    PyDateTime_IMPORT,
    PyDelta_Check,
)

PyDateTime_IMPORT

from cpython.object cimport (
    Py_EQ,
    Py_GE,
    Py_GT,
    Py_LE,
    Py_LT,
    Py_NE,
    PyObject_RichCompare,
)

import cython
from cython import Py_ssize_t
import numpy as np

cimport numpy as cnp
from numpy cimport (
    NPY_QUICKSORT,
    PyArray_ArgSort,
    PyArray_Take,
    float32_t,
    float64_t,
    int32_t,
    int64_t,
    ndarray,
    uint64_t,
)

cnp.import_array()


from pandas._libs cimport util
from pandas._libs.hashtable cimport Int64Vector
from pandas._libs.tslibs.timedeltas cimport _Timedelta
from pandas._libs.tslibs.timestamps cimport _Timestamp
from pandas._libs.tslibs.timezones cimport tz_compare
from pandas._libs.tslibs.util cimport (
    is_float_object,
    is_integer_object,
    is_timedelta64_object,
)

VALID_CLOSED = frozenset(['left', 'right', 'both', 'neither'])


cdef class IntervalMixin:

    @property
    def closed_left(self):
        """
        Check if the interval is closed on the left side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.
        """
        return self.closed in ('left', 'both')

    @property
    def closed_right(self):
        """
        Check if the interval is closed on the right side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.
        """
        return self.closed in ('right', 'both')

    @property
    def open_left(self):
        """
        Check if the interval is open on the left side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.
        """
        return not self.closed_left

    @property
    def open_right(self):
        """
        Check if the interval is open on the right side.

        For the meaning of `closed` and `open` see :class:`~pandas.Interval`.

        Returns
        -------
        bool
            True if the Interval is closed on the left-side.
        """
        return not self.closed_right

    @property
    def mid(self):
        """
        Return the midpoint of the Interval.
        """
        try:
            return 0.5 * (self.left + self.right)
        except TypeError:
            # datetime safe version
            return self.left + 0.5 * self.length

    @property
    def length(self):
        """
        Return the length of the Interval.
        """
        return self.right - self.left

    @property
    def is_empty(self):
        """
        Indicates if an interval is empty, meaning it contains no points.

        .. versionadded:: 0.25.0

        Returns
        -------
        bool or ndarray
            A boolean indicating if a scalar :class:`Interval` is empty, or a
            boolean ``ndarray`` positionally indicating if an ``Interval`` in
            an :class:`~arrays.IntervalArray` or :class:`IntervalIndex` is
            empty.

        Examples
        --------
        An :class:`Interval` that contains points is not empty:

        >>> pd.Interval(0, 1, closed='right').is_empty
        False

        An ``Interval`` that does not contain any points is empty:

        >>> pd.Interval(0, 0, closed='right').is_empty
        True
        >>> pd.Interval(0, 0, closed='left').is_empty
        True
        >>> pd.Interval(0, 0, closed='neither').is_empty
        True

        An ``Interval`` that contains a single point is not empty:

        >>> pd.Interval(0, 0, closed='both').is_empty
        False

        An :class:`~arrays.IntervalArray` or :class:`IntervalIndex` returns a
        boolean ``ndarray`` positionally indicating if an ``Interval`` is
        empty:

        >>> ivs = [pd.Interval(0, 0, closed='neither'),
        ...        pd.Interval(1, 2, closed='neither')]
        >>> pd.arrays.IntervalArray(ivs).is_empty
        array([ True, False])

        Missing values are not considered empty:

        >>> ivs = [pd.Interval(0, 0, closed='neither'), np.nan]
        >>> pd.IntervalIndex(ivs).is_empty
        array([ True, False])
        """
        return (self.right == self.left) & (self.closed != 'both')

    def _check_closed_matches(self, other, name='other'):
        """
        Check if the closed attribute of `other` matches.

        Note that 'left' and 'right' are considered different from 'both'.

        Parameters
        ----------
        other : Interval, IntervalIndex, IntervalArray
        name : str
            Name to use for 'other' in the error message.

        Raises
        ------
        ValueError
            When `other` is not closed exactly the same as self.
        """
        if self.closed != other.closed:
            raise ValueError(f"'{name}.closed' is {repr(other.closed)}, "
                             f"expected {repr(self.closed)}.")


cdef bint _interval_like(other):
    return (hasattr(other, 'left')
            and hasattr(other, 'right')
            and hasattr(other, 'closed'))


cdef class Interval(IntervalMixin):
    """
    Immutable object implementing an Interval, a bounded slice-like interval.

    Parameters
    ----------
    left : orderable scalar
        Left bound for the interval.
    right : orderable scalar
        Right bound for the interval.
    closed : {'right', 'left', 'both', 'neither'}, default 'right'
        Whether the interval is closed on the left-side, right-side, both or
        neither. See the Notes for more detailed explanation.

    See Also
    --------
    IntervalIndex : An Index of Interval objects that are all closed on the
        same side.
    cut : Convert continuous data into discrete bins (Categorical
        of Interval objects).
    qcut : Convert continuous data into bins (Categorical of Interval objects)
        based on quantiles.
    Period : Represents a period of time.

    Notes
    -----
    The parameters `left` and `right` must be from the same type, you must be
    able to compare them and they must satisfy ``left <= right``.

    A closed interval (in mathematics denoted by square brackets) contains
    its endpoints, i.e. the closed interval ``[0, 5]`` is characterized by the
    conditions ``0 <= x <= 5``. This is what ``closed='both'`` stands for.
    An open interval (in mathematics denoted by parentheses) does not contain
    its endpoints, i.e. the open interval ``(0, 5)`` is characterized by the
    conditions ``0 < x < 5``. This is what ``closed='neither'`` stands for.
    Intervals can also be half-open or half-closed, i.e. ``[0, 5)`` is
    described by ``0 <= x < 5`` (``closed='left'``) and ``(0, 5]`` is
    described by ``0 < x <= 5`` (``closed='right'``).

    Examples
    --------
    It is possible to build Intervals of different types, like numeric ones:

    >>> iv = pd.Interval(left=0, right=5)
    >>> iv
    Interval(0, 5, closed='right')

    You can check if an element belongs to it

    >>> 2.5 in iv
    True

    You can test the bounds (``closed='right'``, so ``0 < x <= 5``):

    >>> 0 in iv
    False
    >>> 5 in iv
    True
    >>> 0.0001 in iv
    True

    Calculate its length

    >>> iv.length
    5

    You can operate with `+` and `*` over an Interval and the operation
    is applied to each of its bounds, so the result depends on the type
    of the bound elements

    >>> shifted_iv = iv + 3
    >>> shifted_iv
    Interval(3, 8, closed='right')
    >>> extended_iv = iv * 10.0
    >>> extended_iv
    Interval(0.0, 50.0, closed='right')

    To create a time interval you can use Timestamps as the bounds

    >>> year_2017 = pd.Interval(pd.Timestamp('2017-01-01 00:00:00'),
    ...                         pd.Timestamp('2018-01-01 00:00:00'),
    ...                         closed='left')
    >>> pd.Timestamp('2017-01-01 00:00') in year_2017
    True
    >>> year_2017.length
    Timedelta('365 days 00:00:00')
    """
    _typ = "interval"
    __array_priority__ = 1000

    cdef readonly object left
    """
    Left bound for the interval.
    """

    cdef readonly object right
    """
    Right bound for the interval.
    """

    cdef readonly str closed
    """
    Whether the interval is closed on the left-side, right-side, both or
    neither.
    """

    def __init__(self, left, right, str closed='right'):
        # note: it is faster to just do these checks than to use a special
        # constructor (__cinit__/__new__) to avoid them

        self._validate_endpoint(left)
        self._validate_endpoint(right)

        if closed not in VALID_CLOSED:
            raise ValueError(f"invalid option for 'closed': {closed}")
        if not left <= right:
            raise ValueError("left side of interval must be <= right side")
        if (isinstance(left, _Timestamp) and
                not tz_compare(left.tzinfo, right.tzinfo)):
            # GH 18538
            raise ValueError("left and right must have the same time zone, got "
                             f"{repr(left.tzinfo)}' and {repr(right.tzinfo)}")
        self.left = left
        self.right = right
        self.closed = closed

    def _validate_endpoint(self, endpoint):
        # GH 23013
        if not (is_integer_object(endpoint) or is_float_object(endpoint) or
                isinstance(endpoint, (_Timestamp, _Timedelta))):
            raise ValueError("Only numeric, Timestamp and Timedelta endpoints "
                             "are allowed when constructing an Interval.")

    def __hash__(self):
        return hash((self.left, self.right, self.closed))

    def __contains__(self, key) -> bool:
        if _interval_like(key):
            raise TypeError("__contains__ not defined for two intervals")
        return ((self.left < key if self.open_left else self.left <= key) and
                (key < self.right if self.open_right else key <= self.right))

    def __richcmp__(self, other, op: int):
        if isinstance(other, Interval):
            self_tuple = (self.left, self.right, self.closed)
            other_tuple = (other.left, other.right, other.closed)
            return PyObject_RichCompare(self_tuple, other_tuple, op)
        elif util.is_array(other):
            return np.array(
                [PyObject_RichCompare(self, x, op) for x in other],
                dtype=bool,
            )

        return NotImplemented

    def __reduce__(self):
        args = (self.left, self.right, self.closed)
        return (type(self), args)

    def _repr_base(self):
        left = self.left
        right = self.right

        # TODO: need more general formatting methodology here
        if isinstance(left, _Timestamp) and isinstance(right, _Timestamp):
            left = left._short_repr
            right = right._short_repr

        return left, right

    def __repr__(self) -> str:

        left, right = self._repr_base()
        name = type(self).__name__
        repr_str = f'{name}({repr(left)}, {repr(right)}, closed={repr(self.closed)})'
        return repr_str

    def __str__(self) -> str:

        left, right = self._repr_base()
        start_symbol = '[' if self.closed_left else '('
        end_symbol = ']' if self.closed_right else ')'
        return f'{start_symbol}{left}, {right}{end_symbol}'

    def __add__(self, y):
        if (
            isinstance(y, numbers.Number)
            or PyDelta_Check(y)
            or is_timedelta64_object(y)
        ):
            return Interval(self.left + y, self.right + y, closed=self.closed)
        elif (
            isinstance(y, Interval)
            and (
                isinstance(self, numbers.Number)
                or PyDelta_Check(self)
                or is_timedelta64_object(self)
            )
        ):
            return Interval(y.left + self, y.right + self, closed=y.closed)
        return NotImplemented

    def __sub__(self, y):
        if (
            isinstance(y, numbers.Number)
            or PyDelta_Check(y)
            or is_timedelta64_object(y)
        ):
            return Interval(self.left - y, self.right - y, closed=self.closed)
        return NotImplemented

    def __mul__(self, y):
        if isinstance(y, numbers.Number):
            return Interval(self.left * y, self.right * y, closed=self.closed)
        elif isinstance(y, Interval) and isinstance(self, numbers.Number):
            return Interval(y.left * self, y.right * self, closed=y.closed)
        return NotImplemented

    def __truediv__(self, y):
        if isinstance(y, numbers.Number):
            return Interval(self.left / y, self.right / y, closed=self.closed)
        return NotImplemented

    def __floordiv__(self, y):
        if isinstance(y, numbers.Number):
            return Interval(
                self.left // y, self.right // y, closed=self.closed)
        return NotImplemented

    def overlaps(self, other):
        """
        Check whether two Interval objects overlap.

        Two intervals overlap if they share a common point, including closed
        endpoints. Intervals that only have an open endpoint in common do not
        overlap.

        Parameters
        ----------
        other : Interval
            Interval to check against for an overlap.

        Returns
        -------
        bool
            True if the two intervals overlap.

        See Also
        --------
        IntervalArray.overlaps : The corresponding method for IntervalArray.
        IntervalIndex.overlaps : The corresponding method for IntervalIndex.

        Examples
        --------
        >>> i1 = pd.Interval(0, 2)
        >>> i2 = pd.Interval(1, 3)
        >>> i1.overlaps(i2)
        True
        >>> i3 = pd.Interval(4, 5)
        >>> i1.overlaps(i3)
        False

        Intervals that share closed endpoints overlap:

        >>> i4 = pd.Interval(0, 1, closed='both')
        >>> i5 = pd.Interval(1, 2, closed='both')
        >>> i4.overlaps(i5)
        True

        Intervals that only have an open endpoint in common do not overlap:

        >>> i6 = pd.Interval(1, 2, closed='neither')
        >>> i4.overlaps(i6)
        False
        """
        if not isinstance(other, Interval):
            raise TypeError("`other` must be an Interval, "
                            f"got {type(other).__name__}")

        # equality is okay if both endpoints are closed (overlap at a point)
        op1 = le if (self.closed_left and other.closed_right) else lt
        op2 = le if (other.closed_left and self.closed_right) else lt

        # overlaps is equivalent negation of two interval being disjoint:
        # disjoint = (A.left > B.right) or (B.left > A.right)
        # (simplifying the negation allows this to be done in less operations)
        return op1(self.left, other.right) and op2(other.left, self.right)


@cython.wraparound(False)
@cython.boundscheck(False)
def intervals_to_interval_bounds(ndarray intervals, bint validate_closed=True):
    """
    Parameters
    ----------
    intervals : ndarray
        Object array of Intervals / nulls.

    validate_closed: bool, default True
        Boolean indicating if all intervals must be closed on the same side.
        Mismatching closed will raise if True, else return None for closed.

    Returns
    -------
    tuple of tuples
        left : (ndarray, object, array)
        right : (ndarray, object, array)
        closed: str
    """
    cdef:
        object closed = None, interval
        Py_ssize_t i, n = len(intervals)
        ndarray left, right
        bint seen_closed = False

    left = np.empty(n, dtype=intervals.dtype)
    right = np.empty(n, dtype=intervals.dtype)

    for i in range(n):
        interval = intervals[i]
        if interval is None or util.is_nan(interval):
            left[i] = np.nan
            right[i] = np.nan
            continue

        if not isinstance(interval, Interval):
            raise TypeError(f"type {type(interval)} with value "
                            f"{interval} is not an interval")

        left[i] = interval.left
        right[i] = interval.right
        if not seen_closed:
            seen_closed = True
            closed = interval.closed
        elif closed != interval.closed:
            closed = None
            if validate_closed:
                raise ValueError("intervals must all be closed on the same side")

    return left, right, closed


include "intervaltree.pxi"
