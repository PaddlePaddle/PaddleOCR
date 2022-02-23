# flake8: noqa

"""
Expose public exceptions & warnings
"""

from pandas._config.config import OptionError

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)


class IntCastingNaNError(ValueError):
    """
    raised when attempting an astype operation on an array with NaN to an integer
    dtype.
    """

    pass


class NullFrequencyError(ValueError):
    """
    Error raised when a null `freq` attribute is used in an operation
    that needs a non-null frequency, particularly `DatetimeIndex.shift`,
    `TimedeltaIndex.shift`, `PeriodIndex.shift`.
    """

    pass


class PerformanceWarning(Warning):
    """
    Warning raised when there is a possible performance impact.
    """


class UnsupportedFunctionCall(ValueError):
    """
    Exception raised when attempting to call a numpy function
    on a pandas object, but that function is not supported by
    the object e.g. ``np.cumsum(groupby_object)``.
    """


class UnsortedIndexError(KeyError):
    """
    Error raised when attempting to get a slice of a MultiIndex,
    and the index has not been lexsorted. Subclass of `KeyError`.
    """


class ParserError(ValueError):
    """
    Exception that is raised by an error encountered in parsing file contents.

    This is a generic error raised for errors encountered when functions like
    `read_csv` or `read_html` are parsing contents of a file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_html : Read HTML table into a DataFrame.
    """


class DtypeWarning(Warning):
    """
    Warning raised when reading different dtypes in a column from a file.

    Raised for a dtype incompatibility. This can happen whenever `read_csv`
    or `read_table` encounter non-uniform dtypes in a column(s) of a given
    CSV file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_table : Read general delimited file into a DataFrame.

    Notes
    -----
    This warning is issued when dealing with larger files because the dtype
    checking happens per chunk read.

    Despite the warning, the CSV file is read with mixed types in a single
    column which will be an object type. See the examples below to better
    understand this issue.

    Examples
    --------
    This example creates and reads a large CSV file with a column that contains
    `int` and `str`.

    >>> df = pd.DataFrame({'a': (['1'] * 100000 + ['X'] * 100000 +
    ...                          ['1'] * 100000),
    ...                    'b': ['b'] * 300000})
    >>> df.to_csv('test.csv', index=False)
    >>> df2 = pd.read_csv('test.csv')
    ... # DtypeWarning: Columns (0) have mixed types

    Important to notice that ``df2`` will contain both `str` and `int` for the
    same input, '1'.

    >>> df2.iloc[262140, 0]
    '1'
    >>> type(df2.iloc[262140, 0])
    <class 'str'>
    >>> df2.iloc[262150, 0]
    1
    >>> type(df2.iloc[262150, 0])
    <class 'int'>

    One way to solve this issue is using the `dtype` parameter in the
    `read_csv` and `read_table` functions to explicit the conversion:

    >>> df2 = pd.read_csv('test.csv', sep=',', dtype={'a': str})

    No warning was issued.

    >>> import os
    >>> os.remove('test.csv')
    """


class EmptyDataError(ValueError):
    """
    Exception that is thrown in `pd.read_csv` (by both the C and
    Python engines) when empty data or header is encountered.
    """


class ParserWarning(Warning):
    """
    Warning raised when reading a file that doesn't use the default 'c' parser.

    Raised by `pd.read_csv` and `pd.read_table` when it is necessary to change
    parsers, generally from the default 'c' parser to 'python'.

    It happens due to a lack of support or functionality for parsing a
    particular attribute of a CSV file with the requested engine.

    Currently, 'c' unsupported options include the following parameters:

    1. `sep` other than a single character (e.g. regex separators)
    2. `skipfooter` higher than 0
    3. `sep=None` with `delim_whitespace=False`

    The warning can be avoided by adding `engine='python'` as a parameter in
    `pd.read_csv` and `pd.read_table` methods.

    See Also
    --------
    pd.read_csv : Read CSV (comma-separated) file into DataFrame.
    pd.read_table : Read general delimited file into DataFrame.

    Examples
    --------
    Using a `sep` in `pd.read_csv` other than a single character:

    >>> import io
    >>> csv = '''a;b;c
    ...           1;1,8
    ...           1;2,1'''
    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]')  # doctest: +SKIP
    ... # ParserWarning: Falling back to the 'python' engine...

    Adding `engine='python'` to `pd.read_csv` removes the Warning:

    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]', engine='python')
    """


class MergeError(ValueError):
    """
    Error raised when problems arise during merging due to problems
    with input data. Subclass of `ValueError`.
    """


class AccessorRegistrationWarning(Warning):
    """
    Warning for attribute conflicts in accessor registration.
    """


class AbstractMethodError(NotImplementedError):
    """
    Raise this error instead of NotImplementedError for abstract methods
    while keeping compatibility with Python 2 and Python 3.
    """

    def __init__(self, class_instance, methodtype="method"):
        types = {"method", "classmethod", "staticmethod", "property"}
        if methodtype not in types:
            raise ValueError(
                f"methodtype must be one of {methodtype}, got {types} instead."
            )
        self.methodtype = methodtype
        self.class_instance = class_instance

    def __str__(self) -> str:
        if self.methodtype == "classmethod":
            name = self.class_instance.__name__
        else:
            name = type(self.class_instance).__name__
        return f"This {self.methodtype} must be defined in the concrete class {name}"


class NumbaUtilError(Exception):
    """
    Error raised for unsupported Numba engine routines.
    """


class DuplicateLabelError(ValueError):
    """
    Error raised when an operation would introduce duplicate labels.

    .. versionadded:: 1.2.0

    Examples
    --------
    >>> s = pd.Series([0, 1, 2], index=['a', 'b', 'c']).set_flags(
    ...     allows_duplicate_labels=False
    ... )
    >>> s.reindex(['a', 'a', 'b'])
    Traceback (most recent call last):
       ...
    DuplicateLabelError: Index has duplicates.
          positions
    label
    a        [0, 1]
    """


class InvalidIndexError(Exception):
    """
    Exception raised when attempting to use an invalid index key.

    .. versionadded:: 1.1.0
    """
