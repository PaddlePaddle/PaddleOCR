"""
timedelta support tools
"""

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    NaTType,
)
from pandas._libs.tslibs.timedeltas import (
    Timedelta,
    parse_timedelta_unit,
)

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

from pandas.core.arrays.timedeltas import sequence_to_td64ns


def to_timedelta(arg, unit=None, errors="raise"):
    """
    Convert argument to timedelta.

    Timedeltas are absolute differences in times, expressed in difference
    units (e.g. days, hours, minutes, seconds). This method converts
    an argument from a recognized timedelta format / value into
    a Timedelta type.

    Parameters
    ----------
    arg : str, timedelta, list-like or Series
        The data to be converted to timedelta.

        .. deprecated:: 1.2
            Strings with units 'M', 'Y' and 'y' do not represent
            unambiguous timedelta values and will be removed in a future version

    unit : str, optional
        Denotes the unit of the arg for numeric `arg`. Defaults to ``"ns"``.

        Possible values:

        * 'W'
        * 'D' / 'days' / 'day'
        * 'hours' / 'hour' / 'hr' / 'h'
        * 'm' / 'minute' / 'min' / 'minutes' / 'T'
        * 'S' / 'seconds' / 'sec' / 'second'
        * 'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis' / 'L'
        * 'us' / 'microseconds' / 'microsecond' / 'micro' / 'micros' / 'U'
        * 'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond' / 'N'

        .. versionchanged:: 1.1.0

           Must not be specified when `arg` context strings and
           ``errors="raise"``.

    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'ignore', then invalid parsing will return the input.

    Returns
    -------
    timedelta64 or numpy.array of timedelta64
        Output type returned if parsing succeeded.

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_datetime : Convert argument to datetime.
    convert_dtypes : Convert dtypes.

    Notes
    -----
    If the precision is higher than nanoseconds, the precision of the duration is
    truncated to nanoseconds for string inputs.

    Examples
    --------
    Parsing a single string to a Timedelta:

    >>> pd.to_timedelta('1 days 06:05:01.00003')
    Timedelta('1 days 06:05:01.000030')
    >>> pd.to_timedelta('15.5us')
    Timedelta('0 days 00:00:00.000015500')

    Parsing a list or array of strings:

    >>> pd.to_timedelta(['1 days 06:05:01.00003', '15.5us', 'nan'])
    TimedeltaIndex(['1 days 06:05:01.000030', '0 days 00:00:00.000015500', NaT],
                   dtype='timedelta64[ns]', freq=None)

    Converting numbers by specifying the `unit` keyword argument:

    >>> pd.to_timedelta(np.arange(5), unit='s')
    TimedeltaIndex(['0 days 00:00:00', '0 days 00:00:01', '0 days 00:00:02',
                    '0 days 00:00:03', '0 days 00:00:04'],
                   dtype='timedelta64[ns]', freq=None)
    >>> pd.to_timedelta(np.arange(5), unit='d')
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq=None)
    """
    if unit is not None:
        unit = parse_timedelta_unit(unit)

    if errors not in ("ignore", "raise", "coerce"):
        raise ValueError("errors must be one of 'ignore', 'raise', or 'coerce'.")

    if unit in {"Y", "y", "M"}:
        raise ValueError(
            "Units 'M', 'Y', and 'y' are no longer supported, as they do not "
            "represent unambiguous timedelta values durations."
        )

    if arg is None:
        return arg
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, unit=unit, errors=errors)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, unit=unit, errors=errors, name=arg.name)
    elif isinstance(arg, np.ndarray) and arg.ndim == 0:
        # extract array scalar and process below
        arg = lib.item_from_zerodim(arg)
    elif is_list_like(arg) and getattr(arg, "ndim", 1) == 1:
        return _convert_listlike(arg, unit=unit, errors=errors)
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError(
            "arg must be a string, timedelta, list, tuple, 1-d array, or Series"
        )

    if isinstance(arg, str) and unit is not None:
        raise ValueError("unit must not be specified if the input is/contains a str")

    # ...so it must be a scalar value. Return scalar.
    return _coerce_scalar_to_timedelta_type(arg, unit=unit, errors=errors)


def _coerce_scalar_to_timedelta_type(r, unit="ns", errors="raise"):
    """Convert string 'r' to a timedelta object."""
    result: Timedelta | NaTType  # TODO: alias?

    try:
        result = Timedelta(r, unit)
    except ValueError:
        if errors == "raise":
            raise
        elif errors == "ignore":
            return r

        # coerce
        result = NaT

    return result


def _convert_listlike(arg, unit=None, errors="raise", name=None):
    """Convert a list of objects to a timedelta index object."""
    if isinstance(arg, (list, tuple)) or not hasattr(arg, "dtype"):
        # This is needed only to ensure that in the case where we end up
        #  returning arg (errors == "ignore"), and where the input is a
        #  generator, we return a useful list-like instead of a
        #  used-up generator
        arg = np.array(list(arg), dtype=object)

    try:
        td64arr = sequence_to_td64ns(arg, unit=unit, errors=errors, copy=False)[0]
    except ValueError:
        if errors == "ignore":
            return arg
        else:
            # This else-block accounts for the cases when errors='raise'
            # and errors='coerce'. If errors == 'raise', these errors
            # should be raised. If errors == 'coerce', we shouldn't
            # expect any errors to be raised, since all parsing errors
            # cause coercion to pd.NaT. However, if an error / bug is
            # introduced that causes an Exception to be raised, we would
            # like to surface it.
            raise

    from pandas import TimedeltaIndex

    value = TimedeltaIndex(td64arr, unit="ns", name=name)
    return value
