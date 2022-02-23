from __future__ import annotations

import warnings

import numpy as np

from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
    Timestamp,
    tzconversion,
)
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTH_ALIASES,
    MONTH_NUMBERS,
    MONTHS,
    int_to_weekday,
)
from pandas._libs.tslibs.fields import (
    build_field_sarray,
    month_position_check,
)
from pandas._libs.tslibs.offsets import (  # noqa:F401
    DateOffset,
    Day,
    _get_offset,
    to_offset,
)
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import (
    is_datetime64_dtype,
    is_period_dtype,
    is_timedelta64_dtype,
)
from pandas.core.dtypes.generic import ABCSeries

from pandas.core.algorithms import unique

_ONE_MICRO = 1000
_ONE_MILLI = _ONE_MICRO * 1000
_ONE_SECOND = _ONE_MILLI * 1000
_ONE_MINUTE = 60 * _ONE_SECOND
_ONE_HOUR = 60 * _ONE_MINUTE
_ONE_DAY = 24 * _ONE_HOUR

# ---------------------------------------------------------------------
# Offset names ("time rules") and related functions

_offset_to_period_map = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BM": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQ": "Q",
    "BA": "A",
    "AS": "A",
    "BAS": "A",
    "MS": "M",
    "D": "D",
    "C": "C",
    "B": "B",
    "T": "T",
    "S": "S",
    "L": "L",
    "U": "U",
    "N": "N",
    "H": "H",
    "Q": "Q",
    "A": "A",
    "W": "W",
    "M": "M",
    "Y": "A",
    "BY": "A",
    "YS": "A",
    "BYS": "A",
}

_need_suffix = ["QS", "BQ", "BQS", "YS", "AS", "BY", "BA", "BYS", "BAS"]

for _prefix in _need_suffix:
    for _m in MONTHS:
        key = f"{_prefix}-{_m}"
        _offset_to_period_map[key] = _offset_to_period_map[_prefix]

for _prefix in ["A", "Q"]:
    for _m in MONTHS:
        _alias = f"{_prefix}-{_m}"
        _offset_to_period_map[_alias] = _alias

for _d in DAYS:
    _offset_to_period_map[f"W-{_d}"] = f"W-{_d}"


def get_period_alias(offset_str: str) -> str | None:
    """
    Alias to closest period strings BQ->Q etc.
    """
    return _offset_to_period_map.get(offset_str, None)


def get_offset(name: str) -> DateOffset:
    """
    Return DateOffset object associated with rule name.

    .. deprecated:: 1.0.0

    Examples
    --------
    get_offset('EOM') --> BMonthEnd(1)
    """
    warnings.warn(
        "get_offset is deprecated and will be removed in a future version, "
        "use to_offset instead",
        FutureWarning,
        stacklevel=2,
    )
    return _get_offset(name)


# ---------------------------------------------------------------------
# Period codes


def infer_freq(index, warn: bool = True) -> str | None:
    """
    Infer the most likely frequency given the input index. If the frequency is
    uncertain, a warning will be printed.

    Parameters
    ----------
    index : DatetimeIndex or TimedeltaIndex
      If passed a Series will use the values of the series (NOT THE INDEX).
    warn : bool, default True

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values.

    Examples
    --------
    >>> idx = pd.date_range(start='2020/12/01', end='2020/12/30', periods=30)
    >>> pd.infer_freq(idx)
    'D'
    """
    import pandas as pd

    if isinstance(index, ABCSeries):
        values = index._values
        if not (
            is_datetime64_dtype(values)
            or is_timedelta64_dtype(values)
            or values.dtype == object
        ):
            raise TypeError(
                "cannot infer freq from a non-convertible dtype "
                f"on a Series of {index.dtype}"
            )
        index = values

    inferer: _FrequencyInferer

    if not hasattr(index, "dtype"):
        pass
    elif is_period_dtype(index.dtype):
        raise TypeError(
            "PeriodIndex given. Check the `freq` attribute "
            "instead of using infer_freq."
        )
    elif is_timedelta64_dtype(index.dtype):
        # Allow TimedeltaIndex and TimedeltaArray
        inferer = _TimedeltaFrequencyInferer(index, warn=warn)
        return inferer.get_freq()

    if isinstance(index, pd.Index) and not isinstance(index, pd.DatetimeIndex):
        if isinstance(index, (pd.Int64Index, pd.Float64Index)):
            raise TypeError(
                f"cannot infer freq from a non-convertible index type {type(index)}"
            )
        index = index._values

    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    inferer = _FrequencyInferer(index, warn=warn)
    return inferer.get_freq()


class _FrequencyInferer:
    """
    Not sure if I can avoid the state machine here
    """

    def __init__(self, index, warn: bool = True):
        self.index = index
        self.i8values = index.asi8

        # This moves the values, which are implicitly in UTC, to the
        # the timezone so they are in local time
        if hasattr(index, "tz"):
            if index.tz is not None:
                self.i8values = tzconversion.tz_convert_from_utc(
                    self.i8values, index.tz
                )

        self.warn = warn

        if len(index) < 3:
            raise ValueError("Need at least 3 dates to infer frequency")

        self.is_monotonic = (
            self.index._is_monotonic_increasing or self.index._is_monotonic_decreasing
        )

    @cache_readonly
    def deltas(self):
        return unique_deltas(self.i8values)

    @cache_readonly
    def deltas_asi8(self):
        # NB: we cannot use self.i8values here because we may have converted
        #  the tz in __init__
        return unique_deltas(self.index.asi8)

    @cache_readonly
    def is_unique(self) -> bool:
        return len(self.deltas) == 1

    @cache_readonly
    def is_unique_asi8(self) -> bool:
        return len(self.deltas_asi8) == 1

    def get_freq(self) -> str | None:
        """
        Find the appropriate frequency string to describe the inferred
        frequency of self.i8values

        Returns
        -------
        str or None
        """
        if not self.is_monotonic or not self.index._is_unique:
            return None

        delta = self.deltas[0]
        if delta and _is_multiple(delta, _ONE_DAY):
            return self._infer_daily_rule()

        # Business hourly, maybe. 17: one day / 65: one weekend
        if self.hour_deltas in ([1, 17], [1, 65], [1, 17, 65]):
            return "BH"

        # Possibly intraday frequency.  Here we use the
        # original .asi8 values as the modified values
        # will not work around DST transitions.  See #8772
        if not self.is_unique_asi8:
            return None

        delta = self.deltas_asi8[0]
        if _is_multiple(delta, _ONE_HOUR):
            # Hours
            return _maybe_add_count("H", delta / _ONE_HOUR)
        elif _is_multiple(delta, _ONE_MINUTE):
            # Minutes
            return _maybe_add_count("T", delta / _ONE_MINUTE)
        elif _is_multiple(delta, _ONE_SECOND):
            # Seconds
            return _maybe_add_count("S", delta / _ONE_SECOND)
        elif _is_multiple(delta, _ONE_MILLI):
            # Milliseconds
            return _maybe_add_count("L", delta / _ONE_MILLI)
        elif _is_multiple(delta, _ONE_MICRO):
            # Microseconds
            return _maybe_add_count("U", delta / _ONE_MICRO)
        else:
            # Nanoseconds
            return _maybe_add_count("N", delta)

    @cache_readonly
    def day_deltas(self):
        return [x / _ONE_DAY for x in self.deltas]

    @cache_readonly
    def hour_deltas(self):
        return [x / _ONE_HOUR for x in self.deltas]

    @cache_readonly
    def fields(self):
        return build_field_sarray(self.i8values)

    @cache_readonly
    def rep_stamp(self):
        return Timestamp(self.i8values[0])

    def month_position_check(self):
        return month_position_check(self.fields, self.index.dayofweek)

    @cache_readonly
    def mdiffs(self):
        nmonths = self.fields["Y"] * 12 + self.fields["M"]
        return unique_deltas(nmonths.astype("i8"))

    @cache_readonly
    def ydiffs(self):
        return unique_deltas(self.fields["Y"].astype("i8"))

    def _infer_daily_rule(self) -> str | None:
        annual_rule = self._get_annual_rule()
        if annual_rule:
            nyears = self.ydiffs[0]
            month = MONTH_ALIASES[self.rep_stamp.month]
            alias = f"{annual_rule}-{month}"
            return _maybe_add_count(alias, nyears)

        quarterly_rule = self._get_quarterly_rule()
        if quarterly_rule:
            nquarters = self.mdiffs[0] / 3
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = MONTH_ALIASES[mod_dict[self.rep_stamp.month % 3]]
            alias = f"{quarterly_rule}-{month}"
            return _maybe_add_count(alias, nquarters)

        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            return _maybe_add_count(monthly_rule, self.mdiffs[0])

        if self.is_unique:
            return self._get_daily_rule()

        if self._is_business_daily():
            return "B"

        wom_rule = self._get_wom_rule()
        if wom_rule:
            return wom_rule

        return None

    def _get_daily_rule(self) -> str | None:
        days = self.deltas[0] / _ONE_DAY
        if days % 7 == 0:
            # Weekly
            wd = int_to_weekday[self.rep_stamp.weekday()]
            alias = f"W-{wd}"
            return _maybe_add_count(alias, days / 7)
        else:
            return _maybe_add_count("D", days)

    def _get_annual_rule(self) -> str | None:
        if len(self.ydiffs) > 1:
            return None

        if len(unique(self.fields["M"])) > 1:
            return None

        pos_check = self.month_position_check()
        return {"cs": "AS", "bs": "BAS", "ce": "A", "be": "BA"}.get(pos_check)

    def _get_quarterly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None

        if not self.mdiffs[0] % 3 == 0:
            return None

        pos_check = self.month_position_check()
        return {"cs": "QS", "bs": "BQS", "ce": "Q", "be": "BQ"}.get(pos_check)

    def _get_monthly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None
        pos_check = self.month_position_check()
        return {"cs": "MS", "bs": "BMS", "ce": "M", "be": "BM"}.get(pos_check)

    def _is_business_daily(self) -> bool:
        # quick check: cannot be business daily
        if self.day_deltas != [1, 3]:
            return False

        # probably business daily, but need to confirm
        first_weekday = self.index[0].weekday()
        shifts = np.diff(self.index.asi8)
        shifts = np.floor_divide(shifts, _ONE_DAY)
        weekdays = np.mod(first_weekday + np.cumsum(shifts), 7)
        # error: Incompatible return value type (got "bool_", expected "bool")
        return np.all(  # type: ignore[return-value]
            ((weekdays == 0) & (shifts == 3))
            | ((weekdays > 0) & (weekdays <= 4) & (shifts == 1))
        )

    def _get_wom_rule(self) -> str | None:
        # FIXME: dont leave commented-out
        #         wdiffs = unique(np.diff(self.index.week))
        # We also need -47, -49, -48 to catch index spanning year boundary
        #     if not lib.ismember(wdiffs, set([4, 5, -47, -49, -48])).all():
        #         return None

        weekdays = unique(self.index.weekday)
        if len(weekdays) > 1:
            return None

        week_of_months = unique((self.index.day - 1) // 7)
        # Only attempt to infer up to WOM-4. See #9425
        week_of_months = week_of_months[week_of_months < 4]
        if len(week_of_months) == 0 or len(week_of_months) > 1:
            return None

        # get which week
        week = week_of_months[0] + 1
        wd = int_to_weekday[weekdays[0]]

        return f"WOM-{week}{wd}"


class _TimedeltaFrequencyInferer(_FrequencyInferer):
    def _infer_daily_rule(self):
        if self.is_unique:
            return self._get_daily_rule()


def _is_multiple(us, mult: int) -> bool:
    return us % mult == 0


def _maybe_add_count(base: str, count: float) -> str:
    if count != 1:
        assert count == int(count)
        count = int(count)
        return f"{count}{base}"
    else:
        return base


# ----------------------------------------------------------------------
# Frequency comparison


def is_subperiod(source, target) -> bool:
    """
    Returns True if downsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """

    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)

    if _is_annual(target):
        if _is_quarterly(source):
            return _quarter_months_conform(
                get_rule_month(source), get_rule_month(target)
            )
        return source in {"D", "C", "B", "M", "H", "T", "S", "L", "U", "N"}
    elif _is_quarterly(target):
        return source in {"D", "C", "B", "M", "H", "T", "S", "L", "U", "N"}
    elif _is_monthly(target):
        return source in {"D", "C", "B", "H", "T", "S", "L", "U", "N"}
    elif _is_weekly(target):
        return source in {target, "D", "C", "B", "H", "T", "S", "L", "U", "N"}
    elif target == "B":
        return source in {"B", "H", "T", "S", "L", "U", "N"}
    elif target == "C":
        return source in {"C", "H", "T", "S", "L", "U", "N"}
    elif target == "D":
        return source in {"D", "H", "T", "S", "L", "U", "N"}
    elif target == "H":
        return source in {"H", "T", "S", "L", "U", "N"}
    elif target == "T":
        return source in {"T", "S", "L", "U", "N"}
    elif target == "S":
        return source in {"S", "L", "U", "N"}
    elif target == "L":
        return source in {"L", "U", "N"}
    elif target == "U":
        return source in {"U", "N"}
    elif target == "N":
        return source in {"N"}
    else:
        return False


def is_superperiod(source, target) -> bool:
    """
    Returns True if upsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)

    if _is_annual(source):
        if _is_annual(target):
            return get_rule_month(source) == get_rule_month(target)

        if _is_quarterly(target):
            smonth = get_rule_month(source)
            tmonth = get_rule_month(target)
            return _quarter_months_conform(smonth, tmonth)
        return target in {"D", "C", "B", "M", "H", "T", "S", "L", "U", "N"}
    elif _is_quarterly(source):
        return target in {"D", "C", "B", "M", "H", "T", "S", "L", "U", "N"}
    elif _is_monthly(source):
        return target in {"D", "C", "B", "H", "T", "S", "L", "U", "N"}
    elif _is_weekly(source):
        return target in {source, "D", "C", "B", "H", "T", "S", "L", "U", "N"}
    elif source == "B":
        return target in {"D", "C", "B", "H", "T", "S", "L", "U", "N"}
    elif source == "C":
        return target in {"D", "C", "B", "H", "T", "S", "L", "U", "N"}
    elif source == "D":
        return target in {"D", "C", "B", "H", "T", "S", "L", "U", "N"}
    elif source == "H":
        return target in {"H", "T", "S", "L", "U", "N"}
    elif source == "T":
        return target in {"T", "S", "L", "U", "N"}
    elif source == "S":
        return target in {"S", "L", "U", "N"}
    elif source == "L":
        return target in {"L", "U", "N"}
    elif source == "U":
        return target in {"U", "N"}
    elif source == "N":
        return target in {"N"}
    else:
        return False


def _maybe_coerce_freq(code) -> str:
    """we might need to coerce a code to a rule_code
    and uppercase it

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from

    Returns
    -------
    str
    """
    assert code is not None
    if isinstance(code, DateOffset):
        code = code.rule_code
    return code.upper()


def _quarter_months_conform(source: str, target: str) -> bool:
    snum = MONTH_NUMBERS[source]
    tnum = MONTH_NUMBERS[target]
    return snum % 3 == tnum % 3


def _is_annual(rule: str) -> bool:
    rule = rule.upper()
    return rule == "A" or rule.startswith("A-")


def _is_quarterly(rule: str) -> bool:
    rule = rule.upper()
    return rule == "Q" or rule.startswith("Q-") or rule.startswith("BQ")


def _is_monthly(rule: str) -> bool:
    rule = rule.upper()
    return rule == "M" or rule == "BM"


def _is_weekly(rule: str) -> bool:
    rule = rule.upper()
    return rule == "W" or rule.startswith("W-")
