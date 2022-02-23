from datetime import datetime

import numpy as np

from pandas._libs.tslibs.offsets import BaseOffset

class DateParseError(ValueError): ...

def parse_datetime_string(
    date_string: str,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    **kwargs,
) -> datetime: ...
def parse_time_string(
    arg: str,
    freq: BaseOffset | str | None = ...,
    dayfirst: bool | None = ...,
    yearfirst: bool | None = ...,
) -> tuple[datetime, str]: ...
def _does_string_look_like_datetime(py_string: str) -> bool: ...
def quarter_to_myear(year: int, quarter: int, freq: str) -> tuple[int, int]: ...
def try_parse_dates(
    values: np.ndarray,  # object[:]
    parser=...,
    dayfirst: bool = ...,
    default: datetime | None = ...,
) -> np.ndarray: ...  # np.ndarray[object]
def try_parse_date_and_time(
    dates: np.ndarray,  # object[:]
    times: np.ndarray,  # object[:]
    date_parser=...,
    time_parser=...,
    dayfirst: bool = ...,
    default: datetime | None = ...,
) -> np.ndarray: ...  # np.ndarray[object]
def try_parse_year_month_day(
    years: np.ndarray,  # object[:]
    months: np.ndarray,  # object[:]
    days: np.ndarray,  # object[:]
) -> np.ndarray: ...  # np.ndarray[object]
def try_parse_datetime_components(
    years: np.ndarray,  # object[:]
    months: np.ndarray,  # object[:]
    days: np.ndarray,  # object[:]
    hours: np.ndarray,  # object[:]
    minutes: np.ndarray,  # object[:]
    seconds: np.ndarray,  # object[:]
) -> np.ndarray: ...  # np.ndarray[object]
def format_is_iso(f: str) -> bool: ...
def guess_datetime_format(
    dt_str,
    dayfirst: bool = ...,
    dt_str_parse=...,
    dt_str_split=...,
) -> str | None: ...
def concat_date_cols(
    date_cols: tuple,
    keep_trivial_numbers: bool = ...,
) -> np.ndarray: ...  # np.ndarray[object]
def get_rule_month(source: str) -> str: ...
