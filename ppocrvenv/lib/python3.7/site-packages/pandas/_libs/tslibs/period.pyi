from typing import Literal

import numpy as np

from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.offsets import BaseOffset
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._typing import (
    Frequency,
    Timezone,
)

INVALID_FREQ_ERR_MSG: str
DIFFERENT_FREQ: str

class IncompatibleFrequency(ValueError): ...

def periodarr_to_dt64arr(
    periodarr: np.ndarray,  # const int64_t[:]
    freq: int,
) -> np.ndarray: ...  # np.ndarray[np.int64]
def period_asfreq_arr(
    arr: np.ndarray,  # ndarray[int64_t] arr,
    freq1: int,
    freq2: int,
    end: bool,
) -> np.ndarray: ...  # np.ndarray[np.int64]
def get_period_field_arr(
    field: str,
    arr: np.ndarray,  # const int64_t[:]
    freq: int,
) -> np.ndarray: ...  # np.ndarray[np.int64]
def from_ordinals(
    values: np.ndarray,  # const int64_t[:]
    freq: Frequency,
) -> np.ndarray: ...  # np.ndarray[np.int64]
def extract_ordinals(
    values: np.ndarray,  # np.ndarray[object]
    freq: Frequency | int,
) -> np.ndarray: ...  # np.ndarray[np.int64]
def extract_freq(
    values: np.ndarray,  # np.ndarray[object]
) -> BaseOffset: ...

# exposed for tests
def period_asfreq(ordinal: int, freq1: int, freq2: int, end: bool) -> int: ...
def period_ordinal(
    y: int, m: int, d: int, h: int, min: int, s: int, us: int, ps: int, freq: int
) -> int: ...
def freq_to_dtype_code(freq: BaseOffset) -> int: ...
def validate_end_alias(how: str) -> Literal["E", "S"]: ...

class Period:
    ordinal: int  # int64_t
    freq: BaseOffset

    # error: "__new__" must return a class instance (got "Union[Period, NaTType]")
    def __new__(  # type: ignore[misc]
        cls,
        value=None,
        freq=None,
        ordinal=None,
        year=None,
        month=None,
        quarter=None,
        day=None,
        hour=None,
        minute=None,
        second=None,
    ) -> Period | NaTType: ...
    @classmethod
    def _maybe_convert_freq(cls, freq) -> BaseOffset: ...
    @classmethod
    def _from_ordinal(cls, ordinal: int, freq) -> Period: ...
    @classmethod
    def now(cls, freq=...) -> Period: ...
    def strftime(self, fmt: str) -> str: ...
    def to_timestamp(
        self,
        freq: str | BaseOffset | None = ...,
        how: str = ...,
        tz: Timezone | None = ...,
    ) -> Timestamp: ...
    def asfreq(self, freq, how=...) -> Period: ...
    @property
    def freqstr(self) -> str: ...
    @property
    def is_leap_year(self) -> bool: ...
    @property
    def daysinmonth(self) -> int: ...
    @property
    def days_in_month(self) -> int: ...
    @property
    def qyear(self) -> int: ...
    @property
    def quarter(self) -> int: ...
    @property
    def day_of_year(self) -> int: ...
    @property
    def weekday(self) -> int: ...
    @property
    def day_of_week(self) -> int: ...
    @property
    def week(self) -> int: ...
    @property
    def weekofyear(self) -> int: ...
    @property
    def second(self) -> int: ...
    @property
    def minute(self) -> int: ...
    @property
    def hour(self) -> int: ...
    @property
    def day(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def year(self) -> int: ...
    @property
    def end_time(self) -> Timestamp: ...
    @property
    def start_time(self) -> Timestamp: ...
    def __sub__(self, other) -> Period | BaseOffset: ...
    def __add__(self, other) -> Period: ...
