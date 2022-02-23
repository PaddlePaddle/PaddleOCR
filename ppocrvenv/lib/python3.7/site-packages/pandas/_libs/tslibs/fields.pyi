import numpy as np

def build_field_sarray(
    dtindex: np.ndarray,  # const int64_t[:]
) -> np.ndarray: ...
def month_position_check(fields, weekdays) -> str | None: ...
def get_date_name_field(
    dtindex: np.ndarray,  # const int64_t[:]
    field: str,
    locale=...,
) -> np.ndarray: ...  # np.ndarray[object]
def get_start_end_field(
    dtindex: np.ndarray,  # const int64_t[:]
    field: str,
    freqstr: str | None = ...,
    month_kw: int = ...,
) -> np.ndarray: ...  # np.ndarray[bool]
def get_date_field(
    dtindex: np.ndarray,  # const int64_t[:]
    field: str,
) -> np.ndarray: ...  # np.ndarray[in32]
def get_timedelta_field(
    tdindex: np.ndarray,  # const int64_t[:]
    field: str,
) -> np.ndarray: ...  # np.ndarray[int32]
def isleapyear_arr(
    years: np.ndarray,
) -> np.ndarray: ...  # np.ndarray[bool]
def build_isocalendar_sarray(
    dtindex: np.ndarray,  # const int64_t[:]
) -> np.ndarray: ...
def get_locale_names(name_type: str, locale: object = None): ...

class RoundTo:
    @property
    def MINUS_INFTY(self) -> int: ...
    @property
    def PLUS_INFTY(self) -> int: ...
    @property
    def NEAREST_HALF_EVEN(self) -> int: ...
    @property
    def NEAREST_HALF_PLUS_INFTY(self) -> int: ...
    @property
    def NEAREST_HALF_MINUS_INFTY(self) -> int: ...

def round_nsint64(
    values: np.ndarray,  # np.ndarray[np.int64]
    mode: RoundTo,
    nanos: int,
) -> np.ndarray: ...  # np.ndarray[np.int64]
