import sys
from types import TracebackType
from typing import Any, Optional, Callable, Union, Type

# Using a private class is by no means ideal, but it is simply a consquence
# of a `contextlib.context` returning an instance of aformentioned class
from contextlib import _GeneratorContextManager

from numpy import (
    ndarray,
    generic,
    bool_,
    integer,
    timedelta64,
    datetime64,
    floating,
    complexfloating,
    void,
    str_,
    bytes_,
    longdouble,
    clongdouble,
)
from numpy.typing import ArrayLike, _CharLike_co, _FloatLike_co

if sys.version_info > (3, 8):
    from typing import Literal, TypedDict, SupportsIndex
else:
    from typing_extensions import Literal, TypedDict, SupportsIndex

_FloatMode = Literal["fixed", "unique", "maxprec", "maxprec_equal"]

class _FormatDict(TypedDict, total=False):
    bool: Callable[[bool_], str]
    int: Callable[[integer[Any]], str]
    timedelta: Callable[[timedelta64], str]
    datetime: Callable[[datetime64], str]
    float: Callable[[floating[Any]], str]
    longfloat: Callable[[longdouble], str]
    complexfloat: Callable[[complexfloating[Any, Any]], str]
    longcomplexfloat: Callable[[clongdouble], str]
    void: Callable[[void], str]
    numpystr: Callable[[_CharLike_co], str]
    object: Callable[[object], str]
    all: Callable[[object], str]
    int_kind: Callable[[integer[Any]], str]
    float_kind: Callable[[floating[Any]], str]
    complex_kind: Callable[[complexfloating[Any, Any]], str]
    str_kind: Callable[[_CharLike_co], str]

class _FormatOptions(TypedDict):
    precision: int
    threshold: int
    edgeitems: int
    linewidth: int
    suppress: bool
    nanstr: str
    infstr: str
    formatter: Optional[_FormatDict]
    sign: Literal["-", "+", " "]
    floatmode: _FloatMode
    legacy: Literal[False, "1.13"]

def set_printoptions(
    precision: Optional[SupportsIndex] = ...,
    threshold: Optional[int] = ...,
    edgeitems: Optional[int] = ...,
    linewidth: Optional[int] = ...,
    suppress: Optional[bool] = ...,
    nanstr: Optional[str] = ...,
    infstr: Optional[str] = ...,
    formatter: Optional[_FormatDict] = ...,
    sign: Optional[Literal["-", "+", " "]] = ...,
    floatmode: Optional[_FloatMode] = ...,
    *,
    legacy: Optional[Literal[False, "1.13"]] = ...
) -> None: ...
def get_printoptions() -> _FormatOptions: ...
def array2string(
    a: ndarray[Any, Any],
    max_line_width: Optional[int] = ...,
    precision: Optional[SupportsIndex] = ...,
    suppress_small: Optional[bool] = ...,
    separator: str = ...,
    prefix: str = ...,
    # NOTE: With the `style` argument being deprecated,
    # all arguments between `formatter` and `suffix` are de facto
    # keyworld-only arguments
    *,
    formatter: Optional[_FormatDict] = ...,
    threshold: Optional[int] = ...,
    edgeitems: Optional[int] = ...,
    sign: Optional[Literal["-", "+", " "]] = ...,
    floatmode: Optional[_FloatMode] = ...,
    suffix: str = ...,
    legacy: Optional[Literal[False, "1.13"]] = ...,
) -> str: ...
def format_float_scientific(
    x: _FloatLike_co,
    precision: Optional[int] = ...,
    unique: bool = ...,
    trim: Literal["k", ".", "0", "-"] = ...,
    sign: bool = ...,
    pad_left: Optional[int] = ...,
    exp_digits: Optional[int] = ...,
    min_digits: Optional[int] = ...,
) -> str: ...
def format_float_positional(
    x: _FloatLike_co,
    precision: Optional[int] = ...,
    unique: bool = ...,
    fractional: bool = ...,
    trim: Literal["k", ".", "0", "-"] = ...,
    sign: bool = ...,
    pad_left: Optional[int] = ...,
    pad_right: Optional[int] = ...,
    min_digits: Optional[int] = ...,
) -> str: ...
def array_repr(
    arr: ndarray[Any, Any],
    max_line_width: Optional[int] = ...,
    precision: Optional[SupportsIndex] = ...,
    suppress_small: Optional[bool] = ...,
) -> str: ...
def array_str(
    a: ndarray[Any, Any],
    max_line_width: Optional[int] = ...,
    precision: Optional[SupportsIndex] = ...,
    suppress_small: Optional[bool] = ...,
) -> str: ...
def set_string_function(
    f: Optional[Callable[[ndarray[Any, Any]], str]], repr: bool = ...
) -> None: ...
def printoptions(
    precision: Optional[SupportsIndex] = ...,
    threshold: Optional[int] = ...,
    edgeitems: Optional[int] = ...,
    linewidth: Optional[int] = ...,
    suppress: Optional[bool] = ...,
    nanstr: Optional[str] = ...,
    infstr: Optional[str] = ...,
    formatter: Optional[_FormatDict] = ...,
    sign: Optional[Literal["-", "+", " "]] = ...,
    floatmode: Optional[_FloatMode] = ...,
    *,
    legacy: Optional[Literal[False, "1.13"]] = ...
) -> _GeneratorContextManager[_FormatOptions]: ...
