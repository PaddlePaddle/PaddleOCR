# -*- coding: utf-8 -*-
"""
This module is intended to replicate some of the functionality
from the fastnumbers module in the event that module is not installed.
"""
import unicodedata
from typing import Callable, FrozenSet, Optional, Union

from natsort.unicode_numbers import decimal_chars

_NAN_INF = [
    "INF",
    "INf",
    "Inf",
    "inF",
    "iNF",
    "InF",
    "inf",
    "iNf",
    "NAN",
    "nan",
    "NaN",
    "nAn",
    "naN",
    "NAn",
    "nAN",
    "Nan",
]
_NAN_INF.extend(["+" + x[:2] for x in _NAN_INF] + ["-" + x[:2] for x in _NAN_INF])
NAN_INF = frozenset(_NAN_INF)
ASCII_NUMS = "0123456789+-"
POTENTIAL_FIRST_CHAR = frozenset(decimal_chars + list(ASCII_NUMS + "."))

StrOrFloat = Union[str, float]
StrOrInt = Union[str, int]


# noinspection PyIncorrectDocstring
def fast_float(
    x: str,
    key: Callable[[str], StrOrFloat] = lambda x: x,
    nan: Optional[StrOrFloat] = None,
    _uni: Callable[[str, StrOrFloat], StrOrFloat] = unicodedata.numeric,
    _nan_inf: FrozenSet[str] = NAN_INF,
    _first_char: FrozenSet[str] = POTENTIAL_FIRST_CHAR,
) -> StrOrFloat:
    """
    Convert a string to a float quickly, return input as-is if not possible.

    We don't need to accept all input that the real fast_int accepts because
    natsort is controlling what is passed to this function.

    Parameters
    ----------
    x : str
        String to attempt to convert to a float.
    key : callable
        Single-argument function to apply to *x* if conversion fails.
    nan : object
        Value to return instead of NaN if NaN would be returned.

    Returns
    -------
    *str* or *float*

    """
    if x[0] in _first_char or x.lstrip()[:3] in _nan_inf:
        try:
            ret = float(x)
            return nan if nan is not None and ret != ret else ret
        except ValueError:
            try:
                return _uni(x, key(x)) if len(x) == 1 else key(x)
            except TypeError:  # pragma: no cover
                return key(x)
    else:
        try:
            return _uni(x, key(x)) if len(x) == 1 else key(x)
        except TypeError:  # pragma: no cover
            return key(x)


# noinspection PyIncorrectDocstring
def fast_int(
    x: str,
    key: Callable[[str], StrOrInt] = lambda x: x,
    _uni: Callable[[str, StrOrInt], StrOrInt] = unicodedata.digit,
    _first_char: FrozenSet[str] = POTENTIAL_FIRST_CHAR,
) -> StrOrInt:
    """
    Convert a string to a int quickly, return input as-is if not possible.

    We don't need to accept all input that the real fast_int accepts because
    natsort is controlling what is passed to this function.

    Parameters
    ----------
    x : str
        String to attempt to convert to an int.
    key : callable
        Single-argument function to apply to *x* if conversion fails.

    Returns
    -------
    *str* or *int*

    """
    if x[0] in _first_char:
        try:
            return int(x)
        except ValueError:
            try:
                return _uni(x, key(x)) if len(x) == 1 else key(x)
            except TypeError:  # pragma: no cover
                return key(x)
    else:
        try:
            return _uni(x, key(x)) if len(x) == 1 else key(x)
        except TypeError:  # pragma: no cover
            return key(x)
