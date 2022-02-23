# -*- coding: utf-8 -*-
"""
Interface for natsort to access locale functionality without
having to worry about if it is using PyICU or the built-in locale.
"""
import sys
from typing import Callable, Union, cast

StrOrBytes = Union[str, bytes]
TrxfmFunc = Callable[[str], StrOrBytes]

# This string should be sorted after any other byte string because
# it contains the max unicode character repeated 20 times.
# You would need some odd data to come after that.
null_string = ""
null_string_max = chr(sys.maxunicode) * 20

# This variable could be str or bytes depending on the locale library
# being used, so give the type-checker this information.
null_string_locale: StrOrBytes
null_string_locale_max: StrOrBytes

# strxfrm can be buggy (especially on BSD-based systems),
# so prefer icu if available.
try:  # noqa: C901
    import icu
    from locale import getlocale

    null_string_locale = b""

    # This string should in theory be sorted after any other byte
    # string because it contains the max byte char repeated many times.
    # You would need some odd data to come after that.
    null_string_locale_max = b"x7f" * 50

    def dumb_sort() -> bool:
        return False

    # If using icu, get the locale from the current global locale,
    def get_icu_locale() -> str:
        try:
            return cast(str, icu.Locale(".".join(getlocale())))
        except TypeError:  # pragma: no cover
            return cast(str, icu.Locale())

    def get_strxfrm() -> TrxfmFunc:
        return cast(TrxfmFunc, icu.Collator.createInstance(get_icu_locale()).getSortKey)

    def get_thousands_sep() -> str:
        sep = icu.DecimalFormatSymbols.kGroupingSeparatorSymbol
        return cast(str, icu.DecimalFormatSymbols(get_icu_locale()).getSymbol(sep))

    def get_decimal_point() -> str:
        sep = icu.DecimalFormatSymbols.kDecimalSeparatorSymbol
        return cast(str, icu.DecimalFormatSymbols(get_icu_locale()).getSymbol(sep))

except ImportError:
    import locale
    from locale import strxfrm

    null_string_locale = null_string
    null_string_locale_max = null_string_max

    # On some systems, locale is broken and does not sort in the expected
    # order. We will try to detect this and compensate.
    def dumb_sort() -> bool:
        return strxfrm("A") < strxfrm("a")

    def get_strxfrm() -> TrxfmFunc:
        return strxfrm

    def get_thousands_sep() -> str:
        sep = cast(str, locale.localeconv()["thousands_sep"])
        # If this locale library is broken, some of the thousands separator
        # characters are incorrectly blank. Here is a lookup table of the
        # corrections I am aware of.
        if dumb_sort():
            try:
                loc = ".".join(locale.getlocale())
            except TypeError:  # No locale loaded, default to ','
                return ","
            return {
                "de_DE.ISO8859-15": ".",
                "es_ES.ISO8859-1": ".",
                "de_AT.ISO8859-1": ".",
                "de_at": "\xa0",
                "nl_NL.UTF-8": ".",
                "es_es": ".",
                "fr_CH.ISO8859-15": "\xa0",
                "fr_CA.ISO8859-1": "\xa0",
                "de_CH.ISO8859-1": ".",
                "fr_FR.ISO8859-15": "\xa0",
                "nl_NL.ISO8859-1": ".",
                "ca_ES.UTF-8": ".",
                "nl_NL.ISO8859-15": ".",
                "de_ch": "'",
                "ca_es": ".",
                "de_AT.ISO8859-15": ".",
                "ca_ES.ISO8859-1": ".",
                "de_AT.UTF-8": ".",
                "es_ES.UTF-8": ".",
                "fr_fr": "\xa0",
                "es_ES.ISO8859-15": ".",
                "de_DE.ISO8859-1": ".",
                "nl_nl": ".",
                "fr_ch": "\xa0",
                "fr_ca": "\xa0",
                "de_DE.UTF-8": ".",
                "ca_ES.ISO8859-15": ".",
                "de_CH.ISO8859-15": ".",
                "fr_FR.ISO8859-1": "\xa0",
                "fr_CH.ISO8859-1": "\xa0",
                "de_de": ".",
                "fr_FR.UTF-8": "\xa0",
                "fr_CA.ISO8859-15": "\xa0",
            }.get(loc, sep)
        else:
            return sep

    def get_decimal_point() -> str:
        return cast(str, locale.localeconv()["decimal_point"])
