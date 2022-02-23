# -- encoding: UTF-8 --

from babel._compat import string_types
from babel.core import Locale
from babel.numbers import format_decimal, LC_NUMERIC


class UnknownUnitError(ValueError):
    def __init__(self, unit, locale):
        ValueError.__init__(self, "%s is not a known unit in %s" % (unit, locale))


def get_unit_name(measurement_unit, length='long', locale=LC_NUMERIC):
    """
    Get the display name for a measurement unit in the given locale.

    >>> get_unit_name("radian", locale="en")
    'radians'

    Unknown units will raise exceptions:

    >>> get_unit_name("battery", locale="fi")
    Traceback (most recent call last):
        ...
    UnknownUnitError: battery/long is not a known unit/length in fi

    :param measurement_unit: the code of a measurement unit.
                             Known units can be found in the CLDR Unit Validity XML file:
                             https://unicode.org/repos/cldr/tags/latest/common/validity/unit.xml

    :param length: "short", "long" or "narrow"
    :param locale: the `Locale` object or locale identifier
    :return: The unit display name, or None.
    """
    locale = Locale.parse(locale)
    unit = _find_unit_pattern(measurement_unit, locale=locale)
    if not unit:
        raise UnknownUnitError(unit=measurement_unit, locale=locale)
    return locale.unit_display_names.get(unit, {}).get(length)


def _find_unit_pattern(unit_id, locale=LC_NUMERIC):
    """
    Expand an unit into a qualified form.

    Known units can be found in the CLDR Unit Validity XML file:
    https://unicode.org/repos/cldr/tags/latest/common/validity/unit.xml

    >>> _find_unit_pattern("radian", locale="en")
    'angle-radian'

    Unknown values will return None.

    >>> _find_unit_pattern("horse", locale="en")

    :param unit_id: the code of a measurement unit.
    :return: A key to the `unit_patterns` mapping, or None.
    """
    locale = Locale.parse(locale)
    unit_patterns = locale._data["unit_patterns"]
    if unit_id in unit_patterns:
        return unit_id
    for unit_pattern in sorted(unit_patterns, key=len):
        if unit_pattern.endswith(unit_id):
            return unit_pattern


def format_unit(value, measurement_unit, length='long', format=None, locale=LC_NUMERIC):
    """Format a value of a given unit.

    Values are formatted according to the locale's usual pluralization rules
    and number formats.

    >>> format_unit(12, 'length-meter', locale='ro_RO')
    u'12 metri'
    >>> format_unit(15.5, 'length-mile', locale='fi_FI')
    u'15,5 mailia'
    >>> format_unit(1200, 'pressure-millimeter-ofhg', locale='nb')
    u'1\\xa0200 millimeter kvikks\\xf8lv'
    >>> format_unit(270, 'ton', locale='en')
    u'270 tons'

    Number formats may be overridden with the ``format`` parameter.

    >>> from babel._compat import decimal
    >>> format_unit(decimal.Decimal("-42.774"), 'temperature-celsius', 'short', format='#.0', locale='fr')
    u'-42,8\\u202f\\xb0C'

    The locale's usual pluralization rules are respected.

    >>> format_unit(1, 'length-meter', locale='ro_RO')
    u'1 metru'
    >>> format_unit(0, 'length-mile', locale='cy')
    u'0 mi'
    >>> format_unit(1, 'length-mile', locale='cy')
    u'1 filltir'
    >>> format_unit(3, 'length-mile', locale='cy')
    u'3 milltir'

    >>> format_unit(15, 'length-horse', locale='fi')
    Traceback (most recent call last):
        ...
    UnknownUnitError: length-horse is not a known unit in fi

    .. versionadded:: 2.2.0

    :param value: the value to format. If this is a string, no number formatting will be attempted.
    :param measurement_unit: the code of a measurement unit.
                             Known units can be found in the CLDR Unit Validity XML file:
                             https://unicode.org/repos/cldr/tags/latest/common/validity/unit.xml
    :param length: "short", "long" or "narrow"
    :param format: An optional format, as accepted by `format_decimal`.
    :param locale: the `Locale` object or locale identifier
    """
    locale = Locale.parse(locale)

    q_unit = _find_unit_pattern(measurement_unit, locale=locale)
    if not q_unit:
        raise UnknownUnitError(unit=measurement_unit, locale=locale)
    unit_patterns = locale._data["unit_patterns"][q_unit].get(length, {})

    if isinstance(value, string_types):  # Assume the value is a preformatted singular.
        formatted_value = value
        plural_form = "one"
    else:
        formatted_value = format_decimal(value, format, locale)
        plural_form = locale.plural_form(value)

    if plural_form in unit_patterns:
        return unit_patterns[plural_form].format(formatted_value)

    # Fall back to a somewhat bad representation.
    # nb: This is marked as no-cover, as the current CLDR seemingly has no way for this to happen.
    return '%s %s' % (  # pragma: no cover
        formatted_value,
        (get_unit_name(measurement_unit, length=length, locale=locale) or measurement_unit)
    )


def _find_compound_unit(numerator_unit, denominator_unit, locale=LC_NUMERIC):
    """
    Find a predefined compound unit pattern.

    Used internally by format_compound_unit.

    >>> _find_compound_unit("kilometer", "hour", locale="en")
    'speed-kilometer-per-hour'

    >>> _find_compound_unit("mile", "gallon", locale="en")
    'consumption-mile-per-gallon'

    If no predefined compound pattern can be found, `None` is returned.

    >>> _find_compound_unit("gallon", "mile", locale="en")

    >>> _find_compound_unit("horse", "purple", locale="en")

    :param numerator_unit: The numerator unit's identifier
    :param denominator_unit: The denominator unit's identifier
    :param locale: the `Locale` object or locale identifier
    :return: A key to the `unit_patterns` mapping, or None.
    :rtype: str|None
    """
    locale = Locale.parse(locale)

    # Qualify the numerator and denominator units.  This will turn possibly partial
    # units like "kilometer" or "hour" into actual units like "length-kilometer" and
    # "duration-hour".

    numerator_unit = _find_unit_pattern(numerator_unit, locale=locale)
    denominator_unit = _find_unit_pattern(denominator_unit, locale=locale)

    # If either was not found, we can't possibly build a suitable compound unit either.
    if not (numerator_unit and denominator_unit):
        return None

    # Since compound units are named "speed-kilometer-per-hour", we'll have to slice off
    # the quantities (i.e. "length", "duration") from both qualified units.

    bare_numerator_unit = numerator_unit.split("-", 1)[-1]
    bare_denominator_unit = denominator_unit.split("-", 1)[-1]

    # Now we can try and rebuild a compound unit specifier, then qualify it:

    return _find_unit_pattern("%s-per-%s" % (bare_numerator_unit, bare_denominator_unit), locale=locale)


def format_compound_unit(
    numerator_value, numerator_unit=None,
    denominator_value=1, denominator_unit=None,
    length='long', format=None, locale=LC_NUMERIC
):
    """
    Format a compound number value, i.e. "kilometers per hour" or similar.

    Both unit specifiers are optional to allow for formatting of arbitrary values still according
    to the locale's general "per" formatting specifier.

    >>> format_compound_unit(7, denominator_value=11, length="short", locale="pt")
    '7/11'

    >>> format_compound_unit(150, "kilometer", denominator_unit="hour", locale="sv")
    '150 kilometer per timme'

    >>> format_compound_unit(150, "kilowatt", denominator_unit="year", locale="fi")
    '150 kilowattia / vuosi'

    >>> format_compound_unit(32.5, "ton", 15, denominator_unit="hour", locale="en")
    '32.5 tons per 15 hours'

    >>> format_compound_unit(160, denominator_unit="square-meter", locale="fr")
    '160 par m\\xe8tre carr\\xe9'

    >>> format_compound_unit(4, "meter", "ratakisko", length="short", locale="fi")
    '4 m/ratakisko'

    >>> format_compound_unit(35, "minute", denominator_unit="fathom", locale="sv")
    '35 minuter per famn'

    >>> from babel.numbers import format_currency
    >>> format_compound_unit(format_currency(35, "JPY", locale="de"), denominator_unit="liter", locale="de")
    '35\\xa0\\xa5 pro Liter'

    See https://www.unicode.org/reports/tr35/tr35-general.html#perUnitPatterns

    :param numerator_value: The numerator value. This may be a string,
                            in which case it is considered preformatted and the unit is ignored.
    :param numerator_unit: The numerator unit. See `format_unit`.
    :param denominator_value: The denominator value. This may be a string,
                              in which case it is considered preformatted and the unit is ignored.
    :param denominator_unit: The denominator unit. See `format_unit`.
    :param length: The formatting length. "short", "long" or "narrow"
    :param format: An optional format, as accepted by `format_decimal`.
    :param locale: the `Locale` object or locale identifier
    :return: A formatted compound value.
    """
    locale = Locale.parse(locale)

    # Look for a specific compound unit first...

    if numerator_unit and denominator_unit and denominator_value == 1:
        compound_unit = _find_compound_unit(numerator_unit, denominator_unit, locale=locale)
        if compound_unit:
            return format_unit(numerator_value, compound_unit, length=length, format=format, locale=locale)

    # ... failing that, construct one "by hand".

    if isinstance(numerator_value, string_types):  # Numerator is preformatted
        formatted_numerator = numerator_value
    elif numerator_unit:  # Numerator has unit
        formatted_numerator = format_unit(
            numerator_value, numerator_unit, length=length, format=format, locale=locale
        )
    else:  # Unitless numerator
        formatted_numerator = format_decimal(numerator_value, format=format, locale=locale)

    if isinstance(denominator_value, string_types):  # Denominator is preformatted
        formatted_denominator = denominator_value
    elif denominator_unit:  # Denominator has unit
        if denominator_value == 1:  # support perUnitPatterns when the denominator is 1
            denominator_unit = _find_unit_pattern(denominator_unit, locale=locale)
            per_pattern = locale._data["unit_patterns"].get(denominator_unit, {}).get(length, {}).get("per")
            if per_pattern:
                return per_pattern.format(formatted_numerator)
            # See TR-35's per-unit pattern algorithm, point 3.2.
            # For denominator 1, we replace the value to be formatted with the empty string;
            # this will make `format_unit` return " second" instead of "1 second".
            denominator_value = ""

        formatted_denominator = format_unit(
            denominator_value, denominator_unit, length=length, format=format, locale=locale
        ).strip()
    else:  # Bare denominator
        formatted_denominator = format_decimal(denominator_value, format=format, locale=locale)

    # TODO: this doesn't support "compound_variations" (or "prefix"), and will fall back to the "x/y" representation
    per_pattern = locale._data["compound_unit_patterns"].get("per", {}).get(length, {}).get("compound", "{0}/{1}")

    return per_pattern.format(formatted_numerator, formatted_denominator)
