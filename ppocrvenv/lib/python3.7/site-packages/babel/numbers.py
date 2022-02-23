# -*- coding: utf-8 -*-
"""
    babel.numbers
    ~~~~~~~~~~~~~

    Locale dependent formatting and parsing of numeric data.

    The default locale for the functions in this module is determined by the
    following environment variables, in that order:

     * ``LC_NUMERIC``,
     * ``LC_ALL``, and
     * ``LANG``

    :copyright: (c) 2013-2021 by the Babel Team.
    :license: BSD, see LICENSE for more details.
"""
# TODO:
#  Padding and rounding increments in pattern:
#  - https://www.unicode.org/reports/tr35/ (Appendix G.6)
import re
from datetime import date as date_, datetime as datetime_
import warnings

from babel.core import default_locale, Locale, get_global
from babel._compat import decimal, string_types

try:
    # Python 2
    long
except NameError:
    # Python 3
    long = int


LC_NUMERIC = default_locale('LC_NUMERIC')


class UnknownCurrencyError(Exception):
    """Exception thrown when a currency is requested for which no data is available.
    """

    def __init__(self, identifier):
        """Create the exception.
        :param identifier: the identifier string of the unsupported currency
        """
        Exception.__init__(self, 'Unknown currency %r.' % identifier)

        #: The identifier of the locale that could not be found.
        self.identifier = identifier


def list_currencies(locale=None):
    """ Return a `set` of normalized currency codes.

    .. versionadded:: 2.5.0

    :param locale: filters returned currency codes by the provided locale.
                   Expected to be a locale instance or code. If no locale is
                   provided, returns the list of all currencies from all
                   locales.
    """
    # Get locale-scoped currencies.
    if locale:
        currencies = Locale.parse(locale).currencies.keys()
    else:
        currencies = get_global('all_currencies')
    return set(currencies)


def validate_currency(currency, locale=None):
    """ Check the currency code is recognized by Babel.

    Accepts a ``locale`` parameter for fined-grained validation, working as
    the one defined above in ``list_currencies()`` method.

    Raises a `UnknownCurrencyError` exception if the currency is unknown to Babel.
    """
    if currency not in list_currencies(locale):
        raise UnknownCurrencyError(currency)


def is_currency(currency, locale=None):
    """ Returns `True` only if a currency is recognized by Babel.

    This method always return a Boolean and never raise.
    """
    if not currency or not isinstance(currency, string_types):
        return False
    try:
        validate_currency(currency, locale)
    except UnknownCurrencyError:
        return False
    return True


def normalize_currency(currency, locale=None):
    """Returns the normalized sting of any currency code.

    Accepts a ``locale`` parameter for fined-grained validation, working as
    the one defined above in ``list_currencies()`` method.

    Returns None if the currency is unknown to Babel.
    """
    if isinstance(currency, string_types):
        currency = currency.upper()
    if not is_currency(currency, locale):
        return
    return currency


def get_currency_name(currency, count=None, locale=LC_NUMERIC):
    """Return the name used by the locale for the specified currency.

    >>> get_currency_name('USD', locale='en_US')
    u'US Dollar'

    .. versionadded:: 0.9.4

    :param currency: the currency code.
    :param count: the optional count.  If provided the currency name
                  will be pluralized to that number if possible.
    :param locale: the `Locale` object or locale identifier.
    """
    loc = Locale.parse(locale)
    if count is not None:
        plural_form = loc.plural_form(count)
        plural_names = loc._data['currency_names_plural']
        if currency in plural_names:
            return plural_names[currency][plural_form]
    return loc.currencies.get(currency, currency)


def get_currency_symbol(currency, locale=LC_NUMERIC):
    """Return the symbol used by the locale for the specified currency.

    >>> get_currency_symbol('USD', locale='en_US')
    u'$'

    :param currency: the currency code.
    :param locale: the `Locale` object or locale identifier.
    """
    return Locale.parse(locale).currency_symbols.get(currency, currency)


def get_currency_precision(currency):
    """Return currency's precision.

    Precision is the number of decimals found after the decimal point in the
    currency's format pattern.

    .. versionadded:: 2.5.0

    :param currency: the currency code.
    """
    precisions = get_global('currency_fractions')
    return precisions.get(currency, precisions['DEFAULT'])[0]


def get_currency_unit_pattern(currency, count=None, locale=LC_NUMERIC):
    """
    Return the unit pattern used for long display of a currency value
    for a given locale.
    This is a string containing ``{0}`` where the numeric part
    should be substituted and ``{1}`` where the currency long display
    name should be substituted.

    >>> get_currency_unit_pattern('USD', locale='en_US', count=10)
    u'{0} {1}'

    .. versionadded:: 2.7.0

    :param currency: the currency code.
    :param count: the optional count.  If provided the unit
                  pattern for that number will be returned.
    :param locale: the `Locale` object or locale identifier.
    """
    loc = Locale.parse(locale)
    if count is not None:
        plural_form = loc.plural_form(count)
        try:
            return loc._data['currency_unit_patterns'][plural_form]
        except LookupError:
            # Fall back to 'other'
            pass

    return loc._data['currency_unit_patterns']['other']


def get_territory_currencies(territory, start_date=None, end_date=None,
                             tender=True, non_tender=False,
                             include_details=False):
    """Returns the list of currencies for the given territory that are valid for
    the given date range.  In addition to that the currency database
    distinguishes between tender and non-tender currencies.  By default only
    tender currencies are returned.

    The return value is a list of all currencies roughly ordered by the time
    of when the currency became active.  The longer the currency is being in
    use the more to the left of the list it will be.

    The start date defaults to today.  If no end date is given it will be the
    same as the start date.  Otherwise a range can be defined.  For instance
    this can be used to find the currencies in use in Austria between 1995 and
    2011:

    >>> from datetime import date
    >>> get_territory_currencies('AT', date(1995, 1, 1), date(2011, 1, 1))
    ['ATS', 'EUR']

    Likewise it's also possible to find all the currencies in use on a
    single date:

    >>> get_territory_currencies('AT', date(1995, 1, 1))
    ['ATS']
    >>> get_territory_currencies('AT', date(2011, 1, 1))
    ['EUR']

    By default the return value only includes tender currencies.  This
    however can be changed:

    >>> get_territory_currencies('US')
    ['USD']
    >>> get_territory_currencies('US', tender=False, non_tender=True,
    ...                          start_date=date(2014, 1, 1))
    ['USN', 'USS']

    .. versionadded:: 2.0

    :param territory: the name of the territory to find the currency for.
    :param start_date: the start date.  If not given today is assumed.
    :param end_date: the end date.  If not given the start date is assumed.
    :param tender: controls whether tender currencies should be included.
    :param non_tender: controls whether non-tender currencies should be
                       included.
    :param include_details: if set to `True`, instead of returning currency
                            codes the return value will be dictionaries
                            with detail information.  In that case each
                            dictionary will have the keys ``'currency'``,
                            ``'from'``, ``'to'``, and ``'tender'``.
    """
    currencies = get_global('territory_currencies')
    if start_date is None:
        start_date = date_.today()
    elif isinstance(start_date, datetime_):
        start_date = start_date.date()
    if end_date is None:
        end_date = start_date
    elif isinstance(end_date, datetime_):
        end_date = end_date.date()

    curs = currencies.get(territory.upper(), ())
    # TODO: validate that the territory exists

    def _is_active(start, end):
        return (start is None or start <= end_date) and \
               (end is None or end >= start_date)

    result = []
    for currency_code, start, end, is_tender in curs:
        if start:
            start = date_(*start)
        if end:
            end = date_(*end)
        if ((is_tender and tender) or
                (not is_tender and non_tender)) and _is_active(start, end):
            if include_details:
                result.append({
                    'currency': currency_code,
                    'from': start,
                    'to': end,
                    'tender': is_tender,
                })
            else:
                result.append(currency_code)

    return result


def get_decimal_symbol(locale=LC_NUMERIC):
    """Return the symbol used by the locale to separate decimal fractions.

    >>> get_decimal_symbol('en_US')
    u'.'

    :param locale: the `Locale` object or locale identifier
    """
    return Locale.parse(locale).number_symbols.get('decimal', u'.')


def get_plus_sign_symbol(locale=LC_NUMERIC):
    """Return the plus sign symbol used by the current locale.

    >>> get_plus_sign_symbol('en_US')
    u'+'

    :param locale: the `Locale` object or locale identifier
    """
    return Locale.parse(locale).number_symbols.get('plusSign', u'+')


def get_minus_sign_symbol(locale=LC_NUMERIC):
    """Return the plus sign symbol used by the current locale.

    >>> get_minus_sign_symbol('en_US')
    u'-'

    :param locale: the `Locale` object or locale identifier
    """
    return Locale.parse(locale).number_symbols.get('minusSign', u'-')


def get_exponential_symbol(locale=LC_NUMERIC):
    """Return the symbol used by the locale to separate mantissa and exponent.

    >>> get_exponential_symbol('en_US')
    u'E'

    :param locale: the `Locale` object or locale identifier
    """
    return Locale.parse(locale).number_symbols.get('exponential', u'E')


def get_group_symbol(locale=LC_NUMERIC):
    """Return the symbol used by the locale to separate groups of thousands.

    >>> get_group_symbol('en_US')
    u','

    :param locale: the `Locale` object or locale identifier
    """
    return Locale.parse(locale).number_symbols.get('group', u',')


def format_number(number, locale=LC_NUMERIC):
    u"""Return the given number formatted for a specific locale.

    >>> format_number(1099, locale='en_US')
    u'1,099'
    >>> format_number(1099, locale='de_DE')
    u'1.099'

    .. deprecated:: 2.6.0

       Use babel.numbers.format_decimal() instead.

    :param number: the number to format
    :param locale: the `Locale` object or locale identifier


    """
    warnings.warn('Use babel.numbers.format_decimal() instead.', DeprecationWarning)
    return format_decimal(number, locale=locale)


def get_decimal_precision(number):
    """Return maximum precision of a decimal instance's fractional part.

    Precision is extracted from the fractional part only.
    """
    # Copied from: https://github.com/mahmoud/boltons/pull/59
    assert isinstance(number, decimal.Decimal)
    decimal_tuple = number.normalize().as_tuple()
    if decimal_tuple.exponent >= 0:
        return 0
    return abs(decimal_tuple.exponent)


def get_decimal_quantum(precision):
    """Return minimal quantum of a number, as defined by precision."""
    assert isinstance(precision, (int, long, decimal.Decimal))
    return decimal.Decimal(10) ** (-precision)


def format_decimal(
        number, format=None, locale=LC_NUMERIC, decimal_quantization=True, group_separator=True):
    u"""Return the given decimal number formatted for a specific locale.

    >>> format_decimal(1.2345, locale='en_US')
    u'1.234'
    >>> format_decimal(1.2346, locale='en_US')
    u'1.235'
    >>> format_decimal(-1.2346, locale='en_US')
    u'-1.235'
    >>> format_decimal(1.2345, locale='sv_SE')
    u'1,234'
    >>> format_decimal(1.2345, locale='de')
    u'1,234'

    The appropriate thousands grouping and the decimal separator are used for
    each locale:

    >>> format_decimal(12345.5, locale='en_US')
    u'12,345.5'

    By default the locale is allowed to truncate and round a high-precision
    number by forcing its format pattern onto the decimal part. You can bypass
    this behavior with the `decimal_quantization` parameter:

    >>> format_decimal(1.2346, locale='en_US')
    u'1.235'
    >>> format_decimal(1.2346, locale='en_US', decimal_quantization=False)
    u'1.2346'
    >>> format_decimal(12345.67, locale='fr_CA', group_separator=False)
    u'12345,67'
    >>> format_decimal(12345.67, locale='en_US', group_separator=True)
    u'12,345.67'

    :param number: the number to format
    :param format:
    :param locale: the `Locale` object or locale identifier
    :param decimal_quantization: Truncate and round high-precision numbers to
                                 the format pattern. Defaults to `True`.
    :param group_separator: Boolean to switch group separator on/off in a locale's
                            number format.
    """
    locale = Locale.parse(locale)
    if not format:
        format = locale.decimal_formats.get(format)
    pattern = parse_pattern(format)
    return pattern.apply(
        number, locale, decimal_quantization=decimal_quantization, group_separator=group_separator)


class UnknownCurrencyFormatError(KeyError):
    """Exception raised when an unknown currency format is requested."""


def format_currency(
        number, currency, format=None, locale=LC_NUMERIC, currency_digits=True,
        format_type='standard', decimal_quantization=True, group_separator=True):
    u"""Return formatted currency value.

    >>> format_currency(1099.98, 'USD', locale='en_US')
    u'$1,099.98'
    >>> format_currency(1099.98, 'USD', locale='es_CO')
    u'US$\\xa01.099,98'
    >>> format_currency(1099.98, 'EUR', locale='de_DE')
    u'1.099,98\\xa0\\u20ac'

    The format can also be specified explicitly.  The currency is
    placed with the '¤' sign.  As the sign gets repeated the format
    expands (¤ being the symbol, ¤¤ is the currency abbreviation and
    ¤¤¤ is the full name of the currency):

    >>> format_currency(1099.98, 'EUR', u'\xa4\xa4 #,##0.00', locale='en_US')
    u'EUR 1,099.98'
    >>> format_currency(1099.98, 'EUR', u'#,##0.00 \xa4\xa4\xa4', locale='en_US')
    u'1,099.98 euros'

    Currencies usually have a specific number of decimal digits. This function
    favours that information over the given format:

    >>> format_currency(1099.98, 'JPY', locale='en_US')
    u'\\xa51,100'
    >>> format_currency(1099.98, 'COP', u'#,##0.00', locale='es_ES')
    u'1.099,98'

    However, the number of decimal digits can be overriden from the currency
    information, by setting the last parameter to ``False``:

    >>> format_currency(1099.98, 'JPY', locale='en_US', currency_digits=False)
    u'\\xa51,099.98'
    >>> format_currency(1099.98, 'COP', u'#,##0.00', locale='es_ES', currency_digits=False)
    u'1.099,98'

    If a format is not specified the type of currency format to use
    from the locale can be specified:

    >>> format_currency(1099.98, 'EUR', locale='en_US', format_type='standard')
    u'\\u20ac1,099.98'

    When the given currency format type is not available, an exception is
    raised:

    >>> format_currency('1099.98', 'EUR', locale='root', format_type='unknown')
    Traceback (most recent call last):
        ...
    UnknownCurrencyFormatError: "'unknown' is not a known currency format type"

    >>> format_currency(101299.98, 'USD', locale='en_US', group_separator=False)
    u'$101299.98'

    >>> format_currency(101299.98, 'USD', locale='en_US', group_separator=True)
    u'$101,299.98'

    You can also pass format_type='name' to use long display names. The order of
    the number and currency name, along with the correct localized plural form
    of the currency name, is chosen according to locale:

    >>> format_currency(1, 'USD', locale='en_US', format_type='name')
    u'1.00 US dollar'
    >>> format_currency(1099.98, 'USD', locale='en_US', format_type='name')
    u'1,099.98 US dollars'
    >>> format_currency(1099.98, 'USD', locale='ee', format_type='name')
    u'us ga dollar 1,099.98'

    By default the locale is allowed to truncate and round a high-precision
    number by forcing its format pattern onto the decimal part. You can bypass
    this behavior with the `decimal_quantization` parameter:

    >>> format_currency(1099.9876, 'USD', locale='en_US')
    u'$1,099.99'
    >>> format_currency(1099.9876, 'USD', locale='en_US', decimal_quantization=False)
    u'$1,099.9876'

    :param number: the number to format
    :param currency: the currency code
    :param format: the format string to use
    :param locale: the `Locale` object or locale identifier
    :param currency_digits: use the currency's natural number of decimal digits
    :param format_type: the currency format type to use
    :param decimal_quantization: Truncate and round high-precision numbers to
                                 the format pattern. Defaults to `True`.
    :param group_separator: Boolean to switch group separator on/off in a locale's
                            number format.

    """
    if format_type == 'name':
        return _format_currency_long_name(number, currency, format=format,
                                          locale=locale, currency_digits=currency_digits,
                                          decimal_quantization=decimal_quantization, group_separator=group_separator)
    locale = Locale.parse(locale)
    if format:
        pattern = parse_pattern(format)
    else:
        try:
            pattern = locale.currency_formats[format_type]
        except KeyError:
            raise UnknownCurrencyFormatError(
                "%r is not a known currency format type" % format_type)

    return pattern.apply(
        number, locale, currency=currency, currency_digits=currency_digits,
        decimal_quantization=decimal_quantization, group_separator=group_separator)


def _format_currency_long_name(
        number, currency, format=None, locale=LC_NUMERIC, currency_digits=True,
        format_type='standard', decimal_quantization=True, group_separator=True):
    # Algorithm described here:
    # https://www.unicode.org/reports/tr35/tr35-numbers.html#Currencies
    locale = Locale.parse(locale)
    # Step 1.
    # There are no examples of items with explicit count (0 or 1) in current
    # locale data. So there is no point implementing that.
    # Step 2.

    # Correct number to numeric type, important for looking up plural rules:
    if isinstance(number, string_types):
        number_n = float(number)
    else:
        number_n = number

    # Step 3.
    unit_pattern = get_currency_unit_pattern(currency, count=number_n, locale=locale)

    # Step 4.
    display_name = get_currency_name(currency, count=number_n, locale=locale)

    # Step 5.
    if not format:
        format = locale.decimal_formats.get(format)

    pattern = parse_pattern(format)

    number_part = pattern.apply(
        number, locale, currency=currency, currency_digits=currency_digits,
        decimal_quantization=decimal_quantization, group_separator=group_separator)

    return unit_pattern.format(number_part, display_name)


def format_percent(
        number, format=None, locale=LC_NUMERIC, decimal_quantization=True, group_separator=True):
    """Return formatted percent value for a specific locale.

    >>> format_percent(0.34, locale='en_US')
    u'34%'
    >>> format_percent(25.1234, locale='en_US')
    u'2,512%'
    >>> format_percent(25.1234, locale='sv_SE')
    u'2\\xa0512\\xa0%'

    The format pattern can also be specified explicitly:

    >>> format_percent(25.1234, u'#,##0\u2030', locale='en_US')
    u'25,123\u2030'

    By default the locale is allowed to truncate and round a high-precision
    number by forcing its format pattern onto the decimal part. You can bypass
    this behavior with the `decimal_quantization` parameter:

    >>> format_percent(23.9876, locale='en_US')
    u'2,399%'
    >>> format_percent(23.9876, locale='en_US', decimal_quantization=False)
    u'2,398.76%'

    >>> format_percent(229291.1234, locale='pt_BR', group_separator=False)
    u'22929112%'

    >>> format_percent(229291.1234, locale='pt_BR', group_separator=True)
    u'22.929.112%'

    :param number: the percent number to format
    :param format:
    :param locale: the `Locale` object or locale identifier
    :param decimal_quantization: Truncate and round high-precision numbers to
                                 the format pattern. Defaults to `True`.
    :param group_separator: Boolean to switch group separator on/off in a locale's
                            number format.
    """
    locale = Locale.parse(locale)
    if not format:
        format = locale.percent_formats.get(format)
    pattern = parse_pattern(format)
    return pattern.apply(
        number, locale, decimal_quantization=decimal_quantization, group_separator=group_separator)


def format_scientific(
        number, format=None, locale=LC_NUMERIC, decimal_quantization=True):
    """Return value formatted in scientific notation for a specific locale.

    >>> format_scientific(10000, locale='en_US')
    u'1E4'

    The format pattern can also be specified explicitly:

    >>> format_scientific(1234567, u'##0.##E00', locale='en_US')
    u'1.23E06'

    By default the locale is allowed to truncate and round a high-precision
    number by forcing its format pattern onto the decimal part. You can bypass
    this behavior with the `decimal_quantization` parameter:

    >>> format_scientific(1234.9876, u'#.##E0', locale='en_US')
    u'1.23E3'
    >>> format_scientific(1234.9876, u'#.##E0', locale='en_US', decimal_quantization=False)
    u'1.2349876E3'

    :param number: the number to format
    :param format:
    :param locale: the `Locale` object or locale identifier
    :param decimal_quantization: Truncate and round high-precision numbers to
                                 the format pattern. Defaults to `True`.
    """
    locale = Locale.parse(locale)
    if not format:
        format = locale.scientific_formats.get(format)
    pattern = parse_pattern(format)
    return pattern.apply(
        number, locale, decimal_quantization=decimal_quantization)


class NumberFormatError(ValueError):
    """Exception raised when a string cannot be parsed into a number."""

    def __init__(self, message, suggestions=None):
        super(NumberFormatError, self).__init__(message)
        #: a list of properly formatted numbers derived from the invalid input
        self.suggestions = suggestions


def parse_number(string, locale=LC_NUMERIC):
    """Parse localized number string into an integer.

    >>> parse_number('1,099', locale='en_US')
    1099
    >>> parse_number('1.099', locale='de_DE')
    1099

    When the given string cannot be parsed, an exception is raised:

    >>> parse_number('1.099,98', locale='de')
    Traceback (most recent call last):
        ...
    NumberFormatError: '1.099,98' is not a valid number

    :param string: the string to parse
    :param locale: the `Locale` object or locale identifier
    :return: the parsed number
    :raise `NumberFormatError`: if the string can not be converted to a number
    """
    try:
        return int(string.replace(get_group_symbol(locale), ''))
    except ValueError:
        raise NumberFormatError('%r is not a valid number' % string)


def parse_decimal(string, locale=LC_NUMERIC, strict=False):
    """Parse localized decimal string into a decimal.

    >>> parse_decimal('1,099.98', locale='en_US')
    Decimal('1099.98')
    >>> parse_decimal('1.099,98', locale='de')
    Decimal('1099.98')
    >>> parse_decimal('12 345,123', locale='ru')
    Decimal('12345.123')

    When the given string cannot be parsed, an exception is raised:

    >>> parse_decimal('2,109,998', locale='de')
    Traceback (most recent call last):
        ...
    NumberFormatError: '2,109,998' is not a valid decimal number

    If `strict` is set to `True` and the given string contains a number
    formatted in an irregular way, an exception is raised:

    >>> parse_decimal('30.00', locale='de', strict=True)
    Traceback (most recent call last):
        ...
    NumberFormatError: '30.00' is not a properly formatted decimal number. Did you mean '3.000'? Or maybe '30,00'?

    >>> parse_decimal('0.00', locale='de', strict=True)
    Traceback (most recent call last):
        ...
    NumberFormatError: '0.00' is not a properly formatted decimal number. Did you mean '0'?

    :param string: the string to parse
    :param locale: the `Locale` object or locale identifier
    :param strict: controls whether numbers formatted in a weird way are
                   accepted or rejected
    :raise NumberFormatError: if the string can not be converted to a
                              decimal number
    """
    locale = Locale.parse(locale)
    group_symbol = get_group_symbol(locale)
    decimal_symbol = get_decimal_symbol(locale)

    if not strict and (
        group_symbol == u'\xa0' and  # if the grouping symbol is U+00A0 NO-BREAK SPACE,
        group_symbol not in string and  # and the string to be parsed does not contain it,
        ' ' in string  # but it does contain a space instead,
    ):
        # ... it's reasonable to assume it is taking the place of the grouping symbol.
        string = string.replace(' ', group_symbol)

    try:
        parsed = decimal.Decimal(string.replace(group_symbol, '')
                                       .replace(decimal_symbol, '.'))
    except decimal.InvalidOperation:
        raise NumberFormatError('%r is not a valid decimal number' % string)
    if strict and group_symbol in string:
        proper = format_decimal(parsed, locale=locale, decimal_quantization=False)
        if string != proper and string.rstrip('0') != (proper + decimal_symbol):
            try:
                parsed_alt = decimal.Decimal(string.replace(decimal_symbol, '')
                                                   .replace(group_symbol, '.'))
            except decimal.InvalidOperation:
                raise NumberFormatError((
                    "%r is not a properly formatted decimal number. Did you mean %r?" %
                    (string, proper)
                ), suggestions=[proper])
            else:
                proper_alt = format_decimal(parsed_alt, locale=locale, decimal_quantization=False)
                if proper_alt == proper:
                    raise NumberFormatError((
                            "%r is not a properly formatted decimal number. Did you mean %r?" %
                            (string, proper)
                    ), suggestions=[proper])
                else:
                    raise NumberFormatError((
                        "%r is not a properly formatted decimal number. Did you mean %r? Or maybe %r?" %
                        (string, proper, proper_alt)
                    ), suggestions=[proper, proper_alt])
    return parsed


PREFIX_END = r'[^0-9@#.,]'
NUMBER_TOKEN = r'[0-9@#.,E+]'

PREFIX_PATTERN = r"(?P<prefix>(?:'[^']*'|%s)*)" % PREFIX_END
NUMBER_PATTERN = r"(?P<number>%s*)" % NUMBER_TOKEN
SUFFIX_PATTERN = r"(?P<suffix>.*)"

number_re = re.compile(r"%s%s%s" % (PREFIX_PATTERN, NUMBER_PATTERN,
                                    SUFFIX_PATTERN))


def parse_grouping(p):
    """Parse primary and secondary digit grouping

    >>> parse_grouping('##')
    (1000, 1000)
    >>> parse_grouping('#,###')
    (3, 3)
    >>> parse_grouping('#,####,###')
    (3, 4)
    """
    width = len(p)
    g1 = p.rfind(',')
    if g1 == -1:
        return 1000, 1000
    g1 = width - g1 - 1
    g2 = p[:-g1 - 1].rfind(',')
    if g2 == -1:
        return g1, g1
    g2 = width - g1 - g2 - 2
    return g1, g2


def parse_pattern(pattern):
    """Parse number format patterns"""
    if isinstance(pattern, NumberPattern):
        return pattern

    def _match_number(pattern):
        rv = number_re.search(pattern)
        if rv is None:
            raise ValueError('Invalid number pattern %r' % pattern)
        return rv.groups()

    pos_pattern = pattern

    # Do we have a negative subpattern?
    if ';' in pattern:
        pos_pattern, neg_pattern = pattern.split(';', 1)
        pos_prefix, number, pos_suffix = _match_number(pos_pattern)
        neg_prefix, _, neg_suffix = _match_number(neg_pattern)
    else:
        pos_prefix, number, pos_suffix = _match_number(pos_pattern)
        neg_prefix = '-' + pos_prefix
        neg_suffix = pos_suffix
    if 'E' in number:
        number, exp = number.split('E', 1)
    else:
        exp = None
    if '@' in number:
        if '.' in number and '0' in number:
            raise ValueError('Significant digit patterns can not contain '
                             '"@" or "0"')
    if '.' in number:
        integer, fraction = number.rsplit('.', 1)
    else:
        integer = number
        fraction = ''

    def parse_precision(p):
        """Calculate the min and max allowed digits"""
        min = max = 0
        for c in p:
            if c in '@0':
                min += 1
                max += 1
            elif c == '#':
                max += 1
            elif c == ',':
                continue
            else:
                break
        return min, max

    int_prec = parse_precision(integer)
    frac_prec = parse_precision(fraction)
    if exp:
        exp_plus = exp.startswith('+')
        exp = exp.lstrip('+')
        exp_prec = parse_precision(exp)
    else:
        exp_plus = None
        exp_prec = None
    grouping = parse_grouping(integer)
    return NumberPattern(pattern, (pos_prefix, neg_prefix),
                         (pos_suffix, neg_suffix), grouping,
                         int_prec, frac_prec,
                         exp_prec, exp_plus)


class NumberPattern(object):

    def __init__(self, pattern, prefix, suffix, grouping,
                 int_prec, frac_prec, exp_prec, exp_plus):
        # Metadata of the decomposed parsed pattern.
        self.pattern = pattern
        self.prefix = prefix
        self.suffix = suffix
        self.grouping = grouping
        self.int_prec = int_prec
        self.frac_prec = frac_prec
        self.exp_prec = exp_prec
        self.exp_plus = exp_plus
        self.scale = self.compute_scale()

    def __repr__(self):
        return '<%s %r>' % (type(self).__name__, self.pattern)

    def compute_scale(self):
        """Return the scaling factor to apply to the number before rendering.

        Auto-set to a factor of 2 or 3 if presence of a ``%`` or ``‰`` sign is
        detected in the prefix or suffix of the pattern. Default is to not mess
        with the scale at all and keep it to 0.
        """
        scale = 0
        if '%' in ''.join(self.prefix + self.suffix):
            scale = 2
        elif u'‰' in ''.join(self.prefix + self.suffix):
            scale = 3
        return scale

    def scientific_notation_elements(self, value, locale):
        """ Returns normalized scientific notation components of a value.
        """
        # Normalize value to only have one lead digit.
        exp = value.adjusted()
        value = value * get_decimal_quantum(exp)
        assert value.adjusted() == 0

        # Shift exponent and value by the minimum number of leading digits
        # imposed by the rendering pattern. And always make that number
        # greater or equal to 1.
        lead_shift = max([1, min(self.int_prec)]) - 1
        exp = exp - lead_shift
        value = value * get_decimal_quantum(-lead_shift)

        # Get exponent sign symbol.
        exp_sign = ''
        if exp < 0:
            exp_sign = get_minus_sign_symbol(locale)
        elif self.exp_plus:
            exp_sign = get_plus_sign_symbol(locale)

        # Normalize exponent value now that we have the sign.
        exp = abs(exp)

        return value, exp, exp_sign

    def apply(
        self,
        value,
        locale,
        currency=None,
        currency_digits=True,
        decimal_quantization=True,
        force_frac=None,
        group_separator=True,
    ):
        """Renders into a string a number following the defined pattern.

        Forced decimal quantization is active by default so we'll produce a
        number string that is strictly following CLDR pattern definitions.

        :param value: The value to format. If this is not a Decimal object,
                      it will be cast to one.
        :type value: decimal.Decimal|float|int
        :param locale: The locale to use for formatting.
        :type locale: str|babel.core.Locale
        :param currency: Which currency, if any, to format as.
        :type currency: str|None
        :param currency_digits: Whether or not to use the currency's precision.
                                If false, the pattern's precision is used.
        :type currency_digits: bool
        :param decimal_quantization: Whether decimal numbers should be forcibly
                                     quantized to produce a formatted output
                                     strictly matching the CLDR definition for
                                     the locale.
        :type decimal_quantization: bool
        :param force_frac: DEPRECATED - a forced override for `self.frac_prec`
                           for a single formatting invocation.
        :return: Formatted decimal string.
        :rtype: str
        """
        if not isinstance(value, decimal.Decimal):
            value = decimal.Decimal(str(value))

        value = value.scaleb(self.scale)

        # Separate the absolute value from its sign.
        is_negative = int(value.is_signed())
        value = abs(value).normalize()

        # Prepare scientific notation metadata.
        if self.exp_prec:
            value, exp, exp_sign = self.scientific_notation_elements(value, locale)

        # Adjust the precision of the fractional part and force it to the
        # currency's if necessary.
        if force_frac:
            # TODO (3.x?): Remove this parameter
            warnings.warn('The force_frac parameter to NumberPattern.apply() is deprecated.', DeprecationWarning)
            frac_prec = force_frac
        elif currency and currency_digits:
            frac_prec = (get_currency_precision(currency), ) * 2
        else:
            frac_prec = self.frac_prec

        # Bump decimal precision to the natural precision of the number if it
        # exceeds the one we're about to use. This adaptative precision is only
        # triggered if the decimal quantization is disabled or if a scientific
        # notation pattern has a missing mandatory fractional part (as in the
        # default '#E0' pattern). This special case has been extensively
        # discussed at https://github.com/python-babel/babel/pull/494#issuecomment-307649969 .
        if not decimal_quantization or (self.exp_prec and frac_prec == (0, 0)):
            frac_prec = (frac_prec[0], max([frac_prec[1], get_decimal_precision(value)]))

        # Render scientific notation.
        if self.exp_prec:
            number = ''.join([
                self._quantize_value(value, locale, frac_prec, group_separator),
                get_exponential_symbol(locale),
                exp_sign,
                self._format_int(
                    str(exp), self.exp_prec[0], self.exp_prec[1], locale)])

        # Is it a siginificant digits pattern?
        elif '@' in self.pattern:
            text = self._format_significant(value,
                                            self.int_prec[0],
                                            self.int_prec[1])
            a, sep, b = text.partition(".")
            number = self._format_int(a, 0, 1000, locale)
            if sep:
                number += get_decimal_symbol(locale) + b

        # A normal number pattern.
        else:
            number = self._quantize_value(value, locale, frac_prec, group_separator)

        retval = ''.join([
            self.prefix[is_negative],
            number,
            self.suffix[is_negative]])

        if u'¤' in retval:
            retval = retval.replace(u'¤¤¤',
                                    get_currency_name(currency, value, locale))
            retval = retval.replace(u'¤¤', currency.upper())
            retval = retval.replace(u'¤', get_currency_symbol(currency, locale))

        return retval

    #
    # This is one tricky piece of code.  The idea is to rely as much as possible
    # on the decimal module to minimize the amount of code.
    #
    # Conceptually, the implementation of this method can be summarized in the
    # following steps:
    #
    #   - Move or shift the decimal point (i.e. the exponent) so the maximum
    #     amount of significant digits fall into the integer part (i.e. to the
    #     left of the decimal point)
    #
    #   - Round the number to the nearest integer, discarding all the fractional
    #     part which contained extra digits to be eliminated
    #
    #   - Convert the rounded integer to a string, that will contain the final
    #     sequence of significant digits already trimmed to the maximum
    #
    #   - Restore the original position of the decimal point, potentially
    #     padding with zeroes on either side
    #
    def _format_significant(self, value, minimum, maximum):
        exp = value.adjusted()
        scale = maximum - 1 - exp
        digits = str(value.scaleb(scale).quantize(decimal.Decimal(1)))
        if scale <= 0:
            result = digits + '0' * -scale
        else:
            intpart = digits[:-scale]
            i = len(intpart)
            j = i + max(minimum - i, 0)
            result = "{intpart}.{pad:0<{fill}}{fracpart}{fracextra}".format(
                intpart=intpart or '0',
                pad='',
                fill=-min(exp + 1, 0),
                fracpart=digits[i:j],
                fracextra=digits[j:].rstrip('0'),
            ).rstrip('.')
        return result

    def _format_int(self, value, min, max, locale):
        width = len(value)
        if width < min:
            value = '0' * (min - width) + value
        gsize = self.grouping[0]
        ret = ''
        symbol = get_group_symbol(locale)
        while len(value) > gsize:
            ret = symbol + value[-gsize:] + ret
            value = value[:-gsize]
            gsize = self.grouping[1]
        return value + ret

    def _quantize_value(self, value, locale, frac_prec, group_separator):
        quantum = get_decimal_quantum(frac_prec[1])
        rounded = value.quantize(quantum)
        a, sep, b = "{:f}".format(rounded).partition(".")
        integer_part = a
        if group_separator:
            integer_part = self._format_int(a, self.int_prec[0], self.int_prec[1], locale)
        number = integer_part + self._format_frac(b or '0', locale, frac_prec)
        return number

    def _format_frac(self, value, locale, force_frac=None):
        min, max = force_frac or self.frac_prec
        if len(value) < min:
            value += ('0' * (min - len(value)))
        if max == 0 or (min == 0 and int(value) == 0):
            return ''
        while len(value) > min and value[-1] == '0':
            value = value[:-1]
        return get_decimal_symbol(locale) + value
