# -*- coding: utf-8 -*-
"""
    babel.dates
    ~~~~~~~~~~~

    Locale dependent formatting and parsing of dates and times.

    The default locale for the functions in this module is determined by the
    following environment variables, in that order:

     * ``LC_TIME``,
     * ``LC_ALL``, and
     * ``LANG``

    :copyright: (c) 2013-2021 by the Babel Team.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division

import re
import warnings
import pytz as _pytz

from datetime import date, datetime, time, timedelta
from bisect import bisect_right

from babel.core import default_locale, get_global, Locale
from babel.util import UTC, LOCALTZ
from babel._compat import string_types, integer_types, number_types, PY2

# "If a given short metazone form is known NOT to be understood in a given
#  locale and the parent locale has this value such that it would normally
#  be inherited, the inheritance of this value can be explicitly disabled by
#  use of the 'no inheritance marker' as the value, which is 3 simultaneous [sic]
#  empty set characters ( U+2205 )."
#  - https://www.unicode.org/reports/tr35/tr35-dates.html#Metazone_Names

NO_INHERITANCE_MARKER = u'\u2205\u2205\u2205'


LC_TIME = default_locale('LC_TIME')

# Aliases for use in scopes where the modules are shadowed by local variables
date_ = date
datetime_ = datetime
time_ = time


def _get_dt_and_tzinfo(dt_or_tzinfo):
    """
    Parse a `dt_or_tzinfo` value into a datetime and a tzinfo.

    See the docs for this function's callers for semantics.

    :rtype: tuple[datetime, tzinfo]
    """
    if dt_or_tzinfo is None:
        dt = datetime.now()
        tzinfo = LOCALTZ
    elif isinstance(dt_or_tzinfo, string_types):
        dt = None
        tzinfo = get_timezone(dt_or_tzinfo)
    elif isinstance(dt_or_tzinfo, integer_types):
        dt = None
        tzinfo = UTC
    elif isinstance(dt_or_tzinfo, (datetime, time)):
        dt = _get_datetime(dt_or_tzinfo)
        if dt.tzinfo is not None:
            tzinfo = dt.tzinfo
        else:
            tzinfo = UTC
    else:
        dt = None
        tzinfo = dt_or_tzinfo
    return dt, tzinfo


def _get_tz_name(dt_or_tzinfo):
    """
    Get the timezone name out of a time, datetime, or tzinfo object.

    :rtype: str
    """
    dt, tzinfo = _get_dt_and_tzinfo(dt_or_tzinfo)
    if hasattr(tzinfo, 'zone'):  # pytz object
        return tzinfo.zone
    elif hasattr(tzinfo, 'key') and tzinfo.key is not None:  # ZoneInfo object
        return tzinfo.key
    else:
        return tzinfo.tzname(dt or datetime.utcnow())


def _get_datetime(instant):
    """
    Get a datetime out of an "instant" (date, time, datetime, number).

    .. warning:: The return values of this function may depend on the system clock.

    If the instant is None, the current moment is used.
    If the instant is a time, it's augmented with today's date.

    Dates are converted to naive datetimes with midnight as the time component.

    >>> _get_datetime(date(2015, 1, 1))
    datetime.datetime(2015, 1, 1, 0, 0)

    UNIX timestamps are converted to datetimes.

    >>> _get_datetime(1400000000)
    datetime.datetime(2014, 5, 13, 16, 53, 20)

    Other values are passed through as-is.

    >>> x = datetime(2015, 1, 1)
    >>> _get_datetime(x) is x
    True

    :param instant: date, time, datetime, integer, float or None
    :type instant: date|time|datetime|int|float|None
    :return: a datetime
    :rtype: datetime
    """
    if instant is None:
        return datetime_.utcnow()
    elif isinstance(instant, integer_types) or isinstance(instant, float):
        return datetime_.utcfromtimestamp(instant)
    elif isinstance(instant, time):
        return datetime_.combine(date.today(), instant)
    elif isinstance(instant, date) and not isinstance(instant, datetime):
        return datetime_.combine(instant, time())
    # TODO (3.x): Add an assertion/type check for this fallthrough branch:
    return instant


def _ensure_datetime_tzinfo(datetime, tzinfo=None):
    """
    Ensure the datetime passed has an attached tzinfo.

    If the datetime is tz-naive to begin with, UTC is attached.

    If a tzinfo is passed in, the datetime is normalized to that timezone.

    >>> _ensure_datetime_tzinfo(datetime(2015, 1, 1)).tzinfo.zone
    'UTC'

    >>> tz = get_timezone("Europe/Stockholm")
    >>> _ensure_datetime_tzinfo(datetime(2015, 1, 1, 13, 15, tzinfo=UTC), tzinfo=tz).hour
    14

    :param datetime: Datetime to augment.
    :param tzinfo: Optional tznfo.
    :return: datetime with tzinfo
    :rtype: datetime
    """
    if datetime.tzinfo is None:
        datetime = datetime.replace(tzinfo=UTC)
    if tzinfo is not None:
        datetime = datetime.astimezone(get_timezone(tzinfo))
        if hasattr(tzinfo, 'normalize'):  # pytz
            datetime = tzinfo.normalize(datetime)
    return datetime


def _get_time(time, tzinfo=None):
    """
    Get a timezoned time from a given instant.

    .. warning:: The return values of this function may depend on the system clock.

    :param time: time, datetime or None
    :rtype: time
    """
    if time is None:
        time = datetime.utcnow()
    elif isinstance(time, number_types):
        time = datetime.utcfromtimestamp(time)
    if time.tzinfo is None:
        time = time.replace(tzinfo=UTC)
    if isinstance(time, datetime):
        if tzinfo is not None:
            time = time.astimezone(tzinfo)
            if hasattr(tzinfo, 'normalize'):  # pytz
                time = tzinfo.normalize(time)
        time = time.timetz()
    elif tzinfo is not None:
        time = time.replace(tzinfo=tzinfo)
    return time


def get_timezone(zone=None):
    """Looks up a timezone by name and returns it.  The timezone object
    returned comes from ``pytz`` and corresponds to the `tzinfo` interface and
    can be used with all of the functions of Babel that operate with dates.

    If a timezone is not known a :exc:`LookupError` is raised.  If `zone`
    is ``None`` a local zone object is returned.

    :param zone: the name of the timezone to look up.  If a timezone object
                 itself is passed in, mit's returned unchanged.
    """
    if zone is None:
        return LOCALTZ
    if not isinstance(zone, string_types):
        return zone
    try:
        return _pytz.timezone(zone)
    except _pytz.UnknownTimeZoneError:
        raise LookupError('Unknown timezone %s' % zone)


def get_next_timezone_transition(zone=None, dt=None):
    """Given a timezone it will return a :class:`TimezoneTransition` object
    that holds the information about the next timezone transition that's going
    to happen.  For instance this can be used to detect when the next DST
    change is going to happen and how it looks like.

    The transition is calculated relative to the given datetime object.  The
    next transition that follows the date is used.  If a transition cannot
    be found the return value will be `None`.

    Transition information can only be provided for timezones returned by
    the :func:`get_timezone` function.

    :param zone: the timezone for which the transition should be looked up.
                 If not provided the local timezone is used.
    :param dt: the date after which the next transition should be found.
               If not given the current time is assumed.
    """
    zone = get_timezone(zone)
    dt = _get_datetime(dt).replace(tzinfo=None)

    if not hasattr(zone, '_utc_transition_times'):
        raise TypeError('Given timezone does not have UTC transition '
                        'times.  This can happen because the operating '
                        'system fallback local timezone is used or a '
                        'custom timezone object')

    try:
        idx = max(0, bisect_right(zone._utc_transition_times, dt))
        old_trans = zone._transition_info[idx - 1]
        new_trans = zone._transition_info[idx]
        old_tz = zone._tzinfos[old_trans]
        new_tz = zone._tzinfos[new_trans]
    except (LookupError, ValueError):
        return None

    return TimezoneTransition(
        activates=zone._utc_transition_times[idx],
        from_tzinfo=old_tz,
        to_tzinfo=new_tz,
        reference_date=dt
    )


class TimezoneTransition(object):
    """A helper object that represents the return value from
    :func:`get_next_timezone_transition`.
    """

    def __init__(self, activates, from_tzinfo, to_tzinfo, reference_date=None):
        #: the time of the activation of the timezone transition in UTC.
        self.activates = activates
        #: the timezone from where the transition starts.
        self.from_tzinfo = from_tzinfo
        #: the timezone for after the transition.
        self.to_tzinfo = to_tzinfo
        #: the reference date that was provided.  This is the `dt` parameter
        #: to the :func:`get_next_timezone_transition`.
        self.reference_date = reference_date

    @property
    def from_tz(self):
        """The name of the timezone before the transition."""
        return self.from_tzinfo._tzname

    @property
    def to_tz(self):
        """The name of the timezone after the transition."""
        return self.to_tzinfo._tzname

    @property
    def from_offset(self):
        """The UTC offset in seconds before the transition."""
        return int(self.from_tzinfo._utcoffset.total_seconds())

    @property
    def to_offset(self):
        """The UTC offset in seconds after the transition."""
        return int(self.to_tzinfo._utcoffset.total_seconds())

    def __repr__(self):
        return '<TimezoneTransition %s -> %s (%s)>' % (
            self.from_tz,
            self.to_tz,
            self.activates,
        )


def get_period_names(width='wide', context='stand-alone', locale=LC_TIME):
    """Return the names for day periods (AM/PM) used by the locale.

    >>> get_period_names(locale='en_US')['am']
    u'AM'

    :param width: the width to use, one of "abbreviated", "narrow", or "wide"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).day_periods[context][width]


def get_day_names(width='wide', context='format', locale=LC_TIME):
    """Return the day names used by the locale for the specified format.

    >>> get_day_names('wide', locale='en_US')[1]
    u'Tuesday'
    >>> get_day_names('short', locale='en_US')[1]
    u'Tu'
    >>> get_day_names('abbreviated', locale='es')[1]
    u'mar.'
    >>> get_day_names('narrow', context='stand-alone', locale='de_DE')[1]
    u'D'

    :param width: the width to use, one of "wide", "abbreviated", "short" or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).days[context][width]


def get_month_names(width='wide', context='format', locale=LC_TIME):
    """Return the month names used by the locale for the specified format.

    >>> get_month_names('wide', locale='en_US')[1]
    u'January'
    >>> get_month_names('abbreviated', locale='es')[1]
    u'ene.'
    >>> get_month_names('narrow', context='stand-alone', locale='de_DE')[1]
    u'J'

    :param width: the width to use, one of "wide", "abbreviated", or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).months[context][width]


def get_quarter_names(width='wide', context='format', locale=LC_TIME):
    """Return the quarter names used by the locale for the specified format.

    >>> get_quarter_names('wide', locale='en_US')[1]
    u'1st quarter'
    >>> get_quarter_names('abbreviated', locale='de_DE')[1]
    u'Q1'
    >>> get_quarter_names('narrow', locale='de_DE')[1]
    u'1'

    :param width: the width to use, one of "wide", "abbreviated", or "narrow"
    :param context: the context, either "format" or "stand-alone"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).quarters[context][width]


def get_era_names(width='wide', locale=LC_TIME):
    """Return the era names used by the locale for the specified format.

    >>> get_era_names('wide', locale='en_US')[1]
    u'Anno Domini'
    >>> get_era_names('abbreviated', locale='de_DE')[1]
    u'n. Chr.'

    :param width: the width to use, either "wide", "abbreviated", or "narrow"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).eras[width]


def get_date_format(format='medium', locale=LC_TIME):
    """Return the date formatting patterns used by the locale for the specified
    format.

    >>> get_date_format(locale='en_US')
    <DateTimePattern u'MMM d, y'>
    >>> get_date_format('full', locale='de_DE')
    <DateTimePattern u'EEEE, d. MMMM y'>

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).date_formats[format]


def get_datetime_format(format='medium', locale=LC_TIME):
    """Return the datetime formatting patterns used by the locale for the
    specified format.

    >>> get_datetime_format(locale='en_US')
    u'{1}, {0}'

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string
    """
    patterns = Locale.parse(locale).datetime_formats
    if format not in patterns:
        format = None
    return patterns[format]


def get_time_format(format='medium', locale=LC_TIME):
    """Return the time formatting patterns used by the locale for the specified
    format.

    >>> get_time_format(locale='en_US')
    <DateTimePattern u'h:mm:ss a'>
    >>> get_time_format('full', locale='de_DE')
    <DateTimePattern u'HH:mm:ss zzzz'>

    :param format: the format to use, one of "full", "long", "medium", or
                   "short"
    :param locale: the `Locale` object, or a locale string
    """
    return Locale.parse(locale).time_formats[format]


def get_timezone_gmt(datetime=None, width='long', locale=LC_TIME, return_z=False):
    """Return the timezone associated with the given `datetime` object formatted
    as string indicating the offset from GMT.

    >>> dt = datetime(2007, 4, 1, 15, 30)
    >>> get_timezone_gmt(dt, locale='en')
    u'GMT+00:00'
    >>> get_timezone_gmt(dt, locale='en', return_z=True)
    'Z'
    >>> get_timezone_gmt(dt, locale='en', width='iso8601_short')
    u'+00'
    >>> tz = get_timezone('America/Los_Angeles')
    >>> dt = tz.localize(datetime(2007, 4, 1, 15, 30))
    >>> get_timezone_gmt(dt, locale='en')
    u'GMT-07:00'
    >>> get_timezone_gmt(dt, 'short', locale='en')
    u'-0700'
    >>> get_timezone_gmt(dt, locale='en', width='iso8601_short')
    u'-07'

    The long format depends on the locale, for example in France the acronym
    UTC string is used instead of GMT:

    >>> get_timezone_gmt(dt, 'long', locale='fr_FR')
    u'UTC-07:00'

    .. versionadded:: 0.9

    :param datetime: the ``datetime`` object; if `None`, the current date and
                     time in UTC is used
    :param width: either "long" or "short" or "iso8601" or "iso8601_short"
    :param locale: the `Locale` object, or a locale string
    :param return_z: True or False; Function returns indicator "Z"
                     when local time offset is 0
    """
    datetime = _ensure_datetime_tzinfo(_get_datetime(datetime))
    locale = Locale.parse(locale)

    offset = datetime.tzinfo.utcoffset(datetime)
    seconds = offset.days * 24 * 60 * 60 + offset.seconds
    hours, seconds = divmod(seconds, 3600)
    if return_z and hours == 0 and seconds == 0:
        return 'Z'
    elif seconds == 0 and width == 'iso8601_short':
        return u'%+03d' % hours
    elif width == 'short' or width == 'iso8601_short':
        pattern = u'%+03d%02d'
    elif width == 'iso8601':
        pattern = u'%+03d:%02d'
    else:
        pattern = locale.zone_formats['gmt'] % '%+03d:%02d'
    return pattern % (hours, seconds // 60)


def get_timezone_location(dt_or_tzinfo=None, locale=LC_TIME, return_city=False):
    u"""Return a representation of the given timezone using "location format".

    The result depends on both the local display name of the country and the
    city associated with the time zone:

    >>> tz = get_timezone('America/St_Johns')
    >>> print(get_timezone_location(tz, locale='de_DE'))
    Kanada (St. John’s) Zeit
    >>> print(get_timezone_location(tz, locale='en'))
    Canada (St. John’s) Time
    >>> print(get_timezone_location(tz, locale='en', return_city=True))
    St. John’s
    >>> tz = get_timezone('America/Mexico_City')
    >>> get_timezone_location(tz, locale='de_DE')
    u'Mexiko (Mexiko-Stadt) Zeit'

    If the timezone is associated with a country that uses only a single
    timezone, just the localized country name is returned:

    >>> tz = get_timezone('Europe/Berlin')
    >>> get_timezone_name(tz, locale='de_DE')
    u'Mitteleurop\\xe4ische Zeit'

    .. versionadded:: 0.9

    :param dt_or_tzinfo: the ``datetime`` or ``tzinfo`` object that determines
                         the timezone; if `None`, the current date and time in
                         UTC is assumed
    :param locale: the `Locale` object, or a locale string
    :param return_city: True or False, if True then return exemplar city (location)
                        for the time zone
    :return: the localized timezone name using location format

    """
    locale = Locale.parse(locale)

    zone = _get_tz_name(dt_or_tzinfo)

    # Get the canonical time-zone code
    zone = get_global('zone_aliases').get(zone, zone)

    info = locale.time_zones.get(zone, {})

    # Otherwise, if there is only one timezone for the country, return the
    # localized country name
    region_format = locale.zone_formats['region']
    territory = get_global('zone_territories').get(zone)
    if territory not in locale.territories:
        territory = 'ZZ'  # invalid/unknown
    territory_name = locale.territories[territory]
    if not return_city and territory and len(get_global('territory_zones').get(territory, [])) == 1:
        return region_format % territory_name

    # Otherwise, include the city in the output
    fallback_format = locale.zone_formats['fallback']
    if 'city' in info:
        city_name = info['city']
    else:
        metazone = get_global('meta_zones').get(zone)
        metazone_info = locale.meta_zones.get(metazone, {})
        if 'city' in metazone_info:
            city_name = metazone_info['city']
        elif '/' in zone:
            city_name = zone.split('/', 1)[1].replace('_', ' ')
        else:
            city_name = zone.replace('_', ' ')

    if return_city:
        return city_name
    return region_format % (fallback_format % {
        '0': city_name,
        '1': territory_name
    })


def get_timezone_name(dt_or_tzinfo=None, width='long', uncommon=False,
                      locale=LC_TIME, zone_variant=None, return_zone=False):
    r"""Return the localized display name for the given timezone. The timezone
    may be specified using a ``datetime`` or `tzinfo` object.

    >>> dt = time(15, 30, tzinfo=get_timezone('America/Los_Angeles'))
    >>> get_timezone_name(dt, locale='en_US')
    u'Pacific Standard Time'
    >>> get_timezone_name(dt, locale='en_US', return_zone=True)
    'America/Los_Angeles'
    >>> get_timezone_name(dt, width='short', locale='en_US')
    u'PST'

    If this function gets passed only a `tzinfo` object and no concrete
    `datetime`,  the returned display name is indenpendent of daylight savings
    time. This can be used for example for selecting timezones, or to set the
    time of events that recur across DST changes:

    >>> tz = get_timezone('America/Los_Angeles')
    >>> get_timezone_name(tz, locale='en_US')
    u'Pacific Time'
    >>> get_timezone_name(tz, 'short', locale='en_US')
    u'PT'

    If no localized display name for the timezone is available, and the timezone
    is associated with a country that uses only a single timezone, the name of
    that country is returned, formatted according to the locale:

    >>> tz = get_timezone('Europe/Berlin')
    >>> get_timezone_name(tz, locale='de_DE')
    u'Mitteleurop\xe4ische Zeit'
    >>> get_timezone_name(tz, locale='pt_BR')
    u'Hor\xe1rio da Europa Central'

    On the other hand, if the country uses multiple timezones, the city is also
    included in the representation:

    >>> tz = get_timezone('America/St_Johns')
    >>> get_timezone_name(tz, locale='de_DE')
    u'Neufundland-Zeit'

    Note that short format is currently not supported for all timezones and
    all locales.  This is partially because not every timezone has a short
    code in every locale.  In that case it currently falls back to the long
    format.

    For more information see `LDML Appendix J: Time Zone Display Names
    <https://www.unicode.org/reports/tr35/#Time_Zone_Fallback>`_

    .. versionadded:: 0.9

    .. versionchanged:: 1.0
       Added `zone_variant` support.

    :param dt_or_tzinfo: the ``datetime`` or ``tzinfo`` object that determines
                         the timezone; if a ``tzinfo`` object is used, the
                         resulting display name will be generic, i.e.
                         independent of daylight savings time; if `None`, the
                         current date in UTC is assumed
    :param width: either "long" or "short"
    :param uncommon: deprecated and ignored
    :param zone_variant: defines the zone variation to return.  By default the
                           variation is defined from the datetime object
                           passed in.  If no datetime object is passed in, the
                           ``'generic'`` variation is assumed.  The following
                           values are valid: ``'generic'``, ``'daylight'`` and
                           ``'standard'``.
    :param locale: the `Locale` object, or a locale string
    :param return_zone: True or False. If true then function
                        returns long time zone ID
    """
    dt, tzinfo = _get_dt_and_tzinfo(dt_or_tzinfo)
    locale = Locale.parse(locale)

    zone = _get_tz_name(dt_or_tzinfo)

    if zone_variant is None:
        if dt is None:
            zone_variant = 'generic'
        else:
            dst = tzinfo.dst(dt)
            if dst:
                zone_variant = 'daylight'
            else:
                zone_variant = 'standard'
    else:
        if zone_variant not in ('generic', 'standard', 'daylight'):
            raise ValueError('Invalid zone variation')

    # Get the canonical time-zone code
    zone = get_global('zone_aliases').get(zone, zone)
    if return_zone:
        return zone
    info = locale.time_zones.get(zone, {})
    # Try explicitly translated zone names first
    if width in info:
        if zone_variant in info[width]:
            return info[width][zone_variant]

    metazone = get_global('meta_zones').get(zone)
    if metazone:
        metazone_info = locale.meta_zones.get(metazone, {})
        if width in metazone_info:
            name = metazone_info[width].get(zone_variant)
            if width == 'short' and name == NO_INHERITANCE_MARKER:
                # If the short form is marked no-inheritance,
                # try to fall back to the long name instead.
                name = metazone_info.get('long', {}).get(zone_variant)
            if name:
                return name

    # If we have a concrete datetime, we assume that the result can't be
    # independent of daylight savings time, so we return the GMT offset
    if dt is not None:
        return get_timezone_gmt(dt, width=width, locale=locale)

    return get_timezone_location(dt_or_tzinfo, locale=locale)


def format_date(date=None, format='medium', locale=LC_TIME):
    """Return a date formatted according to the given pattern.

    >>> d = date(2007, 4, 1)
    >>> format_date(d, locale='en_US')
    u'Apr 1, 2007'
    >>> format_date(d, format='full', locale='de_DE')
    u'Sonntag, 1. April 2007'

    If you don't want to use the locale default formats, you can specify a
    custom date pattern:

    >>> format_date(d, "EEE, MMM d, ''yy", locale='en')
    u"Sun, Apr 1, '07"

    :param date: the ``date`` or ``datetime`` object; if `None`, the current
                 date is used
    :param format: one of "full", "long", "medium", or "short", or a custom
                   date/time pattern
    :param locale: a `Locale` object or a locale identifier
    """
    if date is None:
        date = date_.today()
    elif isinstance(date, datetime):
        date = date.date()

    locale = Locale.parse(locale)
    if format in ('full', 'long', 'medium', 'short'):
        format = get_date_format(format, locale=locale)
    pattern = parse_pattern(format)
    return pattern.apply(date, locale)


def format_datetime(datetime=None, format='medium', tzinfo=None,
                    locale=LC_TIME):
    r"""Return a date formatted according to the given pattern.

    >>> dt = datetime(2007, 4, 1, 15, 30)
    >>> format_datetime(dt, locale='en_US')
    u'Apr 1, 2007, 3:30:00 PM'

    For any pattern requiring the display of the time-zone, the third-party
    ``pytz`` package is needed to explicitly specify the time-zone:

    >>> format_datetime(dt, 'full', tzinfo=get_timezone('Europe/Paris'),
    ...                 locale='fr_FR')
    u'dimanche 1 avril 2007 \xe0 17:30:00 heure d\u2019\xe9t\xe9 d\u2019Europe centrale'
    >>> format_datetime(dt, "yyyy.MM.dd G 'at' HH:mm:ss zzz",
    ...                 tzinfo=get_timezone('US/Eastern'), locale='en')
    u'2007.04.01 AD at 11:30:00 EDT'

    :param datetime: the `datetime` object; if `None`, the current date and
                     time is used
    :param format: one of "full", "long", "medium", or "short", or a custom
                   date/time pattern
    :param tzinfo: the timezone to apply to the time for display
    :param locale: a `Locale` object or a locale identifier
    """
    datetime = _ensure_datetime_tzinfo(_get_datetime(datetime), tzinfo)

    locale = Locale.parse(locale)
    if format in ('full', 'long', 'medium', 'short'):
        return get_datetime_format(format, locale=locale) \
            .replace("'", "") \
            .replace('{0}', format_time(datetime, format, tzinfo=None,
                                        locale=locale)) \
            .replace('{1}', format_date(datetime, format, locale=locale))
    else:
        return parse_pattern(format).apply(datetime, locale)


def format_time(time=None, format='medium', tzinfo=None, locale=LC_TIME):
    r"""Return a time formatted according to the given pattern.

    >>> t = time(15, 30)
    >>> format_time(t, locale='en_US')
    u'3:30:00 PM'
    >>> format_time(t, format='short', locale='de_DE')
    u'15:30'

    If you don't want to use the locale default formats, you can specify a
    custom time pattern:

    >>> format_time(t, "hh 'o''clock' a", locale='en')
    u"03 o'clock PM"

    For any pattern requiring the display of the time-zone a
    timezone has to be specified explicitly:

    >>> t = datetime(2007, 4, 1, 15, 30)
    >>> tzinfo = get_timezone('Europe/Paris')
    >>> t = tzinfo.localize(t)
    >>> format_time(t, format='full', tzinfo=tzinfo, locale='fr_FR')
    u'15:30:00 heure d\u2019\xe9t\xe9 d\u2019Europe centrale'
    >>> format_time(t, "hh 'o''clock' a, zzzz", tzinfo=get_timezone('US/Eastern'),
    ...             locale='en')
    u"09 o'clock AM, Eastern Daylight Time"

    As that example shows, when this function gets passed a
    ``datetime.datetime`` value, the actual time in the formatted string is
    adjusted to the timezone specified by the `tzinfo` parameter. If the
    ``datetime`` is "naive" (i.e. it has no associated timezone information),
    it is assumed to be in UTC.

    These timezone calculations are **not** performed if the value is of type
    ``datetime.time``, as without date information there's no way to determine
    what a given time would translate to in a different timezone without
    information about whether daylight savings time is in effect or not. This
    means that time values are left as-is, and the value of the `tzinfo`
    parameter is only used to display the timezone name if needed:

    >>> t = time(15, 30)
    >>> format_time(t, format='full', tzinfo=get_timezone('Europe/Paris'),
    ...             locale='fr_FR')
    u'15:30:00 heure normale d\u2019Europe centrale'
    >>> format_time(t, format='full', tzinfo=get_timezone('US/Eastern'),
    ...             locale='en_US')
    u'3:30:00 PM Eastern Standard Time'

    :param time: the ``time`` or ``datetime`` object; if `None`, the current
                 time in UTC is used
    :param format: one of "full", "long", "medium", or "short", or a custom
                   date/time pattern
    :param tzinfo: the time-zone to apply to the time for display
    :param locale: a `Locale` object or a locale identifier
    """
    time = _get_time(time, tzinfo)

    locale = Locale.parse(locale)
    if format in ('full', 'long', 'medium', 'short'):
        format = get_time_format(format, locale=locale)
    return parse_pattern(format).apply(time, locale)


def format_skeleton(skeleton, datetime=None, tzinfo=None, fuzzy=True, locale=LC_TIME):
    r"""Return a time and/or date formatted according to the given pattern.

    The skeletons are defined in the CLDR data and provide more flexibility
    than the simple short/long/medium formats, but are a bit harder to use.
    The are defined using the date/time symbols without order or punctuation
    and map to a suitable format for the given locale.

    >>> t = datetime(2007, 4, 1, 15, 30)
    >>> format_skeleton('MMMEd', t, locale='fr')
    u'dim. 1 avr.'
    >>> format_skeleton('MMMEd', t, locale='en')
    u'Sun, Apr 1'
    >>> format_skeleton('yMMd', t, locale='fi')  # yMMd is not in the Finnish locale; yMd gets used
    u'1.4.2007'
    >>> format_skeleton('yMMd', t, fuzzy=False, locale='fi')  # yMMd is not in the Finnish locale, an error is thrown
    Traceback (most recent call last):
        ...
    KeyError: yMMd

    After the skeleton is resolved to a pattern `format_datetime` is called so
    all timezone processing etc is the same as for that.

    :param skeleton: A date time skeleton as defined in the cldr data.
    :param datetime: the ``time`` or ``datetime`` object; if `None`, the current
                 time in UTC is used
    :param tzinfo: the time-zone to apply to the time for display
    :param fuzzy: If the skeleton is not found, allow choosing a skeleton that's
                  close enough to it.
    :param locale: a `Locale` object or a locale identifier
    """
    locale = Locale.parse(locale)
    if fuzzy and skeleton not in locale.datetime_skeletons:
        skeleton = match_skeleton(skeleton, locale.datetime_skeletons)
    format = locale.datetime_skeletons[skeleton]
    return format_datetime(datetime, format, tzinfo, locale)


TIMEDELTA_UNITS = (
    ('year', 3600 * 24 * 365),
    ('month', 3600 * 24 * 30),
    ('week', 3600 * 24 * 7),
    ('day', 3600 * 24),
    ('hour', 3600),
    ('minute', 60),
    ('second', 1)
)


def format_timedelta(delta, granularity='second', threshold=.85,
                     add_direction=False, format='long',
                     locale=LC_TIME):
    """Return a time delta according to the rules of the given locale.

    >>> format_timedelta(timedelta(weeks=12), locale='en_US')
    u'3 months'
    >>> format_timedelta(timedelta(seconds=1), locale='es')
    u'1 segundo'

    The granularity parameter can be provided to alter the lowest unit
    presented, which defaults to a second.

    >>> format_timedelta(timedelta(hours=3), granularity='day',
    ...                  locale='en_US')
    u'1 day'

    The threshold parameter can be used to determine at which value the
    presentation switches to the next higher unit. A higher threshold factor
    means the presentation will switch later. For example:

    >>> format_timedelta(timedelta(hours=23), threshold=0.9, locale='en_US')
    u'1 day'
    >>> format_timedelta(timedelta(hours=23), threshold=1.1, locale='en_US')
    u'23 hours'

    In addition directional information can be provided that informs
    the user if the date is in the past or in the future:

    >>> format_timedelta(timedelta(hours=1), add_direction=True, locale='en')
    u'in 1 hour'
    >>> format_timedelta(timedelta(hours=-1), add_direction=True, locale='en')
    u'1 hour ago'

    The format parameter controls how compact or wide the presentation is:

    >>> format_timedelta(timedelta(hours=3), format='short', locale='en')
    u'3 hr'
    >>> format_timedelta(timedelta(hours=3), format='narrow', locale='en')
    u'3h'

    :param delta: a ``timedelta`` object representing the time difference to
                  format, or the delta in seconds as an `int` value
    :param granularity: determines the smallest unit that should be displayed,
                        the value can be one of "year", "month", "week", "day",
                        "hour", "minute" or "second"
    :param threshold: factor that determines at which point the presentation
                      switches to the next higher unit
    :param add_direction: if this flag is set to `True` the return value will
                          include directional information.  For instance a
                          positive timedelta will include the information about
                          it being in the future, a negative will be information
                          about the value being in the past.
    :param format: the format, can be "narrow", "short" or "long". (
                   "medium" is deprecated, currently converted to "long" to
                   maintain compatibility)
    :param locale: a `Locale` object or a locale identifier
    """
    if format not in ('narrow', 'short', 'medium', 'long'):
        raise TypeError('Format must be one of "narrow", "short" or "long"')
    if format == 'medium':
        warnings.warn('"medium" value for format param of format_timedelta'
                      ' is deprecated. Use "long" instead',
                      category=DeprecationWarning)
        format = 'long'
    if isinstance(delta, timedelta):
        seconds = int((delta.days * 86400) + delta.seconds)
    else:
        seconds = delta
    locale = Locale.parse(locale)

    def _iter_patterns(a_unit):
        if add_direction:
            unit_rel_patterns = locale._data['date_fields'][a_unit]
            if seconds >= 0:
                yield unit_rel_patterns['future']
            else:
                yield unit_rel_patterns['past']
        a_unit = 'duration-' + a_unit
        yield locale._data['unit_patterns'].get(a_unit, {}).get(format)

    for unit, secs_per_unit in TIMEDELTA_UNITS:
        value = abs(seconds) / secs_per_unit
        if value >= threshold or unit == granularity:
            if unit == granularity and value > 0:
                value = max(1, value)
            value = int(round(value))
            plural_form = locale.plural_form(value)
            pattern = None
            for patterns in _iter_patterns(unit):
                if patterns is not None:
                    pattern = patterns[plural_form]
                    break
            # This really should not happen
            if pattern is None:
                return u''
            return pattern.replace('{0}', str(value))

    return u''


def _format_fallback_interval(start, end, skeleton, tzinfo, locale):
    if skeleton in locale.datetime_skeletons:  # Use the given skeleton
        format = lambda dt: format_skeleton(skeleton, dt, tzinfo, locale=locale)
    elif all((isinstance(d, date) and not isinstance(d, datetime)) for d in (start, end)):  # Both are just dates
        format = lambda dt: format_date(dt, locale=locale)
    elif all((isinstance(d, time) and not isinstance(d, date)) for d in (start, end)):  # Both are times
        format = lambda dt: format_time(dt, tzinfo=tzinfo, locale=locale)
    else:
        format = lambda dt: format_datetime(dt, tzinfo=tzinfo, locale=locale)

    formatted_start = format(start)
    formatted_end = format(end)

    if formatted_start == formatted_end:
        return format(start)

    return (
        locale.interval_formats.get(None, "{0}-{1}").
        replace("{0}", formatted_start).
        replace("{1}", formatted_end)
    )


def format_interval(start, end, skeleton=None, tzinfo=None, fuzzy=True, locale=LC_TIME):
    """
    Format an interval between two instants according to the locale's rules.

    >>> format_interval(date(2016, 1, 15), date(2016, 1, 17), "yMd", locale="fi")
    u'15.\u201317.1.2016'

    >>> format_interval(time(12, 12), time(16, 16), "Hm", locale="en_GB")
    '12:12\u201316:16'

    >>> format_interval(time(5, 12), time(16, 16), "hm", locale="en_US")
    '5:12 AM \u2013 4:16 PM'

    >>> format_interval(time(16, 18), time(16, 24), "Hm", locale="it")
    '16:18\u201316:24'

    If the start instant equals the end instant, the interval is formatted like the instant.

    >>> format_interval(time(16, 18), time(16, 18), "Hm", locale="it")
    '16:18'

    Unknown skeletons fall back to "default" formatting.

    >>> format_interval(date(2015, 1, 1), date(2017, 1, 1), "wzq", locale="ja")
    '2015/01/01\uff5e2017/01/01'

    >>> format_interval(time(16, 18), time(16, 24), "xxx", locale="ja")
    '16:18:00\uff5e16:24:00'

    >>> format_interval(date(2016, 1, 15), date(2016, 1, 17), "xxx", locale="de")
    '15.01.2016 \u2013 17.01.2016'

    :param start: First instant (datetime/date/time)
    :param end: Second instant (datetime/date/time)
    :param skeleton: The "skeleton format" to use for formatting.
    :param tzinfo: tzinfo to use (if none is already attached)
    :param fuzzy: If the skeleton is not found, allow choosing a skeleton that's
                  close enough to it.
    :param locale: A locale object or identifier.
    :return: Formatted interval
    """
    locale = Locale.parse(locale)

    # NB: The quote comments below are from the algorithm description in
    #     https://www.unicode.org/reports/tr35/tr35-dates.html#intervalFormats

    # > Look for the intervalFormatItem element that matches the "skeleton",
    # > starting in the current locale and then following the locale fallback
    # > chain up to, but not including root.

    interval_formats = locale.interval_formats

    if skeleton not in interval_formats or not skeleton:
        # > If no match was found from the previous step, check what the closest
        # > match is in the fallback locale chain, as in availableFormats. That
        # > is, this allows for adjusting the string value field's width,
        # > including adjusting between "MMM" and "MMMM", and using different
        # > variants of the same field, such as 'v' and 'z'.
        if skeleton and fuzzy:
            skeleton = match_skeleton(skeleton, interval_formats)
        else:
            skeleton = None
        if not skeleton:  # Still no match whatsoever?
            # > Otherwise, format the start and end datetime using the fallback pattern.
            return _format_fallback_interval(start, end, skeleton, tzinfo, locale)

    skel_formats = interval_formats[skeleton]

    if start == end:
        return format_skeleton(skeleton, start, tzinfo, fuzzy=fuzzy, locale=locale)

    start = _ensure_datetime_tzinfo(_get_datetime(start), tzinfo=tzinfo)
    end = _ensure_datetime_tzinfo(_get_datetime(end), tzinfo=tzinfo)

    start_fmt = DateTimeFormat(start, locale=locale)
    end_fmt = DateTimeFormat(end, locale=locale)

    # > If a match is found from previous steps, compute the calendar field
    # > with the greatest difference between start and end datetime. If there
    # > is no difference among any of the fields in the pattern, format as a
    # > single date using availableFormats, and return.

    for field in PATTERN_CHAR_ORDER:  # These are in largest-to-smallest order
        if field in skel_formats:
            if start_fmt.extract(field) != end_fmt.extract(field):
                # > If there is a match, use the pieces of the corresponding pattern to
                # > format the start and end datetime, as above.
                return "".join(
                    parse_pattern(pattern).apply(instant, locale)
                    for pattern, instant
                    in zip(skel_formats[field], (start, end))
                )

    # > Otherwise, format the start and end datetime using the fallback pattern.

    return _format_fallback_interval(start, end, skeleton, tzinfo, locale)


def get_period_id(time, tzinfo=None, type=None, locale=LC_TIME):
    """
    Get the day period ID for a given time.

    This ID can be used as a key for the period name dictionary.

    >>> get_period_names(locale="de")[get_period_id(time(7, 42), locale="de")]
    u'Morgen'

    :param time: The time to inspect.
    :param tzinfo: The timezone for the time. See ``format_time``.
    :param type: The period type to use. Either "selection" or None.
                 The selection type is used for selecting among phrases such as
                 “Your email arrived yesterday evening” or “Your email arrived last night”.
    :param locale: the `Locale` object, or a locale string
    :return: period ID. Something is always returned -- even if it's just "am" or "pm".
    """
    time = _get_time(time, tzinfo)
    seconds_past_midnight = int(time.hour * 60 * 60 + time.minute * 60 + time.second)
    locale = Locale.parse(locale)

    # The LDML rules state that the rules may not overlap, so iterating in arbitrary
    # order should be alright, though `at` periods should be preferred.
    rulesets = locale.day_period_rules.get(type, {}).items()

    for rule_id, rules in rulesets:
        for rule in rules:
            if "at" in rule and rule["at"] == seconds_past_midnight:
                return rule_id

    for rule_id, rules in rulesets:
        for rule in rules:
            start_ok = end_ok = False

            if "from" in rule and seconds_past_midnight >= rule["from"]:
                start_ok = True
            if "to" in rule and seconds_past_midnight <= rule["to"]:
                # This rule type does not exist in the present CLDR data;
                # excuse the lack of test coverage.
                end_ok = True
            if "before" in rule and seconds_past_midnight < rule["before"]:
                end_ok = True
            if "after" in rule:
                raise NotImplementedError("'after' is deprecated as of CLDR 29.")

            if start_ok and end_ok:
                return rule_id

    if seconds_past_midnight < 43200:
        return "am"
    else:
        return "pm"


def parse_date(string, locale=LC_TIME):
    """Parse a date from a string.

    This function uses the date format for the locale as a hint to determine
    the order in which the date fields appear in the string.

    >>> parse_date('4/1/04', locale='en_US')
    datetime.date(2004, 4, 1)
    >>> parse_date('01.04.2004', locale='de_DE')
    datetime.date(2004, 4, 1)

    :param string: the string containing the date
    :param locale: a `Locale` object or a locale identifier
    """
    # TODO: try ISO format first?
    format = get_date_format(locale=locale).pattern.lower()
    year_idx = format.index('y')
    month_idx = format.index('m')
    if month_idx < 0:
        month_idx = format.index('l')
    day_idx = format.index('d')

    indexes = [(year_idx, 'Y'), (month_idx, 'M'), (day_idx, 'D')]
    indexes.sort()
    indexes = dict([(item[1], idx) for idx, item in enumerate(indexes)])

    # FIXME: this currently only supports numbers, but should also support month
    #        names, both in the requested locale, and english

    numbers = re.findall(r'(\d+)', string)
    year = numbers[indexes['Y']]
    if len(year) == 2:
        year = 2000 + int(year)
    else:
        year = int(year)
    month = int(numbers[indexes['M']])
    day = int(numbers[indexes['D']])
    if month > 12:
        month, day = day, month
    return date(year, month, day)


def parse_time(string, locale=LC_TIME):
    """Parse a time from a string.

    This function uses the time format for the locale as a hint to determine
    the order in which the time fields appear in the string.

    >>> parse_time('15:30:00', locale='en_US')
    datetime.time(15, 30)

    :param string: the string containing the time
    :param locale: a `Locale` object or a locale identifier
    :return: the parsed time
    :rtype: `time`
    """
    # TODO: try ISO format first?
    format = get_time_format(locale=locale).pattern.lower()
    hour_idx = format.index('h')
    if hour_idx < 0:
        hour_idx = format.index('k')
    min_idx = format.index('m')
    sec_idx = format.index('s')

    indexes = [(hour_idx, 'H'), (min_idx, 'M'), (sec_idx, 'S')]
    indexes.sort()
    indexes = dict([(item[1], idx) for idx, item in enumerate(indexes)])

    # FIXME: support 12 hour clock, and 0-based hour specification
    #        and seconds should be optional, maybe minutes too
    #        oh, and time-zones, of course

    numbers = re.findall(r'(\d+)', string)
    hour = int(numbers[indexes['H']])
    minute = int(numbers[indexes['M']])
    second = int(numbers[indexes['S']])
    return time(hour, minute, second)


class DateTimePattern(object):

    def __init__(self, pattern, format):
        self.pattern = pattern
        self.format = format

    def __repr__(self):
        return '<%s %r>' % (type(self).__name__, self.pattern)

    def __unicode__(self):
        return self.pattern

    def __str__(self):
        pat = self.pattern
        if PY2:
            pat = pat.encode('utf-8')
        return pat

    def __mod__(self, other):
        if type(other) is not DateTimeFormat:
            return NotImplemented
        return self.format % other

    def apply(self, datetime, locale):
        return self % DateTimeFormat(datetime, locale)


class DateTimeFormat(object):

    def __init__(self, value, locale):
        assert isinstance(value, (date, datetime, time))
        if isinstance(value, (datetime, time)) and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        self.value = value
        self.locale = Locale.parse(locale)

    def __getitem__(self, name):
        char = name[0]
        num = len(name)
        if char == 'G':
            return self.format_era(char, num)
        elif char in ('y', 'Y', 'u'):
            return self.format_year(char, num)
        elif char in ('Q', 'q'):
            return self.format_quarter(char, num)
        elif char in ('M', 'L'):
            return self.format_month(char, num)
        elif char in ('w', 'W'):
            return self.format_week(char, num)
        elif char == 'd':
            return self.format(self.value.day, num)
        elif char == 'D':
            return self.format_day_of_year(num)
        elif char == 'F':
            return self.format_day_of_week_in_month()
        elif char in ('E', 'e', 'c'):
            return self.format_weekday(char, num)
        elif char == 'a':
            # TODO: Add support for the rest of the period formats (a*, b*, B*)
            return self.format_period(char)
        elif char == 'h':
            if self.value.hour % 12 == 0:
                return self.format(12, num)
            else:
                return self.format(self.value.hour % 12, num)
        elif char == 'H':
            return self.format(self.value.hour, num)
        elif char == 'K':
            return self.format(self.value.hour % 12, num)
        elif char == 'k':
            if self.value.hour == 0:
                return self.format(24, num)
            else:
                return self.format(self.value.hour, num)
        elif char == 'm':
            return self.format(self.value.minute, num)
        elif char == 's':
            return self.format(self.value.second, num)
        elif char == 'S':
            return self.format_frac_seconds(num)
        elif char == 'A':
            return self.format_milliseconds_in_day(num)
        elif char in ('z', 'Z', 'v', 'V', 'x', 'X', 'O'):
            return self.format_timezone(char, num)
        else:
            raise KeyError('Unsupported date/time field %r' % char)

    def extract(self, char):
        char = str(char)[0]
        if char == 'y':
            return self.value.year
        elif char == 'M':
            return self.value.month
        elif char == 'd':
            return self.value.day
        elif char == 'H':
            return self.value.hour
        elif char == 'h':
            return self.value.hour % 12 or 12
        elif char == 'm':
            return self.value.minute
        elif char == 'a':
            return int(self.value.hour >= 12)  # 0 for am, 1 for pm
        else:
            raise NotImplementedError("Not implemented: extracting %r from %r" % (char, self.value))

    def format_era(self, char, num):
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[max(3, num)]
        era = int(self.value.year >= 0)
        return get_era_names(width, self.locale)[era]

    def format_year(self, char, num):
        value = self.value.year
        if char.isupper():
            value = self.value.isocalendar()[0]
        year = self.format(value, num)
        if num == 2:
            year = year[-2:]
        return year

    def format_quarter(self, char, num):
        quarter = (self.value.month - 1) // 3 + 1
        if num <= 2:
            return '%0*d' % (num, quarter)
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[num]
        context = {'Q': 'format', 'q': 'stand-alone'}[char]
        return get_quarter_names(width, context, self.locale)[quarter]

    def format_month(self, char, num):
        if num <= 2:
            return '%0*d' % (num, self.value.month)
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow'}[num]
        context = {'M': 'format', 'L': 'stand-alone'}[char]
        return get_month_names(width, context, self.locale)[self.value.month]

    def format_week(self, char, num):
        if char.islower():  # week of year
            day_of_year = self.get_day_of_year()
            week = self.get_week_number(day_of_year)
            if week == 0:
                date = self.value - timedelta(days=day_of_year)
                week = self.get_week_number(self.get_day_of_year(date),
                                            date.weekday())
            return self.format(week, num)
        else:  # week of month
            week = self.get_week_number(self.value.day)
            if week == 0:
                date = self.value - timedelta(days=self.value.day)
                week = self.get_week_number(date.day, date.weekday())
            return '%d' % week

    def format_weekday(self, char='E', num=4):
        """
        Return weekday from parsed datetime according to format pattern.

        >>> format = DateTimeFormat(date(2016, 2, 28), Locale.parse('en_US'))
        >>> format.format_weekday()
        u'Sunday'

        'E': Day of week - Use one through three letters for the abbreviated day name, four for the full (wide) name,
             five for the narrow name, or six for the short name.
        >>> format.format_weekday('E',2)
        u'Sun'

        'e': Local day of week. Same as E except adds a numeric value that will depend on the local starting day of the
             week, using one or two letters. For this example, Monday is the first day of the week.
        >>> format.format_weekday('e',2)
        '01'

        'c': Stand-Alone local day of week - Use one letter for the local numeric value (same as 'e'), three for the
             abbreviated day name, four for the full (wide) name, five for the narrow name, or six for the short name.
        >>> format.format_weekday('c',1)
        '1'

        :param char: pattern format character ('e','E','c')
        :param num: count of format character

        """
        if num < 3:
            if char.islower():
                value = 7 - self.locale.first_week_day + self.value.weekday()
                return self.format(value % 7 + 1, num)
            num = 3
        weekday = self.value.weekday()
        width = {3: 'abbreviated', 4: 'wide', 5: 'narrow', 6: 'short'}[num]
        if char == 'c':
            context = 'stand-alone'
        else:
            context = 'format'
        return get_day_names(width, context, self.locale)[weekday]

    def format_day_of_year(self, num):
        return self.format(self.get_day_of_year(), num)

    def format_day_of_week_in_month(self):
        return '%d' % ((self.value.day - 1) // 7 + 1)

    def format_period(self, char):
        period = {0: 'am', 1: 'pm'}[int(self.value.hour >= 12)]
        for width in ('wide', 'narrow', 'abbreviated'):
            period_names = get_period_names(context='format', width=width, locale=self.locale)
            if period in period_names:
                return period_names[period]
        raise ValueError('Could not format period %s in %s' % (period, self.locale))

    def format_frac_seconds(self, num):
        """ Return fractional seconds.

        Rounds the time's microseconds to the precision given by the number \
        of digits passed in.
        """
        value = self.value.microsecond / 1000000
        return self.format(round(value, num) * 10**num, num)

    def format_milliseconds_in_day(self, num):
        msecs = self.value.microsecond // 1000 + self.value.second * 1000 + \
            self.value.minute * 60000 + self.value.hour * 3600000
        return self.format(msecs, num)

    def format_timezone(self, char, num):
        width = {3: 'short', 4: 'long', 5: 'iso8601'}[max(3, num)]
        if char == 'z':
            return get_timezone_name(self.value, width, locale=self.locale)
        elif char == 'Z':
            if num == 5:
                return get_timezone_gmt(self.value, width, locale=self.locale, return_z=True)
            return get_timezone_gmt(self.value, width, locale=self.locale)
        elif char == 'O':
            if num == 4:
                return get_timezone_gmt(self.value, width, locale=self.locale)
        # TODO: To add support for O:1
        elif char == 'v':
            return get_timezone_name(self.value.tzinfo, width,
                                     locale=self.locale)
        elif char == 'V':
            if num == 1:
                return get_timezone_name(self.value.tzinfo, width,
                                         uncommon=True, locale=self.locale)
            elif num == 2:
                return get_timezone_name(self.value.tzinfo, locale=self.locale, return_zone=True)
            elif num == 3:
                return get_timezone_location(self.value.tzinfo, locale=self.locale, return_city=True)
            return get_timezone_location(self.value.tzinfo, locale=self.locale)
        # Included additional elif condition to add support for 'Xx' in timezone format
        elif char == 'X':
            if num == 1:
                return get_timezone_gmt(self.value, width='iso8601_short', locale=self.locale,
                                        return_z=True)
            elif num in (2, 4):
                return get_timezone_gmt(self.value, width='short', locale=self.locale,
                                        return_z=True)
            elif num in (3, 5):
                return get_timezone_gmt(self.value, width='iso8601', locale=self.locale,
                                        return_z=True)
        elif char == 'x':
            if num == 1:
                return get_timezone_gmt(self.value, width='iso8601_short', locale=self.locale)
            elif num in (2, 4):
                return get_timezone_gmt(self.value, width='short', locale=self.locale)
            elif num in (3, 5):
                return get_timezone_gmt(self.value, width='iso8601', locale=self.locale)

    def format(self, value, length):
        return '%0*d' % (length, value)

    def get_day_of_year(self, date=None):
        if date is None:
            date = self.value
        return (date - date.replace(month=1, day=1)).days + 1

    def get_week_number(self, day_of_period, day_of_week=None):
        """Return the number of the week of a day within a period. This may be
        the week number in a year or the week number in a month.

        Usually this will return a value equal to or greater than 1, but if the
        first week of the period is so short that it actually counts as the last
        week of the previous period, this function will return 0.

        >>> format = DateTimeFormat(date(2006, 1, 8), Locale.parse('de_DE'))
        >>> format.get_week_number(6)
        1

        >>> format = DateTimeFormat(date(2006, 1, 8), Locale.parse('en_US'))
        >>> format.get_week_number(6)
        2

        :param day_of_period: the number of the day in the period (usually
                              either the day of month or the day of year)
        :param day_of_week: the week day; if ommitted, the week day of the
                            current date is assumed
        """
        if day_of_week is None:
            day_of_week = self.value.weekday()
        first_day = (day_of_week - self.locale.first_week_day -
                     day_of_period + 1) % 7
        if first_day < 0:
            first_day += 7
        week_number = (day_of_period + first_day - 1) // 7

        if 7 - first_day >= self.locale.min_week_days:
            week_number += 1

        if self.locale.first_week_day == 0:
            # Correct the weeknumber in case of iso-calendar usage (first_week_day=0).
            # If the weeknumber exceeds the maximum number of weeks for the given year
            # we must count from zero.For example the above calculation gives week 53
            # for 2018-12-31. By iso-calender definition 2018 has a max of 52
            # weeks, thus the weeknumber must be 53-52=1.
            max_weeks = date(year=self.value.year, day=28, month=12).isocalendar()[1]
            if week_number > max_weeks:
                week_number -= max_weeks

        return week_number


PATTERN_CHARS = {
    'G': [1, 2, 3, 4, 5],                                               # era
    'y': None, 'Y': None, 'u': None,                                    # year
    'Q': [1, 2, 3, 4, 5], 'q': [1, 2, 3, 4, 5],                         # quarter
    'M': [1, 2, 3, 4, 5], 'L': [1, 2, 3, 4, 5],                         # month
    'w': [1, 2], 'W': [1],                                              # week
    'd': [1, 2], 'D': [1, 2, 3], 'F': [1], 'g': None,                   # day
    'E': [1, 2, 3, 4, 5, 6], 'e': [1, 2, 3, 4, 5, 6], 'c': [1, 3, 4, 5, 6],  # week day
    'a': [1],                                                           # period
    'h': [1, 2], 'H': [1, 2], 'K': [1, 2], 'k': [1, 2],                 # hour
    'm': [1, 2],                                                        # minute
    's': [1, 2], 'S': None, 'A': None,                                  # second
    'z': [1, 2, 3, 4], 'Z': [1, 2, 3, 4, 5], 'O': [1, 4], 'v': [1, 4],  # zone
    'V': [1, 2, 3, 4], 'x': [1, 2, 3, 4, 5], 'X': [1, 2, 3, 4, 5]       # zone
}

#: The pattern characters declared in the Date Field Symbol Table
#: (https://www.unicode.org/reports/tr35/tr35-dates.html#Date_Field_Symbol_Table)
#: in order of decreasing magnitude.
PATTERN_CHAR_ORDER = "GyYuUQqMLlwWdDFgEecabBChHKkjJmsSAzZOvVXx"

_pattern_cache = {}


def parse_pattern(pattern):
    """Parse date, time, and datetime format patterns.

    >>> parse_pattern("MMMMd").format
    u'%(MMMM)s%(d)s'
    >>> parse_pattern("MMM d, yyyy").format
    u'%(MMM)s %(d)s, %(yyyy)s'

    Pattern can contain literal strings in single quotes:

    >>> parse_pattern("H:mm' Uhr 'z").format
    u'%(H)s:%(mm)s Uhr %(z)s'

    An actual single quote can be used by using two adjacent single quote
    characters:

    >>> parse_pattern("hh' o''clock'").format
    u"%(hh)s o'clock"

    :param pattern: the formatting pattern to parse
    """
    if type(pattern) is DateTimePattern:
        return pattern

    if pattern in _pattern_cache:
        return _pattern_cache[pattern]

    result = []

    for tok_type, tok_value in tokenize_pattern(pattern):
        if tok_type == "chars":
            result.append(tok_value.replace('%', '%%'))
        elif tok_type == "field":
            fieldchar, fieldnum = tok_value
            limit = PATTERN_CHARS[fieldchar]
            if limit and fieldnum not in limit:
                raise ValueError('Invalid length for field: %r'
                                 % (fieldchar * fieldnum))
            result.append('%%(%s)s' % (fieldchar * fieldnum))
        else:
            raise NotImplementedError("Unknown token type: %s" % tok_type)

    _pattern_cache[pattern] = pat = DateTimePattern(pattern, u''.join(result))
    return pat


def tokenize_pattern(pattern):
    """
    Tokenize date format patterns.

    Returns a list of (token_type, token_value) tuples.

    ``token_type`` may be either "chars" or "field".

    For "chars" tokens, the value is the literal value.

    For "field" tokens, the value is a tuple of (field character, repetition count).

    :param pattern: Pattern string
    :type pattern: str
    :rtype: list[tuple]
    """
    result = []
    quotebuf = None
    charbuf = []
    fieldchar = ['']
    fieldnum = [0]

    def append_chars():
        result.append(('chars', ''.join(charbuf).replace('\0', "'")))
        del charbuf[:]

    def append_field():
        result.append(('field', (fieldchar[0], fieldnum[0])))
        fieldchar[0] = ''
        fieldnum[0] = 0

    for idx, char in enumerate(pattern.replace("''", '\0')):
        if quotebuf is None:
            if char == "'":  # quote started
                if fieldchar[0]:
                    append_field()
                elif charbuf:
                    append_chars()
                quotebuf = []
            elif char in PATTERN_CHARS:
                if charbuf:
                    append_chars()
                if char == fieldchar[0]:
                    fieldnum[0] += 1
                else:
                    if fieldchar[0]:
                        append_field()
                    fieldchar[0] = char
                    fieldnum[0] = 1
            else:
                if fieldchar[0]:
                    append_field()
                charbuf.append(char)

        elif quotebuf is not None:
            if char == "'":  # end of quote
                charbuf.extend(quotebuf)
                quotebuf = None
            else:  # inside quote
                quotebuf.append(char)

    if fieldchar[0]:
        append_field()
    elif charbuf:
        append_chars()

    return result


def untokenize_pattern(tokens):
    """
    Turn a date format pattern token stream back into a string.

    This is the reverse operation of ``tokenize_pattern``.

    :type tokens: Iterable[tuple]
    :rtype: str
    """
    output = []
    for tok_type, tok_value in tokens:
        if tok_type == "field":
            output.append(tok_value[0] * tok_value[1])
        elif tok_type == "chars":
            if not any(ch in PATTERN_CHARS for ch in tok_value):  # No need to quote
                output.append(tok_value)
            else:
                output.append("'%s'" % tok_value.replace("'", "''"))
    return "".join(output)


def split_interval_pattern(pattern):
    """
    Split an interval-describing datetime pattern into multiple pieces.

    > The pattern is then designed to be broken up into two pieces by determining the first repeating field.
    - https://www.unicode.org/reports/tr35/tr35-dates.html#intervalFormats

    >>> split_interval_pattern(u'E d.M. \u2013 E d.M.')
    [u'E d.M. \u2013 ', 'E d.M.']
    >>> split_interval_pattern("Y 'text' Y 'more text'")
    ["Y 'text '", "Y 'more text'"]
    >>> split_interval_pattern(u"E, MMM d \u2013 E")
    [u'E, MMM d \u2013 ', u'E']
    >>> split_interval_pattern("MMM d")
    ['MMM d']
    >>> split_interval_pattern("y G")
    ['y G']
    >>> split_interval_pattern(u"MMM d \u2013 d")
    [u'MMM d \u2013 ', u'd']

    :param pattern: Interval pattern string
    :return: list of "subpatterns"
    """

    seen_fields = set()
    parts = [[]]

    for tok_type, tok_value in tokenize_pattern(pattern):
        if tok_type == "field":
            if tok_value[0] in seen_fields:  # Repeated field
                parts.append([])
                seen_fields.clear()
            seen_fields.add(tok_value[0])
        parts[-1].append((tok_type, tok_value))

    return [untokenize_pattern(tokens) for tokens in parts]


def match_skeleton(skeleton, options, allow_different_fields=False):
    """
    Find the closest match for the given datetime skeleton among the options given.

    This uses the rules outlined in the TR35 document.

    >>> match_skeleton('yMMd', ('yMd', 'yMMMd'))
    'yMd'

    >>> match_skeleton('yMMd', ('jyMMd',), allow_different_fields=True)
    'jyMMd'

    >>> match_skeleton('yMMd', ('qyMMd',), allow_different_fields=False)

    >>> match_skeleton('hmz', ('hmv',))
    'hmv'

    :param skeleton: The skeleton to match
    :type skeleton: str
    :param options: An iterable of other skeletons to match against
    :type options: Iterable[str]
    :return: The closest skeleton match, or if no match was found, None.
    :rtype: str|None
    """

    # TODO: maybe implement pattern expansion?

    # Based on the implementation in
    # http://source.icu-project.org/repos/icu/icu4j/trunk/main/classes/core/src/com/ibm/icu/text/DateIntervalInfo.java

    # Filter out falsy values and sort for stability; when `interval_formats` is passed in, there may be a None key.
    options = sorted(option for option in options if option)

    if 'z' in skeleton and not any('z' in option for option in options):
        skeleton = skeleton.replace('z', 'v')

    get_input_field_width = dict(t[1] for t in tokenize_pattern(skeleton) if t[0] == "field").get
    best_skeleton = None
    best_distance = None
    for option in options:
        get_opt_field_width = dict(t[1] for t in tokenize_pattern(option) if t[0] == "field").get
        distance = 0
        for field in PATTERN_CHARS:
            input_width = get_input_field_width(field, 0)
            opt_width = get_opt_field_width(field, 0)
            if input_width == opt_width:
                continue
            if opt_width == 0 or input_width == 0:
                if not allow_different_fields:  # This one is not okay
                    option = None
                    break
                distance += 0x1000  # Magic weight constant for "entirely different fields"
            elif field == 'M' and ((input_width > 2 and opt_width <= 2) or (input_width <= 2 and opt_width > 2)):
                distance += 0x100  # Magic weight for "text turns into a number"
            else:
                distance += abs(input_width - opt_width)

        if not option:  # We lost the option along the way (probably due to "allow_different_fields")
            continue

        if not best_skeleton or distance < best_distance:
            best_skeleton = option
            best_distance = distance

        if distance == 0:  # Found a perfect match!
            break

    return best_skeleton
