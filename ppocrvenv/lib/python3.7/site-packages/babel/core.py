# -*- coding: utf-8 -*-
"""
    babel.core
    ~~~~~~~~~~

    Core locale representation and locale data access.

    :copyright: (c) 2013-2021 by the Babel Team.
    :license: BSD, see LICENSE for more details.
"""

import os

from babel import localedata
from babel._compat import pickle, string_types
from babel.plural import PluralRule

__all__ = ['UnknownLocaleError', 'Locale', 'default_locale', 'negotiate_locale',
           'parse_locale']


_global_data = None
_default_plural_rule = PluralRule({})


def _raise_no_data_error():
    raise RuntimeError('The babel data files are not available. '
                       'This usually happens because you are using '
                       'a source checkout from Babel and you did '
                       'not build the data files.  Just make sure '
                       'to run "python setup.py import_cldr" before '
                       'installing the library.')


def get_global(key):
    """Return the dictionary for the given key in the global data.

    The global data is stored in the ``babel/global.dat`` file and contains
    information independent of individual locales.

    >>> get_global('zone_aliases')['UTC']
    u'Etc/UTC'
    >>> get_global('zone_territories')['Europe/Berlin']
    u'DE'

    The keys available are:

    - ``all_currencies``
    - ``currency_fractions``
    - ``language_aliases``
    - ``likely_subtags``
    - ``parent_exceptions``
    - ``script_aliases``
    - ``territory_aliases``
    - ``territory_currencies``
    - ``territory_languages``
    - ``territory_zones``
    - ``variant_aliases``
    - ``windows_zone_mapping``
    - ``zone_aliases``
    - ``zone_territories``

    .. note:: The internal structure of the data may change between versions.

    .. versionadded:: 0.9

    :param key: the data key
    """
    global _global_data
    if _global_data is None:
        dirname = os.path.join(os.path.dirname(__file__))
        filename = os.path.join(dirname, 'global.dat')
        if not os.path.isfile(filename):
            _raise_no_data_error()
        with open(filename, 'rb') as fileobj:
            _global_data = pickle.load(fileobj)
    return _global_data.get(key, {})


LOCALE_ALIASES = {
    'ar': 'ar_SY', 'bg': 'bg_BG', 'bs': 'bs_BA', 'ca': 'ca_ES', 'cs': 'cs_CZ',
    'da': 'da_DK', 'de': 'de_DE', 'el': 'el_GR', 'en': 'en_US', 'es': 'es_ES',
    'et': 'et_EE', 'fa': 'fa_IR', 'fi': 'fi_FI', 'fr': 'fr_FR', 'gl': 'gl_ES',
    'he': 'he_IL', 'hu': 'hu_HU', 'id': 'id_ID', 'is': 'is_IS', 'it': 'it_IT',
    'ja': 'ja_JP', 'km': 'km_KH', 'ko': 'ko_KR', 'lt': 'lt_LT', 'lv': 'lv_LV',
    'mk': 'mk_MK', 'nl': 'nl_NL', 'nn': 'nn_NO', 'no': 'nb_NO', 'pl': 'pl_PL',
    'pt': 'pt_PT', 'ro': 'ro_RO', 'ru': 'ru_RU', 'sk': 'sk_SK', 'sl': 'sl_SI',
    'sv': 'sv_SE', 'th': 'th_TH', 'tr': 'tr_TR', 'uk': 'uk_UA'
}


class UnknownLocaleError(Exception):
    """Exception thrown when a locale is requested for which no locale data
    is available.
    """

    def __init__(self, identifier):
        """Create the exception.

        :param identifier: the identifier string of the unsupported locale
        """
        Exception.__init__(self, 'unknown locale %r' % identifier)

        #: The identifier of the locale that could not be found.
        self.identifier = identifier


class Locale(object):
    """Representation of a specific locale.

    >>> locale = Locale('en', 'US')
    >>> repr(locale)
    "Locale('en', territory='US')"
    >>> locale.display_name
    u'English (United States)'

    A `Locale` object can also be instantiated from a raw locale string:

    >>> locale = Locale.parse('en-US', sep='-')
    >>> repr(locale)
    "Locale('en', territory='US')"

    `Locale` objects provide access to a collection of locale data, such as
    territory and language names, number and date format patterns, and more:

    >>> locale.number_symbols['decimal']
    u'.'

    If a locale is requested for which no locale data is available, an
    `UnknownLocaleError` is raised:

    >>> Locale.parse('en_XX')
    Traceback (most recent call last):
        ...
    UnknownLocaleError: unknown locale 'en_XX'

    For more information see :rfc:`3066`.
    """

    def __init__(self, language, territory=None, script=None, variant=None):
        """Initialize the locale object from the given identifier components.

        >>> locale = Locale('en', 'US')
        >>> locale.language
        'en'
        >>> locale.territory
        'US'

        :param language: the language code
        :param territory: the territory (country or region) code
        :param script: the script code
        :param variant: the variant code
        :raise `UnknownLocaleError`: if no locale data is available for the
                                     requested locale
        """
        #: the language code
        self.language = language
        #: the territory (country or region) code
        self.territory = territory
        #: the script code
        self.script = script
        #: the variant code
        self.variant = variant
        self.__data = None

        identifier = str(self)
        if not localedata.exists(identifier):
            raise UnknownLocaleError(identifier)

    @classmethod
    def default(cls, category=None, aliases=LOCALE_ALIASES):
        """Return the system default locale for the specified category.

        >>> for name in ['LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LC_MESSAGES']:
        ...     os.environ[name] = ''
        >>> os.environ['LANG'] = 'fr_FR.UTF-8'
        >>> Locale.default('LC_MESSAGES')
        Locale('fr', territory='FR')

        The following fallbacks to the variable are always considered:

        - ``LANGUAGE``
        - ``LC_ALL``
        - ``LC_CTYPE``
        - ``LANG``

        :param category: one of the ``LC_XXX`` environment variable names
        :param aliases: a dictionary of aliases for locale identifiers
        """
        # XXX: use likely subtag expansion here instead of the
        # aliases dictionary.
        locale_string = default_locale(category, aliases=aliases)
        return cls.parse(locale_string)

    @classmethod
    def negotiate(cls, preferred, available, sep='_', aliases=LOCALE_ALIASES):
        """Find the best match between available and requested locale strings.

        >>> Locale.negotiate(['de_DE', 'en_US'], ['de_DE', 'de_AT'])
        Locale('de', territory='DE')
        >>> Locale.negotiate(['de_DE', 'en_US'], ['en', 'de'])
        Locale('de')
        >>> Locale.negotiate(['de_DE', 'de'], ['en_US'])

        You can specify the character used in the locale identifiers to separate
        the differnet components. This separator is applied to both lists. Also,
        case is ignored in the comparison:

        >>> Locale.negotiate(['de-DE', 'de'], ['en-us', 'de-de'], sep='-')
        Locale('de', territory='DE')

        :param preferred: the list of locale identifers preferred by the user
        :param available: the list of locale identifiers available
        :param aliases: a dictionary of aliases for locale identifiers
        """
        identifier = negotiate_locale(preferred, available, sep=sep,
                                      aliases=aliases)
        if identifier:
            return Locale.parse(identifier, sep=sep)

    @classmethod
    def parse(cls, identifier, sep='_', resolve_likely_subtags=True):
        """Create a `Locale` instance for the given locale identifier.

        >>> l = Locale.parse('de-DE', sep='-')
        >>> l.display_name
        u'Deutsch (Deutschland)'

        If the `identifier` parameter is not a string, but actually a `Locale`
        object, that object is returned:

        >>> Locale.parse(l)
        Locale('de', territory='DE')

        This also can perform resolving of likely subtags which it does
        by default.  This is for instance useful to figure out the most
        likely locale for a territory you can use ``'und'`` as the
        language tag:

        >>> Locale.parse('und_AT')
        Locale('de', territory='AT')

        :param identifier: the locale identifier string
        :param sep: optional component separator
        :param resolve_likely_subtags: if this is specified then a locale will
                                       have its likely subtag resolved if the
                                       locale otherwise does not exist.  For
                                       instance ``zh_TW`` by itself is not a
                                       locale that exists but Babel can
                                       automatically expand it to the full
                                       form of ``zh_hant_TW``.  Note that this
                                       expansion is only taking place if no
                                       locale exists otherwise.  For instance
                                       there is a locale ``en`` that can exist
                                       by itself.
        :raise `ValueError`: if the string does not appear to be a valid locale
                             identifier
        :raise `UnknownLocaleError`: if no locale data is available for the
                                     requested locale
        """
        if identifier is None:
            return None
        elif isinstance(identifier, Locale):
            return identifier
        elif not isinstance(identifier, string_types):
            raise TypeError('Unexpected value for identifier: %r' % (identifier,))

        parts = parse_locale(identifier, sep=sep)
        input_id = get_locale_identifier(parts)

        def _try_load(parts):
            try:
                return cls(*parts)
            except UnknownLocaleError:
                return None

        def _try_load_reducing(parts):
            # Success on first hit, return it.
            locale = _try_load(parts)
            if locale is not None:
                return locale

            # Now try without script and variant
            locale = _try_load(parts[:2])
            if locale is not None:
                return locale

        locale = _try_load(parts)
        if locale is not None:
            return locale
        if not resolve_likely_subtags:
            raise UnknownLocaleError(input_id)

        # From here onwards is some very bad likely subtag resolving.  This
        # whole logic is not entirely correct but good enough (tm) for the
        # time being.  This has been added so that zh_TW does not cause
        # errors for people when they upgrade.  Later we should properly
        # implement ICU like fuzzy locale objects and provide a way to
        # maximize and minimize locale tags.

        language, territory, script, variant = parts
        language = get_global('language_aliases').get(language, language)
        territory = get_global('territory_aliases').get(territory, (territory,))[0]
        script = get_global('script_aliases').get(script, script)
        variant = get_global('variant_aliases').get(variant, variant)

        if territory == 'ZZ':
            territory = None
        if script == 'Zzzz':
            script = None

        parts = language, territory, script, variant

        # First match: try the whole identifier
        new_id = get_locale_identifier(parts)
        likely_subtag = get_global('likely_subtags').get(new_id)
        if likely_subtag is not None:
            locale = _try_load_reducing(parse_locale(likely_subtag))
            if locale is not None:
                return locale

        # If we did not find anything so far, try again with a
        # simplified identifier that is just the language
        likely_subtag = get_global('likely_subtags').get(language)
        if likely_subtag is not None:
            language2, _, script2, variant2 = parse_locale(likely_subtag)
            locale = _try_load_reducing((language2, territory, script2, variant2))
            if locale is not None:
                return locale

        raise UnknownLocaleError(input_id)

    def __eq__(self, other):
        for key in ('language', 'territory', 'script', 'variant'):
            if not hasattr(other, key):
                return False
        return (self.language == other.language) and \
            (self.territory == other.territory) and \
            (self.script == other.script) and \
            (self.variant == other.variant)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.language, self.territory, self.script, self.variant))

    def __repr__(self):
        parameters = ['']
        for key in ('territory', 'script', 'variant'):
            value = getattr(self, key)
            if value is not None:
                parameters.append('%s=%r' % (key, value))
        parameter_string = '%r' % self.language + ', '.join(parameters)
        return 'Locale(%s)' % parameter_string

    def __str__(self):
        return get_locale_identifier((self.language, self.territory,
                                      self.script, self.variant))

    @property
    def _data(self):
        if self.__data is None:
            self.__data = localedata.LocaleDataDict(localedata.load(str(self)))
        return self.__data

    def get_display_name(self, locale=None):
        """Return the display name of the locale using the given locale.

        The display name will include the language, territory, script, and
        variant, if those are specified.

        >>> Locale('zh', 'CN', script='Hans').get_display_name('en')
        u'Chinese (Simplified, China)'

        :param locale: the locale to use
        """
        if locale is None:
            locale = self
        locale = Locale.parse(locale)
        retval = locale.languages.get(self.language)
        if retval and (self.territory or self.script or self.variant):
            details = []
            if self.script:
                details.append(locale.scripts.get(self.script))
            if self.territory:
                details.append(locale.territories.get(self.territory))
            if self.variant:
                details.append(locale.variants.get(self.variant))
            details = filter(None, details)
            if details:
                retval += ' (%s)' % u', '.join(details)
        return retval

    display_name = property(get_display_name, doc="""\
        The localized display name of the locale.

        >>> Locale('en').display_name
        u'English'
        >>> Locale('en', 'US').display_name
        u'English (United States)'
        >>> Locale('sv').display_name
        u'svenska'

        :type: `unicode`
        """)

    def get_language_name(self, locale=None):
        """Return the language of this locale in the given locale.

        >>> Locale('zh', 'CN', script='Hans').get_language_name('de')
        u'Chinesisch'

        .. versionadded:: 1.0

        :param locale: the locale to use
        """
        if locale is None:
            locale = self
        locale = Locale.parse(locale)
        return locale.languages.get(self.language)

    language_name = property(get_language_name, doc="""\
        The localized language name of the locale.

        >>> Locale('en', 'US').language_name
        u'English'
    """)

    def get_territory_name(self, locale=None):
        """Return the territory name in the given locale."""
        if locale is None:
            locale = self
        locale = Locale.parse(locale)
        return locale.territories.get(self.territory)

    territory_name = property(get_territory_name, doc="""\
        The localized territory name of the locale if available.

        >>> Locale('de', 'DE').territory_name
        u'Deutschland'
    """)

    def get_script_name(self, locale=None):
        """Return the script name in the given locale."""
        if locale is None:
            locale = self
        locale = Locale.parse(locale)
        return locale.scripts.get(self.script)

    script_name = property(get_script_name, doc="""\
        The localized script name of the locale if available.

        >>> Locale('sr', 'ME', script='Latn').script_name
        u'latinica'
    """)

    @property
    def english_name(self):
        """The english display name of the locale.

        >>> Locale('de').english_name
        u'German'
        >>> Locale('de', 'DE').english_name
        u'German (Germany)'

        :type: `unicode`"""
        return self.get_display_name(Locale('en'))

    # { General Locale Display Names

    @property
    def languages(self):
        """Mapping of language codes to translated language names.

        >>> Locale('de', 'DE').languages['ja']
        u'Japanisch'

        See `ISO 639 <http://www.loc.gov/standards/iso639-2/>`_ for
        more information.
        """
        return self._data['languages']

    @property
    def scripts(self):
        """Mapping of script codes to translated script names.

        >>> Locale('en', 'US').scripts['Hira']
        u'Hiragana'

        See `ISO 15924 <http://www.evertype.com/standards/iso15924/>`_
        for more information.
        """
        return self._data['scripts']

    @property
    def territories(self):
        """Mapping of script codes to translated script names.

        >>> Locale('es', 'CO').territories['DE']
        u'Alemania'

        See `ISO 3166 <http://www.iso.org/iso/en/prods-services/iso3166ma/>`_
        for more information.
        """
        return self._data['territories']

    @property
    def variants(self):
        """Mapping of script codes to translated script names.

        >>> Locale('de', 'DE').variants['1901']
        u'Alte deutsche Rechtschreibung'
        """
        return self._data['variants']

    # { Number Formatting

    @property
    def currencies(self):
        """Mapping of currency codes to translated currency names.  This
        only returns the generic form of the currency name, not the count
        specific one.  If an actual number is requested use the
        :func:`babel.numbers.get_currency_name` function.

        >>> Locale('en').currencies['COP']
        u'Colombian Peso'
        >>> Locale('de', 'DE').currencies['COP']
        u'Kolumbianischer Peso'
        """
        return self._data['currency_names']

    @property
    def currency_symbols(self):
        """Mapping of currency codes to symbols.

        >>> Locale('en', 'US').currency_symbols['USD']
        u'$'
        >>> Locale('es', 'CO').currency_symbols['USD']
        u'US$'
        """
        return self._data['currency_symbols']

    @property
    def number_symbols(self):
        """Symbols used in number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('fr', 'FR').number_symbols['decimal']
        u','
        """
        return self._data['number_symbols']

    @property
    def decimal_formats(self):
        """Locale patterns for decimal number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').decimal_formats[None]
        <NumberPattern u'#,##0.###'>
        """
        return self._data['decimal_formats']

    @property
    def currency_formats(self):
        """Locale patterns for currency number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').currency_formats['standard']
        <NumberPattern u'\\xa4#,##0.00'>
        >>> Locale('en', 'US').currency_formats['accounting']
        <NumberPattern u'\\xa4#,##0.00;(\\xa4#,##0.00)'>
        """
        return self._data['currency_formats']

    @property
    def percent_formats(self):
        """Locale patterns for percent number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').percent_formats[None]
        <NumberPattern u'#,##0%'>
        """
        return self._data['percent_formats']

    @property
    def scientific_formats(self):
        """Locale patterns for scientific number formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').scientific_formats[None]
        <NumberPattern u'#E0'>
        """
        return self._data['scientific_formats']

    # { Calendar Information and Date Formatting

    @property
    def periods(self):
        """Locale display names for day periods (AM/PM).

        >>> Locale('en', 'US').periods['am']
        u'AM'
        """
        try:
            return self._data['day_periods']['stand-alone']['wide']
        except KeyError:
            return {}

    @property
    def day_periods(self):
        """Locale display names for various day periods (not necessarily only AM/PM).

        These are not meant to be used without the relevant `day_period_rules`.
        """
        return self._data['day_periods']

    @property
    def day_period_rules(self):
        """Day period rules for the locale.  Used by `get_period_id`.
        """
        return self._data.get('day_period_rules', {})

    @property
    def days(self):
        """Locale display names for weekdays.

        >>> Locale('de', 'DE').days['format']['wide'][3]
        u'Donnerstag'
        """
        return self._data['days']

    @property
    def months(self):
        """Locale display names for months.

        >>> Locale('de', 'DE').months['format']['wide'][10]
        u'Oktober'
        """
        return self._data['months']

    @property
    def quarters(self):
        """Locale display names for quarters.

        >>> Locale('de', 'DE').quarters['format']['wide'][1]
        u'1. Quartal'
        """
        return self._data['quarters']

    @property
    def eras(self):
        """Locale display names for eras.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').eras['wide'][1]
        u'Anno Domini'
        >>> Locale('en', 'US').eras['abbreviated'][0]
        u'BC'
        """
        return self._data['eras']

    @property
    def time_zones(self):
        """Locale display names for time zones.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').time_zones['Europe/London']['long']['daylight']
        u'British Summer Time'
        >>> Locale('en', 'US').time_zones['America/St_Johns']['city']
        u'St. John\u2019s'
        """
        return self._data['time_zones']

    @property
    def meta_zones(self):
        """Locale display names for meta time zones.

        Meta time zones are basically groups of different Olson time zones that
        have the same GMT offset and daylight savings time.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').meta_zones['Europe_Central']['long']['daylight']
        u'Central European Summer Time'

        .. versionadded:: 0.9
        """
        return self._data['meta_zones']

    @property
    def zone_formats(self):
        """Patterns related to the formatting of time zones.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').zone_formats['fallback']
        u'%(1)s (%(0)s)'
        >>> Locale('pt', 'BR').zone_formats['region']
        u'Hor\\xe1rio %s'

        .. versionadded:: 0.9
        """
        return self._data['zone_formats']

    @property
    def first_week_day(self):
        """The first day of a week, with 0 being Monday.

        >>> Locale('de', 'DE').first_week_day
        0
        >>> Locale('en', 'US').first_week_day
        6
        """
        return self._data['week_data']['first_day']

    @property
    def weekend_start(self):
        """The day the weekend starts, with 0 being Monday.

        >>> Locale('de', 'DE').weekend_start
        5
        """
        return self._data['week_data']['weekend_start']

    @property
    def weekend_end(self):
        """The day the weekend ends, with 0 being Monday.

        >>> Locale('de', 'DE').weekend_end
        6
        """
        return self._data['week_data']['weekend_end']

    @property
    def min_week_days(self):
        """The minimum number of days in a week so that the week is counted as
        the first week of a year or month.

        >>> Locale('de', 'DE').min_week_days
        4
        """
        return self._data['week_data']['min_days']

    @property
    def date_formats(self):
        """Locale patterns for date formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').date_formats['short']
        <DateTimePattern u'M/d/yy'>
        >>> Locale('fr', 'FR').date_formats['long']
        <DateTimePattern u'd MMMM y'>
        """
        return self._data['date_formats']

    @property
    def time_formats(self):
        """Locale patterns for time formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').time_formats['short']
        <DateTimePattern u'h:mm a'>
        >>> Locale('fr', 'FR').time_formats['long']
        <DateTimePattern u'HH:mm:ss z'>
        """
        return self._data['time_formats']

    @property
    def datetime_formats(self):
        """Locale patterns for datetime formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en').datetime_formats['full']
        u"{1} 'at' {0}"
        >>> Locale('th').datetime_formats['medium']
        u'{1} {0}'
        """
        return self._data['datetime_formats']

    @property
    def datetime_skeletons(self):
        """Locale patterns for formatting parts of a datetime.

        >>> Locale('en').datetime_skeletons['MEd']
        <DateTimePattern u'E, M/d'>
        >>> Locale('fr').datetime_skeletons['MEd']
        <DateTimePattern u'E dd/MM'>
        >>> Locale('fr').datetime_skeletons['H']
        <DateTimePattern u"HH 'h'">
        """
        return self._data['datetime_skeletons']

    @property
    def interval_formats(self):
        """Locale patterns for interval formatting.

        .. note:: The format of the value returned may change between
                  Babel versions.

        How to format date intervals in Finnish when the day is the
        smallest changing component:

        >>> Locale('fi_FI').interval_formats['MEd']['d']
        [u'E d. \u2013 ', u'E d.M.']

        .. seealso::

           The primary API to use this data is :py:func:`babel.dates.format_interval`.


        :rtype: dict[str, dict[str, list[str]]]
        """
        return self._data['interval_formats']

    @property
    def plural_form(self):
        """Plural rules for the locale.

        >>> Locale('en').plural_form(1)
        'one'
        >>> Locale('en').plural_form(0)
        'other'
        >>> Locale('fr').plural_form(0)
        'one'
        >>> Locale('ru').plural_form(100)
        'many'
        """
        return self._data.get('plural_form', _default_plural_rule)

    @property
    def list_patterns(self):
        """Patterns for generating lists

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en').list_patterns['standard']['start']
        u'{0}, {1}'
        >>> Locale('en').list_patterns['standard']['end']
        u'{0}, and {1}'
        >>> Locale('en_GB').list_patterns['standard']['end']
        u'{0} and {1}'
        """
        return self._data['list_patterns']

    @property
    def ordinal_form(self):
        """Plural rules for the locale.

        >>> Locale('en').ordinal_form(1)
        'one'
        >>> Locale('en').ordinal_form(2)
        'two'
        >>> Locale('en').ordinal_form(3)
        'few'
        >>> Locale('fr').ordinal_form(2)
        'other'
        >>> Locale('ru').ordinal_form(100)
        'other'
        """
        return self._data.get('ordinal_form', _default_plural_rule)

    @property
    def measurement_systems(self):
        """Localized names for various measurement systems.

        >>> Locale('fr', 'FR').measurement_systems['US']
        u'am\\xe9ricain'
        >>> Locale('en', 'US').measurement_systems['US']
        u'US'

        """
        return self._data['measurement_systems']

    @property
    def character_order(self):
        """The text direction for the language.

        >>> Locale('de', 'DE').character_order
        'left-to-right'
        >>> Locale('ar', 'SA').character_order
        'right-to-left'
        """
        return self._data['character_order']

    @property
    def text_direction(self):
        """The text direction for the language in CSS short-hand form.

        >>> Locale('de', 'DE').text_direction
        'ltr'
        >>> Locale('ar', 'SA').text_direction
        'rtl'
        """
        return ''.join(word[0] for word in self.character_order.split('-'))

    @property
    def unit_display_names(self):
        """Display names for units of measurement.

        .. seealso::

           You may want to use :py:func:`babel.units.get_unit_name` instead.

        .. note:: The format of the value returned may change between
                  Babel versions.

        """
        return self._data['unit_display_names']


def default_locale(category=None, aliases=LOCALE_ALIASES):
    """Returns the system default locale for a given category, based on
    environment variables.

    >>> for name in ['LANGUAGE', 'LC_ALL', 'LC_CTYPE']:
    ...     os.environ[name] = ''
    >>> os.environ['LANG'] = 'fr_FR.UTF-8'
    >>> default_locale('LC_MESSAGES')
    'fr_FR'

    The "C" or "POSIX" pseudo-locales are treated as aliases for the
    "en_US_POSIX" locale:

    >>> os.environ['LC_MESSAGES'] = 'POSIX'
    >>> default_locale('LC_MESSAGES')
    'en_US_POSIX'

    The following fallbacks to the variable are always considered:

    - ``LANGUAGE``
    - ``LC_ALL``
    - ``LC_CTYPE``
    - ``LANG``

    :param category: one of the ``LC_XXX`` environment variable names
    :param aliases: a dictionary of aliases for locale identifiers
    """
    varnames = (category, 'LANGUAGE', 'LC_ALL', 'LC_CTYPE', 'LANG')
    for name in filter(None, varnames):
        locale = os.getenv(name)
        if locale:
            if name == 'LANGUAGE' and ':' in locale:
                # the LANGUAGE variable may contain a colon-separated list of
                # language codes; we just pick the language on the list
                locale = locale.split(':')[0]
            if locale.split('.')[0] in ('C', 'POSIX'):
                locale = 'en_US_POSIX'
            elif aliases and locale in aliases:
                locale = aliases[locale]
            try:
                return get_locale_identifier(parse_locale(locale))
            except ValueError:
                pass


def negotiate_locale(preferred, available, sep='_', aliases=LOCALE_ALIASES):
    """Find the best match between available and requested locale strings.

    >>> negotiate_locale(['de_DE', 'en_US'], ['de_DE', 'de_AT'])
    'de_DE'
    >>> negotiate_locale(['de_DE', 'en_US'], ['en', 'de'])
    'de'

    Case is ignored by the algorithm, the result uses the case of the preferred
    locale identifier:

    >>> negotiate_locale(['de_DE', 'en_US'], ['de_de', 'de_at'])
    'de_DE'

    >>> negotiate_locale(['de_DE', 'en_US'], ['de_de', 'de_at'])
    'de_DE'

    By default, some web browsers unfortunately do not include the territory
    in the locale identifier for many locales, and some don't even allow the
    user to easily add the territory. So while you may prefer using qualified
    locale identifiers in your web-application, they would not normally match
    the language-only locale sent by such browsers. To workaround that, this
    function uses a default mapping of commonly used langauge-only locale
    identifiers to identifiers including the territory:

    >>> negotiate_locale(['ja', 'en_US'], ['ja_JP', 'en_US'])
    'ja_JP'

    Some browsers even use an incorrect or outdated language code, such as "no"
    for Norwegian, where the correct locale identifier would actually be "nb_NO"
    (Bokmål) or "nn_NO" (Nynorsk). The aliases are intended to take care of
    such cases, too:

    >>> negotiate_locale(['no', 'sv'], ['nb_NO', 'sv_SE'])
    'nb_NO'

    You can override this default mapping by passing a different `aliases`
    dictionary to this function, or you can bypass the behavior althogher by
    setting the `aliases` parameter to `None`.

    :param preferred: the list of locale strings preferred by the user
    :param available: the list of locale strings available
    :param sep: character that separates the different parts of the locale
                strings
    :param aliases: a dictionary of aliases for locale identifiers
    """
    available = [a.lower() for a in available if a]
    for locale in preferred:
        ll = locale.lower()
        if ll in available:
            return locale
        if aliases:
            alias = aliases.get(ll)
            if alias:
                alias = alias.replace('_', sep)
                if alias.lower() in available:
                    return alias
        parts = locale.split(sep)
        if len(parts) > 1 and parts[0].lower() in available:
            return parts[0]
    return None


def parse_locale(identifier, sep='_'):
    """Parse a locale identifier into a tuple of the form ``(language,
    territory, script, variant)``.

    >>> parse_locale('zh_CN')
    ('zh', 'CN', None, None)
    >>> parse_locale('zh_Hans_CN')
    ('zh', 'CN', 'Hans', None)

    The default component separator is "_", but a different separator can be
    specified using the `sep` parameter:

    >>> parse_locale('zh-CN', sep='-')
    ('zh', 'CN', None, None)

    If the identifier cannot be parsed into a locale, a `ValueError` exception
    is raised:

    >>> parse_locale('not_a_LOCALE_String')
    Traceback (most recent call last):
      ...
    ValueError: 'not_a_LOCALE_String' is not a valid locale identifier

    Encoding information and locale modifiers are removed from the identifier:

    >>> parse_locale('it_IT@euro')
    ('it', 'IT', None, None)
    >>> parse_locale('en_US.UTF-8')
    ('en', 'US', None, None)
    >>> parse_locale('de_DE.iso885915@euro')
    ('de', 'DE', None, None)

    See :rfc:`4646` for more information.

    :param identifier: the locale identifier string
    :param sep: character that separates the different components of the locale
                identifier
    :raise `ValueError`: if the string does not appear to be a valid locale
                         identifier
    """
    if '.' in identifier:
        # this is probably the charset/encoding, which we don't care about
        identifier = identifier.split('.', 1)[0]
    if '@' in identifier:
        # this is a locale modifier such as @euro, which we don't care about
        # either
        identifier = identifier.split('@', 1)[0]

    parts = identifier.split(sep)
    lang = parts.pop(0).lower()
    if not lang.isalpha():
        raise ValueError('expected only letters, got %r' % lang)

    script = territory = variant = None
    if parts:
        if len(parts[0]) == 4 and parts[0].isalpha():
            script = parts.pop(0).title()

    if parts:
        if len(parts[0]) == 2 and parts[0].isalpha():
            territory = parts.pop(0).upper()
        elif len(parts[0]) == 3 and parts[0].isdigit():
            territory = parts.pop(0)

    if parts:
        if len(parts[0]) == 4 and parts[0][0].isdigit() or \
                len(parts[0]) >= 5 and parts[0][0].isalpha():
            variant = parts.pop()

    if parts:
        raise ValueError('%r is not a valid locale identifier' % identifier)

    return lang, territory, script, variant


def get_locale_identifier(tup, sep='_'):
    """The reverse of :func:`parse_locale`.  It creates a locale identifier out
    of a ``(language, territory, script, variant)`` tuple.  Items can be set to
    ``None`` and trailing ``None``\\s can also be left out of the tuple.

    >>> get_locale_identifier(('de', 'DE', None, '1999'))
    'de_DE_1999'

    .. versionadded:: 1.0

    :param tup: the tuple as returned by :func:`parse_locale`.
    :param sep: the separator for the identifier.
    """
    tup = tuple(tup[:4])
    lang, territory, script, variant = tup + (None,) * (4 - len(tup))
    return sep.join(filter(None, (lang, script, territory, variant)))
