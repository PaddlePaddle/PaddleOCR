# -*- coding: utf-8 -*-
"""
    babel.support
    ~~~~~~~~~~~~~

    Several classes and functions that help with integrating and using Babel
    in applications.

    .. note: the code in this module is not used by Babel itself

    :copyright: (c) 2013-2021 by the Babel Team.
    :license: BSD, see LICENSE for more details.
"""

import gettext
import locale

from babel.core import Locale
from babel.dates import format_date, format_datetime, format_time, \
    format_timedelta
from babel.numbers import format_number, format_decimal, format_currency, \
    format_percent, format_scientific
from babel._compat import PY2, text_type, text_to_native


class Format(object):
    """Wrapper class providing the various date and number formatting functions
    bound to a specific locale and time-zone.

    >>> from babel.util import UTC
    >>> from datetime import date
    >>> fmt = Format('en_US', UTC)
    >>> fmt.date(date(2007, 4, 1))
    u'Apr 1, 2007'
    >>> fmt.decimal(1.2345)
    u'1.234'
    """

    def __init__(self, locale, tzinfo=None):
        """Initialize the formatter.

        :param locale: the locale identifier or `Locale` instance
        :param tzinfo: the time-zone info (a `tzinfo` instance or `None`)
        """
        self.locale = Locale.parse(locale)
        self.tzinfo = tzinfo

    def date(self, date=None, format='medium'):
        """Return a date formatted according to the given pattern.

        >>> from datetime import date
        >>> fmt = Format('en_US')
        >>> fmt.date(date(2007, 4, 1))
        u'Apr 1, 2007'
        """
        return format_date(date, format, locale=self.locale)

    def datetime(self, datetime=None, format='medium'):
        """Return a date and time formatted according to the given pattern.

        >>> from datetime import datetime
        >>> from pytz import timezone
        >>> fmt = Format('en_US', tzinfo=timezone('US/Eastern'))
        >>> fmt.datetime(datetime(2007, 4, 1, 15, 30))
        u'Apr 1, 2007, 11:30:00 AM'
        """
        return format_datetime(datetime, format, tzinfo=self.tzinfo,
                               locale=self.locale)

    def time(self, time=None, format='medium'):
        """Return a time formatted according to the given pattern.

        >>> from datetime import datetime
        >>> from pytz import timezone
        >>> fmt = Format('en_US', tzinfo=timezone('US/Eastern'))
        >>> fmt.time(datetime(2007, 4, 1, 15, 30))
        u'11:30:00 AM'
        """
        return format_time(time, format, tzinfo=self.tzinfo, locale=self.locale)

    def timedelta(self, delta, granularity='second', threshold=.85,
                  format='long', add_direction=False):
        """Return a time delta according to the rules of the given locale.

        >>> from datetime import timedelta
        >>> fmt = Format('en_US')
        >>> fmt.timedelta(timedelta(weeks=11))
        u'3 months'
        """
        return format_timedelta(delta, granularity=granularity,
                                threshold=threshold,
                                format=format, add_direction=add_direction,
                                locale=self.locale)

    def number(self, number):
        """Return an integer number formatted for the locale.

        >>> fmt = Format('en_US')
        >>> fmt.number(1099)
        u'1,099'
        """
        return format_number(number, locale=self.locale)

    def decimal(self, number, format=None):
        """Return a decimal number formatted for the locale.

        >>> fmt = Format('en_US')
        >>> fmt.decimal(1.2345)
        u'1.234'
        """
        return format_decimal(number, format, locale=self.locale)

    def currency(self, number, currency):
        """Return a number in the given currency formatted for the locale.
        """
        return format_currency(number, currency, locale=self.locale)

    def percent(self, number, format=None):
        """Return a number formatted as percentage for the locale.

        >>> fmt = Format('en_US')
        >>> fmt.percent(0.34)
        u'34%'
        """
        return format_percent(number, format, locale=self.locale)

    def scientific(self, number):
        """Return a number formatted using scientific notation for the locale.
        """
        return format_scientific(number, locale=self.locale)


class LazyProxy(object):
    """Class for proxy objects that delegate to a specified function to evaluate
    the actual object.

    >>> def greeting(name='world'):
    ...     return 'Hello, %s!' % name
    >>> lazy_greeting = LazyProxy(greeting, name='Joe')
    >>> print(lazy_greeting)
    Hello, Joe!
    >>> u'  ' + lazy_greeting
    u'  Hello, Joe!'
    >>> u'(%s)' % lazy_greeting
    u'(Hello, Joe!)'

    This can be used, for example, to implement lazy translation functions that
    delay the actual translation until the string is actually used. The
    rationale for such behavior is that the locale of the user may not always
    be available. In web applications, you only know the locale when processing
    a request.

    The proxy implementation attempts to be as complete as possible, so that
    the lazy objects should mostly work as expected, for example for sorting:

    >>> greetings = [
    ...     LazyProxy(greeting, 'world'),
    ...     LazyProxy(greeting, 'Joe'),
    ...     LazyProxy(greeting, 'universe'),
    ... ]
    >>> greetings.sort()
    >>> for greeting in greetings:
    ...     print(greeting)
    Hello, Joe!
    Hello, universe!
    Hello, world!
    """
    __slots__ = ['_func', '_args', '_kwargs', '_value', '_is_cache_enabled', '_attribute_error']

    def __init__(self, func, *args, **kwargs):
        is_cache_enabled = kwargs.pop('enable_cache', True)
        # Avoid triggering our own __setattr__ implementation
        object.__setattr__(self, '_func', func)
        object.__setattr__(self, '_args', args)
        object.__setattr__(self, '_kwargs', kwargs)
        object.__setattr__(self, '_is_cache_enabled', is_cache_enabled)
        object.__setattr__(self, '_value', None)
        object.__setattr__(self, '_attribute_error', None)

    @property
    def value(self):
        if self._value is None:
            try:
                value = self._func(*self._args, **self._kwargs)
            except AttributeError as error:
                object.__setattr__(self, '_attribute_error', error)
                raise

            if not self._is_cache_enabled:
                return value
            object.__setattr__(self, '_value', value)
        return self._value

    def __contains__(self, key):
        return key in self.value

    def __nonzero__(self):
        return bool(self.value)

    def __dir__(self):
        return dir(self.value)

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)

    def __str__(self):
        return str(self.value)

    def __unicode__(self):
        return unicode(self.value)

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __mul__(self, other):
        return self.value * other

    def __rmul__(self, other):
        return other * self.value

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __delattr__(self, name):
        delattr(self.value, name)

    def __getattr__(self, name):
        if self._attribute_error is not None:
            raise self._attribute_error
        return getattr(self.value, name)

    def __setattr__(self, name, value):
        setattr(self.value, name, value)

    def __delitem__(self, key):
        del self.value[key]

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value

    def __copy__(self):
        return LazyProxy(
            self._func,
            enable_cache=self._is_cache_enabled,
            *self._args,
            **self._kwargs
        )

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return LazyProxy(
            deepcopy(self._func, memo),
            enable_cache=deepcopy(self._is_cache_enabled, memo),
            *deepcopy(self._args, memo),
            **deepcopy(self._kwargs, memo)
        )


class NullTranslations(gettext.NullTranslations, object):

    DEFAULT_DOMAIN = None

    def __init__(self, fp=None):
        """Initialize a simple translations class which is not backed by a
        real catalog. Behaves similar to gettext.NullTranslations but also
        offers Babel's on *gettext methods (e.g. 'dgettext()').

        :param fp: a file-like object (ignored in this class)
        """
        # These attributes are set by gettext.NullTranslations when a catalog
        # is parsed (fp != None). Ensure that they are always present because
        # some *gettext methods (including '.gettext()') rely on the attributes.
        self._catalog = {}
        self.plural = lambda n: int(n != 1)
        super(NullTranslations, self).__init__(fp=fp)
        self.files = list(filter(None, [getattr(fp, 'name', None)]))
        self.domain = self.DEFAULT_DOMAIN
        self._domains = {}

    def dgettext(self, domain, message):
        """Like ``gettext()``, but look the message up in the specified
        domain.
        """
        return self._domains.get(domain, self).gettext(message)

    def ldgettext(self, domain, message):
        """Like ``lgettext()``, but look the message up in the specified
        domain.
        """
        return self._domains.get(domain, self).lgettext(message)

    def udgettext(self, domain, message):
        """Like ``ugettext()``, but look the message up in the specified
        domain.
        """
        return self._domains.get(domain, self).ugettext(message)
    # backward compatibility with 0.9
    dugettext = udgettext

    def dngettext(self, domain, singular, plural, num):
        """Like ``ngettext()``, but look the message up in the specified
        domain.
        """
        return self._domains.get(domain, self).ngettext(singular, plural, num)

    def ldngettext(self, domain, singular, plural, num):
        """Like ``lngettext()``, but look the message up in the specified
        domain.
        """
        return self._domains.get(domain, self).lngettext(singular, plural, num)

    def udngettext(self, domain, singular, plural, num):
        """Like ``ungettext()`` but look the message up in the specified
        domain.
        """
        return self._domains.get(domain, self).ungettext(singular, plural, num)
    # backward compatibility with 0.9
    dungettext = udngettext

    # Most of the downwards code, until it get's included in stdlib, from:
    #    https://bugs.python.org/file10036/gettext-pgettext.patch
    #
    # The encoding of a msgctxt and a msgid in a .mo file is
    # msgctxt + "\x04" + msgid (gettext version >= 0.15)
    CONTEXT_ENCODING = '%s\x04%s'

    def pgettext(self, context, message):
        """Look up the `context` and `message` id in the catalog and return the
        corresponding message string, as an 8-bit string encoded with the
        catalog's charset encoding, if known.  If there is no entry in the
        catalog for the `message` id and `context` , and a fallback has been
        set, the look up is forwarded to the fallback's ``pgettext()``
        method. Otherwise, the `message` id is returned.
        """
        ctxt_msg_id = self.CONTEXT_ENCODING % (context, message)
        missing = object()
        tmsg = self._catalog.get(ctxt_msg_id, missing)
        if tmsg is missing:
            if self._fallback:
                return self._fallback.pgettext(context, message)
            return message
        # Encode the Unicode tmsg back to an 8-bit string, if possible
        if self._output_charset:
            return text_to_native(tmsg, self._output_charset)
        elif self._charset:
            return text_to_native(tmsg, self._charset)
        return tmsg

    def lpgettext(self, context, message):
        """Equivalent to ``pgettext()``, but the translation is returned in the
        preferred system encoding, if no other encoding was explicitly set with
        ``bind_textdomain_codeset()``.
        """
        ctxt_msg_id = self.CONTEXT_ENCODING % (context, message)
        missing = object()
        tmsg = self._catalog.get(ctxt_msg_id, missing)
        if tmsg is missing:
            if self._fallback:
                return self._fallback.lpgettext(context, message)
            return message
        if self._output_charset:
            return tmsg.encode(self._output_charset)
        return tmsg.encode(locale.getpreferredencoding())

    def npgettext(self, context, singular, plural, num):
        """Do a plural-forms lookup of a message id.  `singular` is used as the
        message id for purposes of lookup in the catalog, while `num` is used to
        determine which plural form to use.  The returned message string is an
        8-bit string encoded with the catalog's charset encoding, if known.

        If the message id for `context` is not found in the catalog, and a
        fallback is specified, the request is forwarded to the fallback's
        ``npgettext()`` method.  Otherwise, when ``num`` is 1 ``singular`` is
        returned, and ``plural`` is returned in all other cases.
        """
        ctxt_msg_id = self.CONTEXT_ENCODING % (context, singular)
        try:
            tmsg = self._catalog[(ctxt_msg_id, self.plural(num))]
            if self._output_charset:
                return text_to_native(tmsg, self._output_charset)
            elif self._charset:
                return text_to_native(tmsg, self._charset)
            return tmsg
        except KeyError:
            if self._fallback:
                return self._fallback.npgettext(context, singular, plural, num)
            if num == 1:
                return singular
            else:
                return plural

    def lnpgettext(self, context, singular, plural, num):
        """Equivalent to ``npgettext()``, but the translation is returned in the
        preferred system encoding, if no other encoding was explicitly set with
        ``bind_textdomain_codeset()``.
        """
        ctxt_msg_id = self.CONTEXT_ENCODING % (context, singular)
        try:
            tmsg = self._catalog[(ctxt_msg_id, self.plural(num))]
            if self._output_charset:
                return tmsg.encode(self._output_charset)
            return tmsg.encode(locale.getpreferredencoding())
        except KeyError:
            if self._fallback:
                return self._fallback.lnpgettext(context, singular, plural, num)
            if num == 1:
                return singular
            else:
                return plural

    def upgettext(self, context, message):
        """Look up the `context` and `message` id in the catalog and return the
        corresponding message string, as a Unicode string.  If there is no entry
        in the catalog for the `message` id and `context`, and a fallback has
        been set, the look up is forwarded to the fallback's ``upgettext()``
        method.  Otherwise, the `message` id is returned.
        """
        ctxt_message_id = self.CONTEXT_ENCODING % (context, message)
        missing = object()
        tmsg = self._catalog.get(ctxt_message_id, missing)
        if tmsg is missing:
            if self._fallback:
                return self._fallback.upgettext(context, message)
            return text_type(message)
        return tmsg

    def unpgettext(self, context, singular, plural, num):
        """Do a plural-forms lookup of a message id.  `singular` is used as the
        message id for purposes of lookup in the catalog, while `num` is used to
        determine which plural form to use.  The returned message string is a
        Unicode string.

        If the message id for `context` is not found in the catalog, and a
        fallback is specified, the request is forwarded to the fallback's
        ``unpgettext()`` method.  Otherwise, when `num` is 1 `singular` is
        returned, and `plural` is returned in all other cases.
        """
        ctxt_message_id = self.CONTEXT_ENCODING % (context, singular)
        try:
            tmsg = self._catalog[(ctxt_message_id, self.plural(num))]
        except KeyError:
            if self._fallback:
                return self._fallback.unpgettext(context, singular, plural, num)
            if num == 1:
                tmsg = text_type(singular)
            else:
                tmsg = text_type(plural)
        return tmsg

    def dpgettext(self, domain, context, message):
        """Like `pgettext()`, but look the message up in the specified
        `domain`.
        """
        return self._domains.get(domain, self).pgettext(context, message)

    def udpgettext(self, domain, context, message):
        """Like `upgettext()`, but look the message up in the specified
        `domain`.
        """
        return self._domains.get(domain, self).upgettext(context, message)
    # backward compatibility with 0.9
    dupgettext = udpgettext

    def ldpgettext(self, domain, context, message):
        """Equivalent to ``dpgettext()``, but the translation is returned in the
        preferred system encoding, if no other encoding was explicitly set with
        ``bind_textdomain_codeset()``.
        """
        return self._domains.get(domain, self).lpgettext(context, message)

    def dnpgettext(self, domain, context, singular, plural, num):
        """Like ``npgettext``, but look the message up in the specified
        `domain`.
        """
        return self._domains.get(domain, self).npgettext(context, singular,
                                                         plural, num)

    def udnpgettext(self, domain, context, singular, plural, num):
        """Like ``unpgettext``, but look the message up in the specified
        `domain`.
        """
        return self._domains.get(domain, self).unpgettext(context, singular,
                                                          plural, num)
    # backward compatibility with 0.9
    dunpgettext = udnpgettext

    def ldnpgettext(self, domain, context, singular, plural, num):
        """Equivalent to ``dnpgettext()``, but the translation is returned in
        the preferred system encoding, if no other encoding was explicitly set
        with ``bind_textdomain_codeset()``.
        """
        return self._domains.get(domain, self).lnpgettext(context, singular,
                                                          plural, num)

    if not PY2:
        ugettext = gettext.NullTranslations.gettext
        ungettext = gettext.NullTranslations.ngettext


class Translations(NullTranslations, gettext.GNUTranslations):
    """An extended translation catalog class."""

    DEFAULT_DOMAIN = 'messages'

    def __init__(self, fp=None, domain=None):
        """Initialize the translations catalog.

        :param fp: the file-like object the translation should be read from
        :param domain: the message domain (default: 'messages')
        """
        super(Translations, self).__init__(fp=fp)
        self.domain = domain or self.DEFAULT_DOMAIN

    if not PY2:
        ugettext = gettext.GNUTranslations.gettext
        ungettext = gettext.GNUTranslations.ngettext

    @classmethod
    def load(cls, dirname=None, locales=None, domain=None):
        """Load translations from the given directory.

        :param dirname: the directory containing the ``MO`` files
        :param locales: the list of locales in order of preference (items in
                        this list can be either `Locale` objects or locale
                        strings)
        :param domain: the message domain (default: 'messages')
        """
        if locales is not None:
            if not isinstance(locales, (list, tuple)):
                locales = [locales]
            locales = [str(locale) for locale in locales]
        if not domain:
            domain = cls.DEFAULT_DOMAIN
        filename = gettext.find(domain, dirname, locales)
        if not filename:
            return NullTranslations()
        with open(filename, 'rb') as fp:
            return cls(fp=fp, domain=domain)

    def __repr__(self):
        return '<%s: "%s">' % (type(self).__name__,
                               self._info.get('project-id-version'))

    def add(self, translations, merge=True):
        """Add the given translations to the catalog.

        If the domain of the translations is different than that of the
        current catalog, they are added as a catalog that is only accessible
        by the various ``d*gettext`` functions.

        :param translations: the `Translations` instance with the messages to
                             add
        :param merge: whether translations for message domains that have
                      already been added should be merged with the existing
                      translations
        """
        domain = getattr(translations, 'domain', self.DEFAULT_DOMAIN)
        if merge and domain == self.domain:
            return self.merge(translations)

        existing = self._domains.get(domain)
        if merge and existing is not None:
            existing.merge(translations)
        else:
            translations.add_fallback(self)
            self._domains[domain] = translations

        return self

    def merge(self, translations):
        """Merge the given translations into the catalog.

        Message translations in the specified catalog override any messages
        with the same identifier in the existing catalog.

        :param translations: the `Translations` instance with the messages to
                             merge
        """
        if isinstance(translations, gettext.GNUTranslations):
            self._catalog.update(translations._catalog)
            if isinstance(translations, Translations):
                self.files.extend(translations.files)

        return self
