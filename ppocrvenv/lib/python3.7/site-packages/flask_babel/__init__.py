"""
    flaskext.babel
    ~~~~~~~~~~~~~~

    Implements i18n/l10n support for Flask applications based on Babel.

    :copyright: (c) 2013 by Armin Ronacher, Daniel Neuhäuser.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import absolute_import
import os

from datetime import datetime
from contextlib import contextmanager
from flask import current_app, request
from flask.ctx import has_request_context
from flask.helpers import locked_cached_property
from babel import dates, numbers, support, Locale
from pytz import timezone, UTC
from werkzeug.datastructures import ImmutableDict

from flask_babel.speaklater import LazyString


class Babel(object):
    """Central controller class that can be used to configure how
    Flask-Babel behaves.  Each application that wants to use Flask-Babel
    has to create, or run :meth:`init_app` on, an instance of this class
    after the configuration was initialized.
    """

    default_date_formats = ImmutableDict({
        'time':             'medium',
        'date':             'medium',
        'datetime':         'medium',
        'time.short':       None,
        'time.medium':      None,
        'time.full':        None,
        'time.long':        None,
        'date.short':       None,
        'date.medium':      None,
        'date.full':        None,
        'date.long':        None,
        'datetime.short':   None,
        'datetime.medium':  None,
        'datetime.full':    None,
        'datetime.long':    None,
    })

    def __init__(self, app=None, default_locale='en', default_timezone='UTC',
                 default_domain='messages', date_formats=None,
                 configure_jinja=True):
        self._default_locale = default_locale
        self._default_timezone = default_timezone
        self._default_domain = default_domain
        self._date_formats = date_formats
        self._configure_jinja = configure_jinja
        self.app = app
        self.locale_selector_func = None
        self.timezone_selector_func = None

        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        """Set up this instance for use with *app*, if no app was passed to
        the constructor.
        """
        self.app = app
        app.babel_instance = self
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        app.extensions['babel'] = self

        app.config.setdefault('BABEL_DEFAULT_LOCALE', self._default_locale)
        app.config.setdefault('BABEL_DEFAULT_TIMEZONE', self._default_timezone)
        app.config.setdefault('BABEL_DOMAIN', self._default_domain)
        if self._date_formats is None:
            self._date_formats = self.default_date_formats.copy()

        #: a mapping of Babel datetime format strings that can be modified
        #: to change the defaults.  If you invoke :func:`format_datetime`
        #: and do not provide any format string Flask-Babel will do the
        #: following things:
        #:
        #: 1.   look up ``date_formats['datetime']``.  By default ``'medium'``
        #:      is returned to enforce medium length datetime formats.
        #: 2.   ``date_formats['datetime.medium'] (if ``'medium'`` was
        #:      returned in step one) is looked up.  If the return value
        #:      is anything but `None` this is used as new format string.
        #:      otherwise the default for that language is used.
        self.date_formats = self._date_formats

        if self._configure_jinja:
            app.jinja_env.filters.update(
                datetimeformat=format_datetime,
                dateformat=format_date,
                timeformat=format_time,
                timedeltaformat=format_timedelta,

                numberformat=format_number,
                decimalformat=format_decimal,
                currencyformat=format_currency,
                percentformat=format_percent,
                scientificformat=format_scientific,
            )
            app.jinja_env.add_extension('jinja2.ext.i18n')
            app.jinja_env.install_gettext_callables(
                lambda x: get_translations().ugettext(x),
                lambda s, p, n: get_translations().ungettext(s, p, n),
                newstyle=True
            )

    def localeselector(self, f):
        """Registers a callback function for locale selection.  The default
        behaves as if a function was registered that returns `None` all the
        time.  If `None` is returned, the locale falls back to the one from
        the configuration.

        This has to return the locale as string (eg: ``'de_AT'``, ``'en_US'``)
        """
        self.locale_selector_func = f
        return f

    def timezoneselector(self, f):
        """Registers a callback function for timezone selection.  The default
        behaves as if a function was registered that returns `None` all the
        time.  If `None` is returned, the timezone falls back to the one from
        the configuration.

        This has to return the timezone as string (eg: ``'Europe/Vienna'``)
        """
        self.timezone_selector_func = f
        return f

    def list_translations(self):
        """Returns a list of all the locales translations exist for.  The
        list returned will be filled with actual locale objects and not just
        strings.

        .. versionadded:: 0.6
        """
        result = []

        for dirname in self.translation_directories:
            if not os.path.isdir(dirname):
                continue

            for folder in os.listdir(dirname):
                locale_dir = os.path.join(dirname, folder, 'LC_MESSAGES')
                if not os.path.isdir(locale_dir):
                    continue

                if filter(lambda x: x.endswith('.mo'), os.listdir(locale_dir)):
                    result.append(Locale.parse(folder))

        # If not other translations are found, add the default locale.
        if not result:
            result.append(Locale.parse(self._default_locale))

        return result

    @property
    def default_locale(self):
        """The default locale from the configuration as instance of a
        `babel.Locale` object.
        """
        return Locale.parse(self.app.config['BABEL_DEFAULT_LOCALE'])

    @property
    def default_timezone(self):
        """The default timezone from the configuration as instance of a
        `pytz.timezone` object.
        """
        return timezone(self.app.config['BABEL_DEFAULT_TIMEZONE'])

    @property
    def domain(self):
        """The message domain for the translations as a string.
        """
        return self.app.config['BABEL_DOMAIN']

    @locked_cached_property
    def domain_instance(self):
        """The message domain for the translations.
        """
        return Domain(domain=self.app.config['BABEL_DOMAIN'])

    @property
    def translation_directories(self):
        directories = self.app.config.get(
            'BABEL_TRANSLATION_DIRECTORIES',
            'translations'
        ).split(';')

        for path in directories:
            if os.path.isabs(path):
                yield path
            else:
                yield os.path.join(self.app.root_path, path)


def get_translations():
    """Returns the correct gettext translations that should be used for
    this request.  This will never fail and return a dummy translation
    object if used outside of the request or if a translation cannot be
    found.
    """
    return get_domain().get_translations()


def get_locale():
    """Returns the locale that should be used for this request as
    `babel.Locale` object.  This returns `None` if used outside of
    a request.
    """
    ctx = _get_current_context()
    if ctx is None:
        return None
    locale = getattr(ctx, 'babel_locale', None)
    if locale is None:
        babel = current_app.extensions['babel']
        if babel.locale_selector_func is None:
            locale = babel.default_locale
        else:
            rv = babel.locale_selector_func()
            if rv is None:
                locale = babel.default_locale
            else:
                locale = Locale.parse(rv)
        ctx.babel_locale = locale
    return locale


def get_timezone():
    """Returns the timezone that should be used for this request as
    `pytz.timezone` object.  This returns `None` if used outside of
    a request.
    """
    ctx = _get_current_context()
    tzinfo = getattr(ctx, 'babel_tzinfo', None)
    if tzinfo is None:
        babel = current_app.extensions['babel']
        if babel.timezone_selector_func is None:
            tzinfo = babel.default_timezone
        else:
            rv = babel.timezone_selector_func()
            if rv is None:
                tzinfo = babel.default_timezone
            else:
                tzinfo = timezone(rv) if isinstance(rv, str) else rv
        ctx.babel_tzinfo = tzinfo
    return tzinfo


def refresh():
    """Refreshes the cached timezones and locale information.  This can
    be used to switch a translation between a request and if you want
    the changes to take place immediately, not just with the next request::

        user.timezone = request.form['timezone']
        user.locale = request.form['locale']
        refresh()
        flash(gettext('Language was changed'))

    Without that refresh, the :func:`~flask.flash` function would probably
    return English text and a now German page.
    """
    ctx = _get_current_context()
    for key in 'babel_locale', 'babel_tzinfo', 'babel_translations':
        if hasattr(ctx, key):
            delattr(ctx, key)

    if hasattr(ctx, 'forced_babel_locale'):
        ctx.babel_locale = ctx.forced_babel_locale


@contextmanager
def force_locale(locale):
    """Temporarily overrides the currently selected locale.

    Sometimes it is useful to switch the current locale to different one, do
    some tasks and then revert back to the original one. For example, if the
    user uses German on the web site, but you want to send them an email in
    English, you can use this function as a context manager::

        with force_locale('en_US'):
            send_email(gettext('Hello!'), ...)

    :param locale: The locale to temporary switch to (ex: 'en_US').
    """
    ctx = _get_current_context()
    if ctx is None:
        yield
        return

    orig_attrs = {}
    for key in ('babel_translations', 'babel_locale'):
        orig_attrs[key] = getattr(ctx, key, None)

    try:
        ctx.babel_locale = Locale.parse(locale)
        ctx.forced_babel_locale = ctx.babel_locale
        ctx.babel_translations = None
        yield
    finally:
        if hasattr(ctx, 'forced_babel_locale'):
            del ctx.forced_babel_locale

        for key, value in orig_attrs.items():
            setattr(ctx, key, value)


def _get_format(key, format):
    """A small helper for the datetime formatting functions.  Looks up
    format defaults for different kinds.
    """
    babel = current_app.extensions['babel']
    if format is None:
        format = babel.date_formats[key]
    if format in ('short', 'medium', 'full', 'long'):
        rv = babel.date_formats['%s.%s' % (key, format)]
        if rv is not None:
            format = rv
    return format


def to_user_timezone(datetime):
    """Convert a datetime object to the user's timezone.  This automatically
    happens on all date formatting unless rebasing is disabled.  If you need
    to convert a :class:`datetime.datetime` object at any time to the user's
    timezone (as returned by :func:`get_timezone` this function can be used).
    """
    if datetime.tzinfo is None:
        datetime = datetime.replace(tzinfo=UTC)
    tzinfo = get_timezone()
    return tzinfo.normalize(datetime.astimezone(tzinfo))


def to_utc(datetime):
    """Convert a datetime object to UTC and drop tzinfo.  This is the
    opposite operation to :func:`to_user_timezone`.
    """
    if datetime.tzinfo is None:
        datetime = get_timezone().localize(datetime)
    return datetime.astimezone(UTC).replace(tzinfo=None)


def format_datetime(datetime=None, format=None, rebase=True):
    """Return a date formatted according to the given pattern.  If no
    :class:`~datetime.datetime` object is passed, the current time is
    assumed.  By default rebasing happens which causes the object to
    be converted to the users's timezone (as returned by
    :func:`to_user_timezone`).  This function formats both date and
    time.

    The format parameter can either be ``'short'``, ``'medium'``,
    ``'long'`` or ``'full'`` (in which cause the language's default for
    that setting is used, or the default from the :attr:`Babel.date_formats`
    mapping is used) or a format string as documented by Babel.

    This function is also available in the template context as filter
    named `datetimeformat`.
    """
    format = _get_format('datetime', format)
    return _date_format(dates.format_datetime, datetime, format, rebase)


def format_date(date=None, format=None, rebase=True):
    """Return a date formatted according to the given pattern.  If no
    :class:`~datetime.datetime` or :class:`~datetime.date` object is passed,
    the current time is assumed.  By default rebasing happens which causes
    the object to be converted to the users's timezone (as returned by
    :func:`to_user_timezone`).  This function only formats the date part
    of a :class:`~datetime.datetime` object.

    The format parameter can either be ``'short'``, ``'medium'``,
    ``'long'`` or ``'full'`` (in which cause the language's default for
    that setting is used, or the default from the :attr:`Babel.date_formats`
    mapping is used) or a format string as documented by Babel.

    This function is also available in the template context as filter
    named `dateformat`.
    """
    if rebase and isinstance(date, datetime):
        date = to_user_timezone(date)
    format = _get_format('date', format)
    return _date_format(dates.format_date, date, format, rebase)


def format_time(time=None, format=None, rebase=True):
    """Return a time formatted according to the given pattern.  If no
    :class:`~datetime.datetime` object is passed, the current time is
    assumed.  By default rebasing happens which causes the object to
    be converted to the users's timezone (as returned by
    :func:`to_user_timezone`).  This function formats both date and
    time.

    The format parameter can either be ``'short'``, ``'medium'``,
    ``'long'`` or ``'full'`` (in which cause the language's default for
    that setting is used, or the default from the :attr:`Babel.date_formats`
    mapping is used) or a format string as documented by Babel.

    This function is also available in the template context as filter
    named `timeformat`.
    """
    format = _get_format('time', format)
    return _date_format(dates.format_time, time, format, rebase)


def format_timedelta(datetime_or_timedelta, granularity='second',
                     add_direction=False, threshold=0.85):
    """Format the elapsed time from the given date to now or the given
    timedelta.

    This function is also available in the template context as filter
    named `timedeltaformat`.
    """
    if isinstance(datetime_or_timedelta, datetime):
        datetime_or_timedelta = datetime.utcnow() - datetime_or_timedelta
    return dates.format_timedelta(
        datetime_or_timedelta,
        granularity,
        threshold=threshold,
        add_direction=add_direction,
        locale=get_locale()
    )


def _date_format(formatter, obj, format, rebase, **extra):
    """Internal helper that formats the date."""
    locale = get_locale()
    extra = {}
    if formatter is not dates.format_date and rebase:
        extra['tzinfo'] = get_timezone()
    return formatter(obj, format, locale=locale, **extra)


def format_number(number):
    """Return the given number formatted for the locale in request

    :param number: the number to format
    :return: the formatted number
    :rtype: unicode
    """
    locale = get_locale()
    return numbers.format_decimal(number, locale=locale)


def format_decimal(number, format=None):
    """Return the given decimal number formatted for the locale in request

    :param number: the number to format
    :param format: the format to use
    :return: the formatted number
    :rtype: unicode
    """
    locale = get_locale()
    return numbers.format_decimal(number, format=format, locale=locale)


def format_currency(number, currency, format=None, currency_digits=True,
                    format_type='standard'):
    """Return the given number formatted for the locale in request

    :param number: the number to format
    :param currency: the currency code
    :param format: the format to use
    :param currency_digits: use the currency’s number of decimal digits
                            [default: True]
    :param format_type: the currency format type to use
                        [default: standard]
    :return: the formatted number
    :rtype: unicode
    """
    locale = get_locale()
    return numbers.format_currency(
        number,
        currency,
        format=format,
        locale=locale,
        currency_digits=currency_digits,
        format_type=format_type
    )


def format_percent(number, format=None):
    """Return formatted percent value for the locale in request

    :param number: the number to format
    :param format: the format to use
    :return: the formatted percent number
    :rtype: unicode
    """
    locale = get_locale()
    return numbers.format_percent(number, format=format, locale=locale)


def format_scientific(number, format=None):
    """Return value formatted in scientific notation for the locale in request

    :param number: the number to format
    :param format: the format to use
    :return: the formatted percent number
    :rtype: unicode
    """
    locale = get_locale()
    return numbers.format_scientific(number, format=format, locale=locale)


class Domain(object):
    """Localization domain. By default will use look for tranlations in Flask
    application directory and "messages" domain - all message catalogs should
    be called ``messages.mo``.
    """

    def __init__(self, translation_directories=None, domain='messages'):
        if isinstance(translation_directories, str):
            translation_directories = [translation_directories]
        self._translation_directories = translation_directories
        self.domain = domain
        self.cache = {}

    def __repr__(self):
        return '<Domain({!r}, {!r})>'.format(self._translation_directories, self.domain)

    @property
    def translation_directories(self):
        if self._translation_directories is not None:
            return self._translation_directories
        babel = current_app.extensions['babel']
        return babel.translation_directories

    def as_default(self):
        """Set this domain as default for the current request"""
        ctx = _get_current_context()

        if ctx is None:
            raise RuntimeError("No request context")

        ctx.babel_domain = self

    def get_translations_cache(self, ctx):
        """Returns dictionary-like object for translation caching"""
        return self.cache

    def get_translations(self):
        ctx = _get_current_context()

        if ctx is None:
            return support.NullTranslations()

        cache = self.get_translations_cache(ctx)
        locale = get_locale()
        try:
            return cache[str(locale), self.domain]
        except KeyError:
            translations = support.Translations()

            for dirname in self.translation_directories:
                catalog = support.Translations.load(
                    dirname,
                    [locale],
                    self.domain
                )
                translations.merge(catalog)
                # FIXME: Workaround for merge() being really, really stupid. It
                # does not copy _info, plural(), or any other instance variables
                # populated by GNUTranslations. We probably want to stop using
                # `support.Translations.merge` entirely.
                if hasattr(catalog, 'plural'):
                    translations.plural = catalog.plural

            cache[str(locale), self.domain] = translations
            return translations

    def gettext(self, string, **variables):
        """Translates a string with the current locale and passes in the
        given keyword arguments as mapping to a string formatting string.

        ::

            gettext(u'Hello World!')
            gettext(u'Hello %(name)s!', name='World')
        """
        t = self.get_translations()
        s = t.ugettext(string)
        return s if not variables else s % variables

    def ngettext(self, singular, plural, num, **variables):
        """Translates a string with the current locale and passes in the
        given keyword arguments as mapping to a string formatting string.
        The `num` parameter is used to dispatch between singular and various
        plural forms of the message.  It is available in the format string
        as ``%(num)d`` or ``%(num)s``.  The source language should be
        English or a similar language which only has one plural form.

        ::

            ngettext(u'%(num)d Apple', u'%(num)d Apples', num=len(apples))
        """
        variables.setdefault('num', num)
        t = self.get_translations()
        s = t.ungettext(singular, plural, num)
        return s if not variables else s % variables

    def pgettext(self, context, string, **variables):
        """Like :func:`gettext` but with a context.

        .. versionadded:: 0.7
        """
        t = self.get_translations()
        s = t.upgettext(context, string)
        return s if not variables else s % variables

    def npgettext(self, context, singular, plural, num, **variables):
        """Like :func:`ngettext` but with a context.

        .. versionadded:: 0.7
        """
        variables.setdefault('num', num)
        t = self.get_translations()
        s = t.unpgettext(context, singular, plural, num)
        return s if not variables else s % variables

    def lazy_gettext(self, string, **variables):
        """Like :func:`gettext` but the string returned is lazy which means
        it will be translated when it is used as an actual string.

        Example::

            hello = lazy_gettext(u'Hello World')

            @app.route('/')
            def index():
                return unicode(hello)
        """
        return LazyString(self.gettext, string, **variables)

    def lazy_ngettext(self, singular, plural, num, **variables):
        """Like :func:`ngettext` but the string returned is lazy which means
        it will be translated when it is used as an actual string.

        Example::

            apples = lazy_ngettext(u'%(num)d Apple', u'%(num)d Apples', num=len(apples))

            @app.route('/')
            def index():
                return unicode(apples)
        """
        return LazyString(self.ngettext, singular, plural, num, **variables)

    def lazy_pgettext(self, context, string, **variables):
        """Like :func:`pgettext` but the string returned is lazy which means
        it will be translated when it is used as an actual string.

        .. versionadded:: 0.7
        """
        return LazyString(self.pgettext, context, string, **variables)


def _get_current_context():
    if has_request_context():
        return request

    if current_app:
        return current_app


def get_domain():
    ctx = _get_current_context()
    if ctx is None:
        # this will use NullTranslations
        return Domain()

    try:
        return ctx.babel_domain
    except AttributeError:
        pass

    babel = current_app.extensions['babel']
    ctx.babel_domain = babel.domain_instance
    return ctx.babel_domain


# Create shortcuts for the default Flask domain
def gettext(*args, **kwargs):
    return get_domain().gettext(*args, **kwargs)
_ = gettext


def ngettext(*args, **kwargs):
    return get_domain().ngettext(*args, **kwargs)


def pgettext(*args, **kwargs):
    return get_domain().pgettext(*args, **kwargs)


def npgettext(*args, **kwargs):
    return get_domain().npgettext(*args, **kwargs)


def lazy_gettext(*args, **kwargs):
    return LazyString(gettext, *args, **kwargs)


def lazy_pgettext(*args, **kwargs):
    return LazyString(pgettext, *args, **kwargs)


def lazy_ngettext(*args, **kwargs):
    return LazyString(ngettext, *args, **kwargs)
