"""
Helpers for configuring locale settings.

Name `localization` is chosen to avoid overlap with builtin `locale` module.
"""
from contextlib import contextmanager
import locale
import re
import subprocess

from pandas._config.config import options


@contextmanager
def set_locale(new_locale, lc_var: int = locale.LC_ALL):
    """
    Context manager for temporarily setting a locale.

    Parameters
    ----------
    new_locale : str or tuple
        A string of the form <language_country>.<encoding>. For example to set
        the current locale to US English with a UTF8 encoding, you would pass
        "en_US.UTF-8".
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Notes
    -----
    This is useful when you want to run a particular block of code under a
    particular locale, without globally setting the locale. This probably isn't
    thread-safe.
    """
    current_locale = locale.getlocale()

    try:
        locale.setlocale(lc_var, new_locale)
        normalized_locale = locale.getlocale()
        if all(x is not None for x in normalized_locale):
            yield ".".join(normalized_locale)
        else:
            yield new_locale
    finally:
        locale.setlocale(lc_var, current_locale)


def can_set_locale(lc: str, lc_var: int = locale.LC_ALL) -> bool:
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    bool
        Whether the passed locale can be set
    """
    try:
        with set_locale(lc, lc_var=lc_var):
            pass
    except (ValueError, locale.Error):
        # horrible name for a Exception subclass
        return False
    else:
        return True


def _valid_locales(locales, normalize):
    """
    Return a list of normalized locales that do not throw an ``Exception``
    when set.

    Parameters
    ----------
    locales : str
        A string where each locale is separated by a newline.
    normalize : bool
        Whether to call ``locale.normalize`` on each locale.

    Returns
    -------
    valid_locales : list
        A list of valid locales.
    """
    return [
        loc
        for loc in (
            locale.normalize(loc.strip()) if normalize else loc.strip()
            for loc in locales
        )
        if can_set_locale(loc)
    ]


def _default_locale_getter():
    return subprocess.check_output(["locale -a"], shell=True)


def get_locales(prefix=None, normalize=True, locale_getter=_default_locale_getter):
    """
    Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.
    locale_getter : callable
        The function to use to retrieve the current locales. This should return
        a string with each locale separated by a newline character.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return None (no locale available, e.g. Windows)

    """
    try:
        raw_locales = locale_getter()
    except subprocess.CalledProcessError:
        # Raised on (some? all?) Windows platforms because Note: "locale -a"
        #  is not defined
        return None

    try:
        # raw_locales is "\n" separated list of locales
        # it may contain non-decodable parts, so split
        # extract what we can and then rejoin.
        raw_locales = raw_locales.split(b"\n")
        out_locales = []
        for x in raw_locales:
            try:
                out_locales.append(str(x, encoding=options.display.encoding))
            except UnicodeError:
                # 'locale -a' is used to populated 'raw_locales' and on
                # Redhat 7 Linux (and maybe others) prints locale names
                # using windows-1252 encoding.  Bug only triggered by
                # a few special characters and when there is an
                # extensive list of installed locales.
                out_locales.append(str(x, encoding="windows-1252"))

    except TypeError:
        pass

    if prefix is None:
        return _valid_locales(out_locales, normalize)

    pattern = re.compile(f"{prefix}.*")
    found = pattern.findall("\n".join(out_locales))
    return _valid_locales(found, normalize)
