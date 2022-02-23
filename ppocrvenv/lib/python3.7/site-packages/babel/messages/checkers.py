# -*- coding: utf-8 -*-
"""
    babel.messages.checkers
    ~~~~~~~~~~~~~~~~~~~~~~~

    Various routines that help with validation of translations.

    :since: version 0.9

    :copyright: (c) 2013-2021 by the Babel Team.
    :license: BSD, see LICENSE for more details.
"""

from babel.messages.catalog import TranslationError, PYTHON_FORMAT
from babel._compat import string_types, izip


#: list of format chars that are compatible to each other
_string_format_compatibilities = [
    {'i', 'd', 'u'},
    {'x', 'X'},
    {'f', 'F', 'g', 'G'}
]


def num_plurals(catalog, message):
    """Verify the number of plurals in the translation."""
    if not message.pluralizable:
        if not isinstance(message.string, string_types):
            raise TranslationError("Found plural forms for non-pluralizable "
                                   "message")
        return

    # skip further tests if no catalog is provided.
    elif catalog is None:
        return

    msgstrs = message.string
    if not isinstance(msgstrs, (list, tuple)):
        msgstrs = (msgstrs,)
    if len(msgstrs) != catalog.num_plurals:
        raise TranslationError("Wrong number of plural forms (expected %d)" %
                               catalog.num_plurals)


def python_format(catalog, message):
    """Verify the format string placeholders in the translation."""
    if 'python-format' not in message.flags:
        return
    msgids = message.id
    if not isinstance(msgids, (list, tuple)):
        msgids = (msgids,)
    msgstrs = message.string
    if not isinstance(msgstrs, (list, tuple)):
        msgstrs = (msgstrs,)

    for msgid, msgstr in izip(msgids, msgstrs):
        if msgstr:
            _validate_format(msgid, msgstr)


def _validate_format(format, alternative):
    """Test format string `alternative` against `format`.  `format` can be the
    msgid of a message and `alternative` one of the `msgstr`\\s.  The two
    arguments are not interchangeable as `alternative` may contain less
    placeholders if `format` uses named placeholders.

    The behavior of this function is undefined if the string does not use
    string formattings.

    If the string formatting of `alternative` is compatible to `format` the
    function returns `None`, otherwise a `TranslationError` is raised.

    Examples for compatible format strings:

    >>> _validate_format('Hello %s!', 'Hallo %s!')
    >>> _validate_format('Hello %i!', 'Hallo %d!')

    Example for an incompatible format strings:

    >>> _validate_format('Hello %(name)s!', 'Hallo %s!')
    Traceback (most recent call last):
      ...
    TranslationError: the format strings are of different kinds

    This function is used by the `python_format` checker.

    :param format: The original format string
    :param alternative: The alternative format string that should be checked
                        against format
    :raises TranslationError: on formatting errors
    """

    def _parse(string):
        result = []
        for match in PYTHON_FORMAT.finditer(string):
            name, format, typechar = match.groups()
            if typechar == '%' and name is None:
                continue
            result.append((name, str(typechar)))
        return result

    def _compatible(a, b):
        if a == b:
            return True
        for set in _string_format_compatibilities:
            if a in set and b in set:
                return True
        return False

    def _check_positional(results):
        positional = None
        for name, char in results:
            if positional is None:
                positional = name is None
            else:
                if (name is None) != positional:
                    raise TranslationError('format string mixes positional '
                                           'and named placeholders')
        return bool(positional)

    a, b = map(_parse, (format, alternative))

    # now check if both strings are positional or named
    a_positional, b_positional = map(_check_positional, (a, b))
    if a_positional and not b_positional and not b:
        raise TranslationError('placeholders are incompatible')
    elif a_positional != b_positional:
        raise TranslationError('the format strings are of different kinds')

    # if we are operating on positional strings both must have the
    # same number of format chars and those must be compatible
    if a_positional:
        if len(a) != len(b):
            raise TranslationError('positional format placeholders are '
                                   'unbalanced')
        for idx, ((_, first), (_, second)) in enumerate(izip(a, b)):
            if not _compatible(first, second):
                raise TranslationError('incompatible format for placeholder '
                                       '%d: %r and %r are not compatible' %
                                       (idx + 1, first, second))

    # otherwise the second string must not have names the first one
    # doesn't have and the types of those included must be compatible
    else:
        type_map = dict(a)
        for name, typechar in b:
            if name not in type_map:
                raise TranslationError('unknown named placeholder %r' % name)
            elif not _compatible(typechar, type_map[name]):
                raise TranslationError('incompatible format for '
                                       'placeholder %r: '
                                       '%r and %r are not compatible' %
                                       (name, typechar, type_map[name]))


def _find_checkers():
    checkers = []
    try:
        from pkg_resources import working_set
    except ImportError:
        pass
    else:
        for entry_point in working_set.iter_entry_points('babel.checkers'):
            checkers.append(entry_point.load())
    if len(checkers) == 0:
        # if pkg_resources is not available or no usable egg-info was found
        # (see #230), just resort to hard-coded checkers
        return [num_plurals, python_format]
    return checkers


checkers = _find_checkers()
