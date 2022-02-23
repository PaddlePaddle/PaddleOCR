# -*- coding: utf-8 -*-
"""
This module defines the "ns" enum for natsort is used to determine
what algorithm natsort uses.
"""

import enum
import itertools
import typing


_counter = itertools.count(0)


class ns(enum.IntEnum):  # noqa: N801
    """
    Enum to control the `natsort` algorithm.

    This class acts like an enum to control the `natsort` algorithm. The
    user may select several options simultaneously by or'ing the options
    together.  For example, to choose ``ns.INT``, ``ns.PATH``, and
    ``ns.LOCALE``, you could do ``ns.INT | ns.LOCALE | ns.PATH``. Each
    function in the :mod:`natsort` package has an `alg` option that accepts
    this enum to allow fine control over how your input is sorted.

    Each option has a shortened 1- or 2-letter form.

    .. note:: Please read :ref:`locale_issues` before using ``ns.LOCALE``.

    Attributes
    ----------
    INT, I (default)
        The default - parse numbers as integers.
    FLOAT, F
        Tell `natsort` to parse numbers as floats.
    UNSIGNED, U (default)
        Tell `natsort` to ignore any sign (i.e. "-" or "+") to the immediate
        left of a number.  This is the default.
    SIGNED, S
        Tell `natsort` to take into account any sign (i.e. "-" or "+")
        to the immediate left of a number.
    REAL, R
        This is a shortcut for ``ns.FLOAT | ns.SIGNED``, which is useful
        when attempting to sort real numbers.
    NOEXP, N
        Tell `natsort` to not search for exponents as part of a float number.
        For example, with `NOEXP` the number "5.6E5" would be interpreted
        as `5.6`, `"E"`, and `5` instead of `560000`.
    NUMAFTER, NA
        Tell `natsort` to sort numbers after non-numbers. By default
        numbers will be ordered before non-numbers.
    PATH, P
        Tell `natsort` to interpret strings as filesystem paths, so they
        will be split according to the filesystem separator
        (i.e. '/' on UNIX, '\\' on Windows), as well as splitting on the
        file extension, if any. Without this, lists of file paths like
        ``['Folder/', 'Folder (1)/', 'Folder (10)/']`` will not be
        sorted properly; 'Folder/' will be placed at the end, not at the
        front. It is the same as setting the old `as_path` option to
        `True`.
    COMPATIBILITYNORMALIZE, CN
        Use the "NFKD" unicode normalization form on input rather than the
        default "NFD". This will transform characters such as '⑦' into
        '7'. Please see https://stackoverflow.com/a/7934397/1399279,
        https://stackoverflow.com/a/7931547/1399279,
        and https://unicode.org/reports/tr15/ for full details into unicode
        normalization.
    LOCALE, L
        Tell `natsort` to be locale-aware when sorting. This includes both
        proper sorting of alphabetical characters as well as proper
        handling of locale-dependent decimal separators and thousands
        separators. This is a shortcut for
        ``ns.LOCALEALPHA | ns.LOCALENUM``.
        Your sorting results will vary depending on your current locale.
    LOCALEALPHA, LA
        Tell `natsort` to be locale-aware when sorting, but only for
        alphabetical characters.
    LOCALENUM, LN
        Tell `natsort` to be locale-aware when sorting, but only for
        decimal separators and thousands separators.
    IGNORECASE, IC
        Tell `natsort` to ignore case when sorting.  For example,
        ``['Banana', 'apple', 'banana', 'Apple']`` would be sorted as
        ``['apple', 'Apple', 'Banana', 'banana']``.
    LOWERCASEFIRST, LF
        Tell `natsort` to put lowercase letters before uppercase letters
        when sorting.  For example,
        ``['Banana', 'apple', 'banana', 'Apple']`` would be sorted as
        ``['apple', 'banana', 'Apple', 'Banana']`` (the default order
        would be ``['Apple', 'Banana', 'apple', 'banana']`` which is
        the order from a purely ordinal sort).
        Useless when used with `IGNORECASE`. Please note that if used
        with ``LOCALE``, this actually has the reverse effect and will
        put uppercase first (this is because ``LOCALE`` already puts
        lowercase first); you may use this to your advantage if you
        need to modify the order returned with ``LOCALE``.
    GROUPLETTERS, G
        Tell `natsort` to group lowercase and uppercase letters together
        when sorting.  For example,
        ``['Banana', 'apple', 'banana', 'Apple']`` would be sorted as
        ``['Apple', 'apple', 'Banana', 'banana']``.
        Useless when used with `IGNORECASE`; use with `LOWERCASEFIRST`
        to reverse the order of upper and lower case. Generally not
        needed with `LOCALE`.
    CAPITALFIRST, C
        Only used when `LOCALE` is enabled. Tell `natsort` to put all
        capitalized words before non-capitalized words. This is essentially
        the inverse of `GROUPLETTERS`, and is the default Python sorting
        behavior without `LOCALE`.
    UNGROUPLETTERS, UG
        An alias for `CAPITALFIRST`.
    NANLAST, NL
        If an NaN shows up in the input, this instructs `natsort` to
        treat these as +Infinity and place them after all the other numbers.
        By default, an NaN be treated as -Infinity and be placed first.
        Note that this ``None`` is treated like NaN internally.

    Notes
    -----
    If you prefer to use `import natsort as ns` as opposed to
    `from natsort import natsorted, ns`, the `ns` options are
    available as top-level imports.

        >>> import natsort as ns
        >>> a = ['num5.10', 'num-3', 'num5.3', 'num2']
        >>> ns.natsorted(a, alg=ns.REAL) == ns.natsorted(a, alg=ns.ns.REAL)
        True

    """

    # The below are the base ns options. The values will be stored as powers
    # of two so bitmasks can be used to extract the user's requested options.
    FLOAT = F = 1 << next(_counter)
    SIGNED = S = 1 << next(_counter)
    NOEXP = N = 1 << next(_counter)
    PATH = P = 1 << next(_counter)
    LOCALEALPHA = LA = 1 << next(_counter)
    LOCALENUM = LN = 1 << next(_counter)
    IGNORECASE = IC = 1 << next(_counter)
    LOWERCASEFIRST = LF = 1 << next(_counter)
    GROUPLETTERS = G = 1 << next(_counter)
    UNGROUPLETTERS = CAPITALFIRST = C = UG = 1 << next(_counter)
    NANLAST = NL = 1 << next(_counter)
    COMPATIBILITYNORMALIZE = CN = 1 << next(_counter)
    NUMAFTER = NA = 1 << next(_counter)

    # Following were previously options but are now defaults.
    DEFAULT = 0
    INT = I = 0  # noqa: E741
    UNSIGNED = U = 0

    # The following are bitwise-OR combinations of other fields.
    REAL = R = FLOAT | SIGNED
    LOCALE = L = LOCALEALPHA | LOCALENUM


# The below is private for internal use only.
NS_DUMB = 1 << 31

# An integer can be used in place of the ns enum so make the
# type to use for this enum a union of it and an inteter.
NSType = typing.Union[ns, int]
