""":mod:`wand.assertions` --- Input assertion helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module checks user input before calling MagickWands C-API methods.


.. versionadded:: 0.5.4
"""

import numbers
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from .compat import string_type


def assert_bool(**kwargs):
    """Ensure all given values are boolean.

    :raises TypeError: if value is not ``True`` or ``False.

    .. versionadded:: 0.5.4
    """
    for label, subject in kwargs.items():
        if not isinstance(subject, bool):
            fmt = "{0} must be a bool, not {1}"
            msg = fmt.format(label, repr(subject))
            raise TypeError(msg)


def assert_color(**kwargs):
    """Ensure all given values are instances of :class:`~wand.color.Color`.

    :raises TypeError: if value is not :class:`~wand.color.Color`.

    .. versionadded:: 0.5.4
    """
    for label, subject in kwargs.items():
        if not isinstance(subject, Color):
            fmt = "Expecting an instance of wand.color.Color for {0}, not {1}"
            msg = fmt.format(label, repr(subject))
            raise TypeError(msg)


def assert_counting_number(**kwargs):
    """Ensure all given values are natural integer.

    :raises TypeError: if value is not an integer.
    :raises ValueError: if value is less than ``1``.

    .. versionadded:: 0.5.4
    """
    assert_integer(**kwargs)
    for label, subject in kwargs.items():
        if subject < 1:
            fmt = "{0}={1} must be an natural number greater than 0"
            msg = fmt.format(label, subject)
            raise ValueError(msg)


def assert_integer(**kwargs):
    """Ensure all given values are an integer.

    :raises TypeError: if value is not an integer.

    .. versionadded:: 0.5.4
    """
    for label, subject in kwargs.items():
        if not isinstance(subject, numbers.Integral):
            fmt = "{0} must be an integer, not {1}"
            msg = fmt.format(label, repr(subject))
            raise TypeError(msg)


def assert_real(**kwargs):
    """Ensure all given values are real numbers.

    :raises TypeError: if value is not a real number.

    .. versionadded:: 0.5.4
    """
    for label, subject in kwargs.items():
        if not isinstance(subject, numbers.Real):
            fmt = "{0} must be a real number, not {1}"
            msg = fmt.format(label, repr(subject))
            raise TypeError(msg)


def assert_unsigned_integer(**kwargs):
    """Ensure all given values are positive integer.

    :raises TypeError: if value is not an integer.
    :raises ValueError: if value is less than ``0``.

    .. versionadded:: 0.5.4
    """
    assert_integer(**kwargs)
    for label, subject in kwargs.items():
        if subject < 0:
            fmt = "{0}={1} must be a positive integer"
            msg = fmt.format(label, subject)
            raise ValueError(msg)


def assert_coordinate(**kwargs):
    """Ensure all given values are a sequence of 2 real numbers.

    :raises TypeError: if value is not a pair of doubles.

    .. versionadded:: 0.6.0
    """
    for label, subject in kwargs.items():
        if not isinstance(subject, Sequence):
            fmt = "'{0}' must be a pair of real numbers, not {1}"
            msg = fmt.format(label, repr(subject))
            raise TypeError(msg)
        elif len(subject) != 2:
            fmt = "'{0}' must be a exactly 2 real numbers, not {1}"
            msg = fmt.format(label, len(subject))
            raise ValueError(msg)
        elif not isinstance(subject[0], numbers.Real):
            fmt = "first entry of '{0}' must be a real number, not {1}"
            msg = fmt.format(label, repr(subject[0]))
            raise TypeError(msg)
        elif not isinstance(subject[1], numbers.Real):
            fmt = "second entry of '{0}' must be a real number, not {1}"
            msg = fmt.format(label, repr(subject[1]))
            raise TypeError(msg)


def assert_string(**kwargs):
    for label, subject in kwargs.items():
        if not isinstance(subject, string_type):
            fmt = "{0} must be a string, not {1}"
            msg = fmt.format(label, repr(subject))
            raise TypeError(msg)


def in_list(options, label, **kwargs):
    for subject_label, subject in kwargs.items():
        if subject not in options:
            fmt = "{0} must be defined in {1}, not {2}"
            msg = fmt.format(subject_label, label, repr(subject))
            raise ValueError(msg)


def string_in_list(options, label, **kwargs):
    assert_string(**kwargs)
    in_list(options, label, **kwargs)


# Lazy load recursive import
from .color import Color  # noqa: E402
