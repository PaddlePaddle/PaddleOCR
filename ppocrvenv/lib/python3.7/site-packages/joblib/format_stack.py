from sys import version_info

from warnings import warn

"""
Represent an exception with a lot of information.

Provides 2 useful functions:

format_exc: format an exception into a complete traceback, with full
            debugging instruction.

format_outer_frames: format the current position in the stack call.

Adapted from IPython's VerboseTB.

This module is deprecated and will be removed in joblib 0.16.
"""
from joblib import _deprecated_format_stack

_deprecated_names = [
    name for name in dir(_deprecated_format_stack) if
    not name.startswith("__")  # special attributes
]


if version_info[:2] >= (3, 7):
    def __getattr__(name):
        if not name.startswith("__") and name in _deprecated_names:
            warn("{} is deprecated and will be removed from joblib "
                 "in 0.16".format(name), DeprecationWarning)
            return getattr(_deprecated_format_stack, name)
        raise AttributeError
else:
    for name in _deprecated_names:
        globals()[name] = getattr(_deprecated_format_stack, name)
