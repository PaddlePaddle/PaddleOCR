from sys import version_info
from warnings import warn
from . import _deprecated_my_exceptions

"""
Exceptions
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

_deprecated_names = [
    name for name in dir(_deprecated_my_exceptions) if
    not name.startswith("__")
]


if version_info[:2] >= (3, 7):
    def __getattr__(name):
        if not name.startswith("__") and name in _deprecated_names:
            warn("{} is deprecated and will be removed from joblib "
                 "in 0.16".format(name), DeprecationWarning)
            return getattr(_deprecated_my_exceptions, name)
        raise AttributeError
else:
    for name in _deprecated_names:
        globals()[name] = getattr(_deprecated_my_exceptions, name)


class WorkerInterrupt(Exception):
    """ An exception that is not KeyboardInterrupt to allow subprocesses
        to be interrupted.
    """

    pass
