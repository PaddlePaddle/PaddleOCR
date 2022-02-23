"""Top-level module for Flake8.

This module

- initializes logging for the command-line tool
- tracks the version of the package
- provides a way to configure logging for the command-line tool

.. autofunction:: flake8.configure_logging

"""
import logging
import sys
from typing import Type

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

__version__ = "4.0.1"
__version_info__ = tuple(int(i) for i in __version__.split(".") if i.isdigit())


# There is nothing lower than logging.DEBUG (10) in the logging library,
# but we want an extra level to avoid being too verbose when using -vv.
_EXTRA_VERBOSE = 5
logging.addLevelName(_EXTRA_VERBOSE, "VERBOSE")

_VERBOSITY_TO_LOG_LEVEL = {
    # output more than warnings but not debugging info
    1: logging.INFO,  # INFO is a numerical level of 20
    # output debugging information
    2: logging.DEBUG,  # DEBUG is a numerical level of 10
    # output extra verbose debugging information
    3: _EXTRA_VERBOSE,
}

LOG_FORMAT = (
    "%(name)-25s %(processName)-11s %(relativeCreated)6d "
    "%(levelname)-8s %(message)s"
)


def configure_logging(verbosity, filename=None, logformat=LOG_FORMAT):
    """Configure logging for flake8.

    :param int verbosity:
        How verbose to be in logging information.
    :param str filename:
        Name of the file to append log information to.
        If ``None`` this will log to ``sys.stderr``.
        If the name is "stdout" or "stderr" this will log to the appropriate
        stream.
    """
    if verbosity <= 0:
        return
    if verbosity > 3:
        verbosity = 3

    log_level = _VERBOSITY_TO_LOG_LEVEL[verbosity]

    if not filename or filename in ("stderr", "stdout"):
        fileobj = getattr(sys, filename or "stderr")
        handler_cls: Type[logging.Handler] = logging.StreamHandler
    else:
        fileobj = filename
        handler_cls = logging.FileHandler

    handler = handler_cls(fileobj)
    handler.setFormatter(logging.Formatter(logformat))
    LOG.addHandler(handler)
    LOG.setLevel(log_level)
    LOG.debug(
        "Added a %s logging handler to logger root at %s", filename, __name__
    )
