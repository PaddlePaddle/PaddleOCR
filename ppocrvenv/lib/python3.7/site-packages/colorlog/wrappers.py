"""Wrappers around the logging module."""

import functools
import logging
import typing
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    INFO,
    NOTSET,
    StreamHandler,
    WARN,
    WARNING,
    getLogger,
    root,
)

import colorlog.formatter

__all__ = (
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "FATAL",
    "INFO",
    "NOTSET",
    "WARN",
    "WARNING",
    "StreamHandler",
    "basicConfig",
    "critical",
    "debug",
    "error",
    "exception",
    "getLogger",
    "info",
    "log",
    "root",
    "warning",
)


def basicConfig(
    style: str = "%",
    log_colors: typing.Optional[colorlog.formatter.LogColors] = None,
    reset: bool = True,
    secondary_log_colors: typing.Optional[colorlog.formatter.SecondaryLogColors] = None,
    format: str = "%(log_color)s%(levelname)s%(reset)s:%(name)s:%(message)s",
    datefmt: typing.Optional[str] = None,
    **kwargs
) -> None:
    """Call ``logging.basicConfig`` and override the formatter it creates."""
    logging.basicConfig(**kwargs)
    logging._acquireLock()  # type: ignore
    try:
        handler = logging.root.handlers[0]
        handler.setFormatter(
            colorlog.formatter.ColoredFormatter(
                fmt=format,
                datefmt=datefmt,
                style=style,
                log_colors=log_colors,
                reset=reset,
                secondary_log_colors=secondary_log_colors,
                stream=kwargs.get("stream", None),
            )
        )
    finally:
        logging._releaseLock()  # type: ignore


def ensure_configured(func):
    """Modify a function to call our basicConfig() first if no handlers exist."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(logging.root.handlers) == 0:
            basicConfig()
        return func(*args, **kwargs)

    return wrapper


debug = ensure_configured(logging.debug)
info = ensure_configured(logging.info)
warning = ensure_configured(logging.warning)
error = ensure_configured(logging.error)
critical = ensure_configured(logging.critical)
log = ensure_configured(logging.log)
exception = ensure_configured(logging.exception)
