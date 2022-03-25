from typing import Any
from typing import Generic
from typing import Type
from typing import TypeVar

import attr

from _pytest.compat import final


class PytestWarning(UserWarning):
    """Base class for all warnings emitted by pytest."""

    __module__ = "pytest"


@final
class PytestAssertRewriteWarning(PytestWarning):
    """Warning emitted by the pytest assert rewrite module."""

    __module__ = "pytest"


@final
class PytestCacheWarning(PytestWarning):
    """Warning emitted by the cache plugin in various situations."""

    __module__ = "pytest"


@final
class PytestConfigWarning(PytestWarning):
    """Warning emitted for configuration issues."""

    __module__ = "pytest"


@final
class PytestCollectionWarning(PytestWarning):
    """Warning emitted when pytest is not able to collect a file or symbol in a module."""

    __module__ = "pytest"


class PytestDeprecationWarning(PytestWarning, DeprecationWarning):
    """Warning class for features that will be removed in a future version."""

    __module__ = "pytest"


@final
class PytestRemovedIn8Warning(PytestDeprecationWarning):
    """Warning class for features that will be removed in pytest 8."""

    __module__ = "pytest"


@final
class PytestExperimentalApiWarning(PytestWarning, FutureWarning):
    """Warning category used to denote experiments in pytest.

    Use sparingly as the API might change or even be removed completely in a
    future version.
    """

    __module__ = "pytest"

    @classmethod
    def simple(cls, apiname: str) -> "PytestExperimentalApiWarning":
        return cls(
            "{apiname} is an experimental api that may change over time".format(
                apiname=apiname
            )
        )


@final
class PytestUnhandledCoroutineWarning(PytestWarning):
    """Warning emitted for an unhandled coroutine.

    A coroutine was encountered when collecting test functions, but was not
    handled by any async-aware plugin.
    Coroutine test functions are not natively supported.
    """

    __module__ = "pytest"


@final
class PytestUnknownMarkWarning(PytestWarning):
    """Warning emitted on use of unknown markers.

    See :ref:`mark` for details.
    """

    __module__ = "pytest"


@final
class PytestUnraisableExceptionWarning(PytestWarning):
    """An unraisable exception was reported.

    Unraisable exceptions are exceptions raised in :meth:`__del__ <object.__del__>`
    implementations and similar situations when the exception cannot be raised
    as normal.
    """

    __module__ = "pytest"


@final
class PytestUnhandledThreadExceptionWarning(PytestWarning):
    """An unhandled exception occurred in a :class:`~threading.Thread`.

    Such exceptions don't propagate normally.
    """

    __module__ = "pytest"


_W = TypeVar("_W", bound=PytestWarning)


@final
@attr.s(auto_attribs=True)
class UnformattedWarning(Generic[_W]):
    """A warning meant to be formatted during runtime.

    This is used to hold warnings that need to format their message at runtime,
    as opposed to a direct message.
    """

    category: Type["_W"]
    template: str

    def format(self, **kwargs: Any) -> _W:
        """Return an instance of the warning category, formatted with given kwargs."""
        return self.category(self.template.format(**kwargs))
