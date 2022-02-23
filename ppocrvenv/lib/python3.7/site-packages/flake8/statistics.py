"""Statistic collection logic for Flake8."""
import collections
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flake8.style_guide import Violation


class Statistics:
    """Manager of aggregated statistics for a run of Flake8."""

    def __init__(self) -> None:
        """Initialize the underlying dictionary for our statistics."""
        self._store: Dict[Key, "Statistic"] = {}

    def error_codes(self) -> List[str]:
        """Return all unique error codes stored.

        :returns:
            Sorted list of error codes.
        :rtype:
            list(str)
        """
        return sorted({key.code for key in self._store})

    def record(self, error: "Violation") -> None:
        """Add the fact that the error was seen in the file.

        :param error:
            The Violation instance containing the information about the
            violation.
        :type error:
            flake8.style_guide.Violation
        """
        key = Key.create_from(error)
        if key not in self._store:
            self._store[key] = Statistic.create_from(error)
        self._store[key].increment()

    def statistics_for(
        self, prefix: str, filename: Optional[str] = None
    ) -> Generator["Statistic", None, None]:
        """Generate statistics for the prefix and filename.

        If you have a :class:`Statistics` object that has recorded errors,
        you can generate the statistics for a prefix (e.g., ``E``, ``E1``,
        ``W50``, ``W503``) with the optional filter of a filename as well.

        .. code-block:: python

            >>> stats = Statistics()
            >>> stats.statistics_for('E12',
                                     filename='src/flake8/statistics.py')
            <generator ...>
            >>> stats.statistics_for('W')
            <generator ...>

        :param str prefix:
            The error class or specific error code to find statistics for.
        :param str filename:
            (Optional) The filename to further filter results by.
        :returns:
            Generator of instances of :class:`Statistic`
        """
        matching_errors = sorted(
            key for key in self._store if key.matches(prefix, filename)
        )
        for error_code in matching_errors:
            yield self._store[error_code]


class Key(collections.namedtuple("Key", ["filename", "code"])):
    """Simple key structure for the Statistics dictionary.

    To make things clearer, easier to read, and more understandable, we use a
    namedtuple here for all Keys in the underlying dictionary for the
    Statistics object.
    """

    __slots__ = ()

    @classmethod
    def create_from(cls, error: "Violation") -> "Key":
        """Create a Key from :class:`flake8.style_guide.Violation`."""
        return cls(filename=error.filename, code=error.code)

    def matches(self, prefix: str, filename: Optional[str]) -> bool:
        """Determine if this key matches some constraints.

        :param str prefix:
            The error code prefix that this key's error code should start with.
        :param str filename:
            The filename that we potentially want to match on. This can be
            None to only match on error prefix.
        :returns:
            True if the Key's code starts with the prefix and either filename
            is None, or the Key's filename matches the value passed in.
        :rtype:
            bool
        """
        return self.code.startswith(prefix) and (
            filename is None or self.filename == filename
        )


class Statistic:
    """Simple wrapper around the logic of each statistic.

    Instead of maintaining a simple but potentially hard to reason about
    tuple, we create a namedtuple which has attributes and a couple
    convenience methods on it.
    """

    def __init__(
        self, error_code: str, filename: str, message: str, count: int
    ) -> None:
        """Initialize our Statistic."""
        self.error_code = error_code
        self.filename = filename
        self.message = message
        self.count = count

    @classmethod
    def create_from(cls, error: "Violation") -> "Statistic":
        """Create a Statistic from a :class:`flake8.style_guide.Violation`."""
        return cls(
            error_code=error.code,
            filename=error.filename,
            message=error.text,
            count=0,
        )

    def increment(self) -> None:
        """Increment the number of times we've seen this error in this file."""
        self.count += 1
