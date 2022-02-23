from __future__ import annotations


class Timeout(TimeoutError):
    """Raised when the lock could not be acquired in *timeout* seconds."""

    def __init__(self, lock_file: str) -> None:
        #: The path of the file lock.
        self.lock_file = lock_file

    def __str__(self) -> str:
        return f"The file lock '{self.lock_file}' could not be acquired."


__all__ = [
    "Timeout",
]
