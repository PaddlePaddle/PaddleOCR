from __future__ import annotations

import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from threading import Lock
from types import TracebackType
from typing import Any

from ._error import Timeout

_LOGGER = logging.getLogger("filelock")


# This is a helper class which is returned by :meth:`BaseFileLock.acquire` and wraps the lock to make sure __enter__
# is not called twice when entering the with statement. If we would simply return *self*, the lock would be acquired
# again in the *__enter__* method of the BaseFileLock, but not released again automatically. issue #37 (memory leak)
class AcquireReturnProxy:
    """A context aware object that will release the lock file when exiting."""

    def __init__(self, lock: BaseFileLock) -> None:
        self.lock = lock

    def __enter__(self) -> BaseFileLock:
        return self.lock

    def __exit__(
        self,
        exc_type: type[BaseException] | None,  # noqa: U100
        exc_value: BaseException | None,  # noqa: U100
        traceback: TracebackType | None,  # noqa: U100
    ) -> None:
        self.lock.release()


class BaseFileLock(ABC):
    """Abstract base class for a file lock object."""

    def __init__(self, lock_file: str | os.PathLike[Any], timeout: float = -1) -> None:
        """
        Create a new lock object.

        :param lock_file: path to the file
        :param timeout: default timeout when acquiring the lock. It will be used as fallback value in the acquire
        method, if no timeout value (``None``) is given. If you want to disable the timeout, set it to a negative value.
         A timeout of 0 means, that there is exactly one attempt to acquire the file lock.
        """
        # The path to the lock file.
        self._lock_file: str = os.fspath(lock_file)

        # The file descriptor for the *_lock_file* as it is returned by the os.open() function.
        # This file lock is only NOT None, if the object currently holds the lock.
        self._lock_file_fd: int | None = None

        # The default timeout value.
        self.timeout: float = timeout

        # We use this lock primarily for the lock counter.
        self._thread_lock: Lock = Lock()

        # The lock counter is used for implementing the nested locking mechanism. Whenever the lock is acquired, the
        # counter is increased and the lock is only released, when this value is 0 again.
        self._lock_counter: int = 0

    @property
    def lock_file(self) -> str:
        """:return: path to the lock file"""
        return self._lock_file

    @property
    def timeout(self) -> float:
        """
        :return: the default timeout value

        .. versionadded:: 2.0.0
        """
        return self._timeout

    @timeout.setter
    def timeout(self, value: float | str) -> None:
        """
        Change the default timeout value.

        :param value: the new value
        """
        self._timeout = float(value)

    @abstractmethod
    def _acquire(self) -> None:
        """If the file lock could be acquired, self._lock_file_fd holds the file descriptor of the lock file."""
        raise NotImplementedError

    @abstractmethod
    def _release(self) -> None:
        """Releases the lock and sets self._lock_file_fd to None."""
        raise NotImplementedError

    @property
    def is_locked(self) -> bool:
        """

        :return: A boolean indicating if the lock file is holding the lock currently.

        .. versionchanged:: 2.0.0

            This was previously a method and is now a property.
        """
        return self._lock_file_fd is not None

    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        *,
        poll_intervall: float | None = None,
    ) -> AcquireReturnProxy:
        """
        Try to acquire the file lock.

        :param timeout: maximum wait time for acquiring the lock, ``None`` means use the default :attr:`~timeout` is and
         if ``timeout < 0``, there is no timeout and this method will block until the lock could be acquired
        :param poll_interval: interval of trying to acquire the lock file
        :param poll_intervall: deprecated, kept for backwards compatibility, use ``poll_interval`` instead
        :raises Timeout: if fails to acquire lock within the timeout period
        :return: a context object that will unlock the file when the context is exited

        .. code-block:: python

            # You can use this method in the context manager (recommended)
            with lock.acquire():
                pass

            # Or use an equivalent try-finally construct:
            lock.acquire()
            try:
                pass
            finally:
                lock.release()

        .. versionchanged:: 2.0.0

            This method returns now a *proxy* object instead of *self*,
            so that it can be used in a with statement without side effects.

        """
        # Use the default timeout, if no timeout is provided.
        if timeout is None:
            timeout = self.timeout

        if poll_intervall is not None:
            msg = "use poll_interval instead of poll_intervall"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            poll_interval = poll_intervall

        # Increment the number right at the beginning. We can still undo it, if something fails.
        with self._thread_lock:
            self._lock_counter += 1

        lock_id = id(self)
        lock_filename = self._lock_file
        start_time = time.time()
        try:
            while True:
                with self._thread_lock:
                    if not self.is_locked:
                        _LOGGER.debug("Attempting to acquire lock %s on %s", lock_id, lock_filename)
                        self._acquire()

                if self.is_locked:
                    _LOGGER.debug("Lock %s acquired on %s", lock_id, lock_filename)
                    break
                elif 0 <= timeout < time.time() - start_time:
                    _LOGGER.debug("Timeout on acquiring lock %s on %s", lock_id, lock_filename)
                    raise Timeout(self._lock_file)
                else:
                    msg = "Lock %s not acquired on %s, waiting %s seconds ..."
                    _LOGGER.debug(msg, lock_id, lock_filename, poll_interval)
                    time.sleep(poll_interval)
        except BaseException:  # Something did go wrong, so decrement the counter.
            with self._thread_lock:
                self._lock_counter = max(0, self._lock_counter - 1)
            raise
        return AcquireReturnProxy(lock=self)

    def release(self, force: bool = False) -> None:
        """
        Releases the file lock. Please note, that the lock is only completely released, if the lock counter is 0. Also
        note, that the lock file itself is not automatically deleted.

        :param force: If true, the lock counter is ignored and the lock is released in every case/
        """
        with self._thread_lock:

            if self.is_locked:
                self._lock_counter -= 1

                if self._lock_counter == 0 or force:
                    lock_id, lock_filename = id(self), self._lock_file

                    _LOGGER.debug("Attempting to release lock %s on %s", lock_id, lock_filename)
                    self._release()
                    self._lock_counter = 0
                    _LOGGER.debug("Lock %s released on %s", lock_id, lock_filename)

    def __enter__(self) -> BaseFileLock:
        """
        Acquire the lock.

        :return: the lock object
        """
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,  # noqa: U100
        exc_value: BaseException | None,  # noqa: U100
        traceback: TracebackType | None,  # noqa: U100
    ) -> None:
        """
        Release the lock.

        :param exc_type: the exception type if raised
        :param exc_value: the exception value if raised
        :param traceback: the exception traceback if raised
        """
        self.release()

    def __del__(self) -> None:
        """Called when the lock object is deleted."""
        self.release(force=True)


__all__ = [
    "BaseFileLock",
    "AcquireReturnProxy",
]
