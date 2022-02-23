from __future__ import annotations

import os
import sys
from abc import ABC
from errno import ENOENT
from typing import cast

from ._api import BaseFileLock
from ._util import raise_on_exist_ro_file

if sys.platform == "win32":  # pragma: win32 cover
    import msvcrt

    class WindowsFileLock(BaseFileLock):
        """Uses the :func:`msvcrt.locking` function to hard lock the lock file on windows systems."""

        def _acquire(self) -> None:
            raise_on_exist_ro_file(self._lock_file)
            mode = (
                os.O_RDWR  # open for read and write
                | os.O_CREAT  # create file if not exists
                | os.O_TRUNC  # truncate file  if not empty
            )
            try:
                fd = os.open(self._lock_file, mode)
            except OSError as exception:
                if exception.errno == ENOENT:  # No such file or directory
                    raise
            else:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError:
                    os.close(fd)
                else:
                    self._lock_file_fd = fd

        def _release(self) -> None:
            fd = cast(int, self._lock_file_fd)
            self._lock_file_fd = None
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)

            try:
                os.remove(self._lock_file)
            # Probably another instance of the application hat acquired the file lock.
            except OSError:
                pass

else:  # pragma: win32 no cover

    class WindowsFileLock(BaseFileLock, ABC):
        """Uses the :func:`msvcrt.locking` function to hard lock the lock file on windows systems."""


__all__ = [
    "WindowsFileLock",
]
