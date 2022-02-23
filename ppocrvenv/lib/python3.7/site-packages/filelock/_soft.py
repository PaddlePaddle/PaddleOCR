from __future__ import annotations

import os
import sys
from errno import EACCES, EEXIST, ENOENT

from ._api import BaseFileLock
from ._util import raise_on_exist_ro_file


class SoftFileLock(BaseFileLock):
    """Simply watches the existence of the lock file."""

    def _acquire(self) -> None:
        raise_on_exist_ro_file(self._lock_file)
        # first check for exists and read-only mode as the open will mask this case as EEXIST
        mode = (
            os.O_WRONLY  # open for writing only
            | os.O_CREAT
            | os.O_EXCL  # together with above raise EEXIST if the file specified by filename exists
            | os.O_TRUNC  # truncate the file to zero byte
        )
        try:
            fd = os.open(self._lock_file, mode)
        except OSError as exception:
            if exception.errno == EEXIST:  # expected if cannot lock
                pass
            elif exception.errno == ENOENT:  # No such file or directory - parent directory is missing
                raise
            elif exception.errno == EACCES and sys.platform != "win32":  # pragma: win32 no cover
                # Permission denied - parent dir is R/O
                raise  # note windows does not allow you to make a folder r/o only files
        else:
            self._lock_file_fd = fd

    def _release(self) -> None:
        os.close(self._lock_file_fd)  # type: ignore # the lock file is definitely not None
        self._lock_file_fd = None
        try:
            os.remove(self._lock_file)
        except OSError:  # the file is already deleted and that's what we want
            pass


__all__ = [
    "SoftFileLock",
]
