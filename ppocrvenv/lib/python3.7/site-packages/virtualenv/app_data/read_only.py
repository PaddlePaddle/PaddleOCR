import os.path

from virtualenv.util.lock import NoOpFileLock

from .via_disk_folder import AppDataDiskFolder, PyInfoStoreDisk


class ReadOnlyAppData(AppDataDiskFolder):
    can_update = False

    def __init__(self, folder):  # type: (str) -> None
        if not os.path.isdir(folder):
            raise RuntimeError("read-only app data directory {} does not exist".format(folder))
        self.lock = NoOpFileLock(folder)

    def reset(self):  # type: () -> None
        raise RuntimeError("read-only app data does not support reset")

    def py_info_clear(self):  # type: () -> None
        raise NotImplementedError

    def py_info(self, path):
        return _PyInfoStoreDiskReadOnly(self.py_info_at, path)

    def embed_update_log(self, distribution, for_py_version):
        raise NotImplementedError


class _PyInfoStoreDiskReadOnly(PyInfoStoreDisk):
    def write(self, content):
        raise RuntimeError("read-only app data python info cannot be updated")


__all__ = ("ReadOnlyAppData",)
