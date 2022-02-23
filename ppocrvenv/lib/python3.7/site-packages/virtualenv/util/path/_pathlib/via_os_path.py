from __future__ import absolute_import, unicode_literals

import os
import platform
from contextlib import contextmanager

from virtualenv.util.six import ensure_str, ensure_text

IS_PYPY = platform.python_implementation() == "PyPy"


class Path(object):
    def __init__(self, path):
        if isinstance(path, Path):
            _path = path._path
        else:
            _path = ensure_text(path)
            if IS_PYPY:
                _path = _path.encode("utf-8")
        self._path = _path

    def __repr__(self):
        return ensure_str("Path({})".format(ensure_text(self._path)))

    def __unicode__(self):
        return ensure_text(self._path)

    def __str__(self):
        return ensure_str(self._path)

    def __div__(self, other):
        if isinstance(other, Path):
            right = other._path
        else:
            right = ensure_text(other)
            if IS_PYPY:
                right = right.encode("utf-8")
        return Path(os.path.join(self._path, right))

    def __truediv__(self, other):
        return self.__div__(other)

    def __eq__(self, other):
        return self._path == (other._path if isinstance(other, Path) else None)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self._path)

    def exists(self):
        return os.path.exists(self._path)

    @property
    def parent(self):
        return Path(os.path.abspath(os.path.join(self._path, os.path.pardir)))

    def resolve(self):
        return Path(os.path.realpath(self._path))

    @property
    def name(self):
        return os.path.basename(self._path)

    @property
    def parts(self):
        return self._path.split(os.sep)

    def is_file(self):
        return os.path.isfile(self._path)

    def is_dir(self):
        return os.path.isdir(self._path)

    def mkdir(self, parents=True, exist_ok=True):
        try:
            os.makedirs(self._path)
        except OSError:
            if not exist_ok:
                raise

    def read_text(self, encoding="utf-8"):
        return self.read_bytes().decode(encoding)

    def read_bytes(self):
        with open(self._path, "rb") as file_handler:
            return file_handler.read()

    def write_bytes(self, content):
        with open(self._path, "wb") as file_handler:
            file_handler.write(content)

    def write_text(self, text, encoding="utf-8"):
        self.write_bytes(text.encode(encoding))

    def iterdir(self):
        for p in os.listdir(self._path):
            yield Path(os.path.join(self._path, p))

    @property
    def suffix(self):
        _, ext = os.path.splitext(self.name)
        return ext

    @property
    def stem(self):
        base, _ = os.path.splitext(self.name)
        return base

    @contextmanager
    def open(self, mode="r"):
        with open(self._path, mode) as file_handler:
            yield file_handler

    @property
    def parents(self):
        result = []
        parts = self.parts
        for i in range(len(parts) - 1):
            result.append(Path(os.sep.join(parts[0 : i + 1])))
        return result[::-1]

    def unlink(self):
        os.remove(self._path)

    def with_name(self, name):
        return self.parent / name

    def is_symlink(self):
        return os.path.islink(self._path)

    def relative_to(self, other):
        if not self._path.startswith(other._path):
            raise ValueError("{} does not start with {}".format(self._path, other._path))
        return Path(os.sep.join(self.parts[len(other.parts) :]))

    def stat(self):
        return os.stat(self._path)

    def chmod(self, mode):
        os.chmod(self._path, mode)

    def absolute(self):
        return Path(os.path.abspath(self._path))


__all__ = ("Path",)
