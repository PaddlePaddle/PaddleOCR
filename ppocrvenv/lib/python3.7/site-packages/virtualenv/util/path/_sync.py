from __future__ import absolute_import, unicode_literals

import logging
import os
import shutil
from stat import S_IWUSR

from six import PY2

from virtualenv.info import IS_CPYTHON, IS_WIN
from virtualenv.util.six import ensure_text

if PY2 and IS_CPYTHON and IS_WIN:  # CPython2 on Windows supports unicode paths if passed as unicode

    def norm(src):
        return ensure_text(str(src))

else:
    norm = str


def ensure_dir(path):
    if not path.exists():
        logging.debug("create folder %s", ensure_text(str(path)))
        os.makedirs(norm(path))


def ensure_safe_to_do(src, dest):
    if src == dest:
        raise ValueError("source and destination is the same {}".format(src))
    if not dest.exists():
        return
    if dest.is_dir() and not dest.is_symlink():
        logging.debug("remove directory %s", dest)
        safe_delete(dest)
    else:
        logging.debug("remove file %s", dest)
        dest.unlink()


def symlink(src, dest):
    ensure_safe_to_do(src, dest)
    logging.debug("symlink %s", _Debug(src, dest))
    dest.symlink_to(src, target_is_directory=src.is_dir())


def copy(src, dest):
    ensure_safe_to_do(src, dest)
    is_dir = src.is_dir()
    method = copytree if is_dir else shutil.copy
    logging.debug("copy %s", _Debug(src, dest))
    method(norm(src), norm(dest))


def copytree(src, dest):
    for root, _, files in os.walk(src):
        dest_dir = os.path.join(dest, os.path.relpath(root, src))
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        for name in files:
            src_f = os.path.join(root, name)
            dest_f = os.path.join(dest_dir, name)
            shutil.copy(src_f, dest_f)


def safe_delete(dest):
    def onerror(func, path, exc_info):
        if not os.access(path, os.W_OK):
            os.chmod(path, S_IWUSR)
            func(path)
        else:
            raise

    shutil.rmtree(ensure_text(str(dest)), ignore_errors=True, onerror=onerror)


class _Debug(object):
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest

    def __str__(self):
        return "{}{} to {}".format(
            "directory " if self.src.is_dir() else "",
            ensure_text(str(self.src)),
            ensure_text(str(self.dest)),
        )


__all__ = (
    "ensure_dir",
    "symlink",
    "copy",
    "symlink",
    "copytree",
    "safe_delete",
)
