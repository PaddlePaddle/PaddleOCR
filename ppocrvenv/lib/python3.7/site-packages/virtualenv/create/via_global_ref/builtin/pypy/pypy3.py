from __future__ import absolute_import, unicode_literals

import abc

from six import add_metaclass

from virtualenv.create.describe import PosixSupports, Python3Supports, WindowsSupports
from virtualenv.create.via_global_ref.builtin.ref import PathRefToDest
from virtualenv.util.path import Path

from .common import PyPy


@add_metaclass(abc.ABCMeta)
class PyPy3(PyPy, Python3Supports):
    @classmethod
    def exe_stem(cls):
        return "pypy3"

    @classmethod
    def exe_names(cls, interpreter):
        return super(PyPy3, cls).exe_names(interpreter) | {"pypy"}


class PyPy3Posix(PyPy3, PosixSupports):
    """PyPy 3 on POSIX"""

    @property
    def stdlib(self):
        """PyPy3 respects sysconfig only for the host python, virtual envs is instead lib/pythonx.y/site-packages"""
        return self.dest / "lib" / "pypy{}".format(self.interpreter.version_release_str) / "site-packages"

    @classmethod
    def _shared_libs(cls, python_dir):
        # glob for libpypy3-c.so, libpypy3-c.dylib, libpypy3.9-c.so ...
        return python_dir.glob("libpypy3*.*")

    def to_lib(self, src):
        return self.dest / "lib" / src.name

    @classmethod
    def sources(cls, interpreter):
        for src in super(PyPy3Posix, cls).sources(interpreter):
            yield src
        # Also copy/symlink anything under prefix/lib, which, for "portable"
        # PyPy builds, includes the tk,tcl runtime and a number of shared
        # objects. In distro-specific builds or on conda this should be empty
        # (on PyPy3.8+ it will, like on CPython, hold the stdlib).
        host_lib = Path(interpreter.system_prefix) / "lib"
        stdlib = Path(interpreter.system_stdlib)
        if host_lib.exists() and host_lib.is_dir():
            for path in host_lib.iterdir():
                if stdlib == path:
                    # For PyPy3.8+ the stdlib lives in lib/pypy3.8
                    # We need to avoid creating a symlink to it since that
                    # will defeat the purpose of a virtualenv
                    continue
                yield PathRefToDest(path, dest=cls.to_lib)


class Pypy3Windows(PyPy3, WindowsSupports):
    """PyPy 3 on Windows"""

    @property
    def stdlib(self):
        """PyPy3 respects sysconfig only for the host python, virtual envs is instead Lib/site-packages"""
        return self.dest / "Lib" / "site-packages"

    @property
    def bin_dir(self):
        """PyPy3 needs to fallback to pypy definition"""
        return self.dest / "Scripts"

    @classmethod
    def _shared_libs(cls, python_dir):
        # glob for libpypy*.dll and libffi*.dll
        for pattern in ["libpypy*.dll", "libffi*.dll"]:
            srcs = python_dir.glob(pattern)
            for src in srcs:
                yield src
