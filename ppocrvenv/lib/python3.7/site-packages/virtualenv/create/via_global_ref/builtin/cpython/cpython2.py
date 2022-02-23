from __future__ import absolute_import, unicode_literals

import abc
import logging

from six import add_metaclass

from virtualenv.create.via_global_ref.builtin.ref import PathRefToDest
from virtualenv.util.path import Path

from ..python2.python2 import Python2
from .common import CPython, CPythonPosix, CPythonWindows, is_mac_os_framework


@add_metaclass(abc.ABCMeta)
class CPython2(CPython, Python2):
    """Create a CPython version 2  virtual environment"""

    @classmethod
    def sources(cls, interpreter):
        for src in super(CPython2, cls).sources(interpreter):
            yield src
        # include folder needed on Python 2 as we don't have pyenv.cfg
        host_include_marker = cls.host_include_marker(interpreter)
        if host_include_marker.exists():
            yield PathRefToDest(host_include_marker.parent, dest=lambda self, _: self.include)

    @classmethod
    def needs_stdlib_py_module(cls):
        return False

    @classmethod
    def host_include_marker(cls, interpreter):
        return Path(interpreter.system_include) / "Python.h"

    @property
    def include(self):
        # the pattern include the distribution name too at the end, remove that via the parent call
        return (self.dest / self.interpreter.install_path("headers")).parent

    @classmethod
    def modules(cls):
        return [
            "os",  # landmark to set sys.prefix
        ]

    def ensure_directories(self):
        dirs = super(CPython2, self).ensure_directories()
        host_include_marker = self.host_include_marker(self.interpreter)
        if host_include_marker.exists():
            dirs.add(self.include.parent)
        else:
            logging.debug("no include folders as can't find include marker %s", host_include_marker)
        return dirs


@add_metaclass(abc.ABCMeta)
class CPython2PosixBase(CPython2, CPythonPosix):
    """common to macOs framework builds and other posix CPython2"""

    @classmethod
    def sources(cls, interpreter):
        for src in super(CPython2PosixBase, cls).sources(interpreter):
            yield src

        # check if the makefile exists and if so make it available under the virtual environment
        make_file = Path(interpreter.sysconfig["makefile_filename"])
        if make_file.exists() and str(make_file).startswith(interpreter.prefix):
            under_prefix = make_file.relative_to(Path(interpreter.prefix))
            yield PathRefToDest(make_file, dest=lambda self, s: self.dest / under_prefix)


class CPython2Posix(CPython2PosixBase):
    """CPython 2 on POSIX (excluding macOs framework builds)"""

    @classmethod
    def can_describe(cls, interpreter):
        return is_mac_os_framework(interpreter) is False and super(CPython2Posix, cls).can_describe(interpreter)

    @classmethod
    def sources(cls, interpreter):
        for src in super(CPython2Posix, cls).sources(interpreter):
            yield src
        # landmark for exec_prefix
        exec_marker_file, to_path, _ = cls.from_stdlib(cls.mappings(interpreter), "lib-dynload")
        yield PathRefToDest(exec_marker_file, dest=to_path)


class CPython2Windows(CPython2, CPythonWindows):
    """CPython 2 on Windows"""

    @classmethod
    def sources(cls, interpreter):
        for src in super(CPython2Windows, cls).sources(interpreter):
            yield src
        py27_dll = Path(interpreter.system_executable).parent / "python27.dll"
        if py27_dll.exists():  # this might be global in the Windows folder in which case it's alright to be missing
            yield PathRefToDest(py27_dll, dest=cls.to_bin)

        libs = Path(interpreter.system_prefix) / "libs"
        if libs.exists():
            yield PathRefToDest(libs, dest=lambda self, s: self.dest / s.name)
