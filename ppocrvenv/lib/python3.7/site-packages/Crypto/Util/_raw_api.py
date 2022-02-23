# ===================================================================
#
# Copyright (c) 2014, Legrandin <helderijs@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===================================================================

import os
import abc
import sys
from Crypto.Util.py3compat import byte_string
from Crypto.Util._file_system import pycryptodome_filename

#
# List of file suffixes for Python extensions
#
if sys.version_info[0] < 3:

    import imp
    extension_suffixes = []
    for ext, mod, typ in imp.get_suffixes():
        if typ == imp.C_EXTENSION:
            extension_suffixes.append(ext)

else:

    from importlib import machinery
    extension_suffixes = machinery.EXTENSION_SUFFIXES

# Which types with buffer interface we support (apart from byte strings)
_buffer_type = (bytearray, memoryview)


class _VoidPointer(object):
    @abc.abstractmethod
    def get(self):
        """Return the memory location we point to"""
        return

    @abc.abstractmethod
    def address_of(self):
        """Return a raw pointer to this pointer"""
        return


try:
    # Starting from v2.18, pycparser (used by cffi for in-line ABI mode)
    # stops working correctly when PYOPTIMIZE==2 or the parameter -OO is
    # passed. In that case, we fall back to ctypes.
    # Note that PyPy ships with an old version of pycparser so we can keep
    # using cffi there.
    # See https://github.com/Legrandin/pycryptodome/issues/228
    if '__pypy__' not in sys.builtin_module_names and sys.flags.optimize == 2:
        raise ImportError("CFFI with optimize=2 fails due to pycparser bug.")

    from cffi import FFI

    ffi = FFI()
    null_pointer = ffi.NULL
    uint8_t_type = ffi.typeof(ffi.new("const uint8_t*"))

    _Array = ffi.new("uint8_t[1]").__class__.__bases__

    def load_lib(name, cdecl):
        """Load a shared library and return a handle to it.

        @name,  either an absolute path or the name of a library
                in the system search path.

        @cdecl, the C function declarations.
        """

        if hasattr(ffi, "RTLD_DEEPBIND") and not os.getenv('PYCRYPTODOME_DISABLE_DEEPBIND'):
            lib = ffi.dlopen(name, ffi.RTLD_DEEPBIND)
        else:
            lib = ffi.dlopen(name)
        ffi.cdef(cdecl)
        return lib

    def c_ulong(x):
        """Convert a Python integer to unsigned long"""
        return x

    c_ulonglong = c_ulong
    c_uint = c_ulong
    c_ubyte = c_ulong

    def c_size_t(x):
        """Convert a Python integer to size_t"""
        return x

    def create_string_buffer(init_or_size, size=None):
        """Allocate the given amount of bytes (initially set to 0)"""

        if isinstance(init_or_size, bytes):
            size = max(len(init_or_size) + 1, size)
            result = ffi.new("uint8_t[]", size)
            result[:] = init_or_size
        else:
            if size:
                raise ValueError("Size must be specified once only")
            result = ffi.new("uint8_t[]", init_or_size)
        return result

    def get_c_string(c_string):
        """Convert a C string into a Python byte sequence"""
        return ffi.string(c_string)

    def get_raw_buffer(buf):
        """Convert a C buffer into a Python byte sequence"""
        return ffi.buffer(buf)[:]

    def c_uint8_ptr(data):
        if isinstance(data, _buffer_type):
            # This only works for cffi >= 1.7
            return ffi.cast(uint8_t_type, ffi.from_buffer(data))
        elif byte_string(data) or isinstance(data, _Array):
            return data
        else:
            raise TypeError("Object type %s cannot be passed to C code" % type(data))

    class VoidPointer_cffi(_VoidPointer):
        """Model a newly allocated pointer to void"""

        def __init__(self):
            self._pp = ffi.new("void *[1]")

        def get(self):
            return self._pp[0]

        def address_of(self):
            return self._pp

    def VoidPointer():
        return VoidPointer_cffi()

    backend = "cffi"

except ImportError:

    import ctypes
    from ctypes import (CDLL, c_void_p, byref, c_ulong, c_ulonglong, c_size_t,
                        create_string_buffer, c_ubyte, c_uint)
    from ctypes.util import find_library
    from ctypes import Array as _Array

    null_pointer = None
    cached_architecture = []

    def c_ubyte(c):
        if not (0 <= c < 256):
            raise OverflowError()
        return ctypes.c_ubyte(c)

    def load_lib(name, cdecl):
        if not cached_architecture:
            # platform.architecture() creates a subprocess, so caching the
            # result makes successive imports faster.
            import platform
            cached_architecture[:] = platform.architecture()
        bits, linkage = cached_architecture
        if "." not in name and not linkage.startswith("Win"):
            full_name = find_library(name)
            if full_name is None:
                raise OSError("Cannot load library '%s'" % name)
            name = full_name
        return CDLL(name)

    def get_c_string(c_string):
        return c_string.value

    def get_raw_buffer(buf):
        return buf.raw

    # ---- Get raw pointer ---

    _c_ssize_t = ctypes.c_ssize_t

    _PyBUF_SIMPLE = 0
    _PyObject_GetBuffer = ctypes.pythonapi.PyObject_GetBuffer
    _PyBuffer_Release = ctypes.pythonapi.PyBuffer_Release
    _py_object = ctypes.py_object
    _c_ssize_p = ctypes.POINTER(_c_ssize_t)

    # See Include/object.h for CPython
    # and https://github.com/pallets/click/blob/master/src/click/_winconsole.py
    class _Py_buffer(ctypes.Structure):
        _fields_ = [
            ('buf',         c_void_p),
            ('obj',         ctypes.py_object),
            ('len',         _c_ssize_t),
            ('itemsize',    _c_ssize_t),
            ('readonly',    ctypes.c_int),
            ('ndim',        ctypes.c_int),
            ('format',      ctypes.c_char_p),
            ('shape',       _c_ssize_p),
            ('strides',     _c_ssize_p),
            ('suboffsets',  _c_ssize_p),
            ('internal',    c_void_p)
        ]

        # Extra field for CPython 2.6/2.7
        if sys.version_info[0] == 2:
            _fields_.insert(-1, ('smalltable', _c_ssize_t * 2))

    def c_uint8_ptr(data):
        if byte_string(data) or isinstance(data, _Array):
            return data
        elif isinstance(data, _buffer_type):
            obj = _py_object(data)
            buf = _Py_buffer()
            _PyObject_GetBuffer(obj, byref(buf), _PyBUF_SIMPLE)
            try:
                buffer_type = ctypes.c_ubyte * buf.len
                return buffer_type.from_address(buf.buf)
            finally:
                _PyBuffer_Release(byref(buf))
        else:
            raise TypeError("Object type %s cannot be passed to C code" % type(data))

    # ---

    class VoidPointer_ctypes(_VoidPointer):
        """Model a newly allocated pointer to void"""

        def __init__(self):
            self._p = c_void_p()

        def get(self):
            return self._p

        def address_of(self):
            return byref(self._p)

    def VoidPointer():
        return VoidPointer_ctypes()

    backend = "ctypes"


class SmartPointer(object):
    """Class to hold a non-managed piece of memory"""

    def __init__(self, raw_pointer, destructor):
        self._raw_pointer = raw_pointer
        self._destructor = destructor

    def get(self):
        return self._raw_pointer

    def release(self):
        rp, self._raw_pointer = self._raw_pointer, None
        return rp

    def __del__(self):
        try:
            if self._raw_pointer is not None:
                self._destructor(self._raw_pointer)
                self._raw_pointer = None
        except AttributeError:
            pass


def load_pycryptodome_raw_lib(name, cdecl):
    """Load a shared library and return a handle to it.

    @name,  the name of the library expressed as a PyCryptodome module,
            for instance Crypto.Cipher._raw_cbc.

    @cdecl, the C function declarations.
    """

    split = name.split(".")
    dir_comps, basename = split[:-1], split[-1]
    attempts = []
    for ext in extension_suffixes:
        try:
            filename = basename + ext
            full_name = pycryptodome_filename(dir_comps, filename)
            if not os.path.isfile(full_name):
                attempts.append("Not found '%s'" % filename)
                continue
            return load_lib(full_name, cdecl)
        except OSError as exp:
            attempts.append("Cannot load '%s': %s" % (filename, str(exp)))
    raise OSError("Cannot load native module '%s': %s" % (name, ", ".join(attempts)))


def is_buffer(x):
    """Return True if object x supports the buffer interface"""
    return isinstance(x, (bytes, bytearray, memoryview))


def is_writeable_buffer(x):
    return (isinstance(x, bytearray) or
            (isinstance(x, memoryview) and not x.readonly))
