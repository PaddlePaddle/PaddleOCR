""":mod:`wand.cdefs.structures` --- MagickWand C typedefs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 0.5.0
"""
import ctypes
import os
import sys

__all__ = ('c_magick_real_t', 'c_magick_size_t', 'c_ssize_t')


if not hasattr(ctypes, 'c_ssize_t'):
    if ctypes.sizeof(ctypes.c_uint) == ctypes.sizeof(ctypes.c_void_p):
        ctypes.c_ssize_t = ctypes.c_int
    elif ctypes.sizeof(ctypes.c_ulong) == ctypes.sizeof(ctypes.c_void_p):
        ctypes.c_ssize_t = ctypes.c_long
    elif ctypes.sizeof(ctypes.c_ulonglong) == ctypes.sizeof(ctypes.c_void_p):
        ctypes.c_ssize_t = ctypes.c_longlong
c_ssize_t = ctypes.c_ssize_t


env_real = os.getenv('WAND_REAL_TYPE', 'auto')
if env_real in ('double', 'c_double'):
    c_magick_real_t = ctypes.c_double
elif env_real in ('longdouble', 'c_longdouble'):
    c_magick_real_t = ctypes.c_longdouble
else:
    # Attempt to guess MagickRealType size
    if sys.maxsize > 2**32:
        c_magick_real_t = ctypes.c_double
    else:
        c_magick_real_t = ctypes.c_longdouble
del env_real


# FIXME: Might need to rewrite to check against c_void_p size;
# like `c_ssize_t` above, and not against window platform.
if ctypes.sizeof(ctypes.c_size_t) == 8:
    c_magick_size_t = ctypes.c_size_t
elif ctypes.sizeof(ctypes.c_ulonglong) == 8:
    c_magick_size_t = ctypes.c_ulonglong
else:
    c_magick_size_t = ctypes.c_uint64
