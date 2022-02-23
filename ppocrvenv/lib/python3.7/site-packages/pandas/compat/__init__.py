"""
compat
======

Cross-compatible functions for different versions of Python.

Other items:
* platform checker
"""
import platform
import sys
import warnings

from pandas._typing import F
from pandas.compat.numpy import (
    is_numpy_dev,
    np_array_datetime64_compat,
    np_datetime64_compat,
    np_version_under1p18,
    np_version_under1p19,
    np_version_under1p20,
)
from pandas.compat.pyarrow import (
    pa_version_under1p0,
    pa_version_under2p0,
    pa_version_under3p0,
    pa_version_under4p0,
)

PY38 = sys.version_info >= (3, 8)
PY39 = sys.version_info >= (3, 9)
PY310 = sys.version_info >= (3, 10)
PYPY = platform.python_implementation() == "PyPy"
IS64 = sys.maxsize > 2 ** 32


def set_function_name(f: F, name: str, cls) -> F:
    """
    Bind the name/qualname attributes of the function.
    """
    f.__name__ = name
    f.__qualname__ = f"{cls.__name__}.{name}"
    f.__module__ = cls.__module__
    return f


def is_platform_little_endian() -> bool:
    """
    Checking if the running platform is little endian.

    Returns
    -------
    bool
        True if the running platform is little endian.
    """
    return sys.byteorder == "little"


def is_platform_windows() -> bool:
    """
    Checking if the running platform is windows.

    Returns
    -------
    bool
        True if the running platform is windows.
    """
    return sys.platform in ["win32", "cygwin"]


def is_platform_linux() -> bool:
    """
    Checking if the running platform is linux.

    Returns
    -------
    bool
        True if the running platform is linux.
    """
    return sys.platform == "linux"


def is_platform_mac() -> bool:
    """
    Checking if the running platform is mac.

    Returns
    -------
    bool
        True if the running platform is mac.
    """
    return sys.platform == "darwin"


def is_platform_arm() -> bool:
    """
    Checking if he running platform use ARM architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    return platform.machine() in ("arm64", "aarch64")


def import_lzma():
    """
    Importing the `lzma` module.

    Warns
    -----
    When the `lzma` module is not available.
    """
    try:
        import lzma

        return lzma
    except ImportError:
        msg = (
            "Could not import the lzma module. Your installed Python is incomplete. "
            "Attempting to use lzma compression will result in a RuntimeError."
        )
        warnings.warn(msg)


def get_lzma_file(lzma):
    """
    Importing the `LZMAFile` class from the `lzma` module.

    Returns
    -------
    class
        The `LZMAFile` class from the `lzma` module.

    Raises
    ------
    RuntimeError
        If the `lzma` module was not imported correctly, or didn't exist.
    """
    if lzma is None:
        raise RuntimeError(
            "lzma module not available. "
            "A Python re-install with the proper dependencies, "
            "might be required to solve this issue."
        )
    return lzma.LZMAFile


__all__ = [
    "is_numpy_dev",
    "np_array_datetime64_compat",
    "np_datetime64_compat",
    "np_version_under1p18",
    "np_version_under1p19",
    "np_version_under1p20",
    "pa_version_under1p0",
    "pa_version_under2p0",
    "pa_version_under3p0",
    "pa_version_under4p0",
]
