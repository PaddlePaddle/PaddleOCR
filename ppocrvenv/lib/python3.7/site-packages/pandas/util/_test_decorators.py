"""
This module provides decorator functions which can be applied to test objects
in order to skip those objects when certain conditions occur. A sample use case
is to detect if the platform is missing ``matplotlib``. If so, any test objects
which require ``matplotlib`` and decorated with ``@td.skip_if_no_mpl`` will be
skipped by ``pytest`` during the execution of the test suite.

To illustrate, after importing this module:

import pandas.util._test_decorators as td

The decorators can be applied to classes:

@td.skip_if_some_reason
class Foo:
    ...

Or individual functions:

@td.skip_if_some_reason
def test_foo():
    ...

For more information, refer to the ``pytest`` documentation on ``skipif``.
"""
from __future__ import annotations

from contextlib import contextmanager
import locale
from typing import Callable
import warnings

import numpy as np
import pytest

from pandas._config import get_option

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.computation.expressions import (
    NUMEXPR_INSTALLED,
    USE_NUMEXPR,
)
from pandas.util.version import Version


def safe_import(mod_name: str, min_version: str | None = None):
    """
    Parameters
    ----------
    mod_name : str
        Name of the module to be imported
    min_version : str, default None
        Minimum required version of the specified mod_name

    Returns
    -------
    object
        The imported module if successful, or False
    """
    with warnings.catch_warnings():
        # Suppress warnings that we can't do anything about,
        #  e.g. from aiohttp
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="aiohttp",
            message=".*decorator is deprecated since Python 3.8.*",
        )

        try:
            mod = __import__(mod_name)
        except ImportError:
            return False

    if not min_version:
        return mod
    else:
        import sys

        try:
            version = getattr(sys.modules[mod_name], "__version__")
        except AttributeError:
            # xlrd uses a capitalized attribute name
            version = getattr(sys.modules[mod_name], "__VERSION__")
        if version and Version(version) >= Version(min_version):
            return mod

    return False


def _skip_if_no_mpl():
    mod = safe_import("matplotlib")
    if mod:
        mod.use("Agg")
    else:
        return True


def _skip_if_has_locale():
    lang, _ = locale.getlocale()
    if lang is not None:
        return True


def _skip_if_not_us_locale():
    lang, _ = locale.getlocale()
    if lang != "en_US":
        return True


def _skip_if_no_scipy() -> bool:
    return not (
        safe_import("scipy.stats")
        and safe_import("scipy.sparse")
        and safe_import("scipy.interpolate")
        and safe_import("scipy.signal")
    )


# TODO: return type, _pytest.mark.structures.MarkDecorator is not public
# https://github.com/pytest-dev/pytest/issues/7469
def skip_if_installed(package: str):
    """
    Skip a test if a package is installed.

    Parameters
    ----------
    package : str
        The name of the package.
    """
    return pytest.mark.skipif(
        safe_import(package), reason=f"Skipping because {package} is installed."
    )


# TODO: return type, _pytest.mark.structures.MarkDecorator is not public
# https://github.com/pytest-dev/pytest/issues/7469
def skip_if_no(package: str, min_version: str | None = None):
    """
    Generic function to help skip tests when required packages are not
    present on the testing system.

    This function returns a pytest mark with a skip condition that will be
    evaluated during test collection. An attempt will be made to import the
    specified ``package`` and optionally ensure it meets the ``min_version``

    The mark can be used as either a decorator for a test function or to be
    applied to parameters in pytest.mark.parametrize calls or parametrized
    fixtures.

    If the import and version check are unsuccessful, then the test function
    (or test case when used in conjunction with parametrization) will be
    skipped.

    Parameters
    ----------
    package: str
        The name of the required package.
    min_version: str or None, default None
        Optional minimum version of the package.

    Returns
    -------
    _pytest.mark.structures.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
    msg = f"Could not import '{package}'"
    if min_version:
        msg += f" satisfying a min_version of {min_version}"
    return pytest.mark.skipif(
        not safe_import(package, min_version=min_version), reason=msg
    )


skip_if_no_mpl = pytest.mark.skipif(
    _skip_if_no_mpl(), reason="Missing matplotlib dependency"
)
skip_if_mpl = pytest.mark.skipif(not _skip_if_no_mpl(), reason="matplotlib is present")
skip_if_32bit = pytest.mark.skipif(not IS64, reason="skipping for 32 bit")
skip_if_windows = pytest.mark.skipif(is_platform_windows(), reason="Running on Windows")
skip_if_windows_python_3 = pytest.mark.skipif(
    is_platform_windows(), reason="not used on win32"
)
skip_if_has_locale = pytest.mark.skipif(
    _skip_if_has_locale(), reason=f"Specific locale is set {locale.getlocale()[0]}"
)
skip_if_not_us_locale = pytest.mark.skipif(
    _skip_if_not_us_locale(), reason=f"Specific locale is set {locale.getlocale()[0]}"
)
skip_if_no_scipy = pytest.mark.skipif(
    _skip_if_no_scipy(), reason="Missing SciPy requirement"
)
skip_if_no_ne = pytest.mark.skipif(
    not USE_NUMEXPR,
    reason=f"numexpr enabled->{USE_NUMEXPR}, installed->{NUMEXPR_INSTALLED}",
)


# TODO: return type, _pytest.mark.structures.MarkDecorator is not public
# https://github.com/pytest-dev/pytest/issues/7469
def skip_if_np_lt(ver_str: str, *args, reason: str | None = None):
    if reason is None:
        reason = f"NumPy {ver_str} or greater required"
    return pytest.mark.skipif(
        Version(np.__version__) < Version(ver_str),
        *args,
        reason=reason,
    )


def parametrize_fixture_doc(*args):
    """
    Intended for use as a decorator for parametrized fixture,
    this function will wrap the decorated function with a pytest
    ``parametrize_fixture_doc`` mark. That mark will format
    initial fixture docstring by replacing placeholders {0}, {1} etc
    with parameters passed as arguments.

    Parameters
    ----------
    args: iterable
        Positional arguments for docstring.

    Returns
    -------
    function
        The decorated function wrapped within a pytest
        ``parametrize_fixture_doc`` mark
    """

    def documented_fixture(fixture):
        fixture.__doc__ = fixture.__doc__.format(*args)
        return fixture

    return documented_fixture


def check_file_leaks(func) -> Callable:
    """
    Decorate a test function to check that we are not leaking file descriptors.
    """
    with file_leak_context():
        return func


@contextmanager
def file_leak_context():
    """
    ContextManager analogue to check_file_leaks.
    """
    psutil = safe_import("psutil")
    if not psutil:
        yield
    else:
        proc = psutil.Process()
        flist = proc.open_files()
        conns = proc.connections()

        yield

        flist2 = proc.open_files()
        # on some builds open_files includes file position, which we _dont_
        #  expect to remain unchanged, so we need to compare excluding that
        flist_ex = [(x.path, x.fd) for x in flist]
        flist2_ex = [(x.path, x.fd) for x in flist2]
        assert flist2_ex == flist_ex, (flist2, flist)

        conns2 = proc.connections()
        assert conns2 == conns, (conns2, conns)


def async_mark():
    try:
        import_optional_dependency("pytest_asyncio")
        async_mark = pytest.mark.asyncio
    except ImportError:
        async_mark = pytest.mark.skip(reason="Missing dependency pytest-asyncio")

    return async_mark


skip_array_manager_not_yet_implemented = pytest.mark.skipif(
    get_option("mode.data_manager") == "array", reason="JSON C code relies on Blocks"
)

skip_array_manager_invalid_test = pytest.mark.skipif(
    get_option("mode.data_manager") == "array",
    reason="Test that relies on BlockManager internals or specific behaviour",
)
