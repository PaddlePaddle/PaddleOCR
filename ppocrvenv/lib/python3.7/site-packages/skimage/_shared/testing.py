"""
Testing utilities.
"""

import os
import re
import struct
import threading
import functools
from tempfile import NamedTemporaryFile

import numpy as np
from numpy import testing
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less, assert_array_almost_equal_nulp,
                           assert_equal, TestCase, assert_allclose,
                           assert_almost_equal, assert_, assert_warns,
                           assert_no_warnings)

import warnings

from .. import data, io
from ..data._fetchers import _fetch
from ..util import img_as_uint, img_as_float, img_as_int, img_as_ubyte
from ._warnings import expected_warnings


SKIP_RE = re.compile(r"(\s*>>>.*?)(\s*)#\s*skip\s+if\s+(.*)$")

import pytest
skipif = pytest.mark.skipif
xfail = pytest.mark.xfail
parametrize = pytest.mark.parametrize
raises = pytest.raises
fixture = pytest.fixture

# true if python is running in 32bit mode
# Calculate the size of a void * pointer in bits
# https://docs.python.org/3/library/struct.html
arch32 = struct.calcsize("P") * 8 == 32


_error_on_warnings = os.environ.get('SKIMAGE_TEST_STRICT_WARNINGS_GLOBAL', '0')
if _error_on_warnings.lower() == 'true':
    _error_on_warnings = True
elif _error_on_warnings.lower() == 'false':
    _error_on_warnings = False
else:
    try:
        _error_on_warnings = bool(int(_error_on_warnings))
    except ValueError:
        _error_on_warnings = False

def assert_less(a, b, msg=None):
    message = "%r is not lower than %r" % (a, b)
    if msg is not None:
        message += ": " + msg
    assert a < b, message


def assert_greater(a, b, msg=None):
    message = "%r is not greater than %r" % (a, b)
    if msg is not None:
        message += ": " + msg
    assert a > b, message


def doctest_skip_parser(func):
    """ Decorator replaces custom skip test markup in doctests

    Say a function has a docstring::

        >>> something, HAVE_AMODULE, HAVE_BMODULE = 0, False, False
        >>> something # skip if not HAVE_AMODULE
        0
        >>> something # skip if HAVE_BMODULE
        0

    This decorator will evaluate the expression after ``skip if``.  If this
    evaluates to True, then the comment is replaced by ``# doctest: +SKIP``. If
    False, then the comment is just removed. The expression is evaluated in the
    ``globals`` scope of `func`.

    For example, if the module global ``HAVE_AMODULE`` is False, and module
    global ``HAVE_BMODULE`` is False, the returned function will have docstring::

        >>> something # doctest: +SKIP
        >>> something + else # doctest: +SKIP
        >>> something # doctest: +SKIP

    """
    lines = func.__doc__.split('\n')
    new_lines = []
    for line in lines:
        match = SKIP_RE.match(line)
        if match is None:
            new_lines.append(line)
            continue
        code, space, expr = match.groups()

        try:
            # Works as a function decorator
            if eval(expr, func.__globals__):
                code = code + space + "# doctest: +SKIP"
        except AttributeError:
            # Works as a class decorator
            if eval(expr, func.__init__.__globals__):
                code = code + space + "# doctest: +SKIP"

        new_lines.append(code)
    func.__doc__ = "\n".join(new_lines)
    return func


def roundtrip(image, plugin, suffix):
    """Save and read an image using a specified plugin"""
    if '.' not in suffix:
        suffix = '.' + suffix
    temp_file = NamedTemporaryFile(suffix=suffix, delete=False)
    fname = temp_file.name
    temp_file.close()
    io.imsave(fname, image, plugin=plugin)
    new = io.imread(fname, plugin=plugin)
    try:
        os.remove(fname)
    except Exception:
        pass
    return new


def color_check(plugin, fmt='png'):
    """Check roundtrip behavior for color images.

    All major input types should be handled as ubytes and read
    back correctly.
    """
    img = img_as_ubyte(data.chelsea())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2, r2.astype(bool))

    img3 = img_as_float(img)
    r3 = roundtrip(img3, plugin, fmt)
    testing.assert_allclose(r3, img)

    img4 = img_as_int(img)
    if fmt.lower() in (('tif', 'tiff')):
        img4 -= 100
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img4)
    else:
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img_as_ubyte(img4))

    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img)


def mono_check(plugin, fmt='png'):
    """Check the roundtrip behavior for images that support most types.

    All major input types should be handled.
    """

    img = img_as_ubyte(data.moon())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2, r2.astype(bool))

    img3 = img_as_float(img)
    r3 = roundtrip(img3, plugin, fmt)
    if r3.dtype.kind == 'f':
        testing.assert_allclose(img3, r3)
    else:
        testing.assert_allclose(r3, img_as_uint(img))

    img4 = img_as_int(img)
    if fmt.lower() in (('tif', 'tiff')):
        img4 -= 100
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img4)
    else:
        r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img_as_uint(img4))

    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img5)


def setup_test():
    """Default package level setup routine for skimage tests.

    Import packages known to raise warnings, and then
    force warnings to raise errors.

    Also set the random seed to zero.
    """
    warnings.simplefilter('default')

    if _error_on_warnings:
        from scipy import signal, ndimage, special, optimize, linalg
        from scipy.io import loadmat
        from skimage import viewer

        np.random.seed(0)

        warnings.simplefilter('error')

        # do not error on specific warnings from the skimage.io module
        # https://github.com/scikit-image/scikit-image/issues/5337
        warnings.filterwarnings(
            'default', message='TiffFile:', category=DeprecationWarning
        )

        warnings.filterwarnings(
            'default', message='TiffWriter:', category=DeprecationWarning
        )
        # newer tifffile change the start of the warning string
        # e.g. <tifffile.TiffWriter.write> data with shape ...
        warnings.filterwarnings(
            'default',
            message='<tifffile.',
            category=DeprecationWarning
        )

        warnings.filterwarnings(
            'default', message='unclosed file', category=ResourceWarning
        )

        # ignore known FutureWarnings from viewer module
        warnings.filterwarnings(
            'ignore', category=FutureWarning, module='skimage.viewer'
        )

        # Ignore other warnings only seen when using older versions of
        # dependencies.
        warnings.filterwarnings(
            'default',
            message='Conversion of the second argument of issubdtype',
            category=FutureWarning
        )

        warnings.filterwarnings(
            'default',
            message='the matrix subclass is not the recommended way',
            category=PendingDeprecationWarning, module='numpy'
        )

        warnings.filterwarnings(
            'default',
            message='Your installed pillow version',
            category=UserWarning,
            module='skimage.io'
        )

        # match both "Viewer requires Qt" and "Viewer requires matplotlib"
        warnings.filterwarnings(
            'default', message='Viewer requires ', category=UserWarning
        )

        # ignore warning from cycle_spin about Dask not being installed
        warnings.filterwarnings(
            'default',
            message='The optional dask dependency is not installed.',
            category=UserWarning
        )

        warnings.filterwarnings(
            'default',
            message='numpy.ufunc size changed',
            category=RuntimeWarning
        )


def teardown_test():
    """Default package level teardown routine for skimage tests.

    Restore warnings to default behavior
    """
    if _error_on_warnings:
        warnings.resetwarnings()
        warnings.simplefilter('default')


def fetch(data_filename):
    """Attempt to fetch data, but if unavailable, skip the tests."""
    try:
        return _fetch(data_filename)
    except (ConnectionError, ModuleNotFoundError):
        pytest.skip(f'Unable to download {data_filename}',
                    allow_module_level=True)


def test_parallel(num_threads=2, warnings_matching=None):
    """Decorator to run the same function multiple times in parallel.

    This decorator is useful to ensure that separate threads execute
    concurrently and correctly while releasing the GIL.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.

    warnings_matching: list or None
        This parameter is passed on to `expected_warnings` so as not to have
        race conditions with the warnings filters. A single
        `expected_warnings` context manager is used for all threads.
        If None, then no warnings are checked.

    """

    assert num_threads > 0

    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            with expected_warnings(warnings_matching):
                threads = []
                for i in range(num_threads - 1):
                    thread = threading.Thread(target=func, args=args,
                                              kwargs=kwargs)
                    threads.append(thread)
                for thread in threads:
                    thread.start()

                result = func(*args, **kwargs)

                for thread in threads:
                    thread.join()

                return result

        return inner

    return wrapper


if __name__ == '__main__':
    color_check('pil')
    mono_check('pil')
    mono_check('pil', 'bmp')
    mono_check('pil', 'tiff')
