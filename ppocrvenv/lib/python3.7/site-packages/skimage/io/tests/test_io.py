import os
import pathlib
import tempfile

import numpy as np
import pytest

from skimage import io
from skimage._shared.testing import assert_array_equal, fetch
from skimage.data import data_dir


one_by_one_jpeg = (
    b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01'
    b'\x00\x01\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02'
    b'\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04'
    b'\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t'
    b'\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10'
    b'\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10'
    b'\x10\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11'
    b'\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\xff\xc4\x00'
    b'\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    b'\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00'
    b'\x00?\x00*\x9f\xff\xd9'
)


def test_stack_basic():
    x = np.arange(12).reshape(3, 4)
    io.push(x)

    assert_array_equal(io.pop(), x)


def test_stack_non_array():
    with pytest.raises(ValueError):
        io.push([[1, 2, 3]])


def test_imread_file_url():
    # tweak data path so that file URI works on both unix and windows.
    data_path = str(fetch('data/camera.png'))
    data_path = data_path.replace(os.path.sep, '/')
    image_url = f'file:///{data_path}'
    image = io.imread(image_url)
    assert image.shape == (512, 512)


def test_imread_http_url(httpserver):
    # httpserver is a fixture provided by pytest-localserver
    # https://bitbucket.org/pytest-dev/pytest-localserver/
    httpserver.serve_content(one_by_one_jpeg)
    # it will serve anything you provide to it on its url.
    # we add a /test.jpg so that we can identify the content
    # by extension
    image = io.imread(httpserver.url + '/test.jpg' + '?' + 's' * 266)
    assert image.shape == (1, 1)


def test_imread_pathlib_tiff():
    """Tests reading from Path object (issue gh-5545)."""

    # read via fetch
    expected = io.imread(fetch('data/multipage.tif'))

    # read by passing in a pathlib.Path object
    fname = os.path.join(data_dir, 'multipage.tif')
    path = pathlib.Path(fname)
    img = io.imread(path)

    assert img.shape == (2, 15, 10)
    assert_array_equal(expected, img)


def _named_tempfile_func(error_class):
    """Create a mock function for NamedTemporaryFile that always raises.

    Parameters
    ----------
    error_class : exception class
        The error that should be raised when asking for a NamedTemporaryFile.

    Returns
    -------
    named_temp_file : callable
        A function that always raises the desired error.

    Notes
    -----
    Although this function has general utility for raising errors, it is
    expected to be used to raise errors that ``tempfile.NamedTemporaryFile``
    from the Python standard library could raise. As of this writing, these
    are ``FileNotFoundError``, ``FileExistsError``, ``PermissionError``, and
    ``BaseException``. See
    `this comment <https://github.com/scikit-image/scikit-image/issues/3785#issuecomment-486598307>`__  # noqa
    for more information.
    """
    def named_temp_file(*args, **kwargs):
        raise error_class()
    return named_temp_file


@pytest.mark.parametrize(
    'error_class', [
        FileNotFoundError, FileExistsError, PermissionError, BaseException
    ]
)
def test_failed_temporary_file(monkeypatch, error_class):
    fetch('data/camera.png')
    # tweak data path so that file URI works on both unix and windows.
    data_path = data_dir.lstrip(os.path.sep)
    data_path = data_path.replace(os.path.sep, '/')
    image_url = f'file:///{data_path}/camera.png'
    with monkeypatch.context():
        monkeypatch.setattr(
            tempfile, 'NamedTemporaryFile', _named_tempfile_func(error_class)
        )
        with pytest.raises(error_class):
            io.imread(image_url)
