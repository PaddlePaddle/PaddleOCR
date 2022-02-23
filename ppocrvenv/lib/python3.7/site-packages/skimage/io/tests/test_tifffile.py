import pathlib
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from skimage._shared.testing import fetch
from skimage.io import imread, imsave, reset_plugins, use_plugin


def setup():
    use_plugin('tifffile')
    np.random.seed(0)


def teardown():
    reset_plugins()


def test_imread_uint16():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


def test_imread_uint16_big_endian():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


def test_imread_multipage_rgb_tif():
    img = imread(fetch('data/multipage_rgb.tif'))
    assert img.shape == (2, 10, 10, 3), img.shape


def test_tifffile_kwarg_passthrough ():
    img = imread(fetch('data/multipage.tif'), key=[1],
                 multifile=False, multifile_close=True, fastij=True,
                 is_ome=True)
    assert img.shape == (15, 10), img.shape


def test_imread_handle():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    with open(fetch('data/chessboard_GRAY_U16.tif'), 'rb') as fh:
        img = imread(fh)
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip(self, dtype, x, use_pathlib=False):
        f = NamedTemporaryFile(suffix='.tif')
        fname = f.name
        f.close()
        if use_pathlib:
            fname = pathlib.Path(fname)
        imsave(fname, x, check_contrast=False)
        y = imread(fname)
        assert_array_equal(x, y)

    shapes = ((10, 10), (10, 10, 3), (10, 10, 4))
    dtypes = (np.uint8, np.uint16, np.float32, np.int16, np.float64)

    @pytest.mark.parametrize("shape", shapes)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("use_pathlib", [False, True])
    def test_imsave_roundtrip(self, shape, dtype, use_pathlib):
        x = np.random.rand(*shape)

        if not np.issubdtype(dtype, np.floating):
            x = (x * np.iinfo(dtype).max).astype(dtype)
        else:
            x = x.astype(dtype)
        self.roundtrip(dtype, x, use_pathlib)
