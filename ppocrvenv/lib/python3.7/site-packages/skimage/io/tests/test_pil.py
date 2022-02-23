import os
import numpy as np
from io import BytesIO
from tempfile import NamedTemporaryFile

from ... import img_as_float
from .. import imread, imsave, use_plugin, reset_plugins

from PIL import Image
from .._plugins.pil_plugin import (
    pil_to_ndarray, ndarray_to_pil, _palette_is_grayscale)
from ...color import rgb2lab

from skimage._shared import testing
from skimage._shared.testing import (mono_check, color_check,
                                     assert_equal, assert_array_equal,
                                     assert_array_almost_equal,
                                     assert_allclose, fetch)
from skimage._shared._warnings import expected_warnings
from skimage._shared._tempfile import temporary_file

from skimage.metrics import structural_similarity


def setup():
    use_plugin('pil')


def teardown():
    reset_plugins()


def setup_module(self):
    """The effect of the `plugin.use` call may be overridden by later imports.
    Call `use_plugin` directly before the tests to ensure that PIL is used.
    """
    try:
        use_plugin('pil')
    except ImportError:
        pass


def test_png_round_trip():
    f = NamedTemporaryFile(suffix='.png')
    fname = f.name
    f.close()
    I = np.eye(3)
    imsave(fname, I)
    Ip = img_as_float(imread(fname))
    os.remove(fname)
    assert np.sum(np.abs(Ip-I)) < 1e-3


def test_imread_as_gray():
    img = imread(fetch('data/color.png'), as_gray=True)
    assert img.ndim == 2
    assert img.dtype == np.float64
    img = imread(fetch('data/camera.png'), as_gray=True)
    # check that conversion does not happen for a gray image
    assert np.sctype2char(img.dtype) in np.typecodes['AllInteger']


def test_imread_separate_channels():
    # Test that imread returns RGBA values contiguously even when they are
    # stored in separate planes.
    x = np.random.rand(3, 16, 8)
    f = NamedTemporaryFile(suffix='.tif')
    fname = f.name
    f.close()
    imsave(fname, x)
    img = imread(fname)
    os.remove(fname)
    assert img.shape == (16, 8, 3), img.shape


def test_imread_multipage_rgb_tif():
    img = imread(fetch('data/multipage_rgb.tif'))
    assert img.shape == (2, 10, 10, 3), img.shape


def test_imread_palette():
    img = imread(fetch('data/palette_gray.png'))
    assert img.ndim == 2
    img = imread(fetch('data/palette_color.png'))
    assert img.ndim == 3


def test_imread_index_png_with_alpha():
    # The file `foo3x5x4indexed.png` was created with this array
    # (3x5 is (height)x(width)):
    dfoo = np.array([[[127, 0, 255, 255],
                      [127, 0, 255, 255],
                      [127, 0, 255, 255],
                      [127, 0, 255, 255],
                      [127, 0, 255, 255]],
                     [[192, 192, 255, 0],
                      [192, 192, 255, 0],
                      [0, 0, 255, 0],
                      [0, 0, 255, 0],
                      [0, 0, 255, 0]],
                     [[0, 31, 255, 255],
                      [0, 31, 255, 255],
                      [0, 31, 255, 255],
                      [0, 31, 255, 255],
                      [0, 31, 255, 255]]], dtype=np.uint8)
    img = imread(fetch('data/foo3x5x4indexed.png'))
    assert_array_equal(img, dfoo)


def test_palette_is_gray():
    gray = Image.open(fetch('data/palette_gray.png'))
    assert _palette_is_grayscale(gray)
    color = Image.open(fetch('data/palette_color.png'))
    assert not _palette_is_grayscale(color)


def test_bilevel():
    expected = np.zeros((10, 10))
    expected[::2] = 255

    img = imread(fetch('data/checker_bilevel.png'))
    assert_array_equal(img, expected)


def test_imread_uint16():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16.tif'))
    assert np.issubdtype(img.dtype, np.uint16)
    assert_array_almost_equal(img, expected)


def test_imread_truncated_jpg():
    with testing.raises(IOError):
        imread(fetch('data/truncated.jpg'))


def test_jpg_quality_arg():
    chessboard = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    with temporary_file(suffix='.jpg') as jpg:
        imsave(jpg, chessboard, quality=95)
        im = imread(jpg)
        sim = structural_similarity(
            chessboard, im,
            data_range=chessboard.max() - chessboard.min())
        assert sim > 0.99


def test_imread_uint16_big_endian():
    expected = np.load(fetch('data/chessboard_GRAY_U8.npy'))
    img = imread(fetch('data/chessboard_GRAY_U16B.tif'))
    assert img.dtype == np.uint16
    assert_array_almost_equal(img, expected)


class TestSave:
    def roundtrip_file(self, x):
        with temporary_file(suffix='.png') as fname:
            imsave(fname, x)
            y = imread(fname)
            return y

    def roundtrip_pil_image(self, x):
        pil_image = ndarray_to_pil(x)
        y = pil_to_ndarray(pil_image)
        return y

    def verify_roundtrip(self, dtype, x, y, scaling=1):
        assert_array_almost_equal((x * scaling).astype(np.int32), y)

    def verify_imsave_roundtrip(self, roundtrip_function):
        for shape in [(10, 10), (10, 10, 3), (10, 10, 4)]:
            for dtype in (np.uint8, np.uint16, np.float32, np.float64):
                x = np.ones(shape, dtype=dtype) * np.random.rand(*shape)

                if np.issubdtype(dtype, np.floating):
                    yield (self.verify_roundtrip, dtype, x,
                           roundtrip_function(x), 255)
                else:
                    x = (x * 255).astype(dtype)
                    yield (self.verify_roundtrip, dtype, x,
                           roundtrip_function(x))

    def test_imsave_roundtrip_file(self):
        self.verify_imsave_roundtrip(self.roundtrip_file)

    def test_imsave_roundtrip_pil_image(self):
        self.verify_imsave_roundtrip(self.roundtrip_pil_image)


def test_imsave_incorrect_dimension():
    with temporary_file(suffix='.png') as fname:
        with testing.raises(ValueError):
            with expected_warnings([fname + ' is a low contrast image']):
                imsave(fname, np.zeros((2, 3, 3, 1)))
        with testing.raises(ValueError):
            with expected_warnings([fname + ' is a low contrast image']):
                imsave(fname, np.zeros((2, 3, 2)))
        # test that low contrast check is ignored
        with testing.raises(ValueError):
            with expected_warnings([]):
                imsave(fname, np.zeros((2, 3, 2)), check_contrast=False)


def test_imsave_filelike():
    shape = (2, 2)
    image = np.zeros(shape)
    s = BytesIO()

    # save to file-like object
    with expected_warnings(['is a low contrast image']):
        imsave(s, image)

    # read from file-like object
    s.seek(0)
    out = imread(s)
    assert_equal(out.shape, shape)
    assert_allclose(out, image)


def test_imsave_boolean_input():
    shape = (2, 2)
    image = np.eye(*shape, dtype=bool)
    s = BytesIO()

    # save to file-like object
    with expected_warnings(
            ['is a boolean image: setting True to 255 and False to 0']):
        imsave(s, image)

    # read from file-like object
    s.seek(0)
    out = imread(s)
    assert_equal(out.shape, shape)
    assert_allclose(out.astype(bool), image)


def test_imexport_imimport():
    shape = (2, 2)
    image = np.zeros(shape)
    pil_image = ndarray_to_pil(image)
    out = pil_to_ndarray(pil_image)
    assert_equal(out.shape, shape)


def test_all_color():
    with expected_warnings(['.* is a boolean image']):
        color_check('pil')
    with expected_warnings(['.* is a boolean image']):
        color_check('pil', 'bmp')


def test_all_mono():
    with expected_warnings(['.* is a boolean image']):
        mono_check('pil')


def test_multi_page_gif():
    img = imread(fetch('data/no_time_for_that_tiny.gif'))
    assert img.shape == (24, 25, 14, 3), img.shape
    img2 = imread(fetch('data/no_time_for_that_tiny.gif'),
                  img_num=5)
    assert img2.shape == (25, 14, 3)
    assert_allclose(img[5], img2)


def test_cmyk():
    ref = imread(fetch('data/color.png'))

    img = Image.open(fetch('data/color.png'))
    img = img.convert('CMYK')

    f = NamedTemporaryFile(suffix='.jpg')
    fname = f.name
    f.close()
    img.save(fname)
    try:
        img.close()
    except AttributeError:  # `close` not available on PIL
        pass

    new = imread(fname)

    ref_lab = rgb2lab(ref)
    new_lab = rgb2lab(new)

    for i in range(3):
        newi = np.ascontiguousarray(new_lab[:, :, i])
        refi = np.ascontiguousarray(ref_lab[:, :, i])
        sim = structural_similarity(refi, newi,
                                    data_range=refi.max() - refi.min())
        assert sim > 0.99


def test_extreme_palette():
    img = imread(fetch('data/green_palette.png'))
    assert_equal(img.ndim, 3)
