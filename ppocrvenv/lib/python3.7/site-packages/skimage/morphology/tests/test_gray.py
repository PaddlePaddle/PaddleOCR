import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import TestCase, fetch
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte


class TestMorphology():

    # These expected outputs were generated with skimage v0.12.1
    # using:
    #
    #   from skimage.morphology.tests.test_gray import TestMorphology
    #   import numpy as np
    #   output = TestMorphology()._build_expected_output()
    #   np.savez_compressed('gray_morph_output.npz', **output)

    def _build_expected_output(self):
        funcs = (gray.erosion, gray.dilation, gray.opening, gray.closing,
                 gray.white_tophat, gray.black_tophat)
        footprints_2D = (footprints.square, footprints.diamond,
                         footprints.disk, footprints.star)

        image = img_as_ubyte(transform.downscale_local_mean(
            color.rgb2gray(data.coffee()), (20, 20)))

        output = {}
        for n in range(1, 4):
            for strel in footprints_2D:
                for func in funcs:
                    key = f'{strel.__name__}_{n}_{func.__name__}'
                    output[key] = func(image, strel(n))

        return output

    def test_gray_morphology(self):
        expected = dict(np.load(fetch('data/gray_morph_output.npz')))
        calculated = self._build_expected_output()
        assert_equal(expected, calculated)


class TestEccentricStructuringElements(TestCase):
    def setUp(self):
        self.black_pixel = 255 * np.ones((4, 4), dtype=np.uint8)
        self.black_pixel[1, 1] = 0
        self.white_pixel = 255 - self.black_pixel
        self.footprints = [footprints.square(2), footprints.rectangle(2, 2),
                           footprints.rectangle(2, 1),
                           footprints.rectangle(1, 2)]

    def test_dilate_erode_symmetry(self):
        for s in self.footprints:
            c = gray.erosion(self.black_pixel, s)
            d = gray.dilation(self.white_pixel, s)
            assert np.all(c == (255 - d))

    def test_open_black_pixel(self):
        for s in self.footprints:
            gray_open = gray.opening(self.black_pixel, s)
            assert np.all(gray_open == self.black_pixel)

    def test_close_white_pixel(self):
        for s in self.footprints:
            gray_close = gray.closing(self.white_pixel, s)
            assert np.all(gray_close == self.white_pixel)

    def test_open_white_pixel(self):
        for s in self.footprints:
            assert np.all(gray.opening(self.white_pixel, s) == 0)

    def test_close_black_pixel(self):
        for s in self.footprints:
            assert np.all(gray.closing(self.black_pixel, s) == 255)

    def test_white_tophat_white_pixel(self):
        for s in self.footprints:
            tophat = gray.white_tophat(self.white_pixel, s)
            assert np.all(tophat == self.white_pixel)

    def test_black_tophat_black_pixel(self):
        for s in self.footprints:
            tophat = gray.black_tophat(self.black_pixel, s)
            assert np.all(tophat == (255 - self.black_pixel))

    def test_white_tophat_black_pixel(self):
        for s in self.footprints:
            tophat = gray.white_tophat(self.black_pixel, s)
            assert np.all(tophat == 0)

    def test_black_tophat_white_pixel(self):
        for s in self.footprints:
            tophat = gray.black_tophat(self.white_pixel, s)
            assert np.all(tophat == 0)


gray_functions = [gray.erosion, gray.dilation,
                  gray.opening, gray.closing,
                  gray.white_tophat, gray.black_tophat]


@pytest.mark.parametrize("function", gray_functions)
def test_default_footprint(function):
    strel = footprints.diamond(radius=1)
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    im_expected = function(image, strel)
    im_test = function(image)
    assert_array_equal(im_expected, im_test)


def test_3d_fallback_default_footprint():
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    opened = gray.opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = np.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    assert_array_equal(opened, image_expected)


gray_3d_fallback_functions = [gray.closing, gray.opening]


@pytest.mark.parametrize("function", gray_3d_fallback_functions)
def test_3d_fallback_cube_footprint(function):
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    cube = np.ones((3, 3, 3), dtype=np.uint8)

    new_image = function(image, cube)
    assert_array_equal(new_image, image)


def test_3d_fallback_white_tophat():
    image = np.zeros((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 1
    image[3, 2:5, 2:5] = 1
    image[4, 3:5, 3:5] = 1

    with expected_warnings([r'operator.*deprecated|\A\Z']):
        new_image = gray.white_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    with expected_warnings([r'operator.*deprecated|\A\Z']):
        image_expected = ndi.white_tophat(
            image.view(dtype=np.uint8), footprint=footprint)
    assert_array_equal(new_image, image_expected)


def test_3d_fallback_black_tophat():
    image = np.ones((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 0
    image[3, 2:5, 2:5] = 0
    image[4, 3:5, 3:5] = 0

    with expected_warnings([r'operator.*deprecated|\A\Z']):
        new_image = gray.black_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    with expected_warnings([r'operator.*deprecated|\A\Z']):
        image_expected = ndi.black_tophat(
            image.view(dtype=np.uint8), footprint=footprint)
    assert_array_equal(new_image, image_expected)


def test_2d_ndimage_equivalence():
    image = np.zeros((9, 9), np.uint8)
    image[2:-2, 2:-2] = 128
    image[3:-3, 3:-3] = 196
    image[4, 4] = 255

    opened = gray.opening(image)
    closed = gray.closing(image)

    footprint = ndi.generate_binary_structure(2, 1)
    ndimage_opened = ndi.grey_opening(image, footprint=footprint)
    ndimage_closed = ndi.grey_closing(image, footprint=footprint)

    assert_array_equal(opened, ndimage_opened)
    assert_array_equal(closed, ndimage_closed)


# float test images
im = np.array([[ 0.55,  0.72,  0.6 ,  0.54,  0.42],
               [ 0.65,  0.44,  0.89,  0.96,  0.38],
               [ 0.79,  0.53,  0.57,  0.93,  0.07],
               [ 0.09,  0.02,  0.83,  0.78,  0.87],
               [ 0.98,  0.8 ,  0.46,  0.78,  0.12]])

eroded = np.array([[ 0.55,  0.44,  0.54,  0.42,  0.38],
                   [ 0.44,  0.44,  0.44,  0.38,  0.07],
                   [ 0.09,  0.02,  0.53,  0.07,  0.07],
                   [ 0.02,  0.02,  0.02,  0.78,  0.07],
                   [ 0.09,  0.02,  0.46,  0.12,  0.12]])

dilated = np.array([[ 0.72,  0.72,  0.89,  0.96,  0.54],
                    [ 0.79,  0.89,  0.96,  0.96,  0.96],
                    [ 0.79,  0.79,  0.93,  0.96,  0.93],
                    [ 0.98,  0.83,  0.83,  0.93,  0.87],
                    [ 0.98,  0.98,  0.83,  0.78,  0.87]])

opened = np.array([[ 0.55,  0.55,  0.54,  0.54,  0.42],
                   [ 0.55,  0.44,  0.54,  0.44,  0.38],
                   [ 0.44,  0.53,  0.53,  0.78,  0.07],
                   [ 0.09,  0.02,  0.78,  0.78,  0.78],
                   [ 0.09,  0.46,  0.46,  0.78,  0.12]])

closed = np.array([[ 0.72,  0.72,  0.72,  0.54,  0.54],
                   [ 0.72,  0.72,  0.89,  0.96,  0.54],
                   [ 0.79,  0.79,  0.79,  0.93,  0.87],
                   [ 0.79,  0.79,  0.83,  0.78,  0.87],
                   [ 0.98,  0.83,  0.78,  0.78,  0.78]])


def test_float():
    assert_allclose(gray.erosion(im), eroded)
    assert_allclose(gray.dilation(im), dilated)
    assert_allclose(gray.opening(im), opened)
    assert_allclose(gray.closing(im), closed)


def test_uint16():
    im16, eroded16, dilated16, opened16, closed16 = (
        map(img_as_uint, [im, eroded, dilated, opened, closed]))
    assert_allclose(gray.erosion(im16), eroded16)
    assert_allclose(gray.dilation(im16), dilated16)
    assert_allclose(gray.opening(im16), opened16)
    assert_allclose(gray.closing(im16), closed16)


def test_discontiguous_out_array():
    image = np.array([[5, 6, 2],
                      [7, 2, 2],
                      [3, 5, 1]], np.uint8)
    out_array_big = np.zeros((5, 5), np.uint8)
    out_array = out_array_big[::2, ::2]
    expected_dilation = np.array([[7, 0, 6, 0, 6],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 7, 0, 2],
                                  [0, 0, 0, 0, 0],
                                  [7, 0, 5, 0, 5]], np.uint8)
    expected_erosion = np.array([[5, 0, 2, 0, 2],
                                 [0, 0, 0, 0, 0],
                                 [2, 0, 2, 0, 1],
                                 [0, 0, 0, 0, 0],
                                 [3, 0, 1, 0, 1]], np.uint8)
    gray.dilation(image, out=out_array)
    assert_array_equal(out_array_big, expected_dilation)
    gray.erosion(image, out=out_array)
    assert_array_equal(out_array_big, expected_erosion)


def test_1d_erosion():
    image = np.array([1, 2, 3, 2, 1])
    expected = np.array([1, 1, 2, 1, 1])
    eroded = gray.erosion(image)
    assert_array_equal(eroded, expected)


def test_deprecated_import():
    msg = "Importing from skimage.morphology.grey is deprecated."
    with expected_warnings([msg + r"|\A\Z"]):
        from skimage.morphology.grey import erosion


@pytest.mark.parametrize(
    'function', ['erosion', 'dilation', 'closing', 'opening', 'white_tophat',
                 'black_tophat'],
)
def test_selem_kwarg_deprecation(function):
    with expected_warnings(["`selem` is a deprecated argument name"]):
        getattr(gray, function)(np.zeros((4, 4)), selem=np.ones((3, 3)))
