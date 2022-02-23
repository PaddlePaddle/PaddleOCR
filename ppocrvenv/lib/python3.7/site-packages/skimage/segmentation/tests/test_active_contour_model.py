import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose

from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_periodic_reference(dtype):
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    img_smooth = gaussian(img, 3, preserve_range=False).astype(dtype, copy=False)
    snake = active_contour(img_smooth, init, alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001)
    assert snake.dtype == _supported_float_type(dtype)
    refr = [98, 99, 100, 101, 102, 103, 104, 105, 106, 108]
    refc = [299, 298, 298, 298, 298, 297, 297, 296, 296, 295]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_fixed_reference(dtype):
    img = data.text()
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    image_smooth = gaussian(img, 1, preserve_range=False).astype(dtype, copy=False)
    snake = active_contour(image_smooth, init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    assert snake.dtype == _supported_float_type(dtype)
    refr = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    refc = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_free_reference(dtype):
    img = data.text()
    r = np.linspace(70, 40, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    img_smooth = gaussian(img, 3, preserve_range=False).astype(dtype, copy=False)
    snake = active_contour(img_smooth, init, boundary_condition='free',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    assert snake.dtype == _supported_float_type(dtype)
    refr = [76, 76, 75, 74, 73, 72, 71, 70, 69, 69]
    refc = [10, 13, 16, 19, 23, 26, 29, 32, 36, 39]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_RGB(dtype):
    img = gaussian(data.text(), 1, preserve_range=False)
    imgR = np.zeros((img.shape[0], img.shape[1], 3), dtype=dtype)
    imgG = np.zeros((img.shape[0], img.shape[1], 3), dtype=dtype)
    imgRGB = np.zeros((img.shape[0], img.shape[1], 3), dtype=dtype)
    imgR[:, :, 0] = img
    imgG[:, :, 1] = img
    imgRGB[:, :, :] = img[:, :, None]
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    snake = active_contour(imgR, init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    float_dtype = _supported_float_type(dtype)
    assert snake.dtype == float_dtype
    refr = [136, 135, 134, 133, 132, 131, 129, 128, 127, 125]
    refc = [5, 9, 13, 17, 21, 25, 30, 34, 38, 42]
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)
    snake = active_contour(imgG, init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)
    assert snake.dtype == float_dtype
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)
    snake = active_contour(imgRGB, init, boundary_condition='fixed',
                           alpha=0.1, beta=1.0, w_line=-5/3., w_edge=0,
                           gamma=0.1)
    assert snake.dtype == float_dtype
    assert_equal(np.array(snake[:10, 0], dtype=np.int32), refr)
    assert_equal(np.array(snake[:10, 1], dtype=np.int32), refc)


def test_end_points():
    img = data.astronaut()
    img = rgb2gray(img)
    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3), init,
                           boundary_condition='periodic', alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001, max_num_iter=100)
    assert np.sum(np.abs(snake[0, :]-snake[-1, :])) < 2
    snake = active_contour(gaussian(img, 3), init,
                           boundary_condition='free', alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001, max_num_iter=100)
    assert np.sum(np.abs(snake[0, :]-snake[-1, :])) > 2
    snake = active_contour(gaussian(img, 3), init,
                           boundary_condition='fixed', alpha=0.015, beta=10,
                           w_line=0, w_edge=1, gamma=0.001, max_num_iter=100)
    assert_allclose(snake[0, :], [r[0], c[0]], atol=1e-5)


def test_bad_input():
    img = np.zeros((10, 10))
    r = np.linspace(136, 50, 100)
    c = np.linspace(5, 424, 100)
    init = np.array([r, c]).T
    with pytest.raises(ValueError):
        active_contour(img, init, boundary_condition='wrong')
    with pytest.raises(ValueError):
        active_contour(img, init, max_num_iter=-15)
    with expected_warnings(["`max_iterations` is a deprecated argument"]):
        active_contour(img, init, max_iterations=15)


def test_coord_raises():
    img = rgb2gray(data.astronaut())
    s = np.linspace(0, 2*np.pi, 400)
    x = 100 + 100*np.sin(s)
    y = 220 + 100*np.cos(s)
    init = np.array([x, y]).T
    # coordinates='xy' is not valid
    with pytest.raises(ValueError):
        active_contour(gaussian(img, 3), init,
                       boundary_condition='periodic', alpha=0.015,
                       beta=10, w_line=0, w_edge=1, gamma=0.001,
                       max_num_iter=100, coordinates='xy')

    # coordinates=None is not valid
    with pytest.raises(ValueError):
        active_contour(gaussian(img, 3), init,
                       boundary_condition='periodic', alpha=0.015,
                       beta=10, w_line=0, w_edge=1, gamma=0.001,
                       max_num_iter=100, coordinates=None)
