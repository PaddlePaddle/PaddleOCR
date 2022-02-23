import numpy as np

from skimage._shared.testing import assert_array_equal
from skimage._shared import testing

from skimage.util.compare import compare_images


def test_compate_images_ValueError_shape():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.zeros((10, 1), dtype=np.uint8)
    with testing.raises(ValueError):
        compare_images(img1, img2)


def test_compare_images_diff():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img1[3:8, 3:8] = 255
    img2 = np.zeros_like(img1)
    img2[3:8, 0:8] = 255
    expected_result = np.zeros_like(img1, dtype=np.float64)
    expected_result[3:8, 0:3] = 1
    result = compare_images(img1, img2, method='diff')
    assert_array_equal(result, expected_result)


def test_compare_images_blend():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img1[3:8, 3:8] = 255
    img2 = np.zeros_like(img1)
    img2[3:8, 0:8] = 255
    expected_result = np.zeros_like(img1, dtype=np.float64)
    expected_result[3:8, 3:8] = 1
    expected_result[3:8, 0:3] = 0.5
    result = compare_images(img1, img2, method='blend')
    assert_array_equal(result, expected_result)


def test_compare_images_checkerboard_default():
    img1 = np.zeros((2**4, 2**4), dtype=np.uint8)
    img2 = np.full(img1.shape, fill_value=255, dtype=np.uint8)
    res = compare_images(img1, img2, method='checkerboard')
    exp_row1 = np.array([0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.])
    exp_row2 = np.array([1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.])
    for i in (0, 1, 4, 5, 8, 9, 12, 13):
        assert_array_equal(res[i, :], exp_row1)
    for i in (2, 3, 6, 7, 10, 11, 14, 15):
        assert_array_equal(res[i, :], exp_row2)


def test_compare_images_checkerboard_tuple():
    img1 = np.zeros((2**4, 2**4), dtype=np.uint8)
    img2 = np.full(img1.shape, fill_value=255, dtype=np.uint8)
    res = compare_images(img1, img2, method='checkerboard', n_tiles=(4, 8))
    exp_row1 = np.array(
        [0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1.]
    )
    exp_row2 = np.array(
        [1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.]
    )
    for i in (0, 1, 2, 3, 8, 9, 10, 11):
        assert_array_equal(res[i, :], exp_row1)
    for i in (4, 5, 6, 7, 12, 13, 14, 15):
        assert_array_equal(res[i, :], exp_row2)
