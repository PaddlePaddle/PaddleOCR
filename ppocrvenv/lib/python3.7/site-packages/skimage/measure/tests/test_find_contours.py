import numpy as np
from skimage.measure import find_contours

from skimage._shared.testing import assert_array_equal
import pytest


a = np.ones((8, 8), dtype=np.float32)
a[1:-1, 1] = 0
a[1, 1:-1] = 0

x, y = np.mgrid[-1:1:5j, -1:1:5j]
r = np.sqrt(x**2 + y**2)


def test_binary():
    ref = [[6.0, 1.5],
           [5.0, 1.5],
           [4.0, 1.5],
           [3.0, 1.5],
           [2.0, 1.5],
           [1.5, 2.0],
           [1.5, 3.0],
           [1.5, 4.0],
           [1.5, 5.0],
           [1.5, 6.0],
           [1.0, 6.5],
           [0.5, 6.0],
           [0.5, 5.0],
           [0.5, 4.0],
           [0.5, 3.0],
           [0.5, 2.0],
           [0.5, 1.0],
           [1.0, 0.5],
           [2.0, 0.5],
           [3.0, 0.5],
           [4.0, 0.5],
           [5.0, 0.5],
           [6.0, 0.5],
           [6.5, 1.0],
           [6.0, 1.5]]

    contours = find_contours(a, 0.5, positive_orientation='high')
    assert len(contours) == 1
    assert_array_equal(contours[0][::-1], ref)


# target contour for mask tests
mask_contour = [
    [6.0, 0.5],
    [5.0, 0.5],
    [4.0, 0.5],
    [3.0, 0.5],
    [2.0, 0.5],
    [1.0, 0.5],
    [0.5, 1.0],
    [0.5, 2.0],
    [0.5, 3.0],
    [0.5, 4.0],
    [0.5, 5.0],
    [0.5, 6.0],
    [1.0, 6.5],
    [1.5, 6.0],
    [1.5, 5.0],
    [1.5, 4.0],
    [1.5, 3.0],
    [1.5, 2.0],
    [2.0, 1.5],
    [3.0, 1.5],
    [4.0, 1.5],
    [5.0, 1.5],
    [6.0, 1.5],
]

mask = np.ones((8, 8), dtype=bool)
# Some missing data that should result in a hole in the contour:
mask[7, 0:3] = False


@pytest.mark.parametrize("level", [0.5, None])
def test_nodata(level):
    # Test missing data via NaNs in input array
    b = np.copy(a)
    b[~mask] = np.nan
    contours = find_contours(b, level, positive_orientation='high')
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


@pytest.mark.parametrize("level", [0.5, None])
def test_mask(level):
    # Test missing data via explicit masking
    contours = find_contours(a, level, positive_orientation='high', mask=mask)
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


@pytest.mark.parametrize("level", [0, None])
def test_mask_shape(level):
    bad_mask = np.ones((8, 7), dtype=bool)
    with pytest.raises(ValueError, match='shape'):
        find_contours(a, level, mask=bad_mask)


@pytest.mark.parametrize("level", [0, None])
def test_mask_dtype(level):
    bad_mask = np.ones((8, 8), dtype=np.uint8)
    with pytest.raises(TypeError, match='binary'):
        find_contours(a, level, mask=bad_mask)


def test_float():
    contours = find_contours(r, 0.5)
    assert len(contours) == 1
    assert_array_equal(contours[0], [[2., 3.],
                                     [1., 2.],
                                     [2., 1.],
                                     [3., 2.],
                                     [2., 3.]])


@pytest.mark.parametrize("level", [0.5, None])
def test_memory_order(level):
    contours = find_contours(np.ascontiguousarray(r), level)
    assert len(contours) == 1

    contours = find_contours(np.asfortranarray(r), level)
    assert len(contours) == 1


def test_invalid_input():
    with pytest.raises(ValueError):
        find_contours(r, 0.5, 'foo', 'bar')
    with pytest.raises(ValueError):
        find_contours(r[..., None], 0.5)


def test_level_default():
    # image with range [0.9, 0.91]
    image = np.random.random((100, 100)) * 0.01 + 0.9
    contours = find_contours(image)  # use default level
    # many contours should be found
    assert len(contours) > 1


@pytest.mark.parametrize("image", [
    [[0.13680, 0.11220, 0.0, 0.0, 0.0, 0.19417, 0.19417, 0.33701],
     [0.0, 0.15140, 0.10267, 0.0, np.nan, 0.14908, 0.18158, 0.19178],
     [0.0, 0.06949, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01860],
     [0.0, 0.06949, 0.0, 0.17852, 0.08469, 0.02135, 0.08198, np.nan],
     [0.0, 0.08244, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
     [0.12342, 0.21330, 0.0, np.nan, 0.01301, 0.04335, 0.0, 0.0]],

    [[0.08, -0.03, -0.17, -0.08, 0.24, 0.06, 0.17, -0.02],
     [0.12, 0., np.nan, 0.24, 0., -0.53, 0.26, 0.16],
     [0.39, 0., 0., 0., 0., -0.02, -0.3, 0.01],
     [0.28, -0.04, -0.03, 0.16, 0.12, 0.01, -0.87, 0.16],
     [0.26, 0.08, 0.08, 0.08, 0.12, 0.13, 0.11, 0.19],
     [0.27, 0.24, 0., 0.25, 0.32, 0.19, 0.26, 0.22]],

    [[-0.18, np.nan, np.nan, 0.22, -0.14, -0.23, -0.2, -0.17, -0.19, -0.24],
     [0., np.nan, np.nan, np.nan, -0.1, -0.24, -0.15, -0.02, -0.09, -0.21],
     [0.43, 0.19, np.nan, np.nan, -0.01, -0.2, -0.22, -0.18, -0.16, -0.07],
     [0.23, 0., np.nan, -0.06, -0.07, -0.21, -0.24, -0.25, -0.23, -0.13],
     [-0.05, -0.11, 0., 0.1, -0.19, -0.23, -0.23, -0.18, -0.19, -0.16],
     [-0.19, -0.05, 0.13, -0.08, -0.22, -0.23, -0.26, -0.15, -0.12, -0.13],
     [-0.2, -0.11, -0.11, -0.24, -0.29, -0.27, -0.35, -0.36, -0.27, -0.13],
     [-0.28, -0.33, -0.31, -0.36, -0.39, -0.37, -0.38, -0.32, -0.34, -0.2],
     [-0.28, -0.33, -0.39, -0.4, -0.42, -0.38, -0.35, -0.39, -0.35, -0.34],
     [-0.38, -0.35, -0.41, -0.42, -0.39, -0.36, -0.34, -0.36, -0.28, -0.34]]
])
def test_keyerror_fix(image):
    """Failing samples from issue #4830
    """
    find_contours(np.array(image, np.float32), 0)
