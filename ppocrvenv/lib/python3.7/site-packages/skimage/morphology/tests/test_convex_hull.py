import numpy as np
from skimage.morphology import convex_hull_image, convex_hull_object
from skimage.morphology._convex_hull import possible_hull

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
from skimage._shared._warnings import expected_warnings


def test_basic():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    expected = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    assert_array_equal(convex_hull_image(image), expected)


def test_empty_image():
    image = np.zeros((6, 6), dtype=bool)
    with expected_warnings(['entirely zero']):
        assert_array_equal(convex_hull_image(image), image)


def test_qhull_offset_example():
    nonzeros = (([1367, 1368, 1368, 1368, 1369, 1369, 1369, 1369, 1369, 1370,
                  1370, 1370, 1370, 1370, 1370, 1370, 1371, 1371, 1371, 1371,
                  1371, 1371, 1371, 1371, 1371, 1372, 1372, 1372, 1372, 1372,
                  1372, 1372, 1372, 1372, 1373, 1373, 1373, 1373, 1373, 1373,
                  1373, 1373, 1373, 1374, 1374, 1374, 1374, 1374, 1374, 1374,
                  1375, 1375, 1375, 1375, 1375, 1376, 1376, 1376, 1377, 1372]),
                ([151, 150, 151, 152, 149, 150, 151, 152, 153, 148, 149, 150,
                 151, 152, 153, 154, 147, 148, 149, 150, 151, 152, 153, 154,
                 155, 146, 147, 148, 149, 150, 151, 152, 153, 154, 146, 147,
                 148, 149, 150, 151, 152, 153, 154, 147, 148, 149, 150, 151,
                  152, 153, 148, 149, 150, 151, 152, 149, 150, 151, 150, 155]))
    image = np.zeros((1392, 1040), dtype=bool)
    image[nonzeros] = True
    expected = image.copy()
    assert_array_equal(convex_hull_image(image), expected)


def test_pathological_qhull_example():
    image = np.array(
                [[0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0]], dtype=bool)
    expected = np.array(
                [[0, 0, 0, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 0, 0]], dtype=bool)
    assert_array_equal(convex_hull_image(image), expected)


def test_possible_hull():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    expected = np.array([[1, 4],
                         [2, 3],
                         [3, 2],
                         [4, 1],
                         [4, 1],
                         [3, 2],
                         [2, 3],
                         [1, 4],
                         [2, 5],
                         [3, 6],
                         [4, 7],
                         [2, 5],
                         [3, 6],
                         [4, 7],
                         [4, 2],
                         [4, 3],
                         [4, 4],
                         [4, 5],
                         [4, 6]])

    ph = possible_hull(image)
    assert_array_equal(ph, expected)


def test_object():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 1, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    expected_conn_1 = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 0, 1],
         [1, 1, 1, 0, 0, 0, 0, 1, 0],
         [1, 1, 0, 0, 0, 0, 1, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    assert_array_equal(convex_hull_object(image, connectivity=1),
                       expected_conn_1)

    expected_conn_2 = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 0, 1, 1, 1],
         [1, 1, 0, 0, 0, 0, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    assert_array_equal(convex_hull_object(image, connectivity=2),
                       expected_conn_2)

    with testing.raises(ValueError):
        convex_hull_object(image, connectivity=3)

    out = convex_hull_object(image, connectivity=1)
    assert_array_equal(out, expected_conn_1)


def test_non_c_contiguous():
    # 2D Fortran-contiguous
    image = np.ones((2, 2), order='F', dtype=bool)
    assert_array_equal(convex_hull_image(image), image)
    # 3D Fortran-contiguous
    image = np.ones((2, 2, 2), order='F', dtype=bool)
    assert_array_equal(convex_hull_image(image), image)
    # 3D non-contiguous
    image = np.transpose(np.ones((2, 2, 2), dtype=bool), [0, 2, 1])
    assert_array_equal(convex_hull_image(image), image)


@testing.fixture
def images2d3d():
    from ...measure.tests.test_regionprops import SAMPLE as image
    image3d = np.stack((image, image, image))
    return image, image3d


def test_consistent_2d_3d_hulls(images2d3d):
    image, image3d = images2d3d
    chimage = convex_hull_image(image)
    chimage[8, 0] = True  # correct for single point exactly on hull edge
    chimage3d = convex_hull_image(image3d)
    assert_array_equal(chimage3d[1], chimage)


def test_few_points():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    image3d = np.stack([image, image, image])
    with testing.assert_warns(UserWarning):
        chimage3d = convex_hull_image(image3d)
        assert_array_equal(chimage3d, np.zeros(image3d.shape, dtype=bool))
