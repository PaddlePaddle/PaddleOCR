import itertools

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.spatial import distance

from skimage._shared._warnings import expected_warnings
from skimage.metrics import hausdorff_distance, hausdorff_pair


def test_hausdorff_empty():
    empty = np.zeros((0, 2), dtype=bool)
    non_empty = np.zeros((3, 2), dtype=bool)
    assert hausdorff_distance(empty, non_empty) == 0.
    with expected_warnings(["One or both of the images is empty"]):
        assert_array_equal(hausdorff_pair(empty, non_empty), [(), ()])
    assert hausdorff_distance(non_empty, empty) == 0.
    with expected_warnings(["One or both of the images is empty"]):
        assert_array_equal(hausdorff_pair(non_empty, empty), [(), ()])
    assert hausdorff_distance(empty, non_empty) == 0.
    with expected_warnings(["One or both of the images is empty"]):
        assert_array_equal(hausdorff_pair(empty, non_empty), [(), ()])


def test_hausdorff_simple():
    points_a = (3, 0)
    points_b = (6, 0)
    shape = (7, 1)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True
    dist = np.sqrt(sum((ca - cb) ** 2
                       for ca, cb in zip(points_a, points_b)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a,
                                                            points_b))


@pytest.mark.parametrize(
    "points_a, points_b",
    itertools.product([(0, 0), (3, 0), (1, 4), (4, 1)], repeat=2)
)
def test_hausdorff_region_single(points_a, points_b):
    shape = (5, 5)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    dist = np.sqrt(sum((ca - cb) ** 2
                       for ca, cb in zip(points_a, points_b)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a,
                                                            points_b))


@pytest.mark.parametrize(
    "points_a, points_b",
    itertools.product([(5, 4), (4, 5), (3, 4), (4, 3)],
                      [(6, 4), (2, 6), (2, 4), (4, 0)])
)
def test_hausdorff_region_different_points(points_a, points_b):
    shape = (7, 7)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    dist = np.sqrt(sum((ca - cb) ** 2
                       for ca, cb in zip(points_a, points_b)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a,
                                                            points_b))


def test_gallery():
    shape = (60, 60)

    # Create a diamond-like shape where the four corners form the 1st set
    # of points
    x_diamond = 30
    y_diamond = 30
    r = 10

    plt_x = [0, 1, 0, -1]
    plt_y = [1, 0, -1, 0]

    set_ax = [(x_diamond + r * x) for x in plt_x]
    set_ay = [(y_diamond + r * y) for y in plt_y]

    # Create a kite-like shape where the four corners form the 2nd set of
    # points
    x_kite = 30
    y_kite = 30
    x_r = 15
    y_r = 20

    set_bx = [(x_kite + x_r * x) for x in plt_x]
    set_by = [(y_kite + y_r * y) for y in plt_y]

    # Set up the data to compute the Hausdorff distance
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)

    for x, y in zip(set_ax, set_ay):
        coords_a[(x, y)] = True

    for x, y in zip(set_bx, set_by):
        coords_b[(x, y)] = True

    # Test the Hausdorff function on the coordinates
    # Should return 10, the distance between the furthest tip of the kite and
    # its closest point on the diamond, which is the furthest someone can make
    # you travel to encounter your nearest neighboring point on the other set.
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), 10.)

    # There are two pairs of points ((30, 20), (30, 10) or (30, 40), (30, 50)),
    # that are Hausdorff distance apart. This tests for either of them.
    hd_points = hausdorff_pair(coords_a, coords_b)
    assert np.equal(hd_points, ((30, 20), (30, 10))).all() or \
           np.equal(hd_points, ((30, 40), (30, 50))).all()


@pytest.mark.parametrize(
    "points_a, points_b",
    itertools.product([(0, 0, 1), (0, 1, 0), (1, 0, 0)],
                      [(0, 0, 2), (0, 2, 0), (2, 0, 0)])
)
def test_3d_hausdorff_region(points_a, points_b):
    shape = (3, 3, 3)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True

    dist = np.sqrt(sum((ca - cb) ** 2
                       for ca, cb in zip(points_a, points_b)))
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), dist)
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a,
                                                            points_b))


def test_hausdorff_metrics_match():
    # Test that Hausdorff distance is the Euclidean distance between Hausdorff
    # pair
    points_a = (3, 0)
    points_b = (6, 0)
    shape = (7, 1)
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    coords_a[points_a] = True
    coords_b[points_b] = True
    assert_array_equal(hausdorff_pair(coords_a, coords_b), (points_a,
                                                            points_b))
    euclidean_distance = distance.euclidean(points_a, points_b)
    assert_almost_equal(euclidean_distance, hausdorff_distance(coords_a,
                                                               coords_b))
