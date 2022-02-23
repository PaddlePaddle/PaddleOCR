import numpy as np

from skimage._shared import testing
from skimage._shared.testing import assert_equal

from skimage.util._label import label_points


def test_label_points_coords_dimension():
    coords, output_shape = np.array([[1, 2], [3, 4]]), (5, 5, 2)
    with testing.raises(ValueError):
        label_points(coords, output_shape)


def test_label_points_coords_range():
    coords, output_shape = np.array([[0, 0],
                                     [5, 5]]), (5, 5)
    with testing.raises(IndexError):
        label_points(coords, output_shape)


def test_label_points_coords_negative():
    coords, output_shape = np.array([[-1, 0],
                                     [5, 5]]), (5, 5)
    with testing.raises(ValueError):
        label_points(coords, output_shape)


def test_label_points_two_dimensional_output():
    coords, output_shape = np.array([[0, 0],
                                     [1, 1],
                                     [2, 2],
                                     [3, 3],
                                     [4, 4]]), (5, 5)
    mask = label_points(coords, output_shape)
    assert_equal(mask, np.array([[1, 0, 0, 0, 0],
                                 [0, 2, 0, 0, 0],
                                 [0, 0, 3, 0, 0],
                                 [0, 0, 0, 4, 0],
                                 [0, 0, 0, 0, 5]]))


def test_label_points_multi_dimensional_output():
    coords, output_shape = np.array([[0, 0, 0],
                                     [1, 1, 1],
                                     [2, 2, 2],
                                     [3, 3, 0],
                                     [4, 4, 1]]), (5, 5, 3)
    mask = label_points(coords, output_shape)
    result = np.array([
        [
            [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        ],
        [
            [0, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        ],
        [
            [0, 0, 0], [0, 0, 0], [0, 0, 3], [0, 0, 0], [0, 0, 0]
        ],
        [
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [4, 0, 0], [0, 0, 0]
        ],
        [
            [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 5, 0]
        ]
    ])
    assert_equal(mask, result)
