import numpy as np
import skimage.graph.spath as spath

from skimage._shared.testing import assert_equal, assert_array_equal


def test_basic():
    x = np.array([[1, 1, 3],
                  [0, 2, 0],
                  [4, 3, 1]])
    path, cost = spath.shortest_path(x)
    assert_array_equal(path, [0, 0, 1])
    assert_equal(cost, 1)


def test_reach():
    x = np.array([[1, 1, 3],
                  [0, 2, 0],
                  [4, 3, 1]])
    path, cost = spath.shortest_path(x, reach=2)
    assert_array_equal(path, [0, 0, 2])
    assert_equal(cost, 0)


def test_non_square():
    x = np.array([[1, 1, 1, 1, 5, 5, 5],
                  [5, 0, 0, 5, 9, 1, 1],
                  [0, 5, 1, 0, 5, 5, 0],
                  [6, 1, 1, 5, 0, 0, 1]])
    path, cost = spath.shortest_path(x, reach=2)
    assert_array_equal(path, [2, 1, 1, 2, 3, 3, 2])
    assert_equal(cost, 0)
