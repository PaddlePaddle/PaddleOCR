import numpy as np
import skimage.graph.mcp as mcp

from skimage._shared.testing import (assert_array_equal, assert_almost_equal,
                                     parametrize)
from skimage._shared._warnings import expected_warnings


np.random.seed(0)
a = np.ones((8, 8), dtype=np.float32)
a[1:-1, 1] = 0
a[1, 1:-1] = 0

warning_optional = r'|\A\Z'


def test_basic():
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(a, fully_connected=True)
    costs, traceback = m.find_costs([(1, 6)])
    return_path = m.traceback((7, 2))
    assert_array_equal(costs,
                       [[1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                        [1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                        [1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
                        [1.,  0.,  1.,  2.,  2.,  2.,  2.,  2.],
                        [1.,  0.,  1.,  2.,  3.,  3.,  3.,  3.],
                        [1.,  0.,  1.,  2.,  3.,  4.,  4.,  4.],
                        [1.,  0.,  1.,  2.,  3.,  4.,  5.,  5.],
                        [1.,  1.,  1.,  2.,  3.,  4.,  5.,  6.]])

    assert_array_equal(return_path,
                       [(1, 6),
                        (1, 5),
                        (1, 4),
                        (1, 3),
                        (1, 2),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 2)])


def test_neg_inf():
    expected_costs = np.where(a == 1, np.inf, 0)
    expected_path = [(1, 6),
                     (1, 5),
                     (1, 4),
                     (1, 3),
                     (1, 2),
                     (2, 1),
                     (3, 1),
                     (4, 1),
                     (5, 1),
                     (6, 1)]
    test_neg = np.where(a == 1, -1, 0)
    test_inf = np.where(a == 1, np.inf, 0)
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(test_neg, fully_connected=True)
    costs, traceback = m.find_costs([(1, 6)])
    return_path = m.traceback((6, 1))
    assert_array_equal(costs, expected_costs)
    assert_array_equal(return_path, expected_path)
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(test_inf, fully_connected=True)
    costs, traceback = m.find_costs([(1, 6)])
    return_path = m.traceback((6, 1))
    assert_array_equal(costs, expected_costs)
    assert_array_equal(return_path, expected_path)


def test_route():
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        return_path, cost = mcp.route_through_array(a, (1, 6), (7, 2),
                                                    geometric=True)
    assert_almost_equal(cost, np.sqrt(2) / 2)
    assert_array_equal(return_path,
                       [(1, 6),
                        (1, 5),
                        (1, 4),
                        (1, 3),
                        (1, 2),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 2)])


def test_no_diagonal():
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(a, fully_connected=False)
    costs, traceback = m.find_costs([(1, 6)])
    return_path = m.traceback((7, 2))
    assert_array_equal(costs,
                       [[2.,  1.,  1.,  1.,  1.,  1.,  1.,  2.],
                        [1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                        [1.,  0.,  1.,  1.,  1.,  1.,  1.,  2.],
                        [1.,  0.,  1.,  2.,  2.,  2.,  2.,  3.],
                        [1.,  0.,  1.,  2.,  3.,  3.,  3.,  4.],
                        [1.,  0.,  1.,  2.,  3.,  4.,  4.,  5.],
                        [1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.],
                        [2.,  1.,  2.,  3.,  4.,  5.,  6.,  7.]])
    assert_array_equal(return_path,
                       [(1, 6),
                        (1, 5),
                        (1, 4),
                        (1, 3),
                        (1, 2),
                        (1, 1),
                        (2, 1),
                        (3, 1),
                        (4, 1),
                        (5, 1),
                        (6, 1),
                        (7, 1),
                        (7, 2)])


def test_offsets():
    offsets = [(1, i) for i in range(10)] + [(1, -i) for i in range(1, 10)]
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(a, offsets=offsets)
    costs, traceback = m.find_costs([(1, 6)])
    assert_array_equal(traceback,
                       [[-2, -2, -2, -2, -2, -2, -2, -2],
                        [-2, -2, -2, -2, -2, -2, -1, -2],
                        [15, 14, 13, 12, 11, 10,  0,  1],
                        [10,  0,  1,  2,  3,  4,  5,  6],
                        [10,  0,  1,  2,  3,  4,  5,  6],
                        [10,  0,  1,  2,  3,  4,  5,  6],
                        [10,  0,  1,  2,  3,  4,  5,  6],
                        [10,  0,  1,  2,  3,  4,  5,  6]])
    assert hasattr(m, "offsets")
    assert_array_equal(offsets, m.offsets)


@parametrize("shape", [(100, 100), (5, 8, 13, 17)] * 5)
def test_crashing(shape):
    _test_random(shape)


def _test_random(shape):
    # Just tests for crashing -- not for correctness.
    a = np.random.rand(*shape).astype(np.float32)
    starts = [[0] * len(shape), [-1] * len(shape),
              (np.random.rand(len(shape)) * shape).astype(int)]
    ends = [(np.random.rand(len(shape)) * shape).astype(int)
            for i in range(4)]
    with expected_warnings(['Upgrading NumPy' + warning_optional]):
        m = mcp.MCP(a, fully_connected=True)
    costs, offsets = m.find_costs(starts)
    for point in [(np.random.rand(len(shape)) * shape).astype(int)
                  for i in range(4)]:
        m.traceback(point)
    m._reset()
    m.find_costs(starts, ends)
    for end in ends:
        m.traceback(end)
    return a, costs, offsets
