import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
                           assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)

from scipy.ndimage import convolve1d

from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder


def check_polyder(p, m, expected):
    dp = _polyder(p, m)
    assert_array_equal(dp, expected)


def test_polyder():
    cases = [
        ([5], 0, [5]),
        ([5], 1, [0]),
        ([3, 2, 1], 0, [3, 2, 1]),
        ([3, 2, 1], 1, [6, 2]),
        ([3, 2, 1], 2, [6]),
        ([3, 2, 1], 3, [0]),
        ([[3, 2, 1], [5, 6, 7]], 0, [[3, 2, 1], [5, 6, 7]]),
        ([[3, 2, 1], [5, 6, 7]], 1, [[6, 2], [10, 6]]),
        ([[3, 2, 1], [5, 6, 7]], 2, [[6], [10]]),
        ([[3, 2, 1], [5, 6, 7]], 3, [[0], [0]]),
    ]
    for p, m, expected in cases:
        check_polyder(np.array(p).T, m, np.array(expected).T)


#--------------------------------------------------------------------
# savgol_coeffs tests
#--------------------------------------------------------------------

def alt_sg_coeffs(window_length, polyorder, pos):
    """This is an alternative implementation of the SG coefficients.

    It uses numpy.polyfit and numpy.polyval. The results should be
    equivalent to those of savgol_coeffs(), but this implementation
    is slower.

    window_length should be odd.

    """
    if pos is None:
        pos = window_length // 2
    t = np.arange(window_length)
    unit = (t == pos).astype(int)
    h = np.polyval(np.polyfit(t, unit, polyorder), t)
    return h


def test_sg_coeffs_trivial():
    # Test a trivial case of savgol_coeffs: polyorder = window_length - 1
    h = savgol_coeffs(1, 0)
    assert_allclose(h, [1])

    h = savgol_coeffs(3, 2)
    assert_allclose(h, [0, 1, 0], atol=1e-10)

    h = savgol_coeffs(5, 4)
    assert_allclose(h, [0, 0, 1, 0, 0], atol=1e-10)

    h = savgol_coeffs(5, 4, pos=1)
    assert_allclose(h, [0, 0, 0, 1, 0], atol=1e-10)

    h = savgol_coeffs(5, 4, pos=1, use='dot')
    assert_allclose(h, [0, 1, 0, 0, 0], atol=1e-10)


def compare_coeffs_to_alt(window_length, order):
    # For the given window_length and order, compare the results
    # of savgol_coeffs and alt_sg_coeffs for pos from 0 to window_length - 1.
    # Also include pos=None.
    for pos in [None] + list(range(window_length)):
        h1 = savgol_coeffs(window_length, order, pos=pos, use='dot')
        h2 = alt_sg_coeffs(window_length, order, pos=pos)
        assert_allclose(h1, h2, atol=1e-10,
                        err_msg=("window_length = %d, order = %d, pos = %s" %
                                 (window_length, order, pos)))


def test_sg_coeffs_compare():
    # Compare savgol_coeffs() to alt_sg_coeffs().
    for window_length in range(1, 8, 2):
        for order in range(window_length):
            compare_coeffs_to_alt(window_length, order)


def test_sg_coeffs_exact():
    polyorder = 4
    window_length = 9
    halflen = window_length // 2

    x = np.linspace(0, 21, 43)
    delta = x[1] - x[0]

    # The data is a cubic polynomial.  We'll use an order 4
    # SG filter, so the filtered values should equal the input data
    # (except within half window_length of the edges).
    y = 0.5 * x ** 3 - x
    h = savgol_coeffs(window_length, polyorder)
    y0 = convolve1d(y, h)
    assert_allclose(y0[halflen:-halflen], y[halflen:-halflen])

    # Check the same input, but use deriv=1.  dy is the exact result.
    dy = 1.5 * x ** 2 - 1
    h = savgol_coeffs(window_length, polyorder, deriv=1, delta=delta)
    y1 = convolve1d(y, h)
    assert_allclose(y1[halflen:-halflen], dy[halflen:-halflen])

    # Check the same input, but use deriv=2. d2y is the exact result.
    d2y = 3.0 * x
    h = savgol_coeffs(window_length, polyorder, deriv=2, delta=delta)
    y2 = convolve1d(y, h)
    assert_allclose(y2[halflen:-halflen], d2y[halflen:-halflen])


def test_sg_coeffs_deriv():
    # The data in `x` is a sampled parabola, so using savgol_coeffs with an
    # order 2 or higher polynomial should give exact results.
    i = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
    x = i ** 2 / 4
    dx = i / 2
    d2x = np.full_like(i, 0.5)
    for pos in range(x.size):
        coeffs0 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot')
        assert_allclose(coeffs0.dot(x), x[pos], atol=1e-10)
        coeffs1 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=1)
        assert_allclose(coeffs1.dot(x), dx[pos], atol=1e-10)
        coeffs2 = savgol_coeffs(5, 3, pos=pos, delta=2.0, use='dot', deriv=2)
        assert_allclose(coeffs2.dot(x), d2x[pos], atol=1e-10)


def test_sg_coeffs_deriv_gt_polyorder():
    """
    If deriv > polyorder, the coefficients should be all 0.
    This is a regression test for a bug where, e.g.,
        savgol_coeffs(5, polyorder=1, deriv=2)
    raised an error.
    """
    coeffs = savgol_coeffs(5, polyorder=1, deriv=2)
    assert_array_equal(coeffs, np.zeros(5))
    coeffs = savgol_coeffs(7, polyorder=4, deriv=6)
    assert_array_equal(coeffs, np.zeros(7))


def test_sg_coeffs_large():
    # Test that for large values of window_length and polyorder the array of
    # coefficients returned is symmetric. The aim is to ensure that
    # no potential numeric overflow occurs.
    coeffs0 = savgol_coeffs(31, 9)
    assert_array_almost_equal(coeffs0, coeffs0[::-1])
    coeffs1 = savgol_coeffs(31, 9, deriv=1)
    assert_array_almost_equal(coeffs1, -coeffs1[::-1])


#--------------------------------------------------------------------
# savgol_filter tests
#--------------------------------------------------------------------


def test_sg_filter_trivial():
    """ Test some trivial edge cases for savgol_filter()."""
    x = np.array([1.0])
    y = savgol_filter(x, 1, 0)
    assert_equal(y, [1.0])

    # Input is a single value. With a window length of 3 and polyorder 1,
    # the value in y is from the straight-line fit of (-1,0), (0,3) and
    # (1, 0) at 0. This is just the average of the three values, hence 1.0.
    x = np.array([3.0])
    y = savgol_filter(x, 3, 1, mode='constant')
    assert_almost_equal(y, [1.0], decimal=15)

    x = np.array([3.0])
    y = savgol_filter(x, 3, 1, mode='nearest')
    assert_almost_equal(y, [3.0], decimal=15)

    x = np.array([1.0] * 3)
    y = savgol_filter(x, 3, 1, mode='wrap')
    assert_almost_equal(y, [1.0, 1.0, 1.0], decimal=15)


def test_sg_filter_basic():
    # Some basic test cases for savgol_filter().
    x = np.array([1.0, 2.0, 1.0])
    y = savgol_filter(x, 3, 1, mode='constant')
    assert_allclose(y, [1.0, 4.0 / 3, 1.0])

    y = savgol_filter(x, 3, 1, mode='mirror')
    assert_allclose(y, [5.0 / 3, 4.0 / 3, 5.0 / 3])

    y = savgol_filter(x, 3, 1, mode='wrap')
    assert_allclose(y, [4.0 / 3, 4.0 / 3, 4.0 / 3])


def test_sg_filter_2d():
    x = np.array([[1.0, 2.0, 1.0],
                  [2.0, 4.0, 2.0]])
    expected = np.array([[1.0, 4.0 / 3, 1.0],
                         [2.0, 8.0 / 3, 2.0]])
    y = savgol_filter(x, 3, 1, mode='constant')
    assert_allclose(y, expected)

    y = savgol_filter(x.T, 3, 1, mode='constant', axis=0)
    assert_allclose(y, expected.T)


def test_sg_filter_interp_edges():
    # Another test with low degree polynomial data, for which we can easily
    # give the exact results. In this test, we use mode='interp', so
    # savgol_filter should match the exact solution for the entire data set,
    # including the edges.
    t = np.linspace(-5, 5, 21)
    delta = t[1] - t[0]
    # Polynomial test data.
    x = np.array([t,
                  3 * t ** 2,
                  t ** 3 - t])
    dx = np.array([np.ones_like(t),
                   6 * t,
                   3 * t ** 2 - 1.0])
    d2x = np.array([np.zeros_like(t),
                    np.full_like(t, 6),
                    6 * t])

    window_length = 7

    y = savgol_filter(x, window_length, 3, axis=-1, mode='interp')
    assert_allclose(y, x, atol=1e-12)

    y1 = savgol_filter(x, window_length, 3, axis=-1, mode='interp',
                       deriv=1, delta=delta)
    assert_allclose(y1, dx, atol=1e-12)

    y2 = savgol_filter(x, window_length, 3, axis=-1, mode='interp',
                       deriv=2, delta=delta)
    assert_allclose(y2, d2x, atol=1e-12)

    # Transpose everything, and test again with axis=0.

    x = x.T
    dx = dx.T
    d2x = d2x.T

    y = savgol_filter(x, window_length, 3, axis=0, mode='interp')
    assert_allclose(y, x, atol=1e-12)

    y1 = savgol_filter(x, window_length, 3, axis=0, mode='interp',
                       deriv=1, delta=delta)
    assert_allclose(y1, dx, atol=1e-12)

    y2 = savgol_filter(x, window_length, 3, axis=0, mode='interp',
                       deriv=2, delta=delta)
    assert_allclose(y2, d2x, atol=1e-12)


def test_sg_filter_interp_edges_3d():
    # Test mode='interp' with a 3-D array.
    t = np.linspace(-5, 5, 21)
    delta = t[1] - t[0]
    x1 = np.array([t, -t])
    x2 = np.array([t ** 2, 3 * t ** 2 + 5])
    x3 = np.array([t ** 3, 2 * t ** 3 + t ** 2 - 0.5 * t])
    dx1 = np.array([np.ones_like(t), -np.ones_like(t)])
    dx2 = np.array([2 * t, 6 * t])
    dx3 = np.array([3 * t ** 2, 6 * t ** 2 + 2 * t - 0.5])

    # z has shape (3, 2, 21)
    z = np.array([x1, x2, x3])
    dz = np.array([dx1, dx2, dx3])

    y = savgol_filter(z, 7, 3, axis=-1, mode='interp', delta=delta)
    assert_allclose(y, z, atol=1e-10)

    dy = savgol_filter(z, 7, 3, axis=-1, mode='interp', deriv=1, delta=delta)
    assert_allclose(dy, dz, atol=1e-10)

    # z has shape (3, 21, 2)
    z = np.array([x1.T, x2.T, x3.T])
    dz = np.array([dx1.T, dx2.T, dx3.T])

    y = savgol_filter(z, 7, 3, axis=1, mode='interp', delta=delta)
    assert_allclose(y, z, atol=1e-10)

    dy = savgol_filter(z, 7, 3, axis=1, mode='interp', deriv=1, delta=delta)
    assert_allclose(dy, dz, atol=1e-10)

    # z has shape (21, 3, 2)
    z = z.swapaxes(0, 1).copy()
    dz = dz.swapaxes(0, 1).copy()

    y = savgol_filter(z, 7, 3, axis=0, mode='interp', delta=delta)
    assert_allclose(y, z, atol=1e-10)

    dy = savgol_filter(z, 7, 3, axis=0, mode='interp', deriv=1, delta=delta)
    assert_allclose(dy, dz, atol=1e-10)
