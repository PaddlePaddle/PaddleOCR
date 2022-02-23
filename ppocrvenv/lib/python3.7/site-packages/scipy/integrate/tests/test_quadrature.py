import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
                           assert_, suppress_warnings)

from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
                             cumulative_trapezoid, cumtrapz, trapz, trapezoid,
                             quad, simpson, simps, fixed_quad, AccuracyWarning)


class TestFixedQuad:
    def test_scalar(self):
        n = 4
        func = lambda x: x**(2*n - 1)
        expected = 1/(2*n)
        got, _ = fixed_quad(func, 0, 1, n=n)
        # quadrature exact for this input
        assert_allclose(got, expected, rtol=1e-12)

    def test_vector(self):
        n = 4
        p = np.arange(1, 2*n)
        func = lambda x: x**p[:,None]
        expected = 1/(p + 1)
        got, _ = fixed_quad(func, 0, 1, n=n)
        assert_allclose(got, expected, rtol=1e-12)


class TestQuadrature:
    def quad(self, x, a, b, args):
        raise NotImplementedError

    def test_quadrature(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        val, err = quadrature(myfunc, 0, pi, (2, 1.8))
        table_val = 0.30614353532540296487
        assert_almost_equal(val, table_val, decimal=7)

    def test_quadrature_rtol(self):
        def myfunc(x, n, z):       # Bessel function integrand
            return 1e90 * cos(n*x-z*sin(x))/pi
        val, err = quadrature(myfunc, 0, pi, (2, 1.8), rtol=1e-10)
        table_val = 1e90 * 0.30614353532540296487
        assert_allclose(val, table_val, rtol=1e-10)

    def test_quadrature_miniter(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        table_val = 0.30614353532540296487
        for miniter in [5, 52]:
            val, err = quadrature(myfunc, 0, pi, (2, 1.8), miniter=miniter)
            assert_almost_equal(val, table_val, decimal=7)
            assert_(err < 1.0)

    def test_quadrature_single_args(self):
        def myfunc(x, n):
            return 1e90 * cos(n*x-1.8*sin(x))/pi
        val, err = quadrature(myfunc, 0, pi, args=2, rtol=1e-10)
        table_val = 1e90 * 0.30614353532540296487
        assert_allclose(val, table_val, rtol=1e-10)

    def test_romberg(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return cos(n*x-z*sin(x))/pi
        val = romberg(myfunc, 0, pi, args=(2, 1.8))
        table_val = 0.30614353532540296487
        assert_almost_equal(val, table_val, decimal=7)

    def test_romberg_rtol(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):       # Bessel function integrand
            return 1e19*cos(n*x-z*sin(x))/pi
        val = romberg(myfunc, 0, pi, args=(2, 1.8), rtol=1e-10)
        table_val = 1e19*0.30614353532540296487
        assert_allclose(val, table_val, rtol=1e-10)

    def test_romb(self):
        assert_equal(romb(np.arange(17)), 128)

    def test_romb_gh_3731(self):
        # Check that romb makes maximal use of data points
        x = np.arange(2**4+1)
        y = np.cos(0.2*x)
        val = romb(y)
        val2, err = quad(lambda x: np.cos(0.2*x), x.min(), x.max())
        assert_allclose(val, val2, rtol=1e-8, atol=0)

        # should be equal to romb with 2**k+1 samples
        with suppress_warnings() as sup:
            sup.filter(AccuracyWarning, "divmax .4. exceeded")
            val3 = romberg(lambda x: np.cos(0.2*x), x.min(), x.max(), divmax=4)
        assert_allclose(val, val3, rtol=1e-12, atol=0)

    def test_non_dtype(self):
        # Check that we work fine with functions returning float
        import math
        valmath = romberg(math.sin, 0, 1)
        expected_val = 0.45969769413185085
        assert_almost_equal(valmath, expected_val, decimal=7)

    def test_newton_cotes(self):
        """Test the first few degrees, for evenly spaced points."""
        n = 1
        wts, errcoff = newton_cotes(n, 1)
        assert_equal(wts, n*np.array([0.5, 0.5]))
        assert_almost_equal(errcoff, -n**3/12.0)

        n = 2
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([1.0, 4.0, 1.0])/6.0)
        assert_almost_equal(errcoff, -n**5/2880.0)

        n = 3
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([1.0, 3.0, 3.0, 1.0])/8.0)
        assert_almost_equal(errcoff, -n**5/6480.0)

        n = 4
        wts, errcoff = newton_cotes(n, 1)
        assert_almost_equal(wts, n*np.array([7.0, 32.0, 12.0, 32.0, 7.0])/90.0)
        assert_almost_equal(errcoff, -n**7/1935360.0)

    def test_newton_cotes2(self):
        """Test newton_cotes with points that are not evenly spaced."""

        x = np.array([0.0, 1.5, 2.0])
        y = x**2
        wts, errcoff = newton_cotes(x)
        exact_integral = 8.0/3
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)

        x = np.array([0.0, 1.4, 2.1, 3.0])
        y = x**2
        wts, errcoff = newton_cotes(x)
        exact_integral = 9.0
        numeric_integral = np.dot(wts, y)
        assert_almost_equal(numeric_integral, exact_integral)

    def test_simpson(self):
        y = np.arange(17)
        assert_equal(simpson(y), 128)
        assert_equal(simpson(y, dx=0.5), 64)
        assert_equal(simpson(y, x=np.linspace(0, 4, 17)), 32)

        y = np.arange(4)
        x = 2**y
        assert_equal(simpson(y, x=x, even='avg'), 13.875)
        assert_equal(simpson(y, x=x, even='first'), 13.75)
        assert_equal(simpson(y, x=x, even='last'), 14)

    def test_simps(self):
        # Basic coverage test for the alias
        y = np.arange(4)
        x = 2**y
        assert_equal(simpson(y, x=x, dx=0.5, even='first'),
                     simps(y, x=x, dx=0.5, even='first'))


class TestCumulative_trapezoid:
    def test_1d(self):
        x = np.linspace(-2, 2, num=5)
        y = x
        y_int = cumulative_trapezoid(y, x, initial=0)
        y_expected = [0., -1.5, -2., -1.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, x, initial=None)
        assert_allclose(y_int, y_expected[1:])

    def test_y_nd_x_nd(self):
        x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        y = x
        y_int = cumulative_trapezoid(y, x, initial=0)
        y_expected = np.array([[[0., 0.5, 2., 4.5],
                                [0., 4.5, 10., 16.5]],
                               [[0., 8.5, 18., 28.5],
                                [0., 12.5, 26., 40.5]],
                               [[0., 16.5, 34., 52.5],
                                [0., 20.5, 42., 64.5]]])

        assert_allclose(y_int, y_expected)

        # Try with all axes
        shapes = [(2, 2, 4), (3, 1, 4), (3, 2, 3)]
        for axis, shape in zip([0, 1, 2], shapes):
            y_int = cumulative_trapezoid(y, x, initial=3.45, axis=axis)
            assert_equal(y_int.shape, (3, 2, 4))
            y_int = cumulative_trapezoid(y, x, initial=None, axis=axis)
            assert_equal(y_int.shape, shape)

    def test_y_nd_x_1d(self):
        y = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        x = np.arange(4)**2
        # Try with all axes
        ys_expected = (
            np.array([[[4., 5., 6., 7.],
                       [8., 9., 10., 11.]],
                      [[40., 44., 48., 52.],
                       [56., 60., 64., 68.]]]),
            np.array([[[2., 3., 4., 5.]],
                      [[10., 11., 12., 13.]],
                      [[18., 19., 20., 21.]]]),
            np.array([[[0.5, 5., 17.5],
                       [4.5, 21., 53.5]],
                      [[8.5, 37., 89.5],
                       [12.5, 53., 125.5]],
                      [[16.5, 69., 161.5],
                       [20.5, 85., 197.5]]]))

        for axis, y_expected in zip([0, 1, 2], ys_expected):
            y_int = cumulative_trapezoid(y, x=x[:y.shape[axis]], axis=axis,
                                         initial=None)
            assert_allclose(y_int, y_expected)

    def test_x_none(self):
        y = np.linspace(-2, 2, num=5)

        y_int = cumulative_trapezoid(y)
        y_expected = [-1.5, -2., -1.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, initial=1.23)
        y_expected = [1.23, -1.5, -2., -1.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, dx=3)
        y_expected = [-4.5, -6., -4.5, 0.]
        assert_allclose(y_int, y_expected)

        y_int = cumulative_trapezoid(y, dx=3, initial=1.23)
        y_expected = [1.23, -4.5, -6., -4.5, 0.]
        assert_allclose(y_int, y_expected)

    def test_cumtrapz(self):
        # Basic coverage test for the alias
        x = np.arange(3 * 2 * 4).reshape(3, 2, 4)
        y = x
        assert_allclose(cumulative_trapezoid(y, x, dx=0.5, axis=0, initial=0),
                        cumtrapz(y, x, dx=0.5, axis=0, initial=0),
                        rtol=1e-14)


class TestTrapezoid():
    """This function is tested in NumPy more extensive, just do some
    basic due diligence here."""
    def test_trapezoid(self):
        y = np.arange(17)
        assert_equal(trapezoid(y), 128)
        assert_equal(trapezoid(y, dx=0.5), 64)
        assert_equal(trapezoid(y, x=np.linspace(0, 4, 17)), 32)

        y = np.arange(4)
        x = 2**y
        assert_equal(trapezoid(y, x=x, dx=0.1), 13.5)

    def test_trapz(self):
        # Basic coverage test for the alias
        y = np.arange(4)
        x = 2**y
        assert_equal(trapezoid(y, x=x, dx=0.5, axis=0),
                     trapz(y, x=x, dx=0.5, axis=0))

