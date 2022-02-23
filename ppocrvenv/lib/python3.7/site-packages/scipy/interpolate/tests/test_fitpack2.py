# Created by Pearu Peterson, June 2003
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
        assert_array_almost_equal, assert_allclose, suppress_warnings)
from pytest import raises as assert_raises

from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate.fitpack import bisplrep, bisplev
from scipy.interpolate.fitpack2 import (UnivariateSpline,
        LSQUnivariateSpline, InterpolatedUnivariateSpline,
        LSQBivariateSpline, SmoothBivariateSpline, RectBivariateSpline,
        LSQSphereBivariateSpline, SmoothSphereBivariateSpline,
        RectSphereBivariateSpline)


class TestUnivariateSpline:
    def test_linear_constant(self):
        x = [1,2,3]
        y = [3,3,3]
        lut = UnivariateSpline(x,y,k=1)
        assert_array_almost_equal(lut.get_knots(),[1,3])
        assert_array_almost_equal(lut.get_coeffs(),[3,3])
        assert_almost_equal(lut.get_residual(),0.0)
        assert_array_almost_equal(lut([1,1.5,2]),[3,3,3])

    def test_preserve_shape(self):
        x = [1, 2, 3]
        y = [0, 2, 4]
        lut = UnivariateSpline(x, y, k=1)
        arg = 2
        assert_equal(shape(arg), shape(lut(arg)))
        assert_equal(shape(arg), shape(lut(arg, nu=1)))
        arg = [1.5, 2, 2.5]
        assert_equal(shape(arg), shape(lut(arg)))
        assert_equal(shape(arg), shape(lut(arg, nu=1)))

    def test_linear_1d(self):
        x = [1,2,3]
        y = [0,2,4]
        lut = UnivariateSpline(x,y,k=1)
        assert_array_almost_equal(lut.get_knots(),[1,3])
        assert_array_almost_equal(lut.get_coeffs(),[0,4])
        assert_almost_equal(lut.get_residual(),0.0)
        assert_array_almost_equal(lut([1,1.5,2]),[0,1,2])

    def test_subclassing(self):
        # See #731

        class ZeroSpline(UnivariateSpline):
            def __call__(self, x):
                return 0*array(x)

        sp = ZeroSpline([1,2,3,4,5], [3,2,3,2,3], k=2)
        assert_array_equal(sp([1.5, 2.5]), [0., 0.])

    def test_empty_input(self):
        # Test whether empty input returns an empty output. Ticket 1014
        x = [1,3,5,7,9]
        y = [0,4,9,12,21]
        spl = UnivariateSpline(x, y, k=3)
        assert_array_equal(spl([]), array([]))

    def test_resize_regression(self):
        """Regression test for #1375."""
        x = [-1., -0.65016502, -0.58856235, -0.26903553, -0.17370892,
             -0.10011001, 0., 0.10011001, 0.17370892, 0.26903553, 0.58856235,
             0.65016502, 1.]
        y = [1.,0.62928599, 0.5797223, 0.39965815, 0.36322694, 0.3508061,
             0.35214793, 0.3508061, 0.36322694, 0.39965815, 0.5797223,
             0.62928599, 1.]
        w = [1.00000000e+12, 6.88875973e+02, 4.89314737e+02, 4.26864807e+02,
             6.07746770e+02, 4.51341444e+02, 3.17480210e+02, 4.51341444e+02,
             6.07746770e+02, 4.26864807e+02, 4.89314737e+02, 6.88875973e+02,
             1.00000000e+12]
        spl = UnivariateSpline(x=x, y=y, w=w, s=None)
        desired = array([0.35100374, 0.51715855, 0.87789547, 0.98719344])
        assert_allclose(spl([0.1, 0.5, 0.9, 0.99]), desired, atol=5e-4)

    def test_out_of_range_regression(self):
        # Test different extrapolation modes. See ticket 3557
        x = np.arange(5, dtype=float)
        y = x**3

        xp = linspace(-8, 13, 100)
        xp_zeros = xp.copy()
        xp_zeros[np.logical_or(xp_zeros < 0., xp_zeros > 4.)] = 0
        xp_clip = xp.copy()
        xp_clip[xp_clip < x[0]] = x[0]
        xp_clip[xp_clip > x[-1]] = x[-1]

        for cls in [UnivariateSpline, InterpolatedUnivariateSpline]:
            spl = cls(x=x, y=y)
            for ext in [0, 'extrapolate']:
                assert_allclose(spl(xp, ext=ext), xp**3, atol=1e-16)
                assert_allclose(cls(x, y, ext=ext)(xp), xp**3, atol=1e-16)
            for ext in [1, 'zeros']:
                assert_allclose(spl(xp, ext=ext), xp_zeros**3, atol=1e-16)
                assert_allclose(cls(x, y, ext=ext)(xp), xp_zeros**3, atol=1e-16)
            for ext in [2, 'raise']:
                assert_raises(ValueError, spl, xp, **dict(ext=ext))
            for ext in [3, 'const']:
                assert_allclose(spl(xp, ext=ext), xp_clip**3, atol=1e-16)
                assert_allclose(cls(x, y, ext=ext)(xp), xp_clip**3, atol=1e-16)

        # also test LSQUnivariateSpline [which needs explicit knots]
        t = spl.get_knots()[3:4]  # interior knots w/ default k=3
        spl = LSQUnivariateSpline(x, y, t)
        assert_allclose(spl(xp, ext=0), xp**3, atol=1e-16)
        assert_allclose(spl(xp, ext=1), xp_zeros**3, atol=1e-16)
        assert_raises(ValueError, spl, xp, **dict(ext=2))
        assert_allclose(spl(xp, ext=3), xp_clip**3, atol=1e-16)

        # also make sure that unknown values for `ext` are caught early
        for ext in [-1, 'unknown']:
            spl = UnivariateSpline(x, y)
            assert_raises(ValueError, spl, xp, **dict(ext=ext))
            assert_raises(ValueError, UnivariateSpline,
                    **dict(x=x, y=y, ext=ext))

    def test_lsq_fpchec(self):
        xs = np.arange(100) * 1.
        ys = np.arange(100) * 1.
        knots = np.linspace(0, 99, 10)
        bbox = (-1, 101)
        assert_raises(ValueError, LSQUnivariateSpline, xs, ys, knots,
                      bbox=bbox)

    def test_derivative_and_antiderivative(self):
        # Thin wrappers to splder/splantider, so light smoke test only.
        x = np.linspace(0, 1, 70)**3
        y = np.cos(x)

        spl = UnivariateSpline(x, y, s=0)
        spl2 = spl.antiderivative(2).derivative(2)
        assert_allclose(spl(0.3), spl2(0.3))

        spl2 = spl.antiderivative(1)
        assert_allclose(spl2(0.6) - spl2(0.2),
                        spl.integral(0.2, 0.6))

    def test_derivative_extrapolation(self):
        # Regression test for gh-10195: for a const-extrapolation spline
        # its derivative evaluates to zero for extrapolation
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 5]
        f = UnivariateSpline(x_values, y_values, ext='const', k=3)

        x = [-1, 0, -0.5, 9, 9.5, 10]
        assert_allclose(f.derivative()(x), 0, atol=1e-15)

    def test_integral_out_of_bounds(self):
        # Regression test for gh-7906: .integral(a, b) is wrong if both
        # a and b are out-of-bounds
        x = np.linspace(0., 1., 7)
        for ext in range(4):
            f = UnivariateSpline(x, x, s=0, ext=ext)
            for (a, b) in [(1, 1), (1, 5), (2, 5),
                           (0, 0), (-2, 0), (-2, -1)]:
                assert_allclose(f.integral(a, b), 0, atol=1e-15)

    def test_nan(self):
        # bail out early if the input data contains nans
        x = np.arange(10, dtype=float)
        y = x**3
        w = np.ones_like(x)
        # also test LSQUnivariateSpline [which needs explicit knots]
        spl = UnivariateSpline(x, y, check_finite=True)
        t = spl.get_knots()[3:4]  # interior knots w/ default k=3
        y_end = y[-1]
        for z in [np.nan, np.inf, -np.inf]:
            y[-1] = z
            assert_raises(ValueError, UnivariateSpline,
                    **dict(x=x, y=y, check_finite=True))
            assert_raises(ValueError, InterpolatedUnivariateSpline,
                    **dict(x=x, y=y, check_finite=True))
            assert_raises(ValueError, LSQUnivariateSpline,
                    **dict(x=x, y=y, t=t, check_finite=True))
            y[-1] = y_end  # check valid y but invalid w
            w[-1] = z
            assert_raises(ValueError, UnivariateSpline,
                    **dict(x=x, y=y, w=w, check_finite=True))
            assert_raises(ValueError, InterpolatedUnivariateSpline,
                    **dict(x=x, y=y, w=w, check_finite=True))
            assert_raises(ValueError, LSQUnivariateSpline,
                    **dict(x=x, y=y, t=t, w=w, check_finite=True))

    def test_strictly_increasing_x(self):
        # Test the x is required to be strictly increasing for
        # UnivariateSpline if s=0 and for InterpolatedUnivariateSpline,
        # but merely increasing for UnivariateSpline if s>0
        # and for LSQUnivariateSpline; see gh-8535
        xx = np.arange(10, dtype=float)
        yy = xx**3
        x = np.arange(10, dtype=float)
        x[1] = x[0]
        y = x**3
        w = np.ones_like(x)
        # also test LSQUnivariateSpline [which needs explicit knots]
        spl = UnivariateSpline(xx, yy, check_finite=True)
        t = spl.get_knots()[3:4]  # interior knots w/ default k=3
        UnivariateSpline(x=x, y=y, w=w, s=1, check_finite=True)
        LSQUnivariateSpline(x=x, y=y, t=t, w=w, check_finite=True)
        assert_raises(ValueError, UnivariateSpline,
                **dict(x=x, y=y, s=0, check_finite=True))
        assert_raises(ValueError, InterpolatedUnivariateSpline,
                **dict(x=x, y=y, check_finite=True))

    def test_increasing_x(self):
        # Test that x is required to be increasing, see gh-8535
        xx = np.arange(10, dtype=float)
        yy = xx**3
        x = np.arange(10, dtype=float)
        x[1] = x[0] - 1.0
        y = x**3
        w = np.ones_like(x)
        # also test LSQUnivariateSpline [which needs explicit knots]
        spl = UnivariateSpline(xx, yy, check_finite=True)
        t = spl.get_knots()[3:4]  # interior knots w/ default k=3
        assert_raises(ValueError, UnivariateSpline,
                **dict(x=x, y=y, check_finite=True))
        assert_raises(ValueError, InterpolatedUnivariateSpline,
                **dict(x=x, y=y, check_finite=True))
        assert_raises(ValueError, LSQUnivariateSpline,
                **dict(x=x, y=y, t=t, w=w, check_finite=True))

    def test_invalid_input_for_univariate_spline(self):

        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            UnivariateSpline(x_values, y_values)
        assert "x and y should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [-1.0, 1.0, 1.0, 1.0]
            UnivariateSpline(x_values, y_values, w=w_values)
        assert "x, y, and w should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            bbox = (-1)
            UnivariateSpline(x_values, y_values, bbox=bbox)
        assert "bbox shape should be (2,)" in str(info.value)

        with assert_raises(ValueError) as info:
            UnivariateSpline(x_values, y_values, k=6)
        assert "k should be 1 <= k <= 5" in str(info.value)

        with assert_raises(ValueError) as info:
            UnivariateSpline(x_values, y_values, s=-1.0)
        assert "s should be s >= 0.0" in str(info.value)

    def test_invalid_input_for_interpolated_univariate_spline(self):

        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            InterpolatedUnivariateSpline(x_values, y_values)
        assert "x and y should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [-1.0, 1.0, 1.0, 1.0]
            InterpolatedUnivariateSpline(x_values, y_values, w=w_values)
        assert "x, y, and w should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            bbox = (-1)
            InterpolatedUnivariateSpline(x_values, y_values, bbox=bbox)
        assert "bbox shape should be (2,)" in str(info.value)

        with assert_raises(ValueError) as info:
            InterpolatedUnivariateSpline(x_values, y_values, k=6)
        assert "k should be 1 <= k <= 5" in str(info.value)

    def test_invalid_input_for_lsq_univariate_spline(self):

        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
        spl = UnivariateSpline(x_values, y_values, check_finite=True)
        t_values = spl.get_knots()[3:4]  # interior knots w/ default k=3

        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5]
            LSQUnivariateSpline(x_values, y_values, t_values)
        assert "x and y should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            x_values = [1, 2, 4, 6, 8.5]
            y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
            w_values = [1.0, 1.0, 1.0, 1.0]
            LSQUnivariateSpline(x_values, y_values, t_values, w=w_values)
        assert "x, y, and w should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            bbox = (100, -100)
            LSQUnivariateSpline(x_values, y_values, t_values, bbox=bbox)
        assert "Interior knots t must satisfy Schoenberg-Whitney conditions" in str(info.value)

        with assert_raises(ValueError) as info:
            bbox = (-1)
            LSQUnivariateSpline(x_values, y_values, t_values, bbox=bbox)
        assert "bbox shape should be (2,)" in str(info.value)

        with assert_raises(ValueError) as info:
            LSQUnivariateSpline(x_values, y_values, t_values, k=6)
        assert "k should be 1 <= k <= 5" in str(info.value)

    def test_array_like_input(self):
        x_values = np.array([1, 2, 4, 6, 8.5])
        y_values = np.array([0.5, 0.8, 1.3, 2.5, 2.8])
        w_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        bbox = np.array([-100, 100])
        # np.array input
        spl1 = UnivariateSpline(x=x_values, y=y_values, w=w_values,
                                bbox=bbox)
        # list input
        spl2 = UnivariateSpline(x=x_values.tolist(), y=y_values.tolist(),
                                w=w_values.tolist(), bbox=bbox.tolist())

        assert_allclose(spl1([0.1, 0.5, 0.9, 0.99]),
                        spl2([0.1, 0.5, 0.9, 0.99]))


class TestLSQBivariateSpline:
    # NOTE: The systems in this test class are rank-deficient
    def test_linear_constant(self):
        x = [1,1,1,2,2,2,3,3,3]
        y = [1,2,3,1,2,3,1,2,3]
        z = [3,3,3,3,3,3,3,3,3]
        s = 0.1
        tx = [1+s,3-s]
        ty = [1+s,3-s]
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            lut = LSQBivariateSpline(x,y,z,tx,ty,kx=1,ky=1)
            assert_equal(len(r), 1)

        assert_almost_equal(lut(2,2), 3.)

    def test_bilinearity(self):
        x = [1,1,1,2,2,2,3,3,3]
        y = [1,2,3,1,2,3,1,2,3]
        z = [0,7,8,3,4,7,1,3,4]
        s = 0.1
        tx = [1+s,3-s]
        ty = [1+s,3-s]
        with suppress_warnings() as sup:
            # This seems to fail (ier=1, see ticket 1642).
            sup.filter(UserWarning, "\nThe coefficients of the spline")
            lut = LSQBivariateSpline(x,y,z,tx,ty,kx=1,ky=1)

        tx, ty = lut.get_knots()
        for xa, xb in zip(tx[:-1], tx[1:]):
            for ya, yb in zip(ty[:-1], ty[1:]):
                for t in [0.1, 0.5, 0.9]:
                    for s in [0.3, 0.4, 0.7]:
                        xp = xa*(1-t) + xb*t
                        yp = ya*(1-s) + yb*s
                        zp = (+ lut(xa, ya)*(1-t)*(1-s)
                              + lut(xb, ya)*t*(1-s)
                              + lut(xa, yb)*(1-t)*s
                              + lut(xb, yb)*t*s)
                        assert_almost_equal(lut(xp,yp), zp)

    def test_integral(self):
        x = [1,1,1,2,2,2,8,8,8]
        y = [1,2,3,1,2,3,1,2,3]
        z = array([0,7,8,3,4,7,1,3,4])

        s = 0.1
        tx = [1+s,3-s]
        ty = [1+s,3-s]
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
            assert_equal(len(r), 1)
        tx, ty = lut.get_knots()
        tz = lut(tx, ty)
        trpz = .25*(diff(tx)[:,None]*diff(ty)[None,:]
                    * (tz[:-1,:-1]+tz[1:,:-1]+tz[:-1,1:]+tz[1:,1:])).sum()

        assert_almost_equal(lut.integral(tx[0], tx[-1], ty[0], ty[-1]),
                            trpz)

    def test_empty_input(self):
        # Test whether empty inputs returns an empty output. Ticket 1014
        x = [1,1,1,2,2,2,3,3,3]
        y = [1,2,3,1,2,3,1,2,3]
        z = [3,3,3,3,3,3,3,3,3]
        s = 0.1
        tx = [1+s,3-s]
        ty = [1+s,3-s]
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
            assert_equal(len(r), 1)

        assert_array_equal(lut([], []), np.zeros((0,0)))
        assert_array_equal(lut([], [], grid=False), np.zeros((0,)))

    def test_invalid_input(self):
        s = 0.1
        tx = [1 + s, 3 - s]
        ty = [1 + s, 3 - s]

        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0, num=10)
            LSQBivariateSpline(x, y, z, tx, ty)
        assert "x, y, and z should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0)
            w = np.linspace(1.0, 10.0, num=20)
            LSQBivariateSpline(x, y, z, tx, ty, w=w)
        assert "x, y, z, and w should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            w = np.linspace(-1.0, 10.0)
            LSQBivariateSpline(x, y, z, tx, ty, w=w)
        assert "w should be positive" in str(info.value)

        with assert_raises(ValueError) as info:
            bbox = (-100, 100, -100)
            LSQBivariateSpline(x, y, z, tx, ty, bbox=bbox)
        assert "bbox shape should be (4,)" in str(info.value)

        with assert_raises(ValueError) as info:
            LSQBivariateSpline(x, y, z, tx, ty, kx=10, ky=10)
        assert "The length of x, y and z should be at least (kx+1) * (ky+1)" in \
               str(info.value)

        with assert_raises(ValueError) as exc_info:
            LSQBivariateSpline(x, y, z, tx, ty, eps=0.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            LSQBivariateSpline(x, y, z, tx, ty, eps=1.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

    def test_array_like_input(self):
        s = 0.1
        tx = np.array([1 + s, 3 - s])
        ty = np.array([1 + s, 3 - s])
        x = np.linspace(1.0, 10.0)
        y = np.linspace(1.0, 10.0)
        z = np.linspace(1.0, 10.0)
        w = np.linspace(1.0, 10.0)
        bbox = np.array([1.0, 10.0, 1.0, 10.0])

        with suppress_warnings() as sup:
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            # np.array input
            spl1 = LSQBivariateSpline(x, y, z, tx, ty, w=w, bbox=bbox)
            # list input
            spl2 = LSQBivariateSpline(x.tolist(), y.tolist(), z.tolist(),
                                      tx.tolist(), ty.tolist(), w=w.tolist(),
                                      bbox=bbox)
            assert_allclose(spl1(2.0, 2.0), spl2(2.0, 2.0))
            assert_equal(len(r), 2)

    def test_unequal_length_of_knots(self):
        """Test for the case when the input knot-location arrays in x and y are
        of different lengths.
        """
        x, y = np.mgrid[0:100, 0:100]
        x = x.ravel()
        y = y.ravel()
        z = 3.0 * np.ones_like(x)
        tx = np.linspace(0.1, 98.0, 29)
        ty = np.linspace(0.1, 98.0, 33)
        with suppress_warnings() as sup:
            r = sup.record(UserWarning, "\nThe coefficients of the spline")
            lut = LSQBivariateSpline(x,y,z,tx,ty)
            assert_equal(len(r), 1)

        assert_almost_equal(lut(x, y, grid=False), z)


class TestSmoothBivariateSpline:
    def test_linear_constant(self):
        x = [1,1,1,2,2,2,3,3,3]
        y = [1,2,3,1,2,3,1,2,3]
        z = [3,3,3,3,3,3,3,3,3]
        lut = SmoothBivariateSpline(x,y,z,kx=1,ky=1)
        assert_array_almost_equal(lut.get_knots(),([1,1,3,3],[1,1,3,3]))
        assert_array_almost_equal(lut.get_coeffs(),[3,3,3,3])
        assert_almost_equal(lut.get_residual(),0.0)
        assert_array_almost_equal(lut([1,1.5,2],[1,1.5]),[[3,3],[3,3],[3,3]])

    def test_linear_1d(self):
        x = [1,1,1,2,2,2,3,3,3]
        y = [1,2,3,1,2,3,1,2,3]
        z = [0,0,0,2,2,2,4,4,4]
        lut = SmoothBivariateSpline(x,y,z,kx=1,ky=1)
        assert_array_almost_equal(lut.get_knots(),([1,1,3,3],[1,1,3,3]))
        assert_array_almost_equal(lut.get_coeffs(),[0,0,4,4])
        assert_almost_equal(lut.get_residual(),0.0)
        assert_array_almost_equal(lut([1,1.5,2],[1,1.5]),[[0,0],[1,1],[2,2]])

    def test_integral(self):
        x = [1,1,1,2,2,2,4,4,4]
        y = [1,2,3,1,2,3,1,2,3]
        z = array([0,7,8,3,4,7,1,3,4])

        with suppress_warnings() as sup:
            # This seems to fail (ier=1, see ticket 1642).
            sup.filter(UserWarning, "\nThe required storage space")
            lut = SmoothBivariateSpline(x, y, z, kx=1, ky=1, s=0)

        tx = [1,2,4]
        ty = [1,2,3]

        tz = lut(tx, ty)
        trpz = .25*(diff(tx)[:,None]*diff(ty)[None,:]
                    * (tz[:-1,:-1]+tz[1:,:-1]+tz[:-1,1:]+tz[1:,1:])).sum()
        assert_almost_equal(lut.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz)

        lut2 = SmoothBivariateSpline(x, y, z, kx=2, ky=2, s=0)
        assert_almost_equal(lut2.integral(tx[0], tx[-1], ty[0], ty[-1]), trpz,
                            decimal=0)  # the quadratures give 23.75 and 23.85

        tz = lut(tx[:-1], ty[:-1])
        trpz = .25*(diff(tx[:-1])[:,None]*diff(ty[:-1])[None,:]
                    * (tz[:-1,:-1]+tz[1:,:-1]+tz[:-1,1:]+tz[1:,1:])).sum()
        assert_almost_equal(lut.integral(tx[0], tx[-2], ty[0], ty[-2]), trpz)

    def test_rerun_lwrk2_too_small(self):
        # in this setting, lwrk2 is too small in the default run. Here we
        # check for equality with the bisplrep/bisplev output because there,
        # an automatic re-run of the spline representation is done if ier>10.
        x = np.linspace(-2, 2, 80)
        y = np.linspace(-2, 2, 80)
        z = x + y
        xi = np.linspace(-1, 1, 100)
        yi = np.linspace(-2, 2, 100)
        tck = bisplrep(x, y, z)
        res1 = bisplev(xi, yi, tck)
        interp_ = SmoothBivariateSpline(x, y, z)
        res2 = interp_(xi, yi)
        assert_almost_equal(res1, res2)

    def test_invalid_input(self):

        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0, num=10)
            SmoothBivariateSpline(x, y, z)
        assert "x, y, and z should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            x = np.linspace(1.0, 10.0)
            y = np.linspace(1.0, 10.0)
            z = np.linspace(1.0, 10.0)
            w = np.linspace(1.0, 10.0, num=20)
            SmoothBivariateSpline(x, y, z, w=w)
        assert "x, y, z, and w should have a same length" in str(info.value)

        with assert_raises(ValueError) as info:
            w = np.linspace(-1.0, 10.0)
            SmoothBivariateSpline(x, y, z, w=w)
        assert "w should be positive" in str(info.value)

        with assert_raises(ValueError) as info:
            bbox = (-100, 100, -100)
            SmoothBivariateSpline(x, y, z, bbox=bbox)
        assert "bbox shape should be (4,)" in str(info.value)

        with assert_raises(ValueError) as info:
            SmoothBivariateSpline(x, y, z, kx=10, ky=10)
        assert "The length of x, y and z should be at least (kx+1) * (ky+1)" in\
               str(info.value)

        with assert_raises(ValueError) as info:
            SmoothBivariateSpline(x, y, z, s=-1.0)
        assert "s should be s >= 0.0" in str(info.value)

        with assert_raises(ValueError) as exc_info:
            SmoothBivariateSpline(x, y, z, eps=0.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            SmoothBivariateSpline(x, y, z, eps=1.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

    def test_array_like_input(self):
        x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        z = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        w = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        bbox = np.array([1.0, 3.0, 1.0, 3.0])
        # np.array input
        spl1 = SmoothBivariateSpline(x, y, z, w=w, bbox=bbox, kx=1, ky=1)
        # list input
        spl2 = SmoothBivariateSpline(x.tolist(), y.tolist(), z.tolist(),
                                     bbox=bbox.tolist(), w=w.tolist(),
                                     kx=1, ky=1)
        assert_allclose(spl1(0.1, 0.5), spl2(0.1, 0.5))


class TestLSQSphereBivariateSpline:
    def setup_method(self):
        # define the input data and coordinates
        ntheta, nphi = 70, 90
        theta = linspace(0.5/(ntheta - 1), 1 - 0.5/(ntheta - 1), ntheta) * pi
        phi = linspace(0.5/(nphi - 1), 1 - 0.5/(nphi - 1), nphi) * 2. * pi
        data = ones((theta.shape[0], phi.shape[0]))
        # define knots and extract data values at the knots
        knotst = theta[::5]
        knotsp = phi[::5]
        knotdata = data[::5, ::5]
        # calculate spline coefficients
        lats, lons = meshgrid(theta, phi)
        lut_lsq = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                           data.T.ravel(), knotst, knotsp)
        self.lut_lsq = lut_lsq
        self.data = knotdata
        self.new_lons, self.new_lats = knotsp, knotst

    def test_linear_constant(self):
        assert_almost_equal(self.lut_lsq.get_residual(), 0.0)
        assert_array_almost_equal(self.lut_lsq(self.new_lats, self.new_lons),
                                  self.data)

    def test_empty_input(self):
        assert_array_almost_equal(self.lut_lsq([], []), np.zeros((0,0)))
        assert_array_almost_equal(self.lut_lsq([], [], grid=False), np.zeros((0,)))

    def test_invalid_input(self):
        ntheta, nphi = 70, 90
        theta = linspace(0.5 / (ntheta - 1), 1 - 0.5 / (ntheta - 1),
                         ntheta) * pi
        phi = linspace(0.5 / (nphi - 1), 1 - 0.5 / (nphi - 1), nphi) * 2. * pi
        data = ones((theta.shape[0], phi.shape[0]))
        # define knots and extract data values at the knots
        knotst = theta[::5]
        knotsp = phi[::5]

        with assert_raises(ValueError) as exc_info:
            invalid_theta = linspace(-0.1, 1.0, num=ntheta) * pi
            invalid_lats, lons = meshgrid(invalid_theta, phi)
            LSQSphereBivariateSpline(invalid_lats.ravel(), lons.ravel(),
                                     data.T.ravel(), knotst, knotsp)
        assert "theta should be between [0, pi]" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_theta = linspace(0.1, 1.1, num=ntheta) * pi
            invalid_lats, lons = meshgrid(invalid_theta, phi)
            LSQSphereBivariateSpline(invalid_lats.ravel(), lons.ravel(),
                                     data.T.ravel(), knotst, knotsp)
        assert "theta should be between [0, pi]" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_phi = linspace(-0.1, 1.0, num=ntheta) * 2.0 * pi
            lats, invalid_lons = meshgrid(theta, invalid_phi)
            LSQSphereBivariateSpline(lats.ravel(), invalid_lons.ravel(),
                                     data.T.ravel(), knotst, knotsp)
        assert "phi should be between [0, 2pi]" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_phi = linspace(0.0, 1.1, num=ntheta) * 2.0 * pi
            lats, invalid_lons = meshgrid(theta, invalid_phi)
            LSQSphereBivariateSpline(lats.ravel(), invalid_lons.ravel(),
                                     data.T.ravel(), knotst, knotsp)
        assert "phi should be between [0, 2pi]" in str(exc_info.value)

        lats, lons = meshgrid(theta, phi)

        with assert_raises(ValueError) as exc_info:
            invalid_knotst = np.copy(knotst)
            invalid_knotst[0] = -0.1
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                     data.T.ravel(), invalid_knotst, knotsp)
        assert "tt should be between (0, pi)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_knotst = np.copy(knotst)
            invalid_knotst[0] = pi
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                     data.T.ravel(), invalid_knotst, knotsp)
        assert "tt should be between (0, pi)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_knotsp = np.copy(knotsp)
            invalid_knotsp[0] = -0.1
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                     data.T.ravel(), knotst, invalid_knotsp)
        assert "tp should be between (0, 2pi)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_knotsp = np.copy(knotsp)
            invalid_knotsp[0] = 2 * pi
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                     data.T.ravel(), knotst, invalid_knotsp)
        assert "tp should be between (0, 2pi)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_w = array([-1.0, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.0])
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(),
                                     knotst, knotsp, w=invalid_w)
        assert "w should be positive" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(),
                                     knotst, knotsp, eps=0.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            LSQSphereBivariateSpline(lats.ravel(), lons.ravel(), data.T.ravel(),
                                     knotst, knotsp, eps=1.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

    def test_array_like_input(self):
        ntheta, nphi = 70, 90
        theta = linspace(0.5 / (ntheta - 1), 1 - 0.5 / (ntheta - 1),
                         ntheta) * pi
        phi = linspace(0.5 / (nphi - 1), 1 - 0.5 / (nphi - 1),
                       nphi) * 2. * pi
        lats, lons = meshgrid(theta, phi)
        data = ones((theta.shape[0], phi.shape[0]))
        # define knots and extract data values at the knots
        knotst = theta[::5]
        knotsp = phi[::5]
        w = ones((lats.ravel().shape[0]))

        # np.array input
        spl1 = LSQSphereBivariateSpline(lats.ravel(), lons.ravel(),
                                        data.T.ravel(), knotst, knotsp, w=w)
        # list input
        spl2 = LSQSphereBivariateSpline(lats.ravel().tolist(),
                                        lons.ravel().tolist(),
                                        data.T.ravel().tolist(),
                                        knotst.tolist(),
                                        knotsp.tolist(), w=w.tolist())
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))


class TestSmoothSphereBivariateSpline:
    def setup_method(self):
        theta = array([.25*pi, .25*pi, .25*pi, .5*pi, .5*pi, .5*pi, .75*pi,
                       .75*pi, .75*pi])
        phi = array([.5 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi, .5 * pi, pi,
                     1.5 * pi])
        r = array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.lut = SmoothSphereBivariateSpline(theta, phi, r, s=1E10)

    def test_linear_constant(self):
        assert_almost_equal(self.lut.get_residual(), 0.)
        assert_array_almost_equal(self.lut([1, 1.5, 2],[1, 1.5]),
                                  [[3, 3], [3, 3], [3, 3]])

    def test_empty_input(self):
        assert_array_almost_equal(self.lut([], []), np.zeros((0,0)))
        assert_array_almost_equal(self.lut([], [], grid=False), np.zeros((0,)))

    def test_invalid_input(self):
        theta = array([.25 * pi, .25 * pi, .25 * pi, .5 * pi, .5 * pi, .5 * pi,
                       .75 * pi, .75 * pi, .75 * pi])
        phi = array([.5 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi, .5 * pi, pi,
                     1.5 * pi])
        r = array([3, 3, 3, 3, 3, 3, 3, 3, 3])

        with assert_raises(ValueError) as exc_info:
            invalid_theta = array([-0.1 * pi, .25 * pi, .25 * pi, .5 * pi,
                                   .5 * pi, .5 * pi, .75 * pi, .75 * pi,
                                   .75 * pi])
            SmoothSphereBivariateSpline(invalid_theta, phi, r, s=1E10)
        assert "theta should be between [0, pi]" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_theta = array([.25 * pi, .25 * pi, .25 * pi, .5 * pi,
                                   .5 * pi, .5 * pi, .75 * pi, .75 * pi,
                                   1.1 * pi])
            SmoothSphereBivariateSpline(invalid_theta, phi, r, s=1E10)
        assert "theta should be between [0, pi]" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_phi = array([-.1 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi,
                                 .5 * pi, pi, 1.5 * pi])
            SmoothSphereBivariateSpline(theta, invalid_phi, r, s=1E10)
        assert "phi should be between [0, 2pi]" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_phi = array([1.0 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi,
                                 .5 * pi, pi, 2.1 * pi])
            SmoothSphereBivariateSpline(theta, invalid_phi, r, s=1E10)
        assert "phi should be between [0, 2pi]" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            invalid_w = array([-1.0, 1.0, 1.5, 0.5, 1.0, 1.5, 0.5, 1.0, 1.0])
            SmoothSphereBivariateSpline(theta, phi, r, w=invalid_w, s=1E10)
        assert "w should be positive" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            SmoothSphereBivariateSpline(theta, phi, r, s=-1.0)
        assert "s should be positive" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            SmoothSphereBivariateSpline(theta, phi, r, eps=-1.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            SmoothSphereBivariateSpline(theta, phi, r, eps=1.0)
        assert "eps should be between (0, 1)" in str(exc_info.value)

    def test_array_like_input(self):
        theta = np.array([.25 * pi, .25 * pi, .25 * pi, .5 * pi, .5 * pi,
                          .5 * pi, .75 * pi, .75 * pi, .75 * pi])
        phi = np.array([.5 * pi, pi, 1.5 * pi, .5 * pi, pi, 1.5 * pi, .5 * pi,
                        pi, 1.5 * pi])
        r = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # np.array input
        spl1 = SmoothSphereBivariateSpline(theta, phi, r, w=w, s=1E10)

        # list input
        spl2 = SmoothSphereBivariateSpline(theta.tolist(), phi.tolist(),
                                           r.tolist(), w=w.tolist(), s=1E10)
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))


class TestRectBivariateSpline:
    def test_defaults(self):
        x = array([1,2,3,4,5])
        y = array([1,2,3,4,5])
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        lut = RectBivariateSpline(x,y,z)
        assert_array_almost_equal(lut(x,y),z)

    def test_evaluate(self):
        x = array([1,2,3,4,5])
        y = array([1,2,3,4,5])
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        lut = RectBivariateSpline(x,y,z)

        xi = [1, 2.3, 5.3, 0.5, 3.3, 1.2, 3]
        yi = [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]
        zi = lut.ev(xi, yi)
        zi2 = array([lut(xp, yp)[0,0] for xp, yp in zip(xi, yi)])

        assert_almost_equal(zi, zi2)

    def test_derivatives_grid(self):
        x = array([1,2,3,4,5])
        y = array([1,2,3,4,5])
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        dx = array([[0,0,-20,0,0],[0,0,13,0,0],[0,0,4,0,0],
            [0,0,-11,0,0],[0,0,4,0,0]])/6.
        dy = array([[4,-1,0,1,-4],[4,-1,0,1,-4],[0,1.5,0,-1.5,0],
            [2,.25,0,-.25,-2],[4,-1,0,1,-4]])
        dxdy = array([[40,-25,0,25,-40],[-26,16.25,0,-16.25,26],
            [-8,5,0,-5,8],[22,-13.75,0,13.75,-22],[-8,5,0,-5,8]])/6.
        lut = RectBivariateSpline(x,y,z)
        assert_array_almost_equal(lut(x,y,dx=1),dx)
        assert_array_almost_equal(lut(x,y,dy=1),dy)
        assert_array_almost_equal(lut(x,y,dx=1,dy=1),dxdy)

    def test_derivatives(self):
        x = array([1,2,3,4,5])
        y = array([1,2,3,4,5])
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        dx = array([0,0,2./3,0,0])
        dy = array([4,-1,0,-.25,-4])
        dxdy = array([160,65,0,55,32])/24.
        lut = RectBivariateSpline(x,y,z)
        assert_array_almost_equal(lut(x,y,dx=1,grid=False),dx)
        assert_array_almost_equal(lut(x,y,dy=1,grid=False),dy)
        assert_array_almost_equal(lut(x,y,dx=1,dy=1,grid=False),dxdy)

    def test_broadcast(self):
        x = array([1,2,3,4,5])
        y = array([1,2,3,4,5])
        z = array([[1,2,1,2,1],[1,2,1,2,1],[1,2,3,2,1],[1,2,2,2,1],[1,2,1,2,1]])
        lut = RectBivariateSpline(x,y,z)
        assert_allclose(lut(x, y), lut(x[:,None], y[None,:], grid=False))

    def test_invalid_input(self):

        with assert_raises(ValueError) as info:
            x = array([6, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert "x must be strictly increasing" in str(info.value)

        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([2, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert "y must be strictly increasing" in str(info.value)

        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1]])
            RectBivariateSpline(x, y, z)
        assert "x dimension of z must have same number of elements as x"\
               in str(info.value)

        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3, 2],
                       [1, 2, 2, 2], [1, 2, 1, 2]])
            RectBivariateSpline(x, y, z)
        assert "y dimension of z must have same number of elements as y"\
               in str(info.value)

        with assert_raises(ValueError) as info:
            x = array([1, 2, 3, 4, 5])
            y = array([1, 2, 3, 4, 5])
            z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                       [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
            bbox = (-100, 100, -100)
            RectBivariateSpline(x, y, z, bbox=bbox)
        assert "bbox shape should be (4,)" in str(info.value)

        with assert_raises(ValueError) as info:
            RectBivariateSpline(x, y, z, s=-1.0)
        assert "s should be s >= 0.0" in str(info.value)

    def test_array_like_input(self):
        x = array([1, 2, 3, 4, 5])
        y = array([1, 2, 3, 4, 5])
        z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                   [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        bbox = array([1, 5, 1, 5])

        spl1 = RectBivariateSpline(x, y, z, bbox=bbox)
        spl2 = RectBivariateSpline(x.tolist(), y.tolist(), z.tolist(),
                                   bbox=bbox.tolist())
        assert_array_almost_equal(spl1(1.0, 1.0), spl2(1.0, 1.0))

    def test_not_increasing_input(self):
        # gh-8565
        NSamp = 20
        Theta = np.random.uniform(0, np.pi, NSamp)
        Phi = np.random.uniform(0, 2 * np.pi, NSamp)
        Data = np.ones(NSamp)

        Interpolator = SmoothSphereBivariateSpline(Theta, Phi, Data, s=3.5)

        NLon = 6
        NLat = 3
        GridPosLats = np.arange(NLat) / NLat * np.pi
        GridPosLons = np.arange(NLon) / NLon * 2 * np.pi

        # No error
        Interpolator(GridPosLats, GridPosLons)

        nonGridPosLats = GridPosLats.copy()
        nonGridPosLats[2] = 0.001
        with assert_raises(ValueError) as exc_info:
            Interpolator(nonGridPosLats, GridPosLons)
        assert "x must be strictly increasing" in str(exc_info.value)

        nonGridPosLons = GridPosLons.copy()
        nonGridPosLons[2] = 0.001
        with assert_raises(ValueError) as exc_info:
            Interpolator(GridPosLats, nonGridPosLons)
        assert "y must be strictly increasing" in str(exc_info.value)


class TestRectSphereBivariateSpline:
    def test_defaults(self):
        y = linspace(0.01, 2*pi-0.01, 7)
        x = linspace(0.01, pi-0.01, 7)
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])
        lut = RectSphereBivariateSpline(x,y,z)
        assert_array_almost_equal(lut(x,y),z)

    def test_evaluate(self):
        y = linspace(0.01, 2*pi-0.01, 7)
        x = linspace(0.01, pi-0.01, 7)
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])
        lut = RectSphereBivariateSpline(x,y,z)
        yi = [0.2, 1, 2.3, 2.35, 3.0, 3.99, 5.25]
        xi = [1.5, 0.4, 1.1, 0.45, 0.2345, 1., 0.0001]
        zi = lut.ev(xi, yi)
        zi2 = array([lut(xp, yp)[0,0] for xp, yp in zip(xi, yi)])
        assert_almost_equal(zi, zi2)

    def test_derivatives_grid(self):
        y = linspace(0.01, 2*pi-0.01, 7)
        x = linspace(0.01, pi-0.01, 7)
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])

        lut = RectSphereBivariateSpline(x,y,z)

        y = linspace(0.02, 2*pi-0.02, 7)
        x = linspace(0.02, pi-0.02, 7)

        assert_allclose(lut(x, y, dtheta=1), _numdiff_2d(lut, x, y, dx=1),
                        rtol=1e-4, atol=1e-4)
        assert_allclose(lut(x, y, dphi=1), _numdiff_2d(lut, x, y, dy=1),
                        rtol=1e-4, atol=1e-4)
        assert_allclose(lut(x, y, dtheta=1, dphi=1), _numdiff_2d(lut, x, y, dx=1, dy=1, eps=1e-6),
                        rtol=1e-3, atol=1e-3)

    def test_derivatives(self):
        y = linspace(0.01, 2*pi-0.01, 7)
        x = linspace(0.01, pi-0.01, 7)
        z = array([[1,2,1,2,1,2,1],[1,2,1,2,1,2,1],[1,2,3,2,1,2,1],
                   [1,2,2,2,1,2,1],[1,2,1,2,1,2,1],[1,2,2,2,1,2,1],
                   [1,2,1,2,1,2,1]])

        lut = RectSphereBivariateSpline(x,y,z)

        y = linspace(0.02, 2*pi-0.02, 7)
        x = linspace(0.02, pi-0.02, 7)

        assert_equal(lut(x, y, dtheta=1, grid=False).shape, x.shape)
        assert_allclose(lut(x, y, dtheta=1, grid=False),
                        _numdiff_2d(lambda x,y: lut(x,y,grid=False), x, y, dx=1),
                        rtol=1e-4, atol=1e-4)
        assert_allclose(lut(x, y, dphi=1, grid=False),
                        _numdiff_2d(lambda x,y: lut(x,y,grid=False), x, y, dy=1),
                        rtol=1e-4, atol=1e-4)
        assert_allclose(lut(x, y, dtheta=1, dphi=1, grid=False),
                        _numdiff_2d(lambda x,y: lut(x,y,grid=False), x, y, dx=1, dy=1, eps=1e-6),
                        rtol=1e-3, atol=1e-3)

    def test_invalid_input(self):
        data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
                      np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(0, 170, 9) * np.pi / 180.
            lons = np.linspace(0, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "u should be between (0, pi)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 180, 9) * np.pi / 180.
            lons = np.linspace(0, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "u should be between (0, pi)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(-181, 10, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "v[0] should be between [-pi, pi)" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(-10, 360, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data)
        assert "v[-1] should be v[0] + 2pi or less" in str(exc_info.value)

        with assert_raises(ValueError) as exc_info:
            lats = np.linspace(10, 170, 9) * np.pi / 180.
            lons = np.linspace(10, 350, 18) * np.pi / 180.
            RectSphereBivariateSpline(lats, lons, data, s=-1)
        assert "s should be positive" in str(exc_info.value)

    def test_array_like_input(self):
        y = linspace(0.01, 2 * pi - 0.01, 7)
        x = linspace(0.01, pi - 0.01, 7)
        z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1],
                   [1, 2, 3, 2, 1, 2, 1],
                   [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1],
                   [1, 2, 2, 2, 1, 2, 1],
                   [1, 2, 1, 2, 1, 2, 1]])
        # np.array input
        spl1 = RectSphereBivariateSpline(x, y, z)
        # list input
        spl2 = RectSphereBivariateSpline(x.tolist(), y.tolist(), z.tolist())
        assert_array_almost_equal(spl1(x, y), spl2(x, y))

    def test_negative_evaluation(self):
        lats = np.array([25, 30, 35, 40, 45])
        lons = np.array([-90, -85, -80, -75, 70])
        mesh = np.meshgrid(lats, lons)
        data = mesh[0] + mesh[1]  # lon + lat value
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        interpolator = RectSphereBivariateSpline(lat_r, lon_r, data)
        query_lat = np.radians(np.array([35, 37.5]))
        query_lon = np.radians(np.array([-80, -77.5]))
        data_interp = interpolator(query_lat, query_lon)
        ans = np.array([[-45.0, -42.480862],
                        [-49.0625, -46.54315]])
        assert_array_almost_equal(data_interp, ans)


def _numdiff_2d(func, x, y, dx=0, dy=0, eps=1e-8):
    if dx == 0 and dy == 0:
        return func(x, y)
    elif dx == 1 and dy == 0:
        return (func(x + eps, y) - func(x - eps, y)) / (2*eps)
    elif dx == 0 and dy == 1:
        return (func(x, y + eps) - func(x, y - eps)) / (2*eps)
    elif dx == 1 and dy == 1:
        return (func(x + eps, y + eps) - func(x - eps, y + eps)
                - func(x + eps, y - eps) + func(x - eps, y - eps)) / (2*eps)**2
    else:
        raise ValueError("invalid derivative order")
