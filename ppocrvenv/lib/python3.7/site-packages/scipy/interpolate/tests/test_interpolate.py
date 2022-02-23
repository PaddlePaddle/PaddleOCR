import itertools

from numpy.testing import (assert_, assert_equal, assert_almost_equal,
        assert_array_almost_equal, assert_array_equal,
        assert_allclose, assert_warns)
from pytest import raises as assert_raises
import pytest

from numpy import mgrid, pi, sin, ogrid, poly1d, linspace
import numpy as np

from scipy.interpolate import (interp1d, interp2d, lagrange, PPoly, BPoly,
         splrep, splev, splantider, splint, sproot, Akima1DInterpolator,
         RegularGridInterpolator, LinearNDInterpolator, NearestNDInterpolator,
         RectBivariateSpline, interpn, NdPPoly, BSpline,
         CloughTocher2DInterpolator)

from scipy.special import poch, gamma

from scipy.interpolate import _ppoly

from scipy._lib._gcutils import assert_deallocated, IS_PYPY

from scipy.integrate import nquad

from scipy.special import binom

from scipy.sparse.sputils import matrix


class TestInterp2D:
    def test_interp2d(self):
        y, x = mgrid[0:2:20j, 0:pi:21j]
        z = sin(x+0.5*y)
        I = interp2d(x, y, z)
        assert_almost_equal(I(1.0, 2.0), sin(2.0), decimal=2)

        v,u = ogrid[0:2:24j, 0:pi:25j]
        assert_almost_equal(I(u.ravel(), v.ravel()), sin(u+0.5*v), decimal=2)

    def test_interp2d_meshgrid_input(self):
        # Ticket #703
        x = linspace(0, 2, 16)
        y = linspace(0, pi, 21)
        z = sin(x[None,:] + y[:,None]/2.)
        I = interp2d(x, y, z)
        assert_almost_equal(I(1.0, 2.0), sin(2.0), decimal=2)

    def test_interp2d_meshgrid_input_unsorted(self):
        np.random.seed(1234)
        x = linspace(0, 2, 16)
        y = linspace(0, pi, 21)

        z = sin(x[None,:] + y[:,None]/2.)
        ip1 = interp2d(x.copy(), y.copy(), z, kind='cubic')

        np.random.shuffle(x)
        z = sin(x[None,:] + y[:,None]/2.)
        ip2 = interp2d(x.copy(), y.copy(), z, kind='cubic')

        np.random.shuffle(x)
        np.random.shuffle(y)
        z = sin(x[None,:] + y[:,None]/2.)
        ip3 = interp2d(x, y, z, kind='cubic')

        x = linspace(0, 2, 31)
        y = linspace(0, pi, 30)

        assert_equal(ip1(x, y), ip2(x, y))
        assert_equal(ip1(x, y), ip3(x, y))

    def test_interp2d_eval_unsorted(self):
        y, x = mgrid[0:2:20j, 0:pi:21j]
        z = sin(x + 0.5*y)
        func = interp2d(x, y, z)

        xe = np.array([3, 4, 5])
        ye = np.array([5.3, 7.1])
        assert_allclose(func(xe, ye), func(xe, ye[::-1]))

        assert_raises(ValueError, func, xe, ye[::-1], 0, 0, True)

    def test_interp2d_linear(self):
        # Ticket #898
        a = np.zeros([5, 5])
        a[2, 2] = 1.0
        x = y = np.arange(5)
        b = interp2d(x, y, a, 'linear')
        assert_almost_equal(b(2.0, 1.5), np.array([0.5]), decimal=2)
        assert_almost_equal(b(2.0, 2.5), np.array([0.5]), decimal=2)

    def test_interp2d_bounds(self):
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 2, 7)
        z = x[None, :]**2 + y[:, None]

        ix = np.linspace(-1, 3, 31)
        iy = np.linspace(-1, 3, 33)

        b = interp2d(x, y, z, bounds_error=True)
        assert_raises(ValueError, b, ix, iy)

        b = interp2d(x, y, z, fill_value=np.nan)
        iz = b(ix, iy)
        mx = (ix < 0) | (ix > 1)
        my = (iy < 0) | (iy > 2)
        assert_(np.isnan(iz[my,:]).all())
        assert_(np.isnan(iz[:,mx]).all())
        assert_(np.isfinite(iz[~my,:][:,~mx]).all())


class TestInterp1D:

    def setup_method(self):
        self.x5 = np.arange(5.)
        self.x10 = np.arange(10.)
        self.y10 = np.arange(10.)
        self.x25 = self.x10.reshape((2,5))
        self.x2 = np.arange(2.)
        self.y2 = np.arange(2.)
        self.x1 = np.array([0.])
        self.y1 = np.array([0.])

        self.y210 = np.arange(20.).reshape((2, 10))
        self.y102 = np.arange(20.).reshape((10, 2))
        self.y225 = np.arange(20.).reshape((2, 2, 5))
        self.y25 = np.arange(10.).reshape((2, 5))
        self.y235 = np.arange(30.).reshape((2, 3, 5))
        self.y325 = np.arange(30.).reshape((3, 2, 5))

        self.fill_value = -100.0

    def test_validation(self):
        # Make sure that appropriate exceptions are raised when invalid values
        # are given to the constructor.

        # These should all work.
        for kind in ('nearest', 'nearest-up', 'zero', 'linear', 'slinear',
                     'quadratic', 'cubic', 'previous', 'next'):
            interp1d(self.x10, self.y10, kind=kind)
            interp1d(self.x10, self.y10, kind=kind, fill_value="extrapolate")
        interp1d(self.x10, self.y10, kind='linear', fill_value=(-1, 1))
        interp1d(self.x10, self.y10, kind='linear',
                 fill_value=np.array([-1]))
        interp1d(self.x10, self.y10, kind='linear',
                 fill_value=(-1,))
        interp1d(self.x10, self.y10, kind='linear',
                 fill_value=-1)
        interp1d(self.x10, self.y10, kind='linear',
                 fill_value=(-1, -1))
        interp1d(self.x10, self.y10, kind=0)
        interp1d(self.x10, self.y10, kind=1)
        interp1d(self.x10, self.y10, kind=2)
        interp1d(self.x10, self.y10, kind=3)
        interp1d(self.x10, self.y210, kind='linear', axis=-1,
                 fill_value=(-1, -1))
        interp1d(self.x2, self.y210, kind='linear', axis=0,
                 fill_value=np.ones(10))
        interp1d(self.x2, self.y210, kind='linear', axis=0,
                 fill_value=(np.ones(10), np.ones(10)))
        interp1d(self.x2, self.y210, kind='linear', axis=0,
                 fill_value=(np.ones(10), -1))

        # x array must be 1D.
        assert_raises(ValueError, interp1d, self.x25, self.y10)

        # y array cannot be a scalar.
        assert_raises(ValueError, interp1d, self.x10, np.array(0))

        # Check for x and y arrays having the same length.
        assert_raises(ValueError, interp1d, self.x10, self.y2)
        assert_raises(ValueError, interp1d, self.x2, self.y10)
        assert_raises(ValueError, interp1d, self.x10, self.y102)
        interp1d(self.x10, self.y210)
        interp1d(self.x10, self.y102, axis=0)

        # Check for x and y having at least 1 element.
        assert_raises(ValueError, interp1d, self.x1, self.y10)
        assert_raises(ValueError, interp1d, self.x10, self.y1)
        assert_raises(ValueError, interp1d, self.x1, self.y1)

        # Bad fill values
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear',
                      fill_value=(-1, -1, -1))  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear',
                      fill_value=[-1, -1, -1])  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear',
                      fill_value=np.array((-1, -1, -1)))  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear',
                      fill_value=[[-1]])  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear',
                      fill_value=[-1, -1])  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear',
                      fill_value=np.array([]))  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x10, self.y10, kind='linear',
                      fill_value=())  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x2, self.y210, kind='linear',
                      axis=0, fill_value=[-1, -1])  # doesn't broadcast
        assert_raises(ValueError, interp1d, self.x2, self.y210, kind='linear',
                      axis=0, fill_value=(0., [-1, -1]))  # above doesn't bc

    def test_init(self):
        # Check that the attributes are initialized appropriately by the
        # constructor.
        assert_(interp1d(self.x10, self.y10).copy)
        assert_(not interp1d(self.x10, self.y10, copy=False).copy)
        assert_(interp1d(self.x10, self.y10).bounds_error)
        assert_(not interp1d(self.x10, self.y10, bounds_error=False).bounds_error)
        assert_(np.isnan(interp1d(self.x10, self.y10).fill_value))
        assert_equal(interp1d(self.x10, self.y10, fill_value=3.0).fill_value,
                     3.0)
        assert_equal(interp1d(self.x10, self.y10, fill_value=(1.0, 2.0)).fill_value,
                     (1.0, 2.0))
        assert_equal(interp1d(self.x10, self.y10).axis, 0)
        assert_equal(interp1d(self.x10, self.y210).axis, 1)
        assert_equal(interp1d(self.x10, self.y102, axis=0).axis, 0)
        assert_array_equal(interp1d(self.x10, self.y10).x, self.x10)
        assert_array_equal(interp1d(self.x10, self.y10).y, self.y10)
        assert_array_equal(interp1d(self.x10, self.y210).y, self.y210)

    def test_assume_sorted(self):
        # Check for unsorted arrays
        interp10 = interp1d(self.x10, self.y10)
        interp10_unsorted = interp1d(self.x10[::-1], self.y10[::-1])

        assert_array_almost_equal(interp10_unsorted(self.x10), self.y10)
        assert_array_almost_equal(interp10_unsorted(1.2), np.array([1.2]))
        assert_array_almost_equal(interp10_unsorted([2.4, 5.6, 6.0]),
                                  interp10([2.4, 5.6, 6.0]))

        # Check assume_sorted keyword (defaults to False)
        interp10_assume_kw = interp1d(self.x10[::-1], self.y10[::-1],
                                      assume_sorted=False)
        assert_array_almost_equal(interp10_assume_kw(self.x10), self.y10)

        interp10_assume_kw2 = interp1d(self.x10[::-1], self.y10[::-1],
                                       assume_sorted=True)
        # Should raise an error for unsorted input if assume_sorted=True
        assert_raises(ValueError, interp10_assume_kw2, self.x10)

        # Check that if y is a 2-D array, things are still consistent
        interp10_y_2d = interp1d(self.x10, self.y210)
        interp10_y_2d_unsorted = interp1d(self.x10[::-1], self.y210[:, ::-1])
        assert_array_almost_equal(interp10_y_2d(self.x10),
                                  interp10_y_2d_unsorted(self.x10))

    def test_linear(self):
        for kind in ['linear', 'slinear']:
            self._check_linear(kind)

    def _check_linear(self, kind):
        # Check the actual implementation of linear interpolation.
        interp10 = interp1d(self.x10, self.y10, kind=kind)
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array([1.2]))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2.4, 5.6, 6.0]))

        # test fill_value="extrapolate"
        extrapolator = interp1d(self.x10, self.y10, kind=kind,
                                fill_value='extrapolate')
        assert_allclose(extrapolator([-1., 0, 9, 11]),
                        [-1, 0, 9, 11], rtol=1e-14)

        opts = dict(kind=kind,
                    fill_value='extrapolate',
                    bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_linear_dtypes(self):
        # regression test for gh-5898, where 1D linear interpolation has been
        # delegated to numpy.interp for all float dtypes, and the latter was
        # not handling e.g. np.float128.
        for dtyp in np.sctypes["float"]:
            x = np.arange(8, dtype=dtyp)
            y = x
            yp = interp1d(x, y, kind='linear')(x)
            assert_equal(yp.dtype, dtyp)
            assert_allclose(yp, y, atol=1e-15)

    def test_slinear_dtypes(self):
        # regression test for gh-7273: 1D slinear interpolation fails with
        # float32 inputs
        dt_r = [np.float16, np.float32, np.float64]
        dt_rc = dt_r + [np.complex64, np.complex128]
        spline_kinds = ['slinear', 'zero', 'quadratic', 'cubic']
        for dtx in dt_r:
            x = np.arange(0, 10, dtype=dtx)
            for dty in dt_rc:
                y = np.exp(-x/3.0).astype(dty)
                for dtn in dt_r:
                    xnew = x.astype(dtn)
                    for kind in spline_kinds:
                        f = interp1d(x, y, kind=kind, bounds_error=False)
                        assert_allclose(f(xnew), y, atol=1e-7,
                                        err_msg="%s, %s %s" % (dtx, dty, dtn))

    def test_cubic(self):
        # Check the actual implementation of spline interpolation.
        interp10 = interp1d(self.x10, self.y10, kind='cubic')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array([1.2]))
        assert_array_almost_equal(interp10(1.5), np.array([1.5]))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2.4, 5.6, 6.0]),)

    def test_nearest(self):
        # Check the actual implementation of nearest-neighbour interpolation.
        # Nearest asserts that half-integer case (1.5) rounds down to 1
        interp10 = interp1d(self.x10, self.y10, kind='nearest')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.))
        assert_array_almost_equal(interp10(1.5), np.array(1.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2., 6., 6.]),)

        # test fill_value="extrapolate"
        extrapolator = interp1d(self.x10, self.y10, kind='nearest',
                                fill_value='extrapolate')
        assert_allclose(extrapolator([-1., 0, 9, 11]),
                        [0, 0, 9, 9], rtol=1e-14)

        opts = dict(kind='nearest',
                    fill_value='extrapolate',
                    bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_nearest_up(self):
        # Check the actual implementation of nearest-neighbour interpolation.
        # Nearest-up asserts that half-integer case (1.5) rounds up to 2
        interp10 = interp1d(self.x10, self.y10, kind='nearest-up')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.))
        assert_array_almost_equal(interp10(1.5), np.array(2.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2., 6., 6.]),)

        # test fill_value="extrapolate"
        extrapolator = interp1d(self.x10, self.y10, kind='nearest-up',
                                fill_value='extrapolate')
        assert_allclose(extrapolator([-1., 0, 9, 11]),
                        [0, 0, 9, 9], rtol=1e-14)

        opts = dict(kind='nearest-up',
                    fill_value='extrapolate',
                    bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_previous(self):
        # Check the actual implementation of previous interpolation.
        interp10 = interp1d(self.x10, self.y10, kind='previous')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.))
        assert_array_almost_equal(interp10(1.5), np.array(1.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2., 5., 6.]),)

        # test fill_value="extrapolate"
        extrapolator = interp1d(self.x10, self.y10, kind='previous',
                                fill_value='extrapolate')
        assert_allclose(extrapolator([-1., 0, 9, 11]),
                        [0, 0, 9, 9], rtol=1e-14)

        opts = dict(kind='previous',
                    fill_value='extrapolate',
                    bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_next(self):
        # Check the actual implementation of next interpolation.
        interp10 = interp1d(self.x10, self.y10, kind='next')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(2.))
        assert_array_almost_equal(interp10(1.5), np.array(2.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([3., 6., 6.]),)

        # test fill_value="extrapolate"
        extrapolator = interp1d(self.x10, self.y10, kind='next',
                                fill_value='extrapolate')
        assert_allclose(extrapolator([-1., 0, 9, 11]),
                        [0, 0, 9, 9], rtol=1e-14)

        opts = dict(kind='next',
                    fill_value='extrapolate',
                    bounds_error=True)
        assert_raises(ValueError, interp1d, self.x10, self.y10, **opts)

    def test_zero(self):
        # Check the actual implementation of zero-order spline interpolation.
        interp10 = interp1d(self.x10, self.y10, kind='zero')
        assert_array_almost_equal(interp10(self.x10), self.y10)
        assert_array_almost_equal(interp10(1.2), np.array(1.))
        assert_array_almost_equal(interp10(1.5), np.array(1.))
        assert_array_almost_equal(interp10([2.4, 5.6, 6.0]),
                                  np.array([2., 5., 6.]))

    def _bounds_check(self, kind='linear'):
        # Test that our handling of out-of-bounds input is correct.
        extrap10 = interp1d(self.x10, self.y10, fill_value=self.fill_value,
                            bounds_error=False, kind=kind)

        assert_array_equal(extrap10(11.2), np.array(self.fill_value))
        assert_array_equal(extrap10(-3.4), np.array(self.fill_value))
        assert_array_equal(extrap10([[[11.2], [-3.4], [12.6], [19.3]]]),
                           np.array(self.fill_value),)
        assert_array_equal(extrap10._check_bounds(
                               np.array([-1.0, 0.0, 5.0, 9.0, 11.0])),
                           np.array([[True, False, False, False, False],
                                     [False, False, False, False, True]]))

        raises_bounds_error = interp1d(self.x10, self.y10, bounds_error=True,
                                       kind=kind)
        assert_raises(ValueError, raises_bounds_error, -1.0)
        assert_raises(ValueError, raises_bounds_error, 11.0)
        raises_bounds_error([0.0, 5.0, 9.0])

    def _bounds_check_int_nan_fill(self, kind='linear'):
        x = np.arange(10).astype(np.int_)
        y = np.arange(10).astype(np.int_)
        c = interp1d(x, y, kind=kind, fill_value=np.nan, bounds_error=False)
        yi = c(x - 1)
        assert_(np.isnan(yi[0]))
        assert_array_almost_equal(yi, np.r_[np.nan, y[:-1]])

    def test_bounds(self):
        for kind in ('linear', 'cubic', 'nearest', 'previous', 'next',
                     'slinear', 'zero', 'quadratic'):
            self._bounds_check(kind)
            self._bounds_check_int_nan_fill(kind)

    def _check_fill_value(self, kind):
        interp = interp1d(self.x10, self.y10, kind=kind,
                          fill_value=(-100, 100), bounds_error=False)
        assert_array_almost_equal(interp(10), 100)
        assert_array_almost_equal(interp(-10), -100)
        assert_array_almost_equal(interp([-10, 10]), [-100, 100])

        # Proper broadcasting:
        #    interp along axis of length 5
        # other dim=(2, 3), (3, 2), (2, 2), or (2,)

        # one singleton fill_value (works for all)
        for y in (self.y235, self.y325, self.y225, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1,
                              fill_value=100, bounds_error=False)
            assert_array_almost_equal(interp(10), 100)
            assert_array_almost_equal(interp(-10), 100)
            assert_array_almost_equal(interp([-10, 10]), 100)

            # singleton lower, singleton upper
            interp = interp1d(self.x5, y, kind=kind, axis=-1,
                              fill_value=(-100, 100), bounds_error=False)
            assert_array_almost_equal(interp(10), 100)
            assert_array_almost_equal(interp(-10), -100)
            if y.ndim == 3:
                result = [[[-100, 100]] * y.shape[1]] * y.shape[0]
            else:
                result = [[-100, 100]] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)

        # one broadcastable (3,) fill_value
        fill_value = [100, 200, 300]
        for y in (self.y325, self.y225):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind,
                          axis=-1, fill_value=fill_value, bounds_error=False)
        interp = interp1d(self.x5, self.y235, kind=kind, axis=-1,
                          fill_value=fill_value, bounds_error=False)
        assert_array_almost_equal(interp(10), [[100, 200, 300]] * 2)
        assert_array_almost_equal(interp(-10), [[100, 200, 300]] * 2)
        assert_array_almost_equal(interp([-10, 10]), [[[100, 100],
                                                       [200, 200],
                                                       [300, 300]]] * 2)

        # one broadcastable (2,) fill_value
        fill_value = [100, 200]
        assert_raises(ValueError, interp1d, self.x5, self.y235, kind=kind,
                      axis=-1, fill_value=fill_value, bounds_error=False)
        for y in (self.y225, self.y325, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1,
                              fill_value=fill_value, bounds_error=False)
            result = [100, 200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(10), result)
            assert_array_almost_equal(interp(-10), result)
            result = [[100, 100], [200, 200]]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)

        # broadcastable (3,) lower, singleton upper
        fill_value = (np.array([-100, -200, -300]), 100)
        for y in (self.y325, self.y225):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind,
                          axis=-1, fill_value=fill_value, bounds_error=False)
        interp = interp1d(self.x5, self.y235, kind=kind, axis=-1,
                          fill_value=fill_value, bounds_error=False)
        assert_array_almost_equal(interp(10), 100)
        assert_array_almost_equal(interp(-10), [[-100, -200, -300]] * 2)
        assert_array_almost_equal(interp([-10, 10]), [[[-100, 100],
                                                       [-200, 100],
                                                       [-300, 100]]] * 2)

        # broadcastable (2,) lower, singleton upper
        fill_value = (np.array([-100, -200]), 100)
        assert_raises(ValueError, interp1d, self.x5, self.y235, kind=kind,
                      axis=-1, fill_value=fill_value, bounds_error=False)
        for y in (self.y225, self.y325, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1,
                              fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), 100)
            result = [-100, -200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(-10), result)
            result = [[-100, 100], [-200, 100]]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)

        # broadcastable (3,) lower, broadcastable (3,) upper
        fill_value = ([-100, -200, -300], [100, 200, 300])
        for y in (self.y325, self.y225):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind,
                          axis=-1, fill_value=fill_value, bounds_error=False)
        for ii in range(2):  # check ndarray as well as list here
            if ii == 1:
                fill_value = tuple(np.array(f) for f in fill_value)
            interp = interp1d(self.x5, self.y235, kind=kind, axis=-1,
                              fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), [[100, 200, 300]] * 2)
            assert_array_almost_equal(interp(-10), [[-100, -200, -300]] * 2)
            assert_array_almost_equal(interp([-10, 10]), [[[-100, 100],
                                                           [-200, 200],
                                                           [-300, 300]]] * 2)
        # broadcastable (2,) lower, broadcastable (2,) upper
        fill_value = ([-100, -200], [100, 200])
        assert_raises(ValueError, interp1d, self.x5, self.y235, kind=kind,
                      axis=-1, fill_value=fill_value, bounds_error=False)
        for y in (self.y325, self.y225, self.y25):
            interp = interp1d(self.x5, y, kind=kind, axis=-1,
                              fill_value=fill_value, bounds_error=False)
            result = [100, 200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(10), result)
            result = [-100, -200]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp(-10), result)
            result = [[-100, 100], [-200, 200]]
            if y.ndim == 3:
                result = [result] * y.shape[0]
            assert_array_almost_equal(interp([-10, 10]), result)

        # one broadcastable (2, 2) array-like
        fill_value = [[100, 200], [1000, 2000]]
        for y in (self.y235, self.y325, self.y25):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind,
                          axis=-1, fill_value=fill_value, bounds_error=False)
        for ii in range(2):
            if ii == 1:
                fill_value = np.array(fill_value)
            interp = interp1d(self.x5, self.y225, kind=kind, axis=-1,
                              fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), [[100, 200], [1000, 2000]])
            assert_array_almost_equal(interp(-10), [[100, 200], [1000, 2000]])
            assert_array_almost_equal(interp([-10, 10]), [[[100, 100],
                                                           [200, 200]],
                                                          [[1000, 1000],
                                                           [2000, 2000]]])

        # broadcastable (2, 2) lower, broadcastable (2, 2) upper
        fill_value = ([[-100, -200], [-1000, -2000]],
                      [[100, 200], [1000, 2000]])
        for y in (self.y235, self.y325, self.y25):
            assert_raises(ValueError, interp1d, self.x5, y, kind=kind,
                          axis=-1, fill_value=fill_value, bounds_error=False)
        for ii in range(2):
            if ii == 1:
                fill_value = (np.array(fill_value[0]), np.array(fill_value[1]))
            interp = interp1d(self.x5, self.y225, kind=kind, axis=-1,
                              fill_value=fill_value, bounds_error=False)
            assert_array_almost_equal(interp(10), [[100, 200], [1000, 2000]])
            assert_array_almost_equal(interp(-10), [[-100, -200],
                                                    [-1000, -2000]])
            assert_array_almost_equal(interp([-10, 10]), [[[-100, 100],
                                                           [-200, 200]],
                                                          [[-1000, 1000],
                                                           [-2000, 2000]]])

    def test_fill_value(self):
        # test that two-element fill value works
        for kind in ('linear', 'nearest', 'cubic', 'slinear', 'quadratic',
                     'zero', 'previous', 'next'):
            self._check_fill_value(kind)

    def test_fill_value_writeable(self):
        # backwards compat: fill_value is a public writeable attribute
        interp = interp1d(self.x10, self.y10, fill_value=123.0)
        assert_equal(interp.fill_value, 123.0)
        interp.fill_value = 321.0
        assert_equal(interp.fill_value, 321.0)

    def _nd_check_interp(self, kind='linear'):
        # Check the behavior when the inputs and outputs are multidimensional.

        # Multidimensional input.
        interp10 = interp1d(self.x10, self.y10, kind=kind)
        assert_array_almost_equal(interp10(np.array([[3., 5.], [2., 7.]])),
                                  np.array([[3., 5.], [2., 7.]]))

        # Scalar input -> 0-dim scalar array output
        assert_(isinstance(interp10(1.2), np.ndarray))
        assert_equal(interp10(1.2).shape, ())

        # Multidimensional outputs.
        interp210 = interp1d(self.x10, self.y210, kind=kind)
        assert_array_almost_equal(interp210(1.), np.array([1., 11.]))
        assert_array_almost_equal(interp210(np.array([1., 2.])),
                                  np.array([[1., 2.], [11., 12.]]))

        interp102 = interp1d(self.x10, self.y102, axis=0, kind=kind)
        assert_array_almost_equal(interp102(1.), np.array([2.0, 3.0]))
        assert_array_almost_equal(interp102(np.array([1., 3.])),
                                  np.array([[2., 3.], [6., 7.]]))

        # Both at the same time!
        x_new = np.array([[3., 5.], [2., 7.]])
        assert_array_almost_equal(interp210(x_new),
                                  np.array([[[3., 5.], [2., 7.]],
                                            [[13., 15.], [12., 17.]]]))
        assert_array_almost_equal(interp102(x_new),
                                  np.array([[[6., 7.], [10., 11.]],
                                            [[4., 5.], [14., 15.]]]))

    def _nd_check_shape(self, kind='linear'):
        # Check large N-D output shape
        a = [4, 5, 6, 7]
        y = np.arange(np.prod(a)).reshape(*a)
        for n, s in enumerate(a):
            x = np.arange(s)
            z = interp1d(x, y, axis=n, kind=kind)
            assert_array_almost_equal(z(x), y, err_msg=kind)

            x2 = np.arange(2*3*1).reshape((2,3,1)) / 12.
            b = list(a)
            b[n:n+1] = [2,3,1]
            assert_array_almost_equal(z(x2).shape, b, err_msg=kind)

    def test_nd(self):
        for kind in ('linear', 'cubic', 'slinear', 'quadratic', 'nearest',
                     'zero', 'previous', 'next'):
            self._nd_check_interp(kind)
            self._nd_check_shape(kind)

    def _check_complex(self, dtype=np.complex_, kind='linear'):
        x = np.array([1, 2.5, 3, 3.1, 4, 6.4, 7.9, 8.0, 9.5, 10])
        y = x * x ** (1 + 2j)
        y = y.astype(dtype)

        # simple test
        c = interp1d(x, y, kind=kind)
        assert_array_almost_equal(y[:-1], c(x)[:-1])

        # check against interpolating real+imag separately
        xi = np.linspace(1, 10, 31)
        cr = interp1d(x, y.real, kind=kind)
        ci = interp1d(x, y.imag, kind=kind)
        assert_array_almost_equal(c(xi).real, cr(xi))
        assert_array_almost_equal(c(xi).imag, ci(xi))

    def test_complex(self):
        for kind in ('linear', 'nearest', 'cubic', 'slinear', 'quadratic',
                     'zero', 'previous', 'next'):
            self._check_complex(np.complex64, kind)
            self._check_complex(np.complex128, kind)

    @pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
    def test_circular_refs(self):
        # Test interp1d can be automatically garbage collected
        x = np.linspace(0, 1)
        y = np.linspace(0, 1)
        # Confirm interp can be released from memory after use
        with assert_deallocated(interp1d, x, y) as interp:
            interp([0.1, 0.2])
            del interp

    def test_overflow_nearest(self):
        # Test that the x range doesn't overflow when given integers as input
        for kind in ('nearest', 'previous', 'next'):
            x = np.array([0, 50, 127], dtype=np.int8)
            ii = interp1d(x, x, kind=kind)
            assert_array_almost_equal(ii(x), x)

    def test_local_nans(self):
        # check that for local interpolation kinds (slinear, zero) a single nan
        # only affects its local neighborhood
        x = np.arange(10).astype(float)
        y = x.copy()
        y[6] = np.nan
        for kind in ('zero', 'slinear'):
            ir = interp1d(x, y, kind=kind)
            vals = ir([4.9, 7.0])
            assert_(np.isfinite(vals).all())

    def test_spline_nans(self):
        # Backwards compat: a single nan makes the whole spline interpolation
        # return nans in an array of the correct shape. And it doesn't raise,
        # just quiet nans because of backcompat.
        x = np.arange(8).astype(float)
        y = x.copy()
        yn = y.copy()
        yn[3] = np.nan

        for kind in ['quadratic', 'cubic']:
            ir = interp1d(x, y, kind=kind)
            irn = interp1d(x, yn, kind=kind)
            for xnew in (6, [1, 6], [[1, 6], [3, 5]]):
                xnew = np.asarray(xnew)
                out, outn = ir(x), irn(x)
                assert_(np.isnan(outn).all())
                assert_equal(out.shape, outn.shape)

    def test_all_nans(self):
        # regression test for gh-11637: interp1d core dumps with all-nan `x`
        x = np.ones(10) * np.nan
        y = np.arange(10)
        with assert_raises(ValueError):
            interp1d(x, y, kind='cubic')

    def test_read_only(self):
        x = np.arange(0, 10)
        y = np.exp(-x / 3.0)
        xnew = np.arange(0, 9, 0.1)
        # Check both read-only and not read-only:
        for xnew_writeable in (True, False):
            xnew.flags.writeable = xnew_writeable
            x.flags.writeable = False
            for kind in ('linear', 'nearest', 'zero', 'slinear', 'quadratic',
                         'cubic'):
                f = interp1d(x, y, kind=kind)
                vals = f(xnew)
                assert_(np.isfinite(vals).all())


class TestLagrange:

    def test_lagrange(self):
        p = poly1d([5,2,1,4,3])
        xs = np.arange(len(p.coeffs))
        ys = p(xs)
        pl = lagrange(xs,ys)
        assert_array_almost_equal(p.coeffs,pl.coeffs)


class TestAkima1DInterpolator:
    def test_eval(self):
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        ak = Akima1DInterpolator(x, y)
        xi = np.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
            8.6, 9.9, 10.])
        yi = np.array([0., 1.375, 2., 1.5, 1.953125, 2.484375,
            4.1363636363636366866103344, 5.9803623910336236590978842,
            5.5067291516462386624652936, 5.2031367459745245795943447,
            4.1796554159017080820603951, 3.4110386597938129327189927,
            3.])
        assert_allclose(ak(xi), yi)

    def test_eval_2d(self):
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        y = np.column_stack((y, 2. * y))
        ak = Akima1DInterpolator(x, y)
        xi = np.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        yi = np.array([0., 1.375, 2., 1.5, 1.953125, 2.484375,
                       4.1363636363636366866103344,
                       5.9803623910336236590978842,
                       5.5067291516462386624652936,
                       5.2031367459745245795943447,
                       4.1796554159017080820603951,
                       3.4110386597938129327189927, 3.])
        yi = np.column_stack((yi, 2. * yi))
        assert_allclose(ak(xi), yi)

    def test_eval_3d(self):
        x = np.arange(0., 11.)
        y_ = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        y = np.empty((11, 2, 2))
        y[:, 0, 0] = y_
        y[:, 1, 0] = 2. * y_
        y[:, 0, 1] = 3. * y_
        y[:, 1, 1] = 4. * y_
        ak = Akima1DInterpolator(x, y)
        xi = np.array([0., 0.5, 1., 1.5, 2.5, 3.5, 4.5, 5.1, 6.5, 7.2,
                       8.6, 9.9, 10.])
        yi = np.empty((13, 2, 2))
        yi_ = np.array([0., 1.375, 2., 1.5, 1.953125, 2.484375,
                        4.1363636363636366866103344,
                        5.9803623910336236590978842,
                        5.5067291516462386624652936,
                        5.2031367459745245795943447,
                        4.1796554159017080820603951,
                        3.4110386597938129327189927, 3.])
        yi[:, 0, 0] = yi_
        yi[:, 1, 0] = 2. * yi_
        yi[:, 0, 1] = 3. * yi_
        yi[:, 1, 1] = 4. * yi_
        assert_allclose(ak(xi), yi)

    def test_degenerate_case_multidimensional(self):
        # This test is for issue #5683.
        x = np.array([0, 1, 2])
        y = np.vstack((x, x**2)).T
        ak = Akima1DInterpolator(x, y)
        x_eval = np.array([0.5, 1.5])
        y_eval = ak(x_eval)
        assert_allclose(y_eval, np.vstack((x_eval, x_eval**2)).T)

    def test_extend(self):
        x = np.arange(0., 11.)
        y = np.array([0., 2., 1., 3., 2., 6., 5.5, 5.5, 2.7, 5.1, 3.])
        ak = Akima1DInterpolator(x, y)
        match = "Extending a 1-D Akima interpolator is not yet implemented"
        with pytest.raises(NotImplementedError, match=match):
            ak.extend(None, None)


class TestPPolyCommon:
    # test basic functionality for PPoly and BPoly
    def test_sort_check(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 1, 0.5])
        assert_raises(ValueError, PPoly, c, x)
        assert_raises(ValueError, BPoly, c, x)

    def test_ctor_c(self):
        # wrong shape: `c` must be at least 2D
        with assert_raises(ValueError):
            PPoly([1, 2], [0, 1])

    def test_extend(self):
        # Test adding new points to the piecewise polynomial
        np.random.seed(1234)

        order = 3
        x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])
        c = 2*np.random.rand(order+1, len(x)-1, 2, 3) - 1

        for cls in (PPoly, BPoly):
            pp = cls(c[:,:9], x[:10])
            pp.extend(c[:,9:], x[10:])

            pp2 = cls(c[:, 10:], x[10:])
            pp2.extend(c[:, :10], x[:10])

            pp3 = cls(c, x)

            assert_array_equal(pp.c, pp3.c)
            assert_array_equal(pp.x, pp3.x)
            assert_array_equal(pp2.c, pp3.c)
            assert_array_equal(pp2.x, pp3.x)

    def test_extend_diff_orders(self):
        # Test extending polynomial with different order one
        np.random.seed(1234)

        x = np.linspace(0, 1, 6)
        c = np.random.rand(2, 5)

        x2 = np.linspace(1, 2, 6)
        c2 = np.random.rand(4, 5)

        for cls in (PPoly, BPoly):
            pp1 = cls(c, x)
            pp2 = cls(c2, x2)

            pp_comb = cls(c, x)
            pp_comb.extend(c2, x2[1:])

            # NB. doesn't match to pp1 at the endpoint, because pp1 is not
            #     continuous with pp2 as we took random coefs.
            xi1 = np.linspace(0, 1, 300, endpoint=False)
            xi2 = np.linspace(1, 2, 300)

            assert_allclose(pp1(xi1), pp_comb(xi1))
            assert_allclose(pp2(xi2), pp_comb(xi2))

    def test_extend_descending(self):
        np.random.seed(0)

        order = 3
        x = np.sort(np.random.uniform(0, 10, 20))
        c = np.random.rand(order + 1, x.shape[0] - 1, 2, 3)

        for cls in (PPoly, BPoly):
            p = cls(c, x)

            p1 = cls(c[:, :9], x[:10])
            p1.extend(c[:, 9:], x[10:])

            p2 = cls(c[:, 10:], x[10:])
            p2.extend(c[:, :10], x[:10])

            assert_array_equal(p1.c, p.c)
            assert_array_equal(p1.x, p.x)
            assert_array_equal(p2.c, p.c)
            assert_array_equal(p2.x, p.x)

    def test_shape(self):
        np.random.seed(1234)
        c = np.random.rand(8, 12, 5, 6, 7)
        x = np.sort(np.random.rand(13))
        xp = np.random.rand(3, 4)
        for cls in (PPoly, BPoly):
            p = cls(c, x)
            assert_equal(p(xp).shape, (3, 4, 5, 6, 7))

        # 'scalars'
        for cls in (PPoly, BPoly):
            p = cls(c[..., 0, 0, 0], x)

            assert_equal(np.shape(p(0.5)), ())
            assert_equal(np.shape(p(np.array(0.5))), ())

            assert_raises(ValueError, p, np.array([[0.1, 0.2], [0.4]], dtype=object))

    def test_complex_coef(self):
        np.random.seed(12345)
        x = np.sort(np.random.random(13))
        c = np.random.random((8, 12)) * (1. + 0.3j)
        c_re, c_im = c.real, c.imag
        xp = np.random.random(5)
        for cls in (PPoly, BPoly):
            p, p_re, p_im = cls(c, x), cls(c_re, x), cls(c_im, x)
            for nu in [0, 1, 2]:
                assert_allclose(p(xp, nu).real, p_re(xp, nu))
                assert_allclose(p(xp, nu).imag, p_im(xp, nu))

    def test_axis(self):
        np.random.seed(12345)
        c = np.random.rand(3, 4, 5, 6, 7, 8)
        c_s = c.shape
        xp = np.random.random((1, 2))
        for axis in (0, 1, 2, 3):
            m = c.shape[axis+1]
            x = np.sort(np.random.rand(m+1))
            for cls in (PPoly, BPoly):
                p = cls(c, x, axis=axis)
                assert_equal(p.c.shape,
                             c_s[axis:axis+2] + c_s[:axis] + c_s[axis+2:])
                res = p(xp)
                targ_shape = c_s[:axis] + xp.shape + c_s[2+axis:]
                assert_equal(res.shape, targ_shape)

                # deriv/antideriv does not drop the axis
                for p1 in [cls(c, x, axis=axis).derivative(),
                           cls(c, x, axis=axis).derivative(2),
                           cls(c, x, axis=axis).antiderivative(),
                           cls(c, x, axis=axis).antiderivative(2)]:
                    assert_equal(p1.axis, p.axis)

        # c array needs two axes for the coefficients and intervals, so
        # 0 <= axis < c.ndim-1; raise otherwise
        for axis in (-1, 4, 5, 6):
            for cls in (BPoly, PPoly):
                assert_raises(ValueError, cls, **dict(c=c, x=x, axis=axis))


class TestPolySubclassing:
    class P(PPoly):
        pass

    class B(BPoly):
        pass

    def _make_polynomials(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(3))
        c = np.random.random((4, 2))
        return self.P(c, x), self.B(c, x)

    def test_derivative(self):
        pp, bp = self._make_polynomials()
        for p in (pp, bp):
            pd = p.derivative()
            assert_equal(p.__class__, pd.__class__)

        ppa = pp.antiderivative()
        assert_equal(pp.__class__, ppa.__class__)

    def test_from_spline(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0)
        pp = self.P.from_spline(spl)
        assert_equal(pp.__class__, self.P)

    def test_conversions(self):
        pp, bp = self._make_polynomials()

        pp1 = self.P.from_bernstein_basis(bp)
        assert_equal(pp1.__class__, self.P)

        bp1 = self.B.from_power_basis(pp)
        assert_equal(bp1.__class__, self.B)

    def test_from_derivatives(self):
        x = [0, 1, 2]
        y = [[1], [2], [3]]
        bp = self.B.from_derivatives(x, y)
        assert_equal(bp.__class__, self.B)


class TestPPoly:
    def test_simple(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x)
        assert_allclose(p(0.3), 1*0.3**2 + 2*0.3 + 3)
        assert_allclose(p(0.7), 4*(0.7-0.5)**2 + 5*(0.7-0.5) + 6)

    def test_periodic(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x, extrapolate='periodic')

        assert_allclose(p(1.3), 1 * 0.3 ** 2 + 2 * 0.3 + 3)
        assert_allclose(p(-0.3), 4 * (0.7 - 0.5) ** 2 + 5 * (0.7 - 0.5) + 6)

        assert_allclose(p(1.3, 1), 2 * 0.3 + 2)
        assert_allclose(p(-0.3, 1), 8 * (0.7 - 0.5) + 5)

    def test_read_only(self):
        c = np.array([[1, 4], [2, 5], [3, 6]])
        x = np.array([0, 0.5, 1])
        xnew = np.array([0, 0.1, 0.2])
        PPoly(c, x, extrapolate='periodic')

        for writeable in (True, False):
            x.flags.writeable = writeable
            f = PPoly(c, x)
            vals = f(xnew)
            assert_(np.isfinite(vals).all())

    def test_descending(self):
        def binom_matrix(power):
            n = np.arange(power + 1).reshape(-1, 1)
            k = np.arange(power + 1)
            B = binom(n, k)
            return B[::-1, ::-1]

        np.random.seed(0)

        power = 3
        for m in [10, 20, 30]:
            x = np.sort(np.random.uniform(0, 10, m + 1))
            ca = np.random.uniform(-2, 2, size=(power + 1, m))

            h = np.diff(x)
            h_powers = h[None, :] ** np.arange(power + 1)[::-1, None]
            B = binom_matrix(power)
            cap = ca * h_powers
            cdp = np.dot(B.T, cap)
            cd = cdp / h_powers

            pa = PPoly(ca, x, extrapolate=True)
            pd = PPoly(cd[:, ::-1], x[::-1], extrapolate=True)

            x_test = np.random.uniform(-10, 20, 100)
            assert_allclose(pa(x_test), pd(x_test), rtol=1e-13)
            assert_allclose(pa(x_test, 1), pd(x_test, 1), rtol=1e-13)

            pa_d = pa.derivative()
            pd_d = pd.derivative()

            assert_allclose(pa_d(x_test), pd_d(x_test), rtol=1e-13)

            # Antiderivatives won't be equal because fixing continuity is
            # done in the reverse order, but surely the differences should be
            # equal.
            pa_i = pa.antiderivative()
            pd_i = pd.antiderivative()
            for a, b in np.random.uniform(-10, 20, (5, 2)):
                int_a = pa.integrate(a, b)
                int_d = pd.integrate(a, b)
                assert_allclose(int_a, int_d, rtol=1e-13)
                assert_allclose(pa_i(b) - pa_i(a), pd_i(b) - pd_i(a),
                                rtol=1e-13)

            roots_d = pd.roots()
            roots_a = pa.roots()
            assert_allclose(roots_a, np.sort(roots_d), rtol=1e-12)

    def test_multi_shape(self):
        c = np.random.rand(6, 2, 1, 2, 3)
        x = np.array([0, 0.5, 1])
        p = PPoly(c, x)
        assert_equal(p.x.shape, x.shape)
        assert_equal(p.c.shape, c.shape)
        assert_equal(p(0.3).shape, c.shape[2:])

        assert_equal(p(np.random.rand(5, 6)).shape, (5, 6) + c.shape[2:])

        dp = p.derivative()
        assert_equal(dp.c.shape, (5, 2, 1, 2, 3))
        ip = p.antiderivative()
        assert_equal(ip.c.shape, (7, 2, 1, 2, 3))

    def test_construct_fast(self):
        np.random.seed(1234)
        c = np.array([[1, 4], [2, 5], [3, 6]], dtype=float)
        x = np.array([0, 0.5, 1])
        p = PPoly.construct_fast(c, x)
        assert_allclose(p(0.3), 1*0.3**2 + 2*0.3 + 3)
        assert_allclose(p(0.7), 4*(0.7-0.5)**2 + 5*(0.7-0.5) + 6)

    def test_vs_alternative_implementations(self):
        np.random.seed(1234)
        c = np.random.rand(3, 12, 22)
        x = np.sort(np.r_[0, np.random.rand(11), 1])

        p = PPoly(c, x)

        xp = np.r_[0.3, 0.5, 0.33, 0.6]
        expected = _ppoly_eval_1(c, x, xp)
        assert_allclose(p(xp), expected)

        expected = _ppoly_eval_2(c[:,:,0], x, xp)
        assert_allclose(p(xp)[:,0], expected)

    def test_from_spline(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0)
        pp = PPoly.from_spline(spl)

        xi = np.linspace(0, 1, 200)
        assert_allclose(pp(xi), splev(xi, spl))

        # make sure .from_spline accepts BSpline objects
        b = BSpline(*spl)
        ppp = PPoly.from_spline(b)
        assert_allclose(ppp(xi), b(xi))

        # BSpline's extrapolate attribute propagates unless overridden
        t, c, k = spl
        for extrap in (None, True, False):
            b = BSpline(t, c, k, extrapolate=extrap)
            p = PPoly.from_spline(b)
            assert_equal(p.extrapolate, b.extrapolate)

    def test_derivative_simple(self):
        np.random.seed(1234)
        c = np.array([[4, 3, 2, 1]]).T
        dc = np.array([[3*4, 2*3, 2]]).T
        ddc = np.array([[2*3*4, 1*2*3]]).T
        x = np.array([0, 1])

        pp = PPoly(c, x)
        dpp = PPoly(dc, x)
        ddpp = PPoly(ddc, x)

        assert_allclose(pp.derivative().c, dpp.c)
        assert_allclose(pp.derivative(2).c, ddpp.c)

    def test_derivative_eval(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0)
        pp = PPoly.from_spline(spl)

        xi = np.linspace(0, 1, 200)
        for dx in range(0, 3):
            assert_allclose(pp(xi, dx), splev(xi, spl, dx))

    def test_derivative(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)

        xi = np.linspace(0, 1, 200)
        for dx in range(0, 10):
            assert_allclose(pp(xi, dx), pp.derivative(dx)(xi),
                            err_msg="dx=%d" % (dx,))

    def test_antiderivative_of_constant(self):
        # https://github.com/scipy/scipy/issues/4216
        p = PPoly([[1.]], [0, 1])
        assert_equal(p.antiderivative().c, PPoly([[1], [0]], [0, 1]).c)
        assert_equal(p.antiderivative().x, PPoly([[1], [0]], [0, 1]).x)

    def test_antiderivative_regression_4355(self):
        # https://github.com/scipy/scipy/issues/4355
        p = PPoly([[1., 0.5]], [0, 1, 2])
        q = p.antiderivative()
        assert_equal(q.c, [[1, 0.5], [0, 1]])
        assert_equal(q.x, [0, 1, 2])
        assert_allclose(p.integrate(0, 2), 1.5)
        assert_allclose(q(2) - q(0), 1.5)

    def test_antiderivative_simple(self):
        np.random.seed(1234)
        # [ p1(x) = 3*x**2 + 2*x + 1,
        #   p2(x) = 1.6875]
        c = np.array([[3, 2, 1], [0, 0, 1.6875]]).T
        # [ pp1(x) = x**3 + x**2 + x,
        #   pp2(x) = 1.6875*(x - 0.25) + pp1(0.25)]
        ic = np.array([[1, 1, 1, 0], [0, 0, 1.6875, 0.328125]]).T
        # [ ppp1(x) = (1/4)*x**4 + (1/3)*x**3 + (1/2)*x**2,
        #   ppp2(x) = (1.6875/2)*(x - 0.25)**2 + pp1(0.25)*x + ppp1(0.25)]
        iic = np.array([[1/4, 1/3, 1/2, 0, 0],
                        [0, 0, 1.6875/2, 0.328125, 0.037434895833333336]]).T
        x = np.array([0, 0.25, 1])

        pp = PPoly(c, x)
        ipp = pp.antiderivative()
        iipp = pp.antiderivative(2)
        iipp2 = ipp.antiderivative()

        assert_allclose(ipp.x, x)
        assert_allclose(ipp.c.T, ic.T)
        assert_allclose(iipp.c.T, iic.T)
        assert_allclose(iipp2.c.T, iic.T)

    def test_antiderivative_vs_derivative(self):
        np.random.seed(1234)
        x = np.linspace(0, 1, 30)**2
        y = np.random.rand(len(x))
        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)

        for dx in range(0, 10):
            ipp = pp.antiderivative(dx)

            # check that derivative is inverse op
            pp2 = ipp.derivative(dx)
            assert_allclose(pp.c, pp2.c)

            # check continuity
            for k in range(dx):
                pp2 = ipp.derivative(k)

                r = 1e-13
                endpoint = r*pp2.x[:-1] + (1 - r)*pp2.x[1:]

                assert_allclose(pp2(pp2.x[1:]), pp2(endpoint),
                                rtol=1e-7, err_msg="dx=%d k=%d" % (dx, k))

    def test_antiderivative_vs_spline(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)

        for dx in range(0, 10):
            pp2 = pp.antiderivative(dx)
            spl2 = splantider(spl, dx)

            xi = np.linspace(0, 1, 200)
            assert_allclose(pp2(xi), splev(xi, spl2),
                            rtol=1e-7)

    def test_antiderivative_continuity(self):
        c = np.array([[2, 1, 2, 2], [2, 1, 3, 3]]).T
        x = np.array([0, 0.5, 1])

        p = PPoly(c, x)
        ip = p.antiderivative()

        # check continuity
        assert_allclose(ip(0.5 - 1e-9), ip(0.5 + 1e-9), rtol=1e-8)

        # check that only lowest order coefficients were changed
        p2 = ip.derivative()
        assert_allclose(p2.c, p.c)

    def test_integrate(self):
        np.random.seed(1234)
        x = np.sort(np.r_[0, np.random.rand(11), 1])
        y = np.random.rand(len(x))

        spl = splrep(x, y, s=0, k=5)
        pp = PPoly.from_spline(spl)

        a, b = 0.3, 0.9
        ig = pp.integrate(a, b)

        ipp = pp.antiderivative()
        assert_allclose(ig, ipp(b) - ipp(a))
        assert_allclose(ig, splint(a, b, spl))

        a, b = -0.3, 0.9
        ig = pp.integrate(a, b, extrapolate=True)
        assert_allclose(ig, ipp(b) - ipp(a))

        assert_(np.isnan(pp.integrate(a, b, extrapolate=False)).all())

    def test_integrate_readonly(self):
        x = np.array([1, 2, 4])
        c = np.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])

        for writeable in (True, False):
            x.flags.writeable = writeable

            P = PPoly(c, x)
            vals = P.integrate(1, 4)

            assert_(np.isfinite(vals).all())

    def test_integrate_periodic(self):
        x = np.array([1, 2, 4])
        c = np.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])

        P = PPoly(c, x, extrapolate='periodic')
        I = P.antiderivative()

        period_int = I(4) - I(1)

        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)

        assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
        assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5) + 4 * period_int)

        assert_allclose(P.integrate(0, -1), I(2) - I(3))
        assert_allclose(P.integrate(-9, -10), I(2) - I(3))
        assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)

    def test_roots(self):
        x = np.linspace(0, 1, 31)**2
        y = np.sin(30*x)

        spl = splrep(x, y, s=0, k=3)
        pp = PPoly.from_spline(spl)

        r = pp.roots()
        r = r[(r >= 0 - 1e-15) & (r <= 1 + 1e-15)]
        assert_allclose(r, sproot(spl), atol=1e-15)

    def test_roots_idzero(self):
        # Roots for piecewise polynomials with identically zero
        # sections.
        c = np.array([[-1, 0.25], [0, 0], [-1, 0.25]]).T
        x = np.array([0, 0.4, 0.6, 1.0])

        pp = PPoly(c, x)
        assert_array_equal(pp.roots(),
                           [0.25, 0.4, np.nan, 0.6 + 0.25])

        # ditto for p.solve(const) with sections identically equal const
        const = 2.
        c1 = c.copy()
        c1[1, :] += const
        pp1 = PPoly(c1, x)

        assert_array_equal(pp1.solve(const),
                           [0.25, 0.4, np.nan, 0.6 + 0.25])

    def test_roots_all_zero(self):
        # test the code path for the polynomial being identically zero everywhere
        c = [[0], [0]]
        x = [0, 1]
        p = PPoly(c, x)
        assert_array_equal(p.roots(), [0, np.nan])
        assert_array_equal(p.solve(0), [0, np.nan])
        assert_array_equal(p.solve(1), [])

        c = [[0, 0], [0, 0]]
        x = [0, 1, 2]
        p = PPoly(c, x)
        assert_array_equal(p.roots(), [0, np.nan, 1, np.nan])
        assert_array_equal(p.solve(0), [0, np.nan, 1, np.nan])
        assert_array_equal(p.solve(1), [])

    def test_roots_repeated(self):
        # Check roots repeated in multiple sections are reported only
        # once.

        # [(x + 1)**2 - 1, -x**2] ; x == 0 is a repeated root
        c = np.array([[1, 0, -1], [-1, 0, 0]]).T
        x = np.array([-1, 0, 1])

        pp = PPoly(c, x)
        assert_array_equal(pp.roots(), [-2, 0])
        assert_array_equal(pp.roots(extrapolate=False), [0])

    def test_roots_discont(self):
        # Check that a discontinuity across zero is reported as root
        c = np.array([[1], [-1]]).T
        x = np.array([0, 0.5, 1])
        pp = PPoly(c, x)
        assert_array_equal(pp.roots(), [0.5])
        assert_array_equal(pp.roots(discontinuity=False), [])

        # ditto for a discontinuity across y:
        assert_array_equal(pp.solve(0.5), [0.5])
        assert_array_equal(pp.solve(0.5, discontinuity=False), [])

        assert_array_equal(pp.solve(1.5), [])
        assert_array_equal(pp.solve(1.5, discontinuity=False), [])

    def test_roots_random(self):
        # Check high-order polynomials with random coefficients
        np.random.seed(1234)

        num = 0

        for extrapolate in (True, False):
            for order in range(0, 20):
                x = np.unique(np.r_[0, 10 * np.random.rand(30), 10])
                c = 2*np.random.rand(order+1, len(x)-1, 2, 3) - 1

                pp = PPoly(c, x)
                for y in [0, np.random.random()]:
                    r = pp.solve(y, discontinuity=False, extrapolate=extrapolate)

                    for i in range(2):
                        for j in range(3):
                            rr = r[i,j]
                            if rr.size > 0:
                                # Check that the reported roots indeed are roots
                                num += rr.size
                                val = pp(rr, extrapolate=extrapolate)[:,i,j]
                                cmpval = pp(rr, nu=1,
                                            extrapolate=extrapolate)[:,i,j]
                                msg = "(%r) r = %s" % (extrapolate, repr(rr),)
                                assert_allclose((val-y) / cmpval, 0, atol=1e-7,
                                                err_msg=msg)

        # Check that we checked a number of roots
        assert_(num > 100, repr(num))

    def test_roots_croots(self):
        # Test the complex root finding algorithm
        np.random.seed(1234)

        for k in range(1, 15):
            c = np.random.rand(k, 1, 130)

            if k == 3:
                # add a case with zero discriminant
                c[:,0,0] = 1, 2, 1

            for y in [0, np.random.random()]:
                w = np.empty(c.shape, dtype=complex)
                _ppoly._croots_poly1(c, w)

                if k == 1:
                    assert_(np.isnan(w).all())
                    continue

                res = 0
                cres = 0
                for i in range(k):
                    res += c[i,None] * w**(k-1-i)
                    cres += abs(c[i,None] * w**(k-1-i))
                with np.errstate(invalid='ignore'):
                    res /= cres
                res = res.ravel()
                res = res[~np.isnan(res)]
                assert_allclose(res, 0, atol=1e-10)

    def test_extrapolate_attr(self):
        # [ 1 - x**2 ]
        c = np.array([[-1, 0, 1]]).T
        x = np.array([0, 1])

        for extrapolate in [True, False, None]:
            pp = PPoly(c, x, extrapolate=extrapolate)
            pp_d = pp.derivative()
            pp_i = pp.antiderivative()

            if extrapolate is False:
                assert_(np.isnan(pp([-0.1, 1.1])).all())
                assert_(np.isnan(pp_i([-0.1, 1.1])).all())
                assert_(np.isnan(pp_d([-0.1, 1.1])).all())
                assert_equal(pp.roots(), [1])
            else:
                assert_allclose(pp([-0.1, 1.1]), [1-0.1**2, 1-1.1**2])
                assert_(not np.isnan(pp_i([-0.1, 1.1])).any())
                assert_(not np.isnan(pp_d([-0.1, 1.1])).any())
                assert_allclose(pp.roots(), [1, -1])


class TestBPoly:
    def test_simple(self):
        x = [0, 1]
        c = [[3]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.1), 3.)

    def test_simple2(self):
        x = [0, 1]
        c = [[3], [1]]
        bp = BPoly(c, x)   # 3*(1-x) + 1*x
        assert_allclose(bp(0.1), 3*0.9 + 1.*0.1)

    def test_simple3(self):
        x = [0, 1]
        c = [[3], [1], [4]]
        bp = BPoly(c, x)   # 3 * (1-x)**2 + 2 * x (1-x) + 4 * x**2
        assert_allclose(bp(0.2),
                3 * 0.8*0.8 + 1 * 2*0.2*0.8 + 4 * 0.2*0.2)

    def test_simple4(self):
        x = [0, 1]
        c = [[1], [1], [1], [2]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.3), 0.7**3 +
                                 3 * 0.7**2 * 0.3 +
                                 3 * 0.7 * 0.3**2 +
                             2 * 0.3**3)

    def test_simple5(self):
        x = [0, 1]
        c = [[1], [1], [8], [2], [1]]
        bp = BPoly(c, x)
        assert_allclose(bp(0.3), 0.7**4 +
                                 4 * 0.7**3 * 0.3 +
                             8 * 6 * 0.7**2 * 0.3**2 +
                             2 * 4 * 0.7 * 0.3**3 +
                                 0.3**4)

    def test_periodic(self):
        x = [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        # [3*(1-x)**2, 2*((x-1)/2)**2]
        bp = BPoly(c, x, extrapolate='periodic')

        assert_allclose(bp(3.4), 3 * 0.6**2)
        assert_allclose(bp(-1.3), 2 * (0.7/2)**2)

        assert_allclose(bp(3.4, 1), -6 * 0.6)
        assert_allclose(bp(-1.3, 1), 2 * (0.7/2))

    def test_descending(self):
        np.random.seed(0)

        power = 3
        for m in [10, 20, 30]:
            x = np.sort(np.random.uniform(0, 10, m + 1))
            ca = np.random.uniform(-0.1, 0.1, size=(power + 1, m))
            # We need only to flip coefficients to get it right!
            cd = ca[::-1].copy()

            pa = BPoly(ca, x, extrapolate=True)
            pd = BPoly(cd[:, ::-1], x[::-1], extrapolate=True)

            x_test = np.random.uniform(-10, 20, 100)
            assert_allclose(pa(x_test), pd(x_test), rtol=1e-13)
            assert_allclose(pa(x_test, 1), pd(x_test, 1), rtol=1e-13)

            pa_d = pa.derivative()
            pd_d = pd.derivative()

            assert_allclose(pa_d(x_test), pd_d(x_test), rtol=1e-13)

            # Antiderivatives won't be equal because fixing continuity is
            # done in the reverse order, but surely the differences should be
            # equal.
            pa_i = pa.antiderivative()
            pd_i = pd.antiderivative()
            for a, b in np.random.uniform(-10, 20, (5, 2)):
                int_a = pa.integrate(a, b)
                int_d = pd.integrate(a, b)
                assert_allclose(int_a, int_d, rtol=1e-12)
                assert_allclose(pa_i(b) - pa_i(a), pd_i(b) - pd_i(a),
                                rtol=1e-12)

    def test_multi_shape(self):
        c = np.random.rand(6, 2, 1, 2, 3)
        x = np.array([0, 0.5, 1])
        p = BPoly(c, x)
        assert_equal(p.x.shape, x.shape)
        assert_equal(p.c.shape, c.shape)
        assert_equal(p(0.3).shape, c.shape[2:])
        assert_equal(p(np.random.rand(5,6)).shape,
                     (5,6)+c.shape[2:])

        dp = p.derivative()
        assert_equal(dp.c.shape, (5, 2, 1, 2, 3))

    def test_interval_length(self):
        x = [0, 2]
        c = [[3], [1], [4]]
        bp = BPoly(c, x)
        xval = 0.1
        s = xval / 2  # s = (x - xa) / (xb - xa)
        assert_allclose(bp(xval), 3 * (1-s)*(1-s) + 1 * 2*s*(1-s) + 4 * s*s)

    def test_two_intervals(self):
        x = [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        bp = BPoly(c, x)  # [3*(1-x)**2, 2*((x-1)/2)**2]

        assert_allclose(bp(0.4), 3 * 0.6*0.6)
        assert_allclose(bp(1.7), 2 * (0.7/2)**2)

    def test_extrapolate_attr(self):
        x = [0, 2]
        c = [[3], [1], [4]]
        bp = BPoly(c, x)

        for extrapolate in (True, False, None):
            bp = BPoly(c, x, extrapolate=extrapolate)
            bp_d = bp.derivative()
            if extrapolate is False:
                assert_(np.isnan(bp([-0.1, 2.1])).all())
                assert_(np.isnan(bp_d([-0.1, 2.1])).all())
            else:
                assert_(not np.isnan(bp([-0.1, 2.1])).any())
                assert_(not np.isnan(bp_d([-0.1, 2.1])).any())


class TestBPolyCalculus:
    def test_derivative(self):
        x = [0, 1, 3]
        c = [[3, 0], [0, 0], [0, 2]]
        bp = BPoly(c, x)  # [3*(1-x)**2, 2*((x-1)/2)**2]
        bp_der = bp.derivative()
        assert_allclose(bp_der(0.4), -6*(0.6))
        assert_allclose(bp_der(1.7), 0.7)

        # derivatives in-place
        assert_allclose([bp(0.4, nu=1), bp(0.4, nu=2), bp(0.4, nu=3)],
                        [-6*(1-0.4), 6., 0.])
        assert_allclose([bp(1.7, nu=1), bp(1.7, nu=2), bp(1.7, nu=3)],
                        [0.7, 1., 0])

    def test_derivative_ppoly(self):
        # make sure it's consistent w/ power basis
        np.random.seed(1234)
        m, k = 5, 8   # number of intervals, order
        x = np.sort(np.random.random(m))
        c = np.random.random((k, m-1))
        bp = BPoly(c, x)
        pp = PPoly.from_bernstein_basis(bp)

        for d in range(k):
            bp = bp.derivative()
            pp = pp.derivative()
            xp = np.linspace(x[0], x[-1], 21)
            assert_allclose(bp(xp), pp(xp))

    def test_deriv_inplace(self):
        np.random.seed(1234)
        m, k = 5, 8   # number of intervals, order
        x = np.sort(np.random.random(m))
        c = np.random.random((k, m-1))

        # test both real and complex coefficients
        for cc in [c.copy(), c*(1. + 2.j)]:
            bp = BPoly(cc, x)
            xp = np.linspace(x[0], x[-1], 21)
            for i in range(k):
                assert_allclose(bp(xp, i), bp.derivative(i)(xp))

    def test_antiderivative_simple(self):
        # f(x) = x        for x \in [0, 1),
        #        (x-1)/2  for x \in [1, 3]
        #
        # antiderivative is then
        # F(x) = x**2 / 2            for x \in [0, 1),
        #        0.5*x*(x/2 - 1) + A  for x \in [1, 3]
        # where A = 3/4 for continuity at x = 1.
        x = [0, 1, 3]
        c = [[0, 0], [1, 1]]

        bp = BPoly(c, x)
        bi = bp.antiderivative()

        xx = np.linspace(0, 3, 11)
        assert_allclose(bi(xx),
                        np.where(xx < 1, xx**2 / 2.,
                                         0.5 * xx * (xx/2. - 1) + 3./4),
                        atol=1e-12, rtol=1e-12)

    def test_der_antider(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10, 2, 3))
        bp = BPoly(c, x)

        xx = np.linspace(x[0], x[-1], 100)
        assert_allclose(bp.antiderivative().derivative()(xx),
                        bp(xx), atol=1e-12, rtol=1e-12)

    def test_antider_ppoly(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10, 2, 3))
        bp = BPoly(c, x)
        pp = PPoly.from_bernstein_basis(bp)

        xx = np.linspace(x[0], x[-1], 10)

        assert_allclose(bp.antiderivative(2)(xx),
                        pp.antiderivative(2)(xx), atol=1e-12, rtol=1e-12)

    def test_antider_continuous(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10))
        bp = BPoly(c, x).antiderivative()

        xx = bp.x[1:-1]
        assert_allclose(bp(xx - 1e-14),
                        bp(xx + 1e-14), atol=1e-12, rtol=1e-12)

    def test_integrate(self):
        np.random.seed(1234)
        x = np.sort(np.random.random(11))
        c = np.random.random((4, 10))
        bp = BPoly(c, x)
        pp = PPoly.from_bernstein_basis(bp)
        assert_allclose(bp.integrate(0, 1),
                        pp.integrate(0, 1), atol=1e-12, rtol=1e-12)

    def test_integrate_extrap(self):
        c = [[1]]
        x = [0, 1]
        b = BPoly(c, x)

        # default is extrapolate=True
        assert_allclose(b.integrate(0, 2), 2., atol=1e-14)

        # .integrate argument overrides self.extrapolate
        b1 = BPoly(c, x, extrapolate=False)
        assert_(np.isnan(b1.integrate(0, 2)))
        assert_allclose(b1.integrate(0, 2, extrapolate=True), 2., atol=1e-14)

    def test_integrate_periodic(self):
        x = np.array([1, 2, 4])
        c = np.array([[0., 0.], [-1., -1.], [2., -0.], [1., 2.]])

        P = BPoly.from_power_basis(PPoly(c, x), extrapolate='periodic')
        I = P.antiderivative()

        period_int = I(4) - I(1)

        assert_allclose(P.integrate(1, 4), period_int)
        assert_allclose(P.integrate(-10, -7), period_int)
        assert_allclose(P.integrate(-10, -4), 2 * period_int)

        assert_allclose(P.integrate(1.5, 2.5), I(2.5) - I(1.5))
        assert_allclose(P.integrate(3.5, 5), I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5 + 12, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5))
        assert_allclose(P.integrate(3.5, 5 + 12),
                        I(2) - I(1) + I(4) - I(3.5) + 4 * period_int)

        assert_allclose(P.integrate(0, -1), I(2) - I(3))
        assert_allclose(P.integrate(-9, -10), I(2) - I(3))
        assert_allclose(P.integrate(0, -10), I(2) - I(3) - 3 * period_int)

    def test_antider_neg(self):
        # .derivative(-nu) ==> .andiderivative(nu) and vice versa
        c = [[1]]
        x = [0, 1]
        b = BPoly(c, x)

        xx = np.linspace(0, 1, 21)

        assert_allclose(b.derivative(-1)(xx), b.antiderivative()(xx),
                        atol=1e-12, rtol=1e-12)
        assert_allclose(b.derivative(1)(xx), b.antiderivative(-1)(xx),
                        atol=1e-12, rtol=1e-12)


class TestPolyConversions:
    def test_bp_from_pp(self):
        x = [0, 1, 3]
        c = [[3, 2], [1, 8], [4, 3]]
        pp = PPoly(c, x)
        bp = BPoly.from_power_basis(pp)
        pp1 = PPoly.from_bernstein_basis(bp)

        xp = [0.1, 1.4]
        assert_allclose(pp(xp), bp(xp))
        assert_allclose(pp(xp), pp1(xp))

    def test_bp_from_pp_random(self):
        np.random.seed(1234)
        m, k = 5, 8   # number of intervals, order
        x = np.sort(np.random.random(m))
        c = np.random.random((k, m-1))
        pp = PPoly(c, x)
        bp = BPoly.from_power_basis(pp)
        pp1 = PPoly.from_bernstein_basis(bp)

        xp = np.linspace(x[0], x[-1], 21)
        assert_allclose(pp(xp), bp(xp))
        assert_allclose(pp(xp), pp1(xp))

    def test_pp_from_bp(self):
        x = [0, 1, 3]
        c = [[3, 3], [1, 1], [4, 2]]
        bp = BPoly(c, x)
        pp = PPoly.from_bernstein_basis(bp)
        bp1 = BPoly.from_power_basis(pp)

        xp = [0.1, 1.4]
        assert_allclose(bp(xp), pp(xp))
        assert_allclose(bp(xp), bp1(xp))

    def test_broken_conversions(self):
        # regression test for gh-10597: from_power_basis only accepts PPoly etc.
        x = [0, 1, 3]
        c = [[3, 3], [1, 1], [4, 2]]
        pp = PPoly(c, x)
        with assert_raises(TypeError):
            PPoly.from_bernstein_basis(pp)

        bp = BPoly(c, x)
        with assert_raises(TypeError):
            BPoly.from_power_basis(bp)


class TestBPolyFromDerivatives:
    def test_make_poly_1(self):
        c1 = BPoly._construct_from_derivatives(0, 1, [2], [3])
        assert_allclose(c1, [2., 3.])

    def test_make_poly_2(self):
        c1 = BPoly._construct_from_derivatives(0, 1, [1, 0], [1])
        assert_allclose(c1, [1., 1., 1.])

        # f'(0) = 3
        c2 = BPoly._construct_from_derivatives(0, 1, [2, 3], [1])
        assert_allclose(c2, [2., 7./2, 1.])

        # f'(1) = 3
        c3 = BPoly._construct_from_derivatives(0, 1, [2], [1, 3])
        assert_allclose(c3, [2., -0.5, 1.])

    def test_make_poly_3(self):
        # f'(0)=2, f''(0)=3
        c1 = BPoly._construct_from_derivatives(0, 1, [1, 2, 3], [4])
        assert_allclose(c1, [1., 5./3, 17./6, 4.])

        # f'(1)=2, f''(1)=3
        c2 = BPoly._construct_from_derivatives(0, 1, [1], [4, 2, 3])
        assert_allclose(c2, [1., 19./6, 10./3, 4.])

        # f'(0)=2, f'(1)=3
        c3 = BPoly._construct_from_derivatives(0, 1, [1, 2], [4, 3])
        assert_allclose(c3, [1., 5./3, 3., 4.])

    def test_make_poly_12(self):
        np.random.seed(12345)
        ya = np.r_[0, np.random.random(5)]
        yb = np.r_[0, np.random.random(5)]

        c = BPoly._construct_from_derivatives(0, 1, ya, yb)
        pp = BPoly(c[:, None], [0, 1])
        for j in range(6):
            assert_allclose([pp(0.), pp(1.)], [ya[j], yb[j]])
            pp = pp.derivative()

    def test_raise_degree(self):
        np.random.seed(12345)
        x = [0, 1]
        k, d = 8, 5
        c = np.random.random((k, 1, 2, 3, 4))
        bp = BPoly(c, x)

        c1 = BPoly._raise_degree(c, d)
        bp1 = BPoly(c1, x)

        xp = np.linspace(0, 1, 11)
        assert_allclose(bp(xp), bp1(xp))

    def test_xi_yi(self):
        assert_raises(ValueError, BPoly.from_derivatives, [0, 1], [0])

    def test_coords_order(self):
        xi = [0, 0, 1]
        yi = [[0], [0], [0]]
        assert_raises(ValueError, BPoly.from_derivatives, xi, yi)

    def test_zeros(self):
        xi = [0, 1, 2, 3]
        yi = [[0, 0], [0], [0, 0], [0, 0]]  # NB: will have to raise the degree
        pp = BPoly.from_derivatives(xi, yi)
        assert_(pp.c.shape == (4, 3))

        ppd = pp.derivative()
        for xp in [0., 0.1, 1., 1.1, 1.9, 2., 2.5]:
            assert_allclose([pp(xp), ppd(xp)], [0., 0.])

    def _make_random_mk(self, m, k):
        # k derivatives at each breakpoint
        np.random.seed(1234)
        xi = np.asarray([1. * j**2 for j in range(m+1)])
        yi = [np.random.random(k) for j in range(m+1)]
        return xi, yi

    def test_random_12(self):
        m, k = 5, 12
        xi, yi = self._make_random_mk(m, k)
        pp = BPoly.from_derivatives(xi, yi)

        for order in range(k//2):
            assert_allclose(pp(xi), [yy[order] for yy in yi])
            pp = pp.derivative()

    def test_order_zero(self):
        m, k = 5, 12
        xi, yi = self._make_random_mk(m, k)
        assert_raises(ValueError, BPoly.from_derivatives,
                **dict(xi=xi, yi=yi, orders=0))

    def test_orders_too_high(self):
        m, k = 5, 12
        xi, yi = self._make_random_mk(m, k)

        BPoly.from_derivatives(xi, yi, orders=2*k-1)   # this is still ok
        assert_raises(ValueError, BPoly.from_derivatives,   # but this is not
                **dict(xi=xi, yi=yi, orders=2*k))

    def test_orders_global(self):
        m, k = 5, 12
        xi, yi = self._make_random_mk(m, k)

        # ok, this is confusing. Local polynomials will be of the order 5
        # which means that up to the 2nd derivatives will be used at each point
        order = 5
        pp = BPoly.from_derivatives(xi, yi, orders=order)

        for j in range(order//2+1):
            assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
            pp = pp.derivative()
        assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))

        # now repeat with `order` being even: on each interval, it uses
        # order//2 'derivatives' @ the right-hand endpoint and
        # order//2+1 @ 'derivatives' the left-hand endpoint
        order = 6
        pp = BPoly.from_derivatives(xi, yi, orders=order)
        for j in range(order//2):
            assert_allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12))
            pp = pp.derivative()
        assert_(not np.allclose(pp(xi[1:-1] - 1e-12), pp(xi[1:-1] + 1e-12)))

    def test_orders_local(self):
        m, k = 7, 12
        xi, yi = self._make_random_mk(m, k)

        orders = [o + 1 for o in range(m)]
        for i, x in enumerate(xi[1:-1]):
            pp = BPoly.from_derivatives(xi, yi, orders=orders)
            for j in range(orders[i] // 2 + 1):
                assert_allclose(pp(x - 1e-12), pp(x + 1e-12))
                pp = pp.derivative()
            assert_(not np.allclose(pp(x - 1e-12), pp(x + 1e-12)))

    def test_yi_trailing_dims(self):
        m, k = 7, 5
        xi = np.sort(np.random.random(m+1))
        yi = np.random.random((m+1, k, 6, 7, 8))
        pp = BPoly.from_derivatives(xi, yi)
        assert_equal(pp.c.shape, (2*k, m, 6, 7, 8))

    def test_gh_5430(self):
        # At least one of these raises an error unless gh-5430 is
        # fixed. In py2k an int is implemented using a C long, so
        # which one fails depends on your system. In py3k there is only
        # one arbitrary precision integer type, so both should fail.
        orders = np.int32(1)
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        assert_almost_equal(p(0), 0)
        orders = np.int64(1)
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        assert_almost_equal(p(0), 0)
        orders = 1
        # This worked before; make sure it still works
        p = BPoly.from_derivatives([0, 1], [[0], [0]], orders=orders)
        assert_almost_equal(p(0), 0)
        orders = 1


class TestNdPPoly:
    def test_simple_1d(self):
        np.random.seed(1234)

        c = np.random.rand(4, 5)
        x = np.linspace(0, 1, 5+1)

        xi = np.random.rand(200)

        p = NdPPoly(c, (x,))
        v1 = p((xi,))

        v2 = _ppoly_eval_1(c[:,:,None], x, xi).ravel()
        assert_allclose(v1, v2)

    def test_simple_2d(self):
        np.random.seed(1234)

        c = np.random.rand(4, 5, 6, 7)
        x = np.linspace(0, 1, 6+1)
        y = np.linspace(0, 1, 7+1)**2

        xi = np.random.rand(200)
        yi = np.random.rand(200)

        v1 = np.empty([len(xi), 1], dtype=c.dtype)
        v1.fill(np.nan)
        _ppoly.evaluate_nd(c.reshape(4*5, 6*7, 1),
                           (x, y),
                           np.array([4, 5], dtype=np.intc),
                           np.c_[xi, yi],
                           np.array([0, 0], dtype=np.intc),
                           1,
                           v1)
        v1 = v1.ravel()
        v2 = _ppoly2d_eval(c, (x, y), xi, yi)
        assert_allclose(v1, v2)

        p = NdPPoly(c, (x, y))
        for nu in (None, (0, 0), (0, 1), (1, 0), (2, 3), (9, 2)):
            v1 = p(np.c_[xi, yi], nu=nu)
            v2 = _ppoly2d_eval(c, (x, y), xi, yi, nu=nu)
            assert_allclose(v1, v2, err_msg=repr(nu))

    def test_simple_3d(self):
        np.random.seed(1234)

        c = np.random.rand(4, 5, 6, 7, 8, 9)
        x = np.linspace(0, 1, 7+1)
        y = np.linspace(0, 1, 8+1)**2
        z = np.linspace(0, 1, 9+1)**3

        xi = np.random.rand(40)
        yi = np.random.rand(40)
        zi = np.random.rand(40)

        p = NdPPoly(c, (x, y, z))

        for nu in (None, (0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 3, 0),
                   (6, 0, 2)):
            v1 = p((xi, yi, zi), nu=nu)
            v2 = _ppoly3d_eval(c, (x, y, z), xi, yi, zi, nu=nu)
            assert_allclose(v1, v2, err_msg=repr(nu))

    def test_simple_4d(self):
        np.random.seed(1234)

        c = np.random.rand(4, 5, 6, 7, 8, 9, 10, 11)
        x = np.linspace(0, 1, 8+1)
        y = np.linspace(0, 1, 9+1)**2
        z = np.linspace(0, 1, 10+1)**3
        u = np.linspace(0, 1, 11+1)**4

        xi = np.random.rand(20)
        yi = np.random.rand(20)
        zi = np.random.rand(20)
        ui = np.random.rand(20)

        p = NdPPoly(c, (x, y, z, u))
        v1 = p((xi, yi, zi, ui))

        v2 = _ppoly4d_eval(c, (x, y, z, u), xi, yi, zi, ui)
        assert_allclose(v1, v2)

    def test_deriv_1d(self):
        np.random.seed(1234)

        c = np.random.rand(4, 5)
        x = np.linspace(0, 1, 5+1)

        p = NdPPoly(c, (x,))

        # derivative
        dp = p.derivative(nu=[1])
        p1 = PPoly(c, x)
        dp1 = p1.derivative()
        assert_allclose(dp.c, dp1.c)

        # antiderivative
        dp = p.antiderivative(nu=[2])
        p1 = PPoly(c, x)
        dp1 = p1.antiderivative(2)
        assert_allclose(dp.c, dp1.c)

    def test_deriv_3d(self):
        np.random.seed(1234)

        c = np.random.rand(4, 5, 6, 7, 8, 9)
        x = np.linspace(0, 1, 7+1)
        y = np.linspace(0, 1, 8+1)**2
        z = np.linspace(0, 1, 9+1)**3

        p = NdPPoly(c, (x, y, z))

        # differentiate vs x
        p1 = PPoly(c.transpose(0, 3, 1, 2, 4, 5), x)
        dp = p.derivative(nu=[2])
        dp1 = p1.derivative(2)
        assert_allclose(dp.c,
                        dp1.c.transpose(0, 2, 3, 1, 4, 5))

        # antidifferentiate vs y
        p1 = PPoly(c.transpose(1, 4, 0, 2, 3, 5), y)
        dp = p.antiderivative(nu=[0, 1, 0])
        dp1 = p1.antiderivative(1)
        assert_allclose(dp.c,
                        dp1.c.transpose(2, 0, 3, 4, 1, 5))

        # differentiate vs z
        p1 = PPoly(c.transpose(2, 5, 0, 1, 3, 4), z)
        dp = p.derivative(nu=[0, 0, 3])
        dp1 = p1.derivative(3)
        assert_allclose(dp.c,
                        dp1.c.transpose(2, 3, 0, 4, 5, 1))

    def test_deriv_3d_simple(self):
        # Integrate to obtain function x y**2 z**4 / (2! 4!)

        c = np.ones((1, 1, 1, 3, 4, 5))
        x = np.linspace(0, 1, 3+1)**1
        y = np.linspace(0, 1, 4+1)**2
        z = np.linspace(0, 1, 5+1)**3

        p = NdPPoly(c, (x, y, z))
        ip = p.antiderivative((1, 0, 4))
        ip = ip.antiderivative((0, 2, 0))

        xi = np.random.rand(20)
        yi = np.random.rand(20)
        zi = np.random.rand(20)

        assert_allclose(ip((xi, yi, zi)),
                        xi * yi**2 * zi**4 / (gamma(3)*gamma(5)))

    def test_integrate_2d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 16, 17)
        x = np.linspace(0, 1, 16+1)**1
        y = np.linspace(0, 1, 17+1)**2

        # make continuously differentiable so that nquad() has an
        # easier time
        c = c.transpose(0, 2, 1, 3)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        _ppoly.fix_continuity(cx, x, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(0, 2, 1, 3)
        c = c.transpose(1, 3, 0, 2)
        cx = c.reshape(c.shape[0], c.shape[1], -1).copy()
        _ppoly.fix_continuity(cx, y, 2)
        c = cx.reshape(c.shape)
        c = c.transpose(2, 0, 3, 1).copy()

        # Check integration
        p = NdPPoly(c, (x, y))

        for ranges in [[(0, 1), (0, 1)],
                       [(0, 0.5), (0, 1)],
                       [(0, 1), (0, 0.5)],
                       [(0.3, 0.7), (0.6, 0.2)]]:

            ig = p.integrate(ranges)
            ig2, err2 = nquad(lambda x, y: p((x, y)), ranges,
                              opts=[dict(epsrel=1e-5, epsabs=1e-5)]*2)
            assert_allclose(ig, ig2, rtol=1e-5, atol=1e-5,
                            err_msg=repr(ranges))

    def test_integrate_1d(self):
        np.random.seed(1234)
        c = np.random.rand(4, 5, 6, 16, 17, 18)
        x = np.linspace(0, 1, 16+1)**1
        y = np.linspace(0, 1, 17+1)**2
        z = np.linspace(0, 1, 18+1)**3

        # Check 1-D integration
        p = NdPPoly(c, (x, y, z))

        u = np.random.rand(200)
        v = np.random.rand(200)
        a, b = 0.2, 0.7

        px = p.integrate_1d(a, b, axis=0)
        pax = p.antiderivative((1, 0, 0))
        assert_allclose(px((u, v)), pax((b, u, v)) - pax((a, u, v)))

        py = p.integrate_1d(a, b, axis=1)
        pay = p.antiderivative((0, 1, 0))
        assert_allclose(py((u, v)), pay((u, b, v)) - pay((u, a, v)))

        pz = p.integrate_1d(a, b, axis=2)
        paz = p.antiderivative((0, 0, 1))
        assert_allclose(pz((u, v)), paz((u, v, b)) - paz((u, v, a)))


def _ppoly_eval_1(c, x, xps):
    """Evaluate piecewise polynomial manually"""
    out = np.zeros((len(xps), c.shape[2]))
    for i, xp in enumerate(xps):
        if xp < 0 or xp > 1:
            out[i,:] = np.nan
            continue
        j = np.searchsorted(x, xp) - 1
        d = xp - x[j]
        assert_(x[j] <= xp < x[j+1])
        r = sum(c[k,j] * d**(c.shape[0]-k-1)
                for k in range(c.shape[0]))
        out[i,:] = r
    return out


def _ppoly_eval_2(coeffs, breaks, xnew, fill=np.nan):
    """Evaluate piecewise polynomial manually (another way)"""
    a = breaks[0]
    b = breaks[-1]
    K = coeffs.shape[0]

    saveshape = np.shape(xnew)
    xnew = np.ravel(xnew)
    res = np.empty_like(xnew)
    mask = (xnew >= a) & (xnew <= b)
    res[~mask] = fill
    xx = xnew.compress(mask)
    indxs = np.searchsorted(breaks, xx)-1
    indxs = indxs.clip(0, len(breaks))
    pp = coeffs
    diff = xx - breaks.take(indxs)
    V = np.vander(diff, N=K)
    values = np.array([np.dot(V[k, :], pp[:, indxs[k]]) for k in range(len(xx))])
    res[mask] = values
    res.shape = saveshape
    return res


def _dpow(x, y, n):
    """
    d^n (x**y) / dx^n
    """
    if n < 0:
        raise ValueError("invalid derivative order")
    elif n > y:
        return 0
    else:
        return poch(y - n + 1, n) * x**(y - n)


def _ppoly2d_eval(c, xs, xnew, ynew, nu=None):
    """
    Straightforward evaluation of 2-D piecewise polynomial
    """
    if nu is None:
        nu = (0, 0)

    out = np.empty((len(xnew),), dtype=c.dtype)

    nx, ny = c.shape[:2]

    for jout, (x, y) in enumerate(zip(xnew, ynew)):
        if not ((xs[0][0] <= x <= xs[0][-1]) and
                (xs[1][0] <= y <= xs[1][-1])):
            out[jout] = np.nan
            continue

        j1 = np.searchsorted(xs[0], x) - 1
        j2 = np.searchsorted(xs[1], y) - 1

        s1 = x - xs[0][j1]
        s2 = y - xs[1][j2]

        val = 0

        for k1 in range(c.shape[0]):
            for k2 in range(c.shape[1]):
                val += (c[nx-k1-1,ny-k2-1,j1,j2]
                        * _dpow(s1, k1, nu[0])
                        * _dpow(s2, k2, nu[1]))

        out[jout] = val

    return out


def _ppoly3d_eval(c, xs, xnew, ynew, znew, nu=None):
    """
    Straightforward evaluation of 3-D piecewise polynomial
    """
    if nu is None:
        nu = (0, 0, 0)

    out = np.empty((len(xnew),), dtype=c.dtype)

    nx, ny, nz = c.shape[:3]

    for jout, (x, y, z) in enumerate(zip(xnew, ynew, znew)):
        if not ((xs[0][0] <= x <= xs[0][-1]) and
                (xs[1][0] <= y <= xs[1][-1]) and
                (xs[2][0] <= z <= xs[2][-1])):
            out[jout] = np.nan
            continue

        j1 = np.searchsorted(xs[0], x) - 1
        j2 = np.searchsorted(xs[1], y) - 1
        j3 = np.searchsorted(xs[2], z) - 1

        s1 = x - xs[0][j1]
        s2 = y - xs[1][j2]
        s3 = z - xs[2][j3]

        val = 0
        for k1 in range(c.shape[0]):
            for k2 in range(c.shape[1]):
                for k3 in range(c.shape[2]):
                    val += (c[nx-k1-1,ny-k2-1,nz-k3-1,j1,j2,j3]
                            * _dpow(s1, k1, nu[0])
                            * _dpow(s2, k2, nu[1])
                            * _dpow(s3, k3, nu[2]))

        out[jout] = val

    return out


def _ppoly4d_eval(c, xs, xnew, ynew, znew, unew, nu=None):
    """
    Straightforward evaluation of 4-D piecewise polynomial
    """
    if nu is None:
        nu = (0, 0, 0, 0)

    out = np.empty((len(xnew),), dtype=c.dtype)

    mx, my, mz, mu = c.shape[:4]

    for jout, (x, y, z, u) in enumerate(zip(xnew, ynew, znew, unew)):
        if not ((xs[0][0] <= x <= xs[0][-1]) and
                (xs[1][0] <= y <= xs[1][-1]) and
                (xs[2][0] <= z <= xs[2][-1]) and
                (xs[3][0] <= u <= xs[3][-1])):
            out[jout] = np.nan
            continue

        j1 = np.searchsorted(xs[0], x) - 1
        j2 = np.searchsorted(xs[1], y) - 1
        j3 = np.searchsorted(xs[2], z) - 1
        j4 = np.searchsorted(xs[3], u) - 1

        s1 = x - xs[0][j1]
        s2 = y - xs[1][j2]
        s3 = z - xs[2][j3]
        s4 = u - xs[3][j4]

        val = 0
        for k1 in range(c.shape[0]):
            for k2 in range(c.shape[1]):
                for k3 in range(c.shape[2]):
                    for k4 in range(c.shape[3]):
                        val += (c[mx-k1-1,my-k2-1,mz-k3-1,mu-k4-1,j1,j2,j3,j4]
                                * _dpow(s1, k1, nu[0])
                                * _dpow(s2, k2, nu[1])
                                * _dpow(s3, k3, nu[2])
                                * _dpow(s4, k4, nu[3]))

        out[jout] = val

    return out


class TestRegularGridInterpolator:
    def _get_sample_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points = [(0., .5, 1.)] * 4
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def _get_sample_4d_2(self):
        # create another 4-D grid of 3 points in each dimension
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def test_list_input(self):
        points, values = self._get_sample_4d()

        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        for method in ['linear', 'nearest']:
            interp = RegularGridInterpolator(points,
                                             values.tolist(),
                                             method=method)
            v1 = interp(sample.tolist())
            interp = RegularGridInterpolator(points,
                                             values,
                                             method=method)
            v2 = interp(sample)
            assert_allclose(v1, v2)

    def test_complex(self):
        points, values = self._get_sample_4d()
        values = values - 2j*values
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        for method in ['linear', 'nearest']:
            interp = RegularGridInterpolator(points, values,
                                             method=method)
            rinterp = RegularGridInterpolator(points, values.real,
                                              method=method)
            iinterp = RegularGridInterpolator(points, values.imag,
                                              method=method)

            v1 = interp(sample)
            v2 = rinterp(sample) + 1j*iinterp(sample)
            assert_allclose(v1, v2)

    def test_linear_xi1d(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([0.1, 0.1, 10., 9.])
        wanted = 1001.1
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_xi3d(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        wanted = np.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    def test_nearest(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method="nearest")
        sample = np.asarray([0.1, 0.1, .9, .9])
        wanted = 1100.
        assert_array_almost_equal(interp(sample), wanted)
        sample = np.asarray([0.1, 0.1, 0.1, 0.1])
        wanted = 0.
        assert_array_almost_equal(interp(sample), wanted)
        sample = np.asarray([0., 0., 0., 0.])
        wanted = 0.
        assert_array_almost_equal(interp(sample), wanted)
        sample = np.asarray([1., 1., 1., 1.])
        wanted = 1111.
        assert_array_almost_equal(interp(sample), wanted)
        sample = np.asarray([0.1, 0.4, 0.6, 0.9])
        wanted = 1055.
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_edges(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        wanted = np.asarray([0., 1111.])
        assert_array_almost_equal(interp(sample), wanted)

    def test_valid_create(self):
        # create a 2-D grid of 3 points in each dimension
        points = [(0., .5, 1.), (0., 1., .5)]
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis]
        values1 = values[np.newaxis, :]
        values = (values0 + values1 * 10)
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [((0., .5, 1.), ), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, .75, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, 1.), (0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0., .5, 1.), (0., .5, 1.)]
        assert_raises(ValueError, RegularGridInterpolator, points, values,
                      method="undefmethod")

    def test_valid_call(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.]])
        assert_raises(ValueError, interp, sample, "undefmethod")
        sample = np.asarray([[0., 0., 0.], [1., 1., 1.]])
        assert_raises(ValueError, interp, sample)
        sample = np.asarray([[0., 0., 0., 0.], [1., 1., 1., 1.1]])
        assert_raises(ValueError, interp, sample)

    def test_out_of_bounds_extrap(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([0., 1111., 11., 11.])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = np.asarray([-111.1, 1222.1, -11068., -1186.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_extrap2(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=None)
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [21, 2.1, -1.1, -11], [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([0., 11., 11., 11.])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = np.asarray([-12.1, 133.1, -1069., -97.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_fill(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, bounds_error=False,
                                         fill_value=np.nan)
        sample = np.asarray([[-.1, -.1, -.1, -.1], [1.1, 1.1, 1.1, 1.1],
                             [2.1, 2.1, -1.1, -1.1]])
        wanted = np.asarray([np.nan, np.nan, np.nan])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        assert_array_almost_equal(interp(sample, method="linear"), wanted)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        wanted = np.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    def test_nearest_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method="nearest")
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values.reshape(-1)
        interp_qhull = NearestNDInterpolator(points_qhull, values_qhull)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    def test_linear_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        points_qhull = itertools.product(*points)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values.reshape(-1)
        interp_qhull = LinearNDInterpolator(points_qhull, values_qhull)
        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])
        assert_array_almost_equal(interp(sample), interp_qhull(sample))

    def test_duck_typed_values(self):
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)

        values = MyValue((5, 7))

        for method in ('nearest', 'linear'):
            interp = RegularGridInterpolator((x, y), values,
                                             method=method)
            v1 = interp([0.4, 0.7])

            interp = RegularGridInterpolator((x, y), values._v,
                                             method=method)
            v2 = interp([0.4, 0.7])
            assert_allclose(v1, v2)

    def test_invalid_fill_value(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)

        # integers can be cast to floats
        RegularGridInterpolator((x, y), values, fill_value=1)

        # complex values cannot
        assert_raises(ValueError, RegularGridInterpolator,
                      (x, y), values, fill_value=1+2j)

    def test_fillvalue_type(self):
        # from #3703; test that interpolator object construction succeeds
        values = np.ones((10, 20, 30), dtype='>f4')
        points = [np.arange(n) for n in values.shape]
        # xi = [(1, 1, 1)]
        RegularGridInterpolator(points, values)
        RegularGridInterpolator(points, values, fill_value=0.)

    def test_broadcastable_input(self):
        # input data
        np.random.seed(0)
        x = np.random.random(10)
        y = np.random.random(10)
        z = np.hypot(x, y)

        # x-y grid for interpolation
        X = np.linspace(min(x), max(x))
        Y = np.linspace(min(y), max(y))
        X, Y = np.meshgrid(X, Y)
        XY = np.vstack((X.ravel(), Y.ravel())).T

        for interpolator in (NearestNDInterpolator, LinearNDInterpolator,
                             CloughTocher2DInterpolator):
            interp = interpolator(list(zip(x, y)), z)
            # single array input
            interp_points0 = interp(XY)
            # tuple input
            interp_points1 = interp((X, Y))
            interp_points2 = interp((X, 0.0))
            # broadcastable input
            interp_points3 = interp(X, Y)
            interp_points4 = interp(X, 0.0)

            assert_equal(interp_points0.size ==
                         interp_points1.size ==
                         interp_points2.size ==
                         interp_points3.size ==
                         interp_points4.size, True)

    def test_read_only(self):
        # input data
        np.random.seed(0)
        xy = np.random.random((10, 2))
        x, y = xy[:, 0], xy[:, 1]
        z = np.hypot(x, y)
        
        # interpolation points
        XY = np.random.random((50, 2))

        xy.setflags(write=False)
        z.setflags(write=False)
        XY.setflags(write=False)

        for interpolator in (NearestNDInterpolator, LinearNDInterpolator,
                             CloughTocher2DInterpolator):
            interp = interpolator(xy, z)
            interp(XY)


class MyValue:
    """
    Minimal indexable object
    """

    def __init__(self, shape):
        self.ndim = 2
        self.shape = shape
        self._v = np.arange(np.prod(shape)).reshape(shape)

    def __getitem__(self, idx):
        return self._v[idx]

    def __array_interface__(self):
        return None

    def __array__(self):
        raise RuntimeError("No array representation")


class TestInterpN:
    def _sample_2d_data(self):
        x = np.arange(1, 6)
        x = np.array([.5, 2., 3., 4., 5.5])
        y = np.arange(1, 6)
        y = np.array([.5, 2., 3., 4., 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        return x, y, z

    def test_spline_2d(self):
        x, y, z = self._sample_2d_data()
        lut = RectBivariateSpline(x, y, z)

        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
        assert_array_almost_equal(interpn((x, y), z, xi, method="splinef2d"),
                                  lut.ev(xi[:, 0], xi[:, 1]))

    def test_list_input(self):
        x, y, z = self._sample_2d_data()
        xi = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T

        for method in ['nearest', 'linear', 'splinef2d']:
            v1 = interpn((x, y), z, xi, method=method)
            v2 = interpn((x.tolist(), y.tolist()), z.tolist(),
                         xi.tolist(), method=method)
            assert_allclose(v1, v2, err_msg=method)

    def test_spline_2d_outofbounds(self):
        x = np.array([.5, 2., 3., 4., 5.5])
        y = np.array([.5, 2., 3., 4., 5.5])
        z = np.array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
                      [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
        lut = RectBivariateSpline(x, y, z)

        xi = np.array([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3],
                       [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
        actual = interpn((x, y), z, xi, method="splinef2d",
                         bounds_error=False, fill_value=999.99)
        expected = lut.ev(xi[:, 0], xi[:, 1])
        expected[2:4] = 999.99
        assert_array_almost_equal(actual, expected)

        # no extrapolation for splinef2d
        assert_raises(ValueError, interpn, (x, y), z, xi, method="splinef2d",
                      bounds_error=False, fill_value=None)

    def _sample_4d_data(self):
        points = [(0., .5, 1.)] * 2 + [(0., 5., 10.)] * 2
        values = np.asarray([0., .5, 1.])
        values0 = values[:, np.newaxis, np.newaxis, np.newaxis]
        values1 = values[np.newaxis, :, np.newaxis, np.newaxis]
        values2 = values[np.newaxis, np.newaxis, :, np.newaxis]
        values3 = values[np.newaxis, np.newaxis, np.newaxis, :]
        values = (values0 + values1 * 10 + values2 * 100 + values3 * 1000)
        return points, values

    def test_linear_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values)
        sample = np.asarray([[0.1, 0.1, 10., 9.]])
        wanted = interpn(points, values, sample, method="linear")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_linear_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = np.asarray([[0.1, -0.1, 10.1, 9.]])
        wanted = 999.99
        actual = interpn(points, values, sample, method="linear",
                         bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_nearest_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values, method="nearest")
        sample = np.asarray([[0.1, 0.1, 10., 9.]])
        wanted = interpn(points, values, sample, method="nearest")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_nearest_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = np.asarray([[0.1, -0.1, 10.1, 9.]])
        wanted = 999.99
        actual = interpn(points, values, sample, method="nearest",
                         bounds_error=False, fill_value=999.99)
        assert_array_almost_equal(actual, wanted)

    def test_xi_1d(self):
        # verify that 1-D xi works as expected
        points, values = self._sample_4d_data()
        sample = np.asarray([0.1, 0.1, 10., 9.])
        v1 = interpn(points, values, sample, bounds_error=False)
        v2 = interpn(points, values, sample[None,:], bounds_error=False)
        assert_allclose(v1, v2)

    def test_xi_nd(self):
        # verify that higher-d xi works as expected
        points, values = self._sample_4d_data()

        np.random.seed(1234)
        sample = np.random.rand(2, 3, 4)

        v1 = interpn(points, values, sample, method='nearest',
                     bounds_error=False)
        assert_equal(v1.shape, (2, 3))

        v2 = interpn(points, values, sample.reshape(-1, 4),
                     method='nearest', bounds_error=False)
        assert_allclose(v1, v2.reshape(v1.shape))

    def test_xi_broadcast(self):
        # verify that the interpolators broadcast xi
        x, y, values = self._sample_2d_data()
        points = (x, y)

        xi = np.linspace(0, 1, 2)
        yi = np.linspace(0, 3, 3)

        for method in ['nearest', 'linear', 'splinef2d']:
            sample = (xi[:,None], yi[None,:])
            v1 = interpn(points, values, sample, method=method,
                         bounds_error=False)
            assert_equal(v1.shape, (2, 3))

            xx, yy = np.meshgrid(xi, yi)
            sample = np.c_[xx.T.ravel(), yy.T.ravel()]

            v2 = interpn(points, values, sample,
                         method=method, bounds_error=False)
            assert_allclose(v1, v2.reshape(v1.shape))

    def test_nonscalar_values(self):
        # Verify that non-scalar valued values also works
        points, values = self._sample_4d_data()

        np.random.seed(1234)
        values = np.random.rand(3, 3, 3, 3, 6)
        sample = np.random.rand(7, 11, 4)

        for method in ['nearest', 'linear']:
            v = interpn(points, values, sample, method=method,
                        bounds_error=False)
            assert_equal(v.shape, (7, 11, 6), err_msg=method)

            vs = [interpn(points, values[...,j], sample, method=method,
                          bounds_error=False)
                  for j in range(6)]
            v2 = np.array(vs).transpose(1, 2, 0)

            assert_allclose(v, v2, err_msg=method)

        # Vector-valued splines supported with fitpack
        assert_raises(ValueError, interpn, points, values, sample,
                      method='splinef2d')

    def test_complex(self):
        x, y, values = self._sample_2d_data()
        points = (x, y)
        values = values - 2j*values

        sample = np.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
                           [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T

        for method in ['linear', 'nearest']:
            v1 = interpn(points, values, sample, method=method)
            v2r = interpn(points, values.real, sample, method=method)
            v2i = interpn(points, values.imag, sample, method=method)
            v2 = v2r + 1j*v2i
            assert_allclose(v1, v2)

        # Complex-valued data not supported by spline2fd
        assert_warns(np.ComplexWarning, interpn, points, values,
                     sample, method='splinef2d')

    def test_duck_typed_values(self):
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)

        values = MyValue((5, 7))

        for method in ('nearest', 'linear'):
            v1 = interpn((x, y), values, [0.4, 0.7], method=method)
            v2 = interpn((x, y), values._v, [0.4, 0.7], method=method)
            assert_allclose(v1, v2)

    def test_matrix_input(self):
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)

        values = matrix(np.random.rand(5, 7))

        sample = np.random.rand(3, 7, 2)

        for method in ('nearest', 'linear', 'splinef2d'):
            v1 = interpn((x, y), values, sample, method=method)
            v2 = interpn((x, y), np.asarray(values), sample, method=method)
            assert_allclose(v1, v2)
