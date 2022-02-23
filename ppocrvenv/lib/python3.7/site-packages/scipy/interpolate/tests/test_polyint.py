import warnings
import io
import numpy as np

from numpy.testing import (
    assert_almost_equal, assert_array_equal, assert_array_almost_equal,
    assert_allclose, assert_equal, assert_)
from pytest import raises as assert_raises

from scipy.interpolate import (
    KroghInterpolator, krogh_interpolate,
    BarycentricInterpolator, barycentric_interpolate,
    approximate_taylor_polynomial, CubicHermiteSpline, pchip,
    PchipInterpolator, pchip_interpolate, Akima1DInterpolator, CubicSpline,
    make_interp_spline)


def check_shape(interpolator_cls, x_shape, y_shape, deriv_shape=None, axis=0,
                extra_args={}):
    np.random.seed(1234)

    x = [-1, 0, 1, 2, 3, 4]
    s = list(range(1, len(y_shape)+1))
    s.insert(axis % (len(y_shape)+1), 0)
    y = np.random.rand(*((6,) + y_shape)).transpose(s)

    # Cython code chokes on y.shape = (0, 3) etc., skip them
    if y.size == 0:
        return

    xi = np.zeros(x_shape)
    if interpolator_cls is CubicHermiteSpline:
        dydx = np.random.rand(*((6,) + y_shape)).transpose(s)
        yi = interpolator_cls(x, y, dydx, axis=axis, **extra_args)(xi)
    else:
        yi = interpolator_cls(x, y, axis=axis, **extra_args)(xi)

    target_shape = ((deriv_shape or ()) + y.shape[:axis]
                    + x_shape + y.shape[axis:][1:])
    assert_equal(yi.shape, target_shape)

    # check it works also with lists
    if x_shape and y.size > 0:
        if interpolator_cls is CubicHermiteSpline:
            interpolator_cls(list(x), list(y), list(dydx), axis=axis,
                             **extra_args)(list(xi))
        else:
            interpolator_cls(list(x), list(y), axis=axis,
                             **extra_args)(list(xi))

    # check also values
    if xi.size > 0 and deriv_shape is None:
        bs_shape = y.shape[:axis] + (1,)*len(x_shape) + y.shape[axis:][1:]
        yv = y[((slice(None,),)*(axis % y.ndim)) + (1,)]
        yv = yv.reshape(bs_shape)

        yi, y = np.broadcast_arrays(yi, yv)
        assert_allclose(yi, y)


SHAPES = [(), (0,), (1,), (6, 2, 5)]


def test_shapes():

    def spl_interp(x, y, axis):
        return make_interp_spline(x, y, axis=axis)

    for ip in [KroghInterpolator, BarycentricInterpolator, CubicHermiteSpline,
               pchip, Akima1DInterpolator, CubicSpline, spl_interp]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    if ip != CubicSpline:
                        check_shape(ip, s1, s2, None, axis)
                    else:
                        for bc in ['natural', 'clamped']:
                            extra = {'bc_type': bc}
                            check_shape(ip, s1, s2, None, axis, extra)

def test_derivs_shapes():
    def krogh_derivs(x, y, axis=0):
        return KroghInterpolator(x, y, axis).derivatives

    for s1 in SHAPES:
        for s2 in SHAPES:
            for axis in range(-len(s2), len(s2)):
                check_shape(krogh_derivs, s1, s2, (6,), axis)


def test_deriv_shapes():
    def krogh_deriv(x, y, axis=0):
        return KroghInterpolator(x, y, axis).derivative

    def pchip_deriv(x, y, axis=0):
        return pchip(x, y, axis).derivative()

    def pchip_deriv2(x, y, axis=0):
        return pchip(x, y, axis).derivative(2)

    def pchip_antideriv(x, y, axis=0):
        return pchip(x, y, axis).derivative()

    def pchip_antideriv2(x, y, axis=0):
        return pchip(x, y, axis).derivative(2)

    def pchip_deriv_inplace(x, y, axis=0):
        class P(PchipInterpolator):
            def __call__(self, x):
                return PchipInterpolator.__call__(self, x, 1)
            pass
        return P(x, y, axis)

    def akima_deriv(x, y, axis=0):
        return Akima1DInterpolator(x, y, axis).derivative()

    def akima_antideriv(x, y, axis=0):
        return Akima1DInterpolator(x, y, axis).antiderivative()

    def cspline_deriv(x, y, axis=0):
        return CubicSpline(x, y, axis).derivative()

    def cspline_antideriv(x, y, axis=0):
        return CubicSpline(x, y, axis).antiderivative()

    def bspl_deriv(x, y, axis=0):
        return make_interp_spline(x, y, axis=axis).derivative()

    def bspl_antideriv(x, y, axis=0):
        return make_interp_spline(x, y, axis=axis).antiderivative()

    for ip in [krogh_deriv, pchip_deriv, pchip_deriv2, pchip_deriv_inplace,
               pchip_antideriv, pchip_antideriv2, akima_deriv, akima_antideriv,
               cspline_deriv, cspline_antideriv, bspl_deriv, bspl_antideriv]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    check_shape(ip, s1, s2, (), axis)


def test_complex():
    x = [1, 2, 3, 4]
    y = [1, 2, 1j, 3]

    for ip in [KroghInterpolator, BarycentricInterpolator, pchip, CubicSpline]:
        p = ip(x, y)
        assert_allclose(y, p(x))

    dydx = [0, -1j, 2, 3j]
    p = CubicHermiteSpline(x, y, dydx)
    assert_allclose(y, p(x))
    assert_allclose(dydx, p(x, 1))


class TestKrogh:
    def setup_method(self):
        self.true_poly = np.poly1d([-2,3,1,5,-4])
        self.test_xs = np.linspace(-1,1,100)
        self.xs = np.linspace(-1,1,5)
        self.ys = self.true_poly(self.xs)

    def test_lagrange(self):
        P = KroghInterpolator(self.xs,self.ys)
        assert_almost_equal(self.true_poly(self.test_xs),P(self.test_xs))

    def test_scalar(self):
        P = KroghInterpolator(self.xs,self.ys)
        assert_almost_equal(self.true_poly(7),P(7))
        assert_almost_equal(self.true_poly(np.array(7)), P(np.array(7)))

    def test_derivatives(self):
        P = KroghInterpolator(self.xs,self.ys)
        D = P.derivatives(self.test_xs)
        for i in range(D.shape[0]):
            assert_almost_equal(self.true_poly.deriv(i)(self.test_xs),
                                D[i])

    def test_low_derivatives(self):
        P = KroghInterpolator(self.xs,self.ys)
        D = P.derivatives(self.test_xs,len(self.xs)+2)
        for i in range(D.shape[0]):
            assert_almost_equal(self.true_poly.deriv(i)(self.test_xs),
                                D[i])

    def test_derivative(self):
        P = KroghInterpolator(self.xs,self.ys)
        m = 10
        r = P.derivatives(self.test_xs,m)
        for i in range(m):
            assert_almost_equal(P.derivative(self.test_xs,i),r[i])

    def test_high_derivative(self):
        P = KroghInterpolator(self.xs,self.ys)
        for i in range(len(self.xs), 2*len(self.xs)):
            assert_almost_equal(P.derivative(self.test_xs,i),
                                np.zeros(len(self.test_xs)))

    def test_hermite(self):
        P = KroghInterpolator(self.xs,self.ys)
        assert_almost_equal(self.true_poly(self.test_xs),P(self.test_xs))

    def test_vector(self):
        xs = [0, 1, 2]
        ys = np.array([[0,1],[1,0],[2,1]])
        P = KroghInterpolator(xs,ys)
        Pi = [KroghInterpolator(xs,ys[:,i]) for i in range(ys.shape[1])]
        test_xs = np.linspace(-1,3,100)
        assert_almost_equal(P(test_xs),
                np.rollaxis(np.asarray([p(test_xs) for p in Pi]),-1))
        assert_almost_equal(P.derivatives(test_xs),
                np.transpose(np.asarray([p.derivatives(test_xs) for p in Pi]),
                    (1,2,0)))

    def test_empty(self):
        P = KroghInterpolator(self.xs,self.ys)
        assert_array_equal(P([]), [])

    def test_shapes_scalarvalue(self):
        P = KroghInterpolator(self.xs,self.ys)
        assert_array_equal(np.shape(P(0)), ())
        assert_array_equal(np.shape(P(np.array(0))), ())
        assert_array_equal(np.shape(P([0])), (1,))
        assert_array_equal(np.shape(P([0,1])), (2,))

    def test_shapes_scalarvalue_derivative(self):
        P = KroghInterpolator(self.xs,self.ys)
        n = P.n
        assert_array_equal(np.shape(P.derivatives(0)), (n,))
        assert_array_equal(np.shape(P.derivatives(np.array(0))), (n,))
        assert_array_equal(np.shape(P.derivatives([0])), (n,1))
        assert_array_equal(np.shape(P.derivatives([0,1])), (n,2))

    def test_shapes_vectorvalue(self):
        P = KroghInterpolator(self.xs,np.outer(self.ys,np.arange(3)))
        assert_array_equal(np.shape(P(0)), (3,))
        assert_array_equal(np.shape(P([0])), (1,3))
        assert_array_equal(np.shape(P([0,1])), (2,3))

    def test_shapes_1d_vectorvalue(self):
        P = KroghInterpolator(self.xs,np.outer(self.ys,[1]))
        assert_array_equal(np.shape(P(0)), (1,))
        assert_array_equal(np.shape(P([0])), (1,1))
        assert_array_equal(np.shape(P([0,1])), (2,1))

    def test_shapes_vectorvalue_derivative(self):
        P = KroghInterpolator(self.xs,np.outer(self.ys,np.arange(3)))
        n = P.n
        assert_array_equal(np.shape(P.derivatives(0)), (n,3))
        assert_array_equal(np.shape(P.derivatives([0])), (n,1,3))
        assert_array_equal(np.shape(P.derivatives([0,1])), (n,2,3))

    def test_wrapper(self):
        P = KroghInterpolator(self.xs, self.ys)
        ki = krogh_interpolate
        assert_almost_equal(P(self.test_xs), ki(self.xs, self.ys, self.test_xs))
        assert_almost_equal(P.derivative(self.test_xs, 2),
                            ki(self.xs, self.ys, self.test_xs, der=2))
        assert_almost_equal(P.derivatives(self.test_xs, 2),
                            ki(self.xs, self.ys, self.test_xs, der=[0, 1]))

    def test_int_inputs(self):
        # Check input args are cast correctly to floats, gh-3669
        x = [0, 234, 468, 702, 936, 1170, 1404, 2340, 3744, 6084, 8424,
             13104, 60000]
        offset_cdf = np.array([-0.95, -0.86114777, -0.8147762, -0.64072425,
                               -0.48002351, -0.34925329, -0.26503107,
                               -0.13148093, -0.12988833, -0.12979296,
                               -0.12973574, -0.08582937, 0.05])
        f = KroghInterpolator(x, offset_cdf)

        assert_allclose(abs((f(x) - offset_cdf) / f.derivative(x, 1)),
                        0, atol=1e-10)

    def test_derivatives_complex(self):
        # regression test for gh-7381: krogh.derivatives(0) fails complex y
        x, y = np.array([-1, -1, 0, 1, 1]), np.array([1, 1.0j, 0, -1, 1.0j])
        func = KroghInterpolator(x, y)
        cmplx = func.derivatives(0)

        cmplx2 = (KroghInterpolator(x, y.real).derivatives(0) +
                  1j*KroghInterpolator(x, y.imag).derivatives(0))
        assert_allclose(cmplx, cmplx2, atol=1e-15)


class TestTaylor:
    def test_exponential(self):
        degree = 5
        p = approximate_taylor_polynomial(np.exp, 0, degree, 1, 15)
        for i in range(degree+1):
            assert_almost_equal(p(0),1)
            p = p.deriv()
        assert_almost_equal(p(0),0)


class TestBarycentric:
    def setup_method(self):
        self.true_poly = np.poly1d([-2, 3, 1, 5, -4])
        self.test_xs = np.linspace(-1, 1, 100)
        self.xs = np.linspace(-1, 1, 5)
        self.ys = self.true_poly(self.xs)

    def test_lagrange(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    def test_scalar(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        assert_almost_equal(self.true_poly(7), P(7))
        assert_almost_equal(self.true_poly(np.array(7)), P(np.array(7)))

    def test_delayed(self):
        P = BarycentricInterpolator(self.xs)
        P.set_yi(self.ys)
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    def test_append(self):
        P = BarycentricInterpolator(self.xs[:3], self.ys[:3])
        P.add_xi(self.xs[3:], self.ys[3:])
        assert_almost_equal(self.true_poly(self.test_xs), P(self.test_xs))

    def test_vector(self):
        xs = [0, 1, 2]
        ys = np.array([[0, 1], [1, 0], [2, 1]])
        BI = BarycentricInterpolator
        P = BI(xs, ys)
        Pi = [BI(xs, ys[:, i]) for i in range(ys.shape[1])]
        test_xs = np.linspace(-1, 3, 100)
        assert_almost_equal(P(test_xs),
                np.rollaxis(np.asarray([p(test_xs) for p in Pi]), -1))

    def test_shapes_scalarvalue(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        assert_array_equal(np.shape(P(0)), ())
        assert_array_equal(np.shape(P(np.array(0))), ())
        assert_array_equal(np.shape(P([0])), (1,))
        assert_array_equal(np.shape(P([0, 1])), (2,))

    def test_shapes_vectorvalue(self):
        P = BarycentricInterpolator(self.xs, np.outer(self.ys, np.arange(3)))
        assert_array_equal(np.shape(P(0)), (3,))
        assert_array_equal(np.shape(P([0])), (1, 3))
        assert_array_equal(np.shape(P([0, 1])), (2, 3))

    def test_shapes_1d_vectorvalue(self):
        P = BarycentricInterpolator(self.xs, np.outer(self.ys, [1]))
        assert_array_equal(np.shape(P(0)), (1,))
        assert_array_equal(np.shape(P([0])), (1, 1))
        assert_array_equal(np.shape(P([0,1])), (2, 1))

    def test_wrapper(self):
        P = BarycentricInterpolator(self.xs, self.ys)
        values = barycentric_interpolate(self.xs, self.ys, self.test_xs)
        assert_almost_equal(P(self.test_xs), values)

    def test_int_input(self):
        x = 1000 * np.arange(1, 11)  # np.prod(x[-1] - x[:-1]) overflows
        y = np.arange(1, 11)
        value = barycentric_interpolate(x, y, 1000 * 9.5)
        assert_almost_equal(value, 9.5)


class TestPCHIP:
    def _make_random(self, npts=20):
        np.random.seed(1234)
        xi = np.sort(np.random.random(npts))
        yi = np.random.random(npts)
        return pchip(xi, yi), xi, yi

    def test_overshoot(self):
        # PCHIP should not overshoot
        p, xi, yi = self._make_random()
        for i in range(len(xi)-1):
            x1, x2 = xi[i], xi[i+1]
            y1, y2 = yi[i], yi[i+1]
            if y1 > y2:
                y1, y2 = y2, y1
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert_(((y1 <= yp + 1e-15) & (yp <= y2 + 1e-15)).all())

    def test_monotone(self):
        # PCHIP should preserve monotonicty
        p, xi, yi = self._make_random()
        for i in range(len(xi)-1):
            x1, x2 = xi[i], xi[i+1]
            y1, y2 = yi[i], yi[i+1]
            xp = np.linspace(x1, x2, 10)
            yp = p(xp)
            assert_(((y2-y1) * (yp[1:] - yp[:1]) > 0).all())

    def test_cast(self):
        # regression test for integer input data, see gh-3453
        data = np.array([[0, 4, 12, 27, 47, 60, 79, 87, 99, 100],
                         [-33, -33, -19, -2, 12, 26, 38, 45, 53, 55]])
        xx = np.arange(100)
        curve = pchip(data[0], data[1])(xx)

        data1 = data * 1.0
        curve1 = pchip(data1[0], data1[1])(xx)

        assert_allclose(curve, curve1, atol=1e-14, rtol=1e-14)

    def test_nag(self):
        # Example from NAG C implementation,
        # http://nag.com/numeric/cl/nagdoc_cl25/html/e01/e01bec.html
        # suggested in gh-5326 as a smoke test for the way the derivatives
        # are computed (see also gh-3453)
        dataStr = '''
          7.99   0.00000E+0
          8.09   0.27643E-4
          8.19   0.43750E-1
          8.70   0.16918E+0
          9.20   0.46943E+0
         10.00   0.94374E+0
         12.00   0.99864E+0
         15.00   0.99992E+0
         20.00   0.99999E+0
        '''
        data = np.loadtxt(io.StringIO(dataStr))
        pch = pchip(data[:,0], data[:,1])

        resultStr = '''
           7.9900       0.0000
           9.1910       0.4640
          10.3920       0.9645
          11.5930       0.9965
          12.7940       0.9992
          13.9950       0.9998
          15.1960       0.9999
          16.3970       1.0000
          17.5980       1.0000
          18.7990       1.0000
          20.0000       1.0000
        '''
        result = np.loadtxt(io.StringIO(resultStr))
        assert_allclose(result[:,1], pch(result[:,0]), rtol=0., atol=5e-5)

    def test_endslopes(self):
        # this is a smoke test for gh-3453: PCHIP interpolator should not
        # set edge slopes to zero if the data do not suggest zero edge derivatives
        x = np.array([0.0, 0.1, 0.25, 0.35])
        y1 = np.array([279.35, 0.5e3, 1.0e3, 2.5e3])
        y2 = np.array([279.35, 2.5e3, 1.50e3, 1.0e3])
        for pp in (pchip(x, y1), pchip(x, y2)):
            for t in (x[0], x[-1]):
                assert_(pp(t, 1) != 0)

    def test_all_zeros(self):
        x = np.arange(10)
        y = np.zeros_like(x)

        # this should work and not generate any warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            pch = pchip(x, y)

        xx = np.linspace(0, 9, 101)
        assert_equal(pch(xx), 0.)

    def test_two_points(self):
        # regression test for gh-6222: pchip([0, 1], [0, 1]) fails because
        # it tries to use a three-point scheme to estimate edge derivatives,
        # while there are only two points available.
        # Instead, it should construct a linear interpolator.
        x = np.linspace(0, 1, 11)
        p = pchip([0, 1], [0, 2])
        assert_allclose(p(x), 2*x, atol=1e-15)

    def test_pchip_interpolate(self):
        assert_array_almost_equal(
            pchip_interpolate([1,2,3], [4,5,6], [0.5], der=1),
            [1.])

        assert_array_almost_equal(
            pchip_interpolate([1,2,3], [4,5,6], [0.5], der=0),
            [3.5])

        assert_array_almost_equal(
            pchip_interpolate([1,2,3], [4,5,6], [0.5], der=[0, 1]),
            [[3.5], [1]])

    def test_roots(self):
        # regression test for gh-6357: .roots method should work
        p = pchip([0, 1], [-1, 1])
        r = p.roots()
        assert_allclose(r, 0.5)


class TestCubicSpline:
    @staticmethod
    def check_correctness(S, bc_start='not-a-knot', bc_end='not-a-knot',
                          tol=1e-14):
        """Check that spline coefficients satisfy the continuity and boundary
        conditions."""
        x = S.x
        c = S.c
        dx = np.diff(x)
        dx = dx.reshape([dx.shape[0]] + [1] * (c.ndim - 2))
        dxi = dx[:-1]

        # Check C2 continuity.
        assert_allclose(c[3, 1:], c[0, :-1] * dxi**3 + c[1, :-1] * dxi**2 +
                        c[2, :-1] * dxi + c[3, :-1], rtol=tol, atol=tol)
        assert_allclose(c[2, 1:], 3 * c[0, :-1] * dxi**2 +
                        2 * c[1, :-1] * dxi + c[2, :-1], rtol=tol, atol=tol)
        assert_allclose(c[1, 1:], 3 * c[0, :-1] * dxi + c[1, :-1],
                        rtol=tol, atol=tol)

        # Check that we found a parabola, the third derivative is 0.
        if x.size == 3 and bc_start == 'not-a-knot' and bc_end == 'not-a-knot':
            assert_allclose(c[0], 0, rtol=tol, atol=tol)
            return

        # Check periodic boundary conditions.
        if bc_start == 'periodic':
            assert_allclose(S(x[0], 0), S(x[-1], 0), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 1), S(x[-1], 1), rtol=tol, atol=tol)
            assert_allclose(S(x[0], 2), S(x[-1], 2), rtol=tol, atol=tol)
            return

        # Check other boundary conditions.
        if bc_start == 'not-a-knot':
            if x.size == 2:
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[0], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, 0], c[0, 1], rtol=tol, atol=tol)
        elif bc_start == 'clamped':
            assert_allclose(S(x[0], 1), 0, rtol=tol, atol=tol)
        elif bc_start == 'natural':
            assert_allclose(S(x[0], 2), 0, rtol=tol, atol=tol)
        else:
            order, value = bc_start
            assert_allclose(S(x[0], order), value, rtol=tol, atol=tol)

        if bc_end == 'not-a-knot':
            if x.size == 2:
                slope = (S(x[1]) - S(x[0])) / dx[0]
                assert_allclose(S(x[1], 1), slope, rtol=tol, atol=tol)
            else:
                assert_allclose(c[0, -1], c[0, -2], rtol=tol, atol=tol)
        elif bc_end == 'clamped':
            assert_allclose(S(x[-1], 1), 0, rtol=tol, atol=tol)
        elif bc_end == 'natural':
            assert_allclose(S(x[-1], 2), 0, rtol=2*tol, atol=2*tol)
        else:
            order, value = bc_end
            assert_allclose(S(x[-1], order), value, rtol=tol, atol=tol)

    def check_all_bc(self, x, y, axis):
        deriv_shape = list(y.shape)
        del deriv_shape[axis]
        first_deriv = np.empty(deriv_shape)
        first_deriv.fill(2)
        second_deriv = np.empty(deriv_shape)
        second_deriv.fill(-1)
        bc_all = [
            'not-a-knot',
            'natural',
            'clamped',
            (1, first_deriv),
            (2, second_deriv)
        ]
        for bc in bc_all[:3]:
            S = CubicSpline(x, y, axis=axis, bc_type=bc)
            self.check_correctness(S, bc, bc)

        for bc_start in bc_all:
            for bc_end in bc_all:
                S = CubicSpline(x, y, axis=axis, bc_type=(bc_start, bc_end))
                self.check_correctness(S, bc_start, bc_end, tol=2e-14)

    def test_general(self):
        x = np.array([-1, 0, 0.5, 2, 4, 4.5, 5.5, 9])
        y = np.array([0, -0.5, 2, 3, 2.5, 1, 1, 0.5])
        for n in [2, 3, x.size]:
            self.check_all_bc(x[:n], y[:n], 0)

            Y = np.empty((2, n, 2))
            Y[0, :, 0] = y[:n]
            Y[0, :, 1] = y[:n] - 1
            Y[1, :, 0] = y[:n] + 2
            Y[1, :, 1] = y[:n] + 3
            self.check_all_bc(x[:n], Y, 1)

    def test_periodic(self):
        for n in [2, 3, 5]:
            x = np.linspace(0, 2 * np.pi, n)
            y = np.cos(x)
            S = CubicSpline(x, y, bc_type='periodic')
            self.check_correctness(S, 'periodic', 'periodic')

            Y = np.empty((2, n, 2))
            Y[0, :, 0] = y
            Y[0, :, 1] = y + 2
            Y[1, :, 0] = y - 1
            Y[1, :, 1] = y + 5
            S = CubicSpline(x, Y, axis=1, bc_type='periodic')
            self.check_correctness(S, 'periodic', 'periodic')

    def test_periodic_eval(self):
        x = np.linspace(0, 2 * np.pi, 10)
        y = np.cos(x)
        S = CubicSpline(x, y, bc_type='periodic')
        assert_almost_equal(S(1), S(1 + 2 * np.pi), decimal=15)

    def test_second_derivative_continuity_gh_11758(self):
        # gh-11758: C2 continuity fail
        x = np.array([0.9, 1.3, 1.9, 2.1, 2.6, 3.0, 3.9, 4.4, 4.7, 5.0, 6.0,
                      7.0, 8.0, 9.2, 10.5, 11.3, 11.6, 12.0, 12.6, 13.0, 13.3])
        y = np.array([1.3, 1.5, 1.85, 2.1, 2.6, 2.7, 2.4, 2.15, 2.05, 2.1,
                      2.25, 2.3, 2.25, 1.95, 1.4, 0.9, 0.7, 0.6, 0.5, 0.4, 1.3])
        S = CubicSpline(x, y, bc_type='periodic', extrapolate='periodic')
        self.check_correctness(S, 'periodic', 'periodic')

    def test_three_points(self):
        # gh-11758: Fails computing a_m2_m1
        # In this case, s (first derivatives) could be found manually by solving
        # system of 2 linear equations. Due to solution of this system,
        # s[i] = (h1m2 + h2m1) / (h1 + h2), where h1 = x[1] - x[0], h2 = x[2] - x[1],
        # m1 = (y[1] - y[0]) / h1, m2 = (y[2] - y[1]) / h2
        x = np.array([1.0, 2.75, 3.0])
        y = np.array([1.0, 15.0, 1.0])
        S = CubicSpline(x, y, bc_type='periodic')
        self.check_correctness(S, 'periodic', 'periodic')
        assert_allclose(S.derivative(1)(x), np.array([-48.0, -48.0, -48.0]))

    def test_dtypes(self):
        x = np.array([0, 1, 2, 3], dtype=int)
        y = np.array([-5, 2, 3, 1], dtype=int)
        S = CubicSpline(x, y)
        self.check_correctness(S)

        y = np.array([-1+1j, 0.0, 1-1j, 0.5-1.5j])
        S = CubicSpline(x, y)
        self.check_correctness(S)

        S = CubicSpline(x, x ** 3, bc_type=("natural", (1, 2j)))
        self.check_correctness(S, "natural", (1, 2j))

        y = np.array([-5, 2, 3, 1])
        S = CubicSpline(x, y, bc_type=[(1, 2 + 0.5j), (2, 0.5 - 1j)])
        self.check_correctness(S, (1, 2 + 0.5j), (2, 0.5 - 1j))

    def test_small_dx(self):
        rng = np.random.RandomState(0)
        x = np.sort(rng.uniform(size=100))
        y = 1e4 + rng.uniform(size=100)
        S = CubicSpline(x, y)
        self.check_correctness(S, tol=1e-13)

    def test_incorrect_inputs(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])
        xc = np.array([1 + 1j, 2, 3, 4])
        xn = np.array([np.nan, 2, 3, 4])
        xo = np.array([2, 1, 3, 4])
        yn = np.array([np.nan, 2, 3, 4])
        y3 = [1, 2, 3]
        x1 = [1]
        y1 = [1]

        assert_raises(ValueError, CubicSpline, xc, y)
        assert_raises(ValueError, CubicSpline, xn, y)
        assert_raises(ValueError, CubicSpline, x, yn)
        assert_raises(ValueError, CubicSpline, xo, y)
        assert_raises(ValueError, CubicSpline, x, y3)
        assert_raises(ValueError, CubicSpline, x[:, np.newaxis], y)
        assert_raises(ValueError, CubicSpline, x1, y1)

        wrong_bc = [('periodic', 'clamped'),
                    ((2, 0), (3, 10)),
                    ((1, 0), ),
                    (0., 0.),
                    'not-a-typo']

        for bc_type in wrong_bc:
            assert_raises(ValueError, CubicSpline, x, y, 0, bc_type, True)

        # Shapes mismatch when giving arbitrary derivative values:
        Y = np.c_[y, y]
        bc1 = ('clamped', (1, 0))
        bc2 = ('clamped', (1, [0, 0, 0]))
        bc3 = ('clamped', (1, [[0, 0]]))
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc1, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc2, True)
        assert_raises(ValueError, CubicSpline, x, Y, 0, bc3, True)

        # periodic condition, y[-1] must be equal to y[0]:
        assert_raises(ValueError, CubicSpline, x, y, 0, 'periodic', True)


def test_CubicHermiteSpline_correctness():
    x = [0, 2, 7]
    y = [-1, 2, 3]
    dydx = [0, 3, 7]
    s = CubicHermiteSpline(x, y, dydx)
    assert_allclose(s(x), y, rtol=1e-15)
    assert_allclose(s(x, 1), dydx, rtol=1e-15)


def test_CubicHermiteSpline_error_handling():
    x = [1, 2, 3]
    y = [0, 3, 5]
    dydx = [1, -1, 2, 3]
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx)

    dydx_with_nan = [1, 0, np.nan]
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx_with_nan)


def test_roots_extrapolate_gh_11185():
    x = np.array([0.001, 0.002])
    y = np.array([1.66066935e-06, 1.10410807e-06])
    dy = np.array([-1.60061854, -1.600619])
    p = CubicHermiteSpline(x, y, dy)

    # roots(extrapolate=True) for a polynomial with a single interval
    # should return all three real roots
    r = p.roots(extrapolate=True)
    assert_equal(p.c.shape[1], 1)
    assert_equal(r.size, 3)
