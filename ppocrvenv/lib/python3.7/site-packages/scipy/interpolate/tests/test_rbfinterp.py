import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
    _AVAILABLE, _SCALE_INVARIANT, _NAME_TO_MIN_DEGREE, _monomial_powers,
    RBFInterpolator
    )
from scipy.interpolate import _rbfinterp_pythran


def _vandermonde(x, degree):
    # Returns a matrix of monomials that span polynomials with the specified
    # degree evaluated at x.
    powers = _monomial_powers(x.shape[1], degree)
    return _rbfinterp_pythran._polynomial_matrix(x, powers)


def _1d_test_function(x):
    # Test function used in Wahba's "Spline Models for Observational Data".
    # domain ~= (0, 3), range ~= (-1.0, 0.2)
    x = x[:, 0]
    y = 4.26*(np.exp(-x) - 4*np.exp(-2*x) + 3*np.exp(-3*x))
    return y


def _2d_test_function(x):
    # Franke's test function.
    # domain ~= (0, 1) X (0, 1), range ~= (0.0, 1.2)
    x1, x2 = x[:, 0], x[:, 1]
    term1 = 0.75 * np.exp(-(9*x1-2)**2/4 - (9*x2-2)**2/4)
    term2 = 0.75 * np.exp(-(9*x1+1)**2/49 - (9*x2+1)/10)
    term3 = 0.5 * np.exp(-(9*x1-7)**2/4 - (9*x2-3)**2/4)
    term4 = -0.2 * np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    y = term1 + term2 + term3 + term4
    return y


def _is_conditionally_positive_definite(kernel, m):
    # Tests whether the kernel is conditionally positive definite of order m.
    # See chapter 7 of Fasshauer's "Meshfree Approximation Methods with
    # MATLAB".
    nx = 10
    ntests = 100
    for ndim in [1, 2, 3, 4, 5]:
        # Generate sample points with a Halton sequence to avoid samples that
        # are too close to eachother, which can make the matrix singular.
        seq = Halton(ndim, scramble=False, seed=np.random.RandomState())
        for _ in range(ntests):
            x = 2*seq.random(nx) - 1
            A = _rbfinterp_pythran._kernel_matrix(x, kernel)
            P = _vandermonde(x, m - 1)
            Q, R = np.linalg.qr(P, mode='complete')
            # Q2 forms a basis spanning the space where P.T.dot(x) = 0. Project
            # A onto this space, and then see if it is positive definite using
            # the Cholesky decomposition. If not, then the kernel is not c.p.d.
            # of order m.
            Q2 = Q[:, P.shape[1]:]
            B = Q2.T.dot(A).dot(Q2)
            try:
                np.linalg.cholesky(B)
            except np.linalg.LinAlgError:
                return False

    return True


# Sorting the parametrize arguments is necessary to avoid a parallelization
# issue described here: https://github.com/pytest-dev/pytest-xdist/issues/432.
@pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
def test_conditionally_positive_definite(kernel):
    # Test if each kernel in _AVAILABLE is conditionally positive definite of
    # order m, where m comes from _NAME_TO_MIN_DEGREE. This is a necessary
    # condition for the smoothed RBF interpolant to be well-posed in general.
    m = _NAME_TO_MIN_DEGREE.get(kernel, -1) + 1
    assert _is_conditionally_positive_definite(kernel, m)


class _TestRBFInterpolator:
    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_1d(self, kernel):
        # Verify that the functions in _SCALE_INVARIANT are insensitive to the
        # shape parameter (when smoothing == 0) in 1d.
        seq = Halton(1, scramble=False, seed=np.random.RandomState())
        x = 3*seq.random(50)
        y = _1d_test_function(x)
        xitp = 3*seq.random(50)
        yitp1 = self.build(x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(x, y, epsilon=2.0, kernel=kernel)(xitp)
        assert_allclose(yitp1, yitp2, atol=1e-8)

    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_2d(self, kernel):
        # Verify that the functions in _SCALE_INVARIANT are insensitive to the
        # shape parameter (when smoothing == 0) in 2d.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        x = seq.random(100)
        y = _2d_test_function(x)
        xitp = seq.random(100)
        yitp1 = self.build(x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(x, y, epsilon=2.0, kernel=kernel)(xitp)
        assert_allclose(yitp1, yitp2, atol=1e-8)

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_extreme_domains(self, kernel):
        # Make sure the interpolant remains numerically stable for very
        # large/small domains.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        scale = 1e50
        shift = 1e55

        x = seq.random(100)
        y = _2d_test_function(x)
        xitp = seq.random(100)

        if kernel in _SCALE_INVARIANT:
            yitp1 = self.build(x, y, kernel=kernel)(xitp)
            yitp2 = self.build(
                x*scale + shift, y,
                kernel=kernel
                )(xitp*scale + shift)
        else:
            yitp1 = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)
            yitp2 = self.build(
                x*scale + shift, y,
                epsilon=5.0/scale,
                kernel=kernel
                )(xitp*scale + shift)

        assert_allclose(yitp1, yitp2, atol=1e-8)

    def test_polynomial_reproduction(self):
        # If the observed data comes from a polynomial, then the interpolant
        # should be able to reproduce the polynomial exactly, provided that
        # `degree` is sufficiently high.
        rng = np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3

        x = seq.random(50)
        xitp = seq.random(50)

        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)

        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])

        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        yitp2 = self.build(x, y, degree=degree)(xitp)

        assert_allclose(yitp1, yitp2, atol=1e-8)

    def test_vector_data(self):
        # Make sure interpolating a vector field is the same as interpolating
        # each component separately.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        x = seq.random(100)
        xitp = seq.random(100)

        y = np.array([_2d_test_function(x),
                      _2d_test_function(x[:, ::-1])]).T

        yitp1 = self.build(x, y)(xitp)
        yitp2 = self.build(x, y[:, 0])(xitp)
        yitp3 = self.build(x, y[:, 1])(xitp)

        assert_allclose(yitp1[:, 0], yitp2)
        assert_allclose(yitp1[:, 1], yitp3)

    def test_complex_data(self):
        # Interpolating complex input should be the same as interpolating the
        # real and complex components.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        x = seq.random(100)
        xitp = seq.random(100)

        y = _2d_test_function(x) + 1j*_2d_test_function(x[:, ::-1])

        yitp1 = self.build(x, y)(xitp)
        yitp2 = self.build(x, y.real)(xitp)
        yitp3 = self.build(x, y.imag)(xitp)

        assert_allclose(yitp1.real, yitp2)
        assert_allclose(yitp1.imag, yitp3)

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_1d(self, kernel):
        # Make sure that each kernel, with its default `degree` and an
        # appropriate `epsilon`, does a good job at interpolation in 1d.
        seq = Halton(1, scramble=False, seed=np.random.RandomState())

        x = 3*seq.random(50)
        xitp = 3*seq.random(50)

        y = _1d_test_function(x)
        ytrue = _1d_test_function(xitp)
        yitp = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)

        mse = np.mean((yitp - ytrue)**2)
        assert mse < 1.0e-4

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_2d(self, kernel):
        # Make sure that each kernel, with its default `degree` and an
        # appropriate `epsilon`, does a good job at interpolation in 2d.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        x = seq.random(100)
        xitp = seq.random(100)

        y = _2d_test_function(x)
        ytrue = _2d_test_function(xitp)
        yitp = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)

        mse = np.mean((yitp - ytrue)**2)
        assert mse < 2.0e-4

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_smoothing_misfit(self, kernel):
        # Make sure we can find a smoothing parameter for each kernel that
        # removes a sufficient amount of noise.
        rng = np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)

        noise = 0.2
        rmse_tol = 0.1
        smoothing_range = 10**np.linspace(-4, 1, 20)

        x = 3*seq.random(100)
        y = _1d_test_function(x) + rng.normal(0.0, noise, (100,))
        ytrue = _1d_test_function(x)
        rmse_within_tol = False
        for smoothing in smoothing_range:
            ysmooth = self.build(
                x, y,
                epsilon=1.0,
                smoothing=smoothing,
                kernel=kernel)(x)
            rmse = np.sqrt(np.mean((ysmooth - ytrue)**2))
            if rmse < rmse_tol:
                rmse_within_tol = True
                break

        assert rmse_within_tol

    def test_array_smoothing(self):
        # Test using an array for `smoothing` to give less weight to a known
        # outlier.
        rng = np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)
        degree = 2

        x = seq.random(50)
        P = _vandermonde(x, degree)
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
        y = P.dot(poly_coeffs)
        y_with_outlier = np.copy(y)
        y_with_outlier[10] += 1.0
        smoothing = np.zeros((50,))
        smoothing[10] = 1000.0
        yitp = self.build(x, y_with_outlier, smoothing=smoothing)(x)
        # Should be able to reproduce the uncorrupted data almost exactly.
        assert_allclose(yitp, y, atol=1e-4)

    def test_inconsistent_x_dimensions_error(self):
        # ValueError should be raised if the observation points and evaluation
        # points have a different number of dimensions.
        y = Halton(2, scramble=False, seed=np.random.RandomState()).random(10)
        d = _2d_test_function(y)
        x = Halton(1, scramble=False, seed=np.random.RandomState()).random(10)
        match = 'Expected the second axis of `x`'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)(x)

    def test_inconsistent_d_length_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(1)
        match = 'Expected the first axis of `d`'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)

    def test_y_not_2d_error(self):
        y = np.linspace(0, 1, 5)
        d = np.zeros(5)
        match = '`y` must be a 2-dimensional array.'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)

    def test_inconsistent_smoothing_length_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        smoothing = np.ones(1)
        match = 'Expected `smoothing` to be'
        with pytest.raises(ValueError, match=match):
            self.build(y, d, smoothing=smoothing)

    def test_invalid_kernel_name_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        match = '`kernel` must be one of'
        with pytest.raises(ValueError, match=match):
            self.build(y, d, kernel='test')

    def test_epsilon_not_specified_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        for kernel in _AVAILABLE:
            if kernel in _SCALE_INVARIANT:
                continue

            match = '`epsilon` must be specified'
            with pytest.raises(ValueError, match=match):
                self.build(y, d, kernel=kernel)

    def test_x_not_2d_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        x = np.linspace(0, 1, 5)
        d = np.zeros(5)
        match = '`x` must be a 2-dimensional array.'
        with pytest.raises(ValueError, match=match):
            self.build(y, d)(x)

    def test_not_enough_observations_error(self):
        y = np.linspace(0, 1, 1)[:, None]
        d = np.zeros(1)
        match = 'At least 2 data points are required'
        with pytest.raises(ValueError, match=match):
            self.build(y, d, kernel='thin_plate_spline')

    def test_degree_warning(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(5)
        for kernel, deg in _NAME_TO_MIN_DEGREE.items():
            match = f'`degree` should not be below {deg}'
            with pytest.warns(Warning, match=match):
                self.build(y, d, epsilon=1.0, kernel=kernel, degree=deg-1)

    def test_rank_error(self):
        # An error should be raised when `kernel` is "thin_plate_spline" and
        # observations are 2-D and collinear.
        y = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        d = np.array([0.0, 0.0, 0.0])
        match = 'does not have full column rank'
        with pytest.raises(LinAlgError, match=match):
            self.build(y, d, kernel='thin_plate_spline')(y)

    def test_single_point(self):
        # Make sure interpolation still works with only one point (in 1, 2, and
        # 3 dimensions).
        for dim in [1, 2, 3]:
            y = np.zeros((1, dim))
            d = np.ones((1,))
            f = self.build(y, d, kernel='linear')(y)
            assert_allclose(d, f)

    def test_pickleable(self):
        # Make sure we can pickle and unpickle the interpolant without any
        # changes in the behavior.
        seq = Halton(1, scramble=False, seed=np.random.RandomState())

        x = 3*seq.random(50)
        xitp = 3*seq.random(50)

        y = _1d_test_function(x)

        interp = self.build(x, y)

        yitp1 = interp(xitp)
        yitp2 = pickle.loads(pickle.dumps(interp))(xitp)

        assert_array_equal(yitp1, yitp2)


class TestRBFInterpolatorNeighborsNone(_TestRBFInterpolator):
    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs)

    def test_smoothing_limit_1d(self):
        # For large smoothing parameters, the interpolant should approach a
        # least squares fit of a polynomial with the specified degree.
        seq = Halton(1, scramble=False, seed=np.random.RandomState())

        degree = 3
        smoothing = 1e8

        x = 3*seq.random(50)
        xitp = 3*seq.random(50)

        y = _1d_test_function(x)

        yitp1 = self.build(
            x, y,
            degree=degree,
            smoothing=smoothing
            )(xitp)

        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(np.linalg.lstsq(P, y, rcond=None)[0])

        assert_allclose(yitp1, yitp2, atol=1e-8)

    def test_smoothing_limit_2d(self):
        # For large smoothing parameters, the interpolant should approach a
        # least squares fit of a polynomial with the specified degree.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        degree = 3
        smoothing = 1e8

        x = seq.random(100)
        xitp = seq.random(100)

        y = _2d_test_function(x)

        yitp1 = self.build(
            x, y,
            degree=degree,
            smoothing=smoothing
            )(xitp)

        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(np.linalg.lstsq(P, y, rcond=None)[0])

        assert_allclose(yitp1, yitp2, atol=1e-8)


class TestRBFInterpolatorNeighbors20(_TestRBFInterpolator):
    # RBFInterpolator using 20 nearest neighbors.
    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs, neighbors=20)

    def test_equivalent_to_rbf_interpolator(self):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        x = seq.random(100)
        xitp = seq.random(100)

        y = _2d_test_function(x)

        yitp1 = self.build(x, y)(xitp)

        yitp2 = []
        tree = cKDTree(x)
        for xi in xitp:
            _, nbr = tree.query(xi, 20)
            yitp2.append(RBFInterpolator(x[nbr], y[nbr])(xi[None])[0])

        assert_allclose(yitp1, yitp2, atol=1e-8)


class TestRBFInterpolatorNeighborsInf(TestRBFInterpolatorNeighborsNone):
    # RBFInterpolator using neighbors=np.inf. This should give exactly the same
    # results as neighbors=None, but it will be slower.
    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs, neighbors=np.inf)

    def test_equivalent_to_rbf_interpolator(self):
        seq = Halton(1, scramble=False, seed=np.random.RandomState())

        x = 3*seq.random(50)
        xitp = 3*seq.random(50)

        y = _1d_test_function(x)
        yitp1 = self.build(x, y)(xitp)
        yitp2 = RBFInterpolator(x, y)(xitp)

        assert_allclose(yitp1, yitp2, atol=1e-8)
