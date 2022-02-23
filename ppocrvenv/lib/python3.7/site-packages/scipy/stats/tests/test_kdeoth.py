from scipy import stats
import numpy as np
from numpy.testing import (assert_almost_equal, assert_,
    assert_array_almost_equal, assert_array_almost_equal_nulp, assert_allclose)
import pytest
from pytest import raises as assert_raises


def test_kde_1d():
    #some basic tests comparing to normal distribution
    np.random.seed(8765678)
    n_basesample = 500
    xn = np.random.randn(n_basesample)
    xnmean = xn.mean()
    xnstd = xn.std(ddof=1)

    # get kde for original sample
    gkde = stats.gaussian_kde(xn)

    # evaluate the density function for the kde for some points
    xs = np.linspace(-7,7,501)
    kdepdf = gkde.evaluate(xs)
    normpdf = stats.norm.pdf(xs, loc=xnmean, scale=xnstd)
    intervall = xs[1] - xs[0]

    assert_(np.sum((kdepdf - normpdf)**2)*intervall < 0.01)
    prob1 = gkde.integrate_box_1d(xnmean, np.inf)
    prob2 = gkde.integrate_box_1d(-np.inf, xnmean)
    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box(xnmean, np.inf), prob1, decimal=13)
    assert_almost_equal(gkde.integrate_box(-np.inf, xnmean), prob2, decimal=13)

    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*intervall, decimal=2)
    assert_almost_equal(gkde.integrate_gaussian(xnmean, xnstd**2),
                        (kdepdf*normpdf).sum()*intervall, decimal=2)


def test_kde_1d_weighted():
    #some basic tests comparing to normal distribution
    np.random.seed(8765678)
    n_basesample = 500
    xn = np.random.randn(n_basesample)
    wn = np.random.rand(n_basesample)
    xnmean = np.average(xn, weights=wn)
    xnstd = np.sqrt(np.average((xn-xnmean)**2, weights=wn))

    # get kde for original sample
    gkde = stats.gaussian_kde(xn, weights=wn)

    # evaluate the density function for the kde for some points
    xs = np.linspace(-7,7,501)
    kdepdf = gkde.evaluate(xs)
    normpdf = stats.norm.pdf(xs, loc=xnmean, scale=xnstd)
    intervall = xs[1] - xs[0]

    assert_(np.sum((kdepdf - normpdf)**2)*intervall < 0.01)
    prob1 = gkde.integrate_box_1d(xnmean, np.inf)
    prob2 = gkde.integrate_box_1d(-np.inf, xnmean)
    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_box(xnmean, np.inf), prob1, decimal=13)
    assert_almost_equal(gkde.integrate_box(-np.inf, xnmean), prob2, decimal=13)

    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*intervall, decimal=2)
    assert_almost_equal(gkde.integrate_gaussian(xnmean, xnstd**2),
                        (kdepdf*normpdf).sum()*intervall, decimal=2)


@pytest.mark.slow
def test_kde_2d():
    #some basic tests comparing to normal distribution
    np.random.seed(8765678)
    n_basesample = 500

    mean = np.array([1.0, 3.0])
    covariance = np.array([[1.0, 2.0], [2.0, 6.0]])

    # Need transpose (shape (2, 500)) for kde
    xn = np.random.multivariate_normal(mean, covariance, size=n_basesample).T

    # get kde for original sample
    gkde = stats.gaussian_kde(xn)

    # evaluate the density function for the kde for some points
    x, y = np.mgrid[-7:7:500j, -7:7:500j]
    grid_coords = np.vstack([x.ravel(), y.ravel()])
    kdepdf = gkde.evaluate(grid_coords)
    kdepdf = kdepdf.reshape(500, 500)

    normpdf = stats.multivariate_normal.pdf(np.dstack([x, y]), mean=mean, cov=covariance)
    intervall = y.ravel()[1] - y.ravel()[0]

    assert_(np.sum((kdepdf - normpdf)**2) * (intervall**2) < 0.01)

    small = -1e100
    large = 1e100
    prob1 = gkde.integrate_box([small, mean[1]], [large, large])
    prob2 = gkde.integrate_box([small, small], [large, mean[1]])

    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*(intervall**2), decimal=2)
    assert_almost_equal(gkde.integrate_gaussian(mean, covariance),
                        (kdepdf*normpdf).sum()*(intervall**2), decimal=2)


@pytest.mark.slow
def test_kde_2d_weighted():
    #some basic tests comparing to normal distribution
    np.random.seed(8765678)
    n_basesample = 500

    mean = np.array([1.0, 3.0])
    covariance = np.array([[1.0, 2.0], [2.0, 6.0]])

    # Need transpose (shape (2, 500)) for kde
    xn = np.random.multivariate_normal(mean, covariance, size=n_basesample).T
    wn = np.random.rand(n_basesample)

    # get kde for original sample
    gkde = stats.gaussian_kde(xn, weights=wn)

    # evaluate the density function for the kde for some points
    x, y = np.mgrid[-7:7:500j, -7:7:500j]
    grid_coords = np.vstack([x.ravel(), y.ravel()])
    kdepdf = gkde.evaluate(grid_coords)
    kdepdf = kdepdf.reshape(500, 500)

    normpdf = stats.multivariate_normal.pdf(np.dstack([x, y]), mean=mean, cov=covariance)
    intervall = y.ravel()[1] - y.ravel()[0]

    assert_(np.sum((kdepdf - normpdf)**2) * (intervall**2) < 0.01)

    small = -1e100
    large = 1e100
    prob1 = gkde.integrate_box([small, mean[1]], [large, large])
    prob2 = gkde.integrate_box([small, small], [large, mean[1]])

    assert_almost_equal(prob1, 0.5, decimal=1)
    assert_almost_equal(prob2, 0.5, decimal=1)
    assert_almost_equal(gkde.integrate_kde(gkde),
                        (kdepdf**2).sum()*(intervall**2), decimal=2)
    assert_almost_equal(gkde.integrate_gaussian(mean, covariance),
                        (kdepdf*normpdf).sum()*(intervall**2), decimal=2)


def test_kde_bandwidth_method():
    def scotts_factor(kde_obj):
        """Same as default, just check that it works."""
        return np.power(kde_obj.n, -1./(kde_obj.d+4))

    np.random.seed(8765678)
    n_basesample = 50
    xn = np.random.randn(n_basesample)

    # Default
    gkde = stats.gaussian_kde(xn)
    # Supply a callable
    gkde2 = stats.gaussian_kde(xn, bw_method=scotts_factor)
    # Supply a scalar
    gkde3 = stats.gaussian_kde(xn, bw_method=gkde.factor)

    xs = np.linspace(-7,7,51)
    kdepdf = gkde.evaluate(xs)
    kdepdf2 = gkde2.evaluate(xs)
    assert_almost_equal(kdepdf, kdepdf2)
    kdepdf3 = gkde3.evaluate(xs)
    assert_almost_equal(kdepdf, kdepdf3)

    assert_raises(ValueError, stats.gaussian_kde, xn, bw_method='wrongstring')


def test_kde_bandwidth_method_weighted():
    def scotts_factor(kde_obj):
        """Same as default, just check that it works."""
        return np.power(kde_obj.neff, -1./(kde_obj.d+4))

    np.random.seed(8765678)
    n_basesample = 50
    xn = np.random.randn(n_basesample)

    # Default
    gkde = stats.gaussian_kde(xn)
    # Supply a callable
    gkde2 = stats.gaussian_kde(xn, bw_method=scotts_factor)
    # Supply a scalar
    gkde3 = stats.gaussian_kde(xn, bw_method=gkde.factor)

    xs = np.linspace(-7,7,51)
    kdepdf = gkde.evaluate(xs)
    kdepdf2 = gkde2.evaluate(xs)
    assert_almost_equal(kdepdf, kdepdf2)
    kdepdf3 = gkde3.evaluate(xs)
    assert_almost_equal(kdepdf, kdepdf3)

    assert_raises(ValueError, stats.gaussian_kde, xn, bw_method='wrongstring')


# Subclasses that should stay working (extracted from various sources).
# Unfortunately the earlier design of gaussian_kde made it necessary for users
# to create these kinds of subclasses, or call _compute_covariance() directly.

class _kde_subclass1(stats.gaussian_kde):
    def __init__(self, dataset):
        self.dataset = np.atleast_2d(dataset)
        self.d, self.n = self.dataset.shape
        self.covariance_factor = self.scotts_factor
        self._compute_covariance()


class _kde_subclass2(stats.gaussian_kde):
    def __init__(self, dataset):
        self.covariance_factor = self.scotts_factor
        super().__init__(dataset)


class _kde_subclass3(stats.gaussian_kde):
    def __init__(self, dataset, covariance):
        self.covariance = covariance
        stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance(self):
        self.inv_cov = np.linalg.inv(self.covariance)
        self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance))


class _kde_subclass4(stats.gaussian_kde):
    def covariance_factor(self):
        return 0.5 * self.silverman_factor()


def test_gaussian_kde_subclassing():
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=50)

    # gaussian_kde itself
    kde = stats.gaussian_kde(x1)
    ys = kde(xs)

    # subclass 1
    kde1 = _kde_subclass1(x1)
    y1 = kde1(xs)
    assert_array_almost_equal_nulp(ys, y1, nulp=10)

    # subclass 2
    kde2 = _kde_subclass2(x1)
    y2 = kde2(xs)
    assert_array_almost_equal_nulp(ys, y2, nulp=10)

    # subclass 3
    kde3 = _kde_subclass3(x1, kde.covariance)
    y3 = kde3(xs)
    assert_array_almost_equal_nulp(ys, y3, nulp=10)

    # subclass 4
    kde4 = _kde_subclass4(x1)
    y4 = kde4(x1)
    y_expected = [0.06292987, 0.06346938, 0.05860291, 0.08657652, 0.07904017]

    assert_array_almost_equal(y_expected, y4, decimal=6)

    # Not a subclass, but check for use of _compute_covariance()
    kde5 = kde
    kde5.covariance_factor = lambda: kde.factor
    kde5._compute_covariance()
    y5 = kde5(xs)
    assert_array_almost_equal_nulp(ys, y5, nulp=10)


def test_gaussian_kde_covariance_caching():
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=5)
    # These expected values are from scipy 0.10, before some changes to
    # gaussian_kde.  They were not compared with any external reference.
    y_expected = [0.02463386, 0.04689208, 0.05395444, 0.05337754, 0.01664475]

    # Set the bandwidth, then reset it to the default.
    kde = stats.gaussian_kde(x1)
    kde.set_bandwidth(bw_method=0.5)
    kde.set_bandwidth(bw_method='scott')
    y2 = kde(xs)

    assert_array_almost_equal(y_expected, y2, decimal=7)


def test_gaussian_kde_monkeypatch():
    """Ugly, but people may rely on this.  See scipy pull request 123,
    specifically the linked ML thread "Width of the Gaussian in stats.kde".
    If it is necessary to break this later on, that is to be discussed on ML.
    """
    x1 = np.array([-7, -5, 1, 4, 5], dtype=float)
    xs = np.linspace(-10, 10, num=50)

    # The old monkeypatched version to get at Silverman's Rule.
    kde = stats.gaussian_kde(x1)
    kde.covariance_factor = kde.silverman_factor
    kde._compute_covariance()
    y1 = kde(xs)

    # The new saner version.
    kde2 = stats.gaussian_kde(x1, bw_method='silverman')
    y2 = kde2(xs)

    assert_array_almost_equal_nulp(y1, y2, nulp=10)


def test_kde_integer_input():
    """Regression test for #1181."""
    x1 = np.arange(5)
    kde = stats.gaussian_kde(x1)
    y_expected = [0.13480721, 0.18222869, 0.19514935, 0.18222869, 0.13480721]
    assert_array_almost_equal(kde(x1), y_expected, decimal=6)


_ftypes = ['float32', 'float64', 'float96', 'float128', 'int32', 'int64']

@pytest.mark.parametrize("bw_type", _ftypes + ["scott", "silverman"])
@pytest.mark.parametrize("weights_type", _ftypes)
@pytest.mark.parametrize("dataset_type", _ftypes)
@pytest.mark.parametrize("point_type", _ftypes)
def test_kde_output_dtype(point_type, dataset_type, weights_type, bw_type):
    # Check whether the datatypes are available
    point_type = getattr(np, point_type, None)
    dataset_type = getattr(np, weights_type, None)
    weights_type = getattr(np, weights_type, None)

    if bw_type in ["scott", "silverman"]:
        bw = bw_type
    else:
        bw_type = getattr(np, bw_type, None)
        bw = bw_type(3) if bw_type else None

    if any(dt is None for dt in [point_type, dataset_type, weights_type, bw]):
        pytest.skip()

    weights = np.arange(5, dtype=weights_type)
    dataset = np.arange(5, dtype=dataset_type)
    k = stats.kde.gaussian_kde(dataset, bw_method=bw, weights=weights)
    points = np.arange(5, dtype=point_type)
    result = k(points)
    # weights are always cast to float64
    assert result.dtype == np.result_type(dataset, points, np.float64(weights),
                                          k.factor)


def test_pdf_logpdf():
    np.random.seed(1)
    n_basesample = 50
    xn = np.random.randn(n_basesample)

    # Default
    gkde = stats.gaussian_kde(xn)

    xs = np.linspace(-15, 12, 25)
    pdf = gkde.evaluate(xs)
    pdf2 = gkde.pdf(xs)
    assert_almost_equal(pdf, pdf2, decimal=12)

    logpdf = np.log(pdf)
    logpdf2 = gkde.logpdf(xs)
    assert_almost_equal(logpdf, logpdf2, decimal=12)

    # There are more points than data
    gkde = stats.gaussian_kde(xs)
    pdf = np.log(gkde.evaluate(xn))
    pdf2 = gkde.logpdf(xn)
    assert_almost_equal(pdf, pdf2, decimal=12)


def test_pdf_logpdf_weighted():
    np.random.seed(1)
    n_basesample = 50
    xn = np.random.randn(n_basesample)
    wn = np.random.rand(n_basesample)

    # Default
    gkde = stats.gaussian_kde(xn, weights=wn)

    xs = np.linspace(-15, 12, 25)
    pdf = gkde.evaluate(xs)
    pdf2 = gkde.pdf(xs)
    assert_almost_equal(pdf, pdf2, decimal=12)

    logpdf = np.log(pdf)
    logpdf2 = gkde.logpdf(xs)
    assert_almost_equal(logpdf, logpdf2, decimal=12)

    # There are more points than data
    gkde = stats.gaussian_kde(xs, weights=np.random.rand(len(xs)))
    pdf = np.log(gkde.evaluate(xn))
    pdf2 = gkde.logpdf(xn)
    assert_almost_equal(pdf, pdf2, decimal=12)


@pytest.mark.xslow
def test_logpdf_overflow():
    # regression test for gh-12988; testing against linalg instability for
    # very high dimensionality kde
    np.random.seed(1)
    n_dimensions = 2500
    n_samples = 5000
    xn = np.array([np.random.randn(n_samples) + (n) for n in range(
        0, n_dimensions)])

    # Default
    gkde = stats.gaussian_kde(xn)

    logpdf = gkde.logpdf(np.arange(0, n_dimensions))
    np.testing.assert_equal(np.isneginf(logpdf[0]), False)
    np.testing.assert_equal(np.isnan(logpdf[0]), False)


def test_weights_intact():
    # regression test for gh-9709: weights are not modified
    np.random.seed(12345)
    vals = np.random.lognormal(size=100)
    weights = np.random.choice([1.0, 10.0, 100], size=vals.size)
    orig_weights = weights.copy()

    stats.gaussian_kde(np.log10(vals), weights=weights)
    assert_allclose(weights, orig_weights, atol=1e-14, rtol=1e-14)


def test_weights_integer():
    # integer weights are OK, cf gh-9709 (comment)
    np.random.seed(12345)
    values = [0.2, 13.5, 21.0, 75.0, 99.0]
    weights = [1, 2, 4, 8, 16]  # a list of integers
    pdf_i = stats.gaussian_kde(values, weights=weights)
    pdf_f = stats.gaussian_kde(values, weights=np.float64(weights))

    xn = [0.3, 11, 88]
    assert_allclose(pdf_i.evaluate(xn),
                    pdf_f.evaluate(xn), atol=1e-14, rtol=1e-14)


def test_seed():
    # Test the seed option of the resample method
    def test_seed_sub(gkde_trail):
        n_sample = 200
        # The results should be different without using seed
        samp1 = gkde_trail.resample(n_sample)
        samp2 = gkde_trail.resample(n_sample)
        assert_raises(
            AssertionError, assert_allclose, samp1, samp2, atol=1e-13
        )
        # Use integer seed
        seed = 831
        samp1 = gkde_trail.resample(n_sample, seed=seed)
        samp2 = gkde_trail.resample(n_sample, seed=seed)
        assert_allclose(samp1, samp2, atol=1e-13)
        # Use RandomState
        rstate1 = np.random.RandomState(seed=138)
        samp1 = gkde_trail.resample(n_sample, seed=rstate1)
        rstate2 = np.random.RandomState(seed=138)
        samp2 = gkde_trail.resample(n_sample, seed=rstate2)
        assert_allclose(samp1, samp2, atol=1e-13)

        # check that np.random.Generator can be used (numpy >= 1.17)
        if hasattr(np.random, 'default_rng'):
            # obtain a np.random.Generator object
            rng = np.random.default_rng(1234)
            gkde_trail.resample(n_sample, seed=rng)

    np.random.seed(8765678)
    n_basesample = 500
    wn = np.random.rand(n_basesample)
    # Test 1D case
    xn_1d = np.random.randn(n_basesample)

    gkde_1d = stats.gaussian_kde(xn_1d)
    test_seed_sub(gkde_1d)
    gkde_1d_weighted = stats.gaussian_kde(xn_1d, weights=wn)
    test_seed_sub(gkde_1d_weighted)

    # Test 2D case
    mean = np.array([1.0, 3.0])
    covariance = np.array([[1.0, 2.0], [2.0, 6.0]])
    xn_2d = np.random.multivariate_normal(mean, covariance, size=n_basesample).T

    gkde_2d = stats.gaussian_kde(xn_2d)
    test_seed_sub(gkde_2d)
    gkde_2d_weighted = stats.gaussian_kde(xn_2d, weights=wn)
    test_seed_sub(gkde_2d_weighted)
