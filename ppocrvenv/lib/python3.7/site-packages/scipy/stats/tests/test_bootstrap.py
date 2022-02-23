import numpy as np
import pytest
from scipy.stats import bootstrap
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
from .. import _bootstrap as _bootstrap
from scipy._lib._util import rng_integers


def test_bootstrap_iv():

    message = "`data` must be a sequence of samples."
    with pytest.raises(ValueError, match=message):
        bootstrap(1, np.mean)

    message = "`data` must contain at least one sample."
    with pytest.raises(ValueError, match=message):
        bootstrap(tuple(), np.mean)

    message = "each sample in `data` must contain two or more observations..."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3], [1]), np.mean)

    message = ("When `paired is True`, all samples must have the same length ")
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3], [1, 2, 3, 4]), np.mean, paired=True)

    message = "`vectorized` must be `True` or `False`."
    with pytest.raises(ValueError, match=message):
        bootstrap(1, np.mean, vectorized='ekki')

    message = "`axis` must be an integer."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, axis=1.5)

    message = "could not convert string to float"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, confidence_level='ni')

    message = "`n_resamples` must be a positive integer."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=-1000)

    message = "`n_resamples` must be a positive integer."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=1000.5)

    message = "`batch` must be a positive integer or None."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, batch=-1000)

    message = "`batch` must be a positive integer or None."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, batch=1000.5)

    message = "`method` must be in"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, method='ekki')

    message = "`method = 'BCa' is only available for one-sample statistics"

    def statistic(x, y, axis):
        mean1 = np.mean(x, axis)
        mean2 = np.mean(y, axis)
        return mean1 - mean2

    with pytest.raises(ValueError, match=message):
        bootstrap(([.1, .2, .3], [.1, .2, .3]), statistic, method='BCa')

    message = "'herring' cannot be used to seed a"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, random_state='herring')


@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_bootstrap_batch(method, axis):
    # for one-sample statistics, batch size shouldn't affect the result
    np.random.seed(0)

    x = np.random.rand(10, 11, 12)
    res1 = bootstrap((x,), np.mean, batch=None, method=method,
                     random_state=0, axis=axis, n_resamples=100)
    res2 = bootstrap((x,), np.mean, batch=10, method=method,
                     random_state=0, axis=axis, n_resamples=100)

    assert_equal(res2.confidence_interval.low, res1.confidence_interval.low)
    assert_equal(res2.confidence_interval.high, res1.confidence_interval.high)
    assert_equal(res2.standard_error, res1.standard_error)


@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
def test_bootstrap_paired(method):
    # test that `paired` works as expected
    np.random.seed(0)
    n = 100
    x = np.random.rand(n)
    y = np.random.rand(n)

    def my_statistic(x, y, axis=-1):
        return ((x-y)**2).mean(axis=axis)

    def my_paired_statistic(i, axis=-1):
        a = x[i]
        b = y[i]
        res = my_statistic(a, b)
        return res

    i = np.arange(len(x))

    res1 = bootstrap((i,), my_paired_statistic, random_state=0)
    res2 = bootstrap((x, y), my_statistic, paired=True, random_state=0)

    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    assert_allclose(res1.standard_error, res2.standard_error)


@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("paired", [True, False])
def test_bootstrap_vectorized(method, axis, paired):
    # test that paired is vectorized as expected: when samples are tiled,
    # CI and standard_error of each axis-slice is the same as those of the
    # original 1d sample

    if not paired and method == 'BCa':
        # should re-assess when BCa is extended
        pytest.xfail(reason="BCa currently for 1-sample statistics only")
    np.random.seed(0)

    def my_statistic(x, y, z, axis=-1):
        return x.mean(axis=axis) + y.mean(axis=axis) + z.mean(axis=axis)

    shape = 10, 11, 12
    n_samples = shape[axis]

    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    z = np.random.rand(n_samples)
    res1 = bootstrap((x, y, z), my_statistic, paired=paired, method=method,
                     random_state=0, axis=0, n_resamples=100)

    reshape = [1, 1, 1]
    reshape[axis] = n_samples
    x = np.broadcast_to(x.reshape(reshape), shape)
    y = np.broadcast_to(y.reshape(reshape), shape)
    z = np.broadcast_to(z.reshape(reshape), shape)
    res2 = bootstrap((x, y, z), my_statistic, paired=paired, method=method,
                     random_state=0, axis=axis, n_resamples=100)

    assert_allclose(res2.confidence_interval.low,
                    res1.confidence_interval.low)
    assert_allclose(res2.confidence_interval.high,
                    res1.confidence_interval.high)
    assert_allclose(res2.standard_error, res1.standard_error)

    result_shape = list(shape)
    result_shape.pop(axis)

    assert_equal(res2.confidence_interval.low.shape, result_shape)
    assert_equal(res2.confidence_interval.high.shape, result_shape)
    assert_equal(res2.standard_error.shape, result_shape)


@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
def test_bootstrap_against_theory(method):
    # based on https://www.statology.org/confidence-intervals-python/
    data = stats.norm.rvs(loc=5, scale=2, size=5000, random_state=0)
    alpha = 0.95
    dist = stats.t(df=len(data)-1, loc=np.mean(data), scale=stats.sem(data))
    expected_interval = dist.interval(alpha=alpha)
    expected_se = dist.std()

    res = bootstrap((data,), np.mean, n_resamples=5000,
                    confidence_level=alpha, method=method,
                    random_state=0)
    assert_allclose(res.confidence_interval, expected_interval, rtol=5e-4)
    assert_allclose(res.standard_error, expected_se, atol=3e-4)


tests_R = {"basic": (23.77, 79.12),
           "percentile": (28.86, 84.21),
           "BCa": (32.31, 91.43)}


@pytest.mark.parametrize("method, expected", tests_R.items())
def test_bootstrap_against_R(method, expected):
    # Compare against R's "boot" library
    # library(boot)

    # stat <- function (x, a) {
    #     mean(x[a])
    # }

    # x <- c(10, 12, 12.5, 12.5, 13.9, 15, 21, 22,
    #        23, 34, 50, 81, 89, 121, 134, 213)

    # # Use a large value so we get a few significant digits for the CI.
    # n = 1000000
    # bootresult = boot(x, stat, n)
    # result <- boot.ci(bootresult)
    # print(result)
    x = np.array([10, 12, 12.5, 12.5, 13.9, 15, 21, 22,
                  23, 34, 50, 81, 89, 121, 134, 213])
    res = bootstrap((x,), np.mean, n_resamples=1000000, method=method,
                    random_state=0)
    assert_allclose(res.confidence_interval, expected, rtol=0.005)


tests_against_itself_1samp = {"basic": 1780,
                              "percentile": 1784,
                              "BCa": 1784}


@pytest.mark.parametrize("method, expected",
                         tests_against_itself_1samp.items())
def test_bootstrap_against_itself_1samp(method, expected):
    # The expected values in this test were generated using bootstrap
    # to check for unintended changes in behavior. The test also makes sure
    # that bootstrap works with multi-sample statistics and that the
    # `axis` argument works as expected / function is vectorized.
    np.random.seed(0)

    n = 100  # size of sample
    n_resamples = 999  # number of bootstrap resamples used to form each CI
    confidence_level = 0.9

    # The true mean is 5
    dist = stats.norm(loc=5, scale=1)
    stat_true = dist.mean()

    # Do the same thing 2000 times. (The code is fully vectorized.)
    n_replications = 2000
    data = dist.rvs(size=(n_replications, n))
    res = bootstrap((data,),
                    statistic=np.mean,
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    batch=50,
                    method=method,
                    axis=-1)
    ci = res.confidence_interval

    # ci contains vectors of lower and upper confidence interval bounds
    ci_contains_true = np.sum((ci[0] < stat_true) & (stat_true < ci[1]))
    assert ci_contains_true == expected

    # ci_contains_true is not inconsistent with confidence_level
    pvalue = stats.binomtest(ci_contains_true, n_replications,
                             confidence_level).pvalue
    assert pvalue > 0.1


tests_against_itself_2samp = {"basic": 892,
                              "percentile": 890}


@pytest.mark.parametrize("method, expected",
                         tests_against_itself_2samp.items())
def test_bootstrap_against_itself_2samp(method, expected):
    # The expected values in this test were generated using bootstrap
    # to check for unintended changes in behavior. The test also makes sure
    # that bootstrap works with multi-sample statistics and that the
    # `axis` argument works as expected / function is vectorized.
    np.random.seed(0)

    n1 = 100  # size of sample 1
    n2 = 120  # size of sample 2
    n_resamples = 999  # number of bootstrap resamples used to form each CI
    confidence_level = 0.9

    # The statistic we're interested in is the difference in means
    def my_stat(data1, data2, axis=-1):
        mean1 = np.mean(data1, axis=axis)
        mean2 = np.mean(data2, axis=axis)
        return mean1 - mean2

    # The true difference in the means is -0.1
    dist1 = stats.norm(loc=0, scale=1)
    dist2 = stats.norm(loc=0.1, scale=1)
    stat_true = dist1.mean() - dist2.mean()

    # Do the same thing 1000 times. (The code is fully vectorized.)
    n_replications = 1000
    data1 = dist1.rvs(size=(n_replications, n1))
    data2 = dist2.rvs(size=(n_replications, n2))
    res = bootstrap((data1, data2),
                    statistic=my_stat,
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    batch=50,
                    method=method,
                    axis=-1)
    ci = res.confidence_interval

    # ci contains vectors of lower and upper confidence interval bounds
    ci_contains_true = np.sum((ci[0] < stat_true) & (stat_true < ci[1]))
    assert ci_contains_true == expected

    # ci_contains_true is not inconsistent with confidence_level
    pvalue = stats.binomtest(ci_contains_true, n_replications,
                             confidence_level).pvalue
    assert pvalue > 0.1


@pytest.mark.parametrize("method", ["basic", "percentile"])
@pytest.mark.parametrize("axis", [0, 1])
def test_bootstrap_vectorized_3samp(method, axis):
    def statistic(*data, axis=0):
        # an arbitrary, vectorized statistic
        return sum((sample.mean(axis) for sample in data))

    def statistic_1d(*data):
        # the same statistic, not vectorized
        for sample in data:
            assert sample.ndim == 1
        return statistic(*data, axis=0)

    np.random.seed(0)
    x = np.random.rand(4, 5)
    y = np.random.rand(4, 5)
    z = np.random.rand(4, 5)
    res1 = bootstrap((x, y, z), statistic, vectorized=True,
                     axis=axis, n_resamples=100, method=method, random_state=0)
    res2 = bootstrap((x, y, z), statistic_1d, vectorized=False,
                     axis=axis, n_resamples=100, method=method, random_state=0)
    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    assert_allclose(res1.standard_error, res2.standard_error)


@pytest.mark.xfail_on_32bit("Failure is not concerning; see gh-14107")
@pytest.mark.parametrize("method", ["basic", "percentile", "BCa"])
@pytest.mark.parametrize("axis", [0, 1])
def test_bootstrap_vectorized_1samp(method, axis):
    def statistic(x, axis=0):
        # an arbitrary, vectorized statistic
        return x.mean(axis=axis)

    def statistic_1d(x):
        # the same statistic, not vectorized
        assert x.ndim == 1
        return statistic(x, axis=0)

    np.random.seed(0)
    x = np.random.rand(4, 5)
    res1 = bootstrap((x,), statistic, vectorized=True, axis=axis,
                     n_resamples=100, batch=None, method=method,
                     random_state=0)
    res2 = bootstrap((x,), statistic_1d, vectorized=False, axis=axis,
                     n_resamples=100, batch=10, method=method,
                     random_state=0)
    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    assert_allclose(res1.standard_error, res2.standard_error)


def test_jackknife_resample():
    shape = 3, 4, 5, 6
    np.random.seed(0)
    x = np.random.rand(*shape)
    y = next(_bootstrap._jackknife_resample(x))

    for i in range(shape[-1]):
        # each resample is indexed along second to last axis
        # (last axis is the one the statistic will be taken over / consumed)
        slc = y[..., i, :]
        expected = np.delete(x, i, axis=-1)

        assert np.array_equal(slc, expected)

    y2 = np.concatenate(list(_bootstrap._jackknife_resample(x, batch=2)),
                        axis=-2)
    assert np.array_equal(y2, y)


@pytest.mark.parametrize("rng_name", ["RandomState", "default_rng"])
def test_bootstrap_resample(rng_name):
    rng = getattr(np.random, rng_name, None)
    if rng is None:
        pytest.skip(f"{rng_name} not available.")
    rng1 = rng(0)
    rng2 = rng(0)

    n_resamples = 10
    shape = 3, 4, 5, 6

    np.random.seed(0)
    x = np.random.rand(*shape)
    y = _bootstrap._bootstrap_resample(x, n_resamples, random_state=rng1)

    for i in range(n_resamples):
        # each resample is indexed along second to last axis
        # (last axis is the one the statistic will be taken over / consumed)
        slc = y[..., i, :]

        js = rng_integers(rng2, 0, shape[-1], shape[-1])
        expected = x[..., js]

        assert np.array_equal(slc, expected)


@pytest.mark.parametrize("score", [0, 0.5, 1])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_percentile_of_score(score, axis):
    shape = 10, 20, 30
    np.random.seed(0)
    x = np.random.rand(*shape)
    p = _bootstrap._percentile_of_score(x, score, axis=-1)

    def vectorized_pos(a, score, axis):
        return np.apply_along_axis(stats.percentileofscore, axis, a, score)

    p2 = vectorized_pos(x, score, axis=-1)/100

    assert_allclose(p, p2, 1e-15)


def test_percentile_along_axis():
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice

    shape = 10, 20
    np.random.seed(0)
    x = np.random.rand(*shape)
    q = np.random.rand(*shape[:-1]) * 100
    y = _bootstrap._percentile_along_axis(x, q)

    for i in range(shape[0]):
        res = y[i]
        expected = np.percentile(x[i], q[i], axis=-1)
        assert_allclose(res, expected, 1e-15)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_vectorize_statistic(axis):
    # test that _vectorize_statistic vectorizes a statistic along `axis`

    def statistic(*data, axis):
        # an arbitrary, vectorized statistic
        return sum((sample.mean(axis) for sample in data))

    def statistic_1d(*data):
        # the same statistic, not vectorized
        for sample in data:
            assert sample.ndim == 1
        return statistic(*data, axis=0)

    # vectorize the non-vectorized statistic
    statistic2 = _bootstrap._vectorize_statistic(statistic_1d)

    np.random.seed(0)
    x = np.random.rand(4, 5, 6)
    y = np.random.rand(4, 1, 6)
    z = np.random.rand(1, 5, 6)

    res1 = statistic(x, y, z, axis=axis)
    res2 = statistic2(x, y, z, axis=axis)
    assert_allclose(res1, res2)
