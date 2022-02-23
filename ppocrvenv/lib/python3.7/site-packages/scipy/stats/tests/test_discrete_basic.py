import numpy.testing as npt
import numpy as np
import pytest

from scipy import stats
from .common_tests import (check_normalization, check_moment, check_mean_expect,
                           check_var_expect, check_skew_expect,
                           check_kurt_expect, check_entropy,
                           check_private_entropy, check_edge_support,
                           check_named_args, check_random_state_property,
                           check_pickling, check_rvs_broadcast, check_freezing)
from scipy.stats._distr_params import distdiscrete, invdistdiscrete

vals = ([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
distdiscrete += [[stats.rv_discrete(values=vals), ()]]

# For these distributions, test_discrete_basic only runs with test mode full
distslow = {'zipfian', 'nhypergeom'}


def cases_test_discrete_basic():
    seen = set()
    for distname, arg in distdiscrete:
        if distname in distslow:
            yield pytest.param(distname, arg, distname, marks=pytest.mark.slow)
        else:
            yield distname, arg, distname not in seen
        seen.add(distname)


@pytest.mark.parametrize('distname,arg,first_case', cases_test_discrete_basic())
def test_discrete_basic(distname, arg, first_case):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'sample distribution'
    np.random.seed(9765456)
    rvs = distfn.rvs(size=2000, *arg)
    supp = np.unique(rvs)
    m, v = distfn.stats(*arg)
    check_cdf_ppf(distfn, arg, supp, distname + ' cdf_ppf')

    check_pmf_cdf(distfn, arg, distname)
    check_oth(distfn, arg, supp, distname + ' oth')
    check_edge_support(distfn, arg)

    alpha = 0.01
    check_discrete_chisquare(distfn, arg, rvs, alpha,
           distname + ' chisquare')

    if first_case:
        locscale_defaults = (0,)
        meths = [distfn.pmf, distfn.logpmf, distfn.cdf, distfn.logcdf,
                 distfn.logsf]
        # make sure arguments are within support
        # for some distributions, this needs to be overridden
        spec_k = {'randint': 11, 'hypergeom': 4, 'bernoulli': 0,
                  'nchypergeom_wallenius': 6}
        k = spec_k.get(distname, 1)
        check_named_args(distfn, k, arg, locscale_defaults, meths)
        if distname != 'sample distribution':
            check_scale_docstring(distfn)
        check_random_state_property(distfn, arg)
        check_pickling(distfn, arg)
        check_freezing(distfn, arg)

        # Entropy
        check_entropy(distfn, arg, distname)
        if distfn.__class__._entropy != stats.rv_discrete._entropy:
            check_private_entropy(distfn, arg, stats.rv_discrete)


@pytest.mark.parametrize('distname,arg', distdiscrete)
def test_moments(distname, arg):
    try:
        distfn = getattr(stats, distname)
    except TypeError:
        distfn = distname
        distname = 'sample distribution'
    m, v, s, k = distfn.stats(*arg, moments='mvsk')
    check_normalization(distfn, arg, distname)

    # compare `stats` and `moment` methods
    check_moment(distfn, arg, m, v, distname)
    check_mean_expect(distfn, arg, m, distname)
    check_var_expect(distfn, arg, m, v, distname)
    check_skew_expect(distfn, arg, m, v, s, distname)
    if distname not in ['zipf', 'yulesimon']:
        check_kurt_expect(distfn, arg, m, v, k, distname)

    # frozen distr moments
    check_moment_frozen(distfn, arg, m, 1)
    check_moment_frozen(distfn, arg, v+m*m, 2)


@pytest.mark.parametrize('dist,shape_args', distdiscrete)
def test_rvs_broadcast(dist, shape_args):
    # If shape_only is True, it means the _rvs method of the
    # distribution uses more than one random number to generate a random
    # variate.  That means the result of using rvs with broadcasting or
    # with a nontrivial size will not necessarily be the same as using the
    # numpy.vectorize'd version of rvs(), so we can only compare the shapes
    # of the results, not the values.
    # Whether or not a distribution is in the following list is an
    # implementation detail of the distribution, not a requirement.  If
    # the implementation the rvs() method of a distribution changes, this
    # test might also have to be changed.
    shape_only = dist in ['betabinom', 'skellam', 'yulesimon', 'dlaplace',
                          'nchypergeom_fisher', 'nchypergeom_wallenius']

    try:
        distfunc = getattr(stats, dist)
    except TypeError:
        distfunc = dist
        dist = 'rv_discrete(values=(%r, %r))' % (dist.xk, dist.pk)
    loc = np.zeros(2)
    nargs = distfunc.numargs
    allargs = []
    bshape = []
    # Generate shape parameter arguments...
    for k in range(nargs):
        shp = (k + 3,) + (1,)*(k + 1)
        param_val = shape_args[k]
        allargs.append(np.full(shp, param_val))
        bshape.insert(0, shp[0])
    allargs.append(loc)
    bshape.append(loc.size)
    # bshape holds the expected shape when loc, scale, and the shape
    # parameters are all broadcast together.
    check_rvs_broadcast(distfunc, dist, allargs, bshape, shape_only, [np.int_])


@pytest.mark.parametrize('dist,args', distdiscrete)
def test_ppf_with_loc(dist, args):
    try:
        distfn = getattr(stats, dist)
    except TypeError:
        distfn = dist
    #check with a negative, no and positive relocation.
    np.random.seed(1942349)
    re_locs = [np.random.randint(-10, -1), 0, np.random.randint(1, 10)]
    _a, _b = distfn.support(*args)
    for loc in re_locs:
        npt.assert_array_equal(
            [_a-1+loc, _b+loc],
            [distfn.ppf(0.0, *args, loc=loc), distfn.ppf(1.0, *args, loc=loc)]
            )


def check_cdf_ppf(distfn, arg, supp, msg):
    # cdf is a step function, and ppf(q) = min{k : cdf(k) >= q, k integer}
    npt.assert_array_equal(distfn.ppf(distfn.cdf(supp, *arg), *arg),
                           supp, msg + '-roundtrip')
    npt.assert_array_equal(distfn.ppf(distfn.cdf(supp, *arg) - 1e-8, *arg),
                           supp, msg + '-roundtrip')

    if not hasattr(distfn, 'xk'):
        _a, _b = distfn.support(*arg)
        supp1 = supp[supp < _b]
        npt.assert_array_equal(distfn.ppf(distfn.cdf(supp1, *arg) + 1e-8, *arg),
                               supp1 + distfn.inc, msg + ' ppf-cdf-next')
        # -1e-8 could cause an error if pmf < 1e-8


def check_pmf_cdf(distfn, arg, distname):
    if hasattr(distfn, 'xk'):
        index = distfn.xk
    else:
        startind = int(distfn.ppf(0.01, *arg) - 1)
        index = list(range(startind, startind + 10))
    cdfs = distfn.cdf(index, *arg)
    pmfs_cum = distfn.pmf(index, *arg).cumsum()

    atol, rtol = 1e-10, 1e-10
    if distname == 'skellam':    # ncx2 accuracy
        atol, rtol = 1e-5, 1e-5
    npt.assert_allclose(cdfs - cdfs[0], pmfs_cum - pmfs_cum[0],
                        atol=atol, rtol=rtol)


def check_moment_frozen(distfn, arg, m, k):
    npt.assert_allclose(distfn(*arg).moment(k), m,
                        atol=1e-10, rtol=1e-10)


def check_oth(distfn, arg, supp, msg):
    # checking other methods of distfn
    npt.assert_allclose(distfn.sf(supp, *arg), 1. - distfn.cdf(supp, *arg),
                        atol=1e-10, rtol=1e-10)

    q = np.linspace(0.01, 0.99, 20)
    npt.assert_allclose(distfn.isf(q, *arg), distfn.ppf(1. - q, *arg),
                        atol=1e-10, rtol=1e-10)

    median_sf = distfn.isf(0.5, *arg)
    npt.assert_(distfn.sf(median_sf - 1, *arg) > 0.5)
    npt.assert_(distfn.cdf(median_sf + 1, *arg) > 0.5)


def check_discrete_chisquare(distfn, arg, rvs, alpha, msg):
    """Perform chisquare test for random sample of a discrete distribution

    Parameters
    ----------
    distname : string
        name of distribution function
    arg : sequence
        parameters of distribution
    alpha : float
        significance level, threshold for p-value

    Returns
    -------
    result : bool
        0 if test passes, 1 if test fails

    """
    wsupp = 0.05

    # construct intervals with minimum mass `wsupp`.
    # intervals are left-half-open as in a cdf difference
    _a, _b = distfn.support(*arg)
    lo = int(max(_a, -1000))
    high = int(min(_b, 1000)) + 1
    distsupport = range(lo, high)
    last = 0
    distsupp = [lo]
    distmass = []
    for ii in distsupport:
        current = distfn.cdf(ii, *arg)
        if current - last >= wsupp - 1e-14:
            distsupp.append(ii)
            distmass.append(current - last)
            last = current
            if current > (1 - wsupp):
                break
    if distsupp[-1] < _b:
        distsupp.append(_b)
        distmass.append(1 - last)
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)

    # convert intervals to right-half-open as required by histogram
    histsupp = distsupp + 1e-8
    histsupp[0] = _a

    # find sample frequencies and perform chisquare test
    freq, hsupp = np.histogram(rvs, histsupp)
    chis, pval = stats.chisquare(np.array(freq), len(rvs)*distmass)

    npt.assert_(pval > alpha,
                'chisquare - test for %s at arg = %s with pval = %s' %
                (msg, str(arg), str(pval)))


def check_scale_docstring(distfn):
    if distfn.__doc__ is not None:
        # Docstrings can be stripped if interpreter is run with -OO
        npt.assert_('scale' not in distfn.__doc__)


@pytest.mark.parametrize('method', ['pmf', 'logpmf', 'cdf', 'logcdf',
                                    'sf', 'logsf', 'ppf', 'isf'])
@pytest.mark.parametrize('distname, args', distdiscrete)
def test_methods_with_lists(method, distname, args):
    # Test that the discrete distributions can accept Python lists
    # as arguments.
    try:
        dist = getattr(stats, distname)
    except TypeError:
        return
    if method in ['ppf', 'isf']:
        z = [0.1, 0.2]
    else:
        z = [0, 1]
    p2 = [[p]*2 for p in args]
    loc = [0, 1]
    result = dist.pmf(z, *p2, loc=loc)
    npt.assert_allclose(result,
                        [dist.pmf(*v) for v in zip(z, *p2, loc)],
                        rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize('distname, args', invdistdiscrete)
def test_cdf_gh13280_regression(distname, args):
    # Test for nan output when shape parameters are invalid
    dist = getattr(stats, distname)
    x = np.arange(-2, 15)
    vals = dist.cdf(x, *args)
    expected = np.nan
    npt.assert_equal(vals, expected)
