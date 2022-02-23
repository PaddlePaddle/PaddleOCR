import pickle

import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises

import numpy.ma.testutils as ma_npt

from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy import stats


def check_named_results(res, attributes, ma=False):
    for i, attr in enumerate(attributes):
        if ma:
            ma_npt.assert_equal(res[i], getattr(res, attr))
        else:
            npt.assert_equal(res[i], getattr(res, attr))


def check_normalization(distfn, args, distname):
    norm_moment = distfn.moment(0, *args)
    npt.assert_allclose(norm_moment, 1.0)

    # this is a temporary plug: either ncf or expect is problematic;
    # best be marked as a knownfail, but I've no clue how to do it.
    if distname == "ncf":
        atol, rtol = 1e-5, 0
    else:
        atol, rtol = 1e-7, 1e-7

    normalization_expect = distfn.expect(lambda x: 1, args=args)
    npt.assert_allclose(normalization_expect, 1.0, atol=atol, rtol=rtol,
            err_msg=distname, verbose=True)

    _a, _b = distfn.support(*args)
    normalization_cdf = distfn.cdf(_b, *args)
    npt.assert_allclose(normalization_cdf, 1.0)


def check_moment(distfn, arg, m, v, msg):
    m1 = distfn.moment(1, *arg)
    m2 = distfn.moment(2, *arg)
    if not np.isinf(m):
        npt.assert_almost_equal(m1, m, decimal=10, err_msg=msg +
                            ' - 1st moment')
    else:                     # or np.isnan(m1),
        npt.assert_(np.isinf(m1),
               msg + ' - 1st moment -infinite, m1=%s' % str(m1))

    if not np.isinf(v):
        npt.assert_almost_equal(m2 - m1 * m1, v, decimal=10, err_msg=msg +
                            ' - 2ndt moment')
    else:                     # or np.isnan(m2),
        npt.assert_(np.isinf(m2),
               msg + ' - 2nd moment -infinite, m2=%s' % str(m2))


def check_mean_expect(distfn, arg, m, msg):
    if np.isfinite(m):
        m1 = distfn.expect(lambda x: x, arg)
        npt.assert_almost_equal(m1, m, decimal=5, err_msg=msg +
                            ' - 1st moment (expect)')


def check_var_expect(distfn, arg, m, v, msg):
    if np.isfinite(v):
        m2 = distfn.expect(lambda x: x*x, arg)
        npt.assert_almost_equal(m2, v + m*m, decimal=5, err_msg=msg +
                            ' - 2st moment (expect)')


def check_skew_expect(distfn, arg, m, v, s, msg):
    if np.isfinite(s):
        m3e = distfn.expect(lambda x: np.power(x-m, 3), arg)
        npt.assert_almost_equal(m3e, s * np.power(v, 1.5),
                decimal=5, err_msg=msg + ' - skew')
    else:
        npt.assert_(np.isnan(s))


def check_kurt_expect(distfn, arg, m, v, k, msg):
    if np.isfinite(k):
        m4e = distfn.expect(lambda x: np.power(x-m, 4), arg)
        npt.assert_allclose(m4e, (k + 3.) * np.power(v, 2), atol=1e-5, rtol=1e-5,
                err_msg=msg + ' - kurtosis')
    elif not np.isposinf(k):
        npt.assert_(np.isnan(k))


def check_entropy(distfn, arg, msg):
    ent = distfn.entropy(*arg)
    npt.assert_(not np.isnan(ent), msg + 'test Entropy is nan')


def check_private_entropy(distfn, args, superclass):
    # compare a generic _entropy with the distribution-specific implementation
    npt.assert_allclose(distfn._entropy(*args),
                        superclass._entropy(distfn, *args))


def check_entropy_vect_scale(distfn, arg):
    # check 2-d
    sc = np.asarray([[1, 2], [3, 4]])
    v_ent = distfn.entropy(*arg, scale=sc)
    s_ent = [distfn.entropy(*arg, scale=s) for s in sc.ravel()]
    s_ent = np.asarray(s_ent).reshape(v_ent.shape)
    assert_allclose(v_ent, s_ent, atol=1e-14)

    # check invalid value, check cast
    sc = [1, 2, -3]
    v_ent = distfn.entropy(*arg, scale=sc)
    s_ent = [distfn.entropy(*arg, scale=s) for s in sc]
    s_ent = np.asarray(s_ent).reshape(v_ent.shape)
    assert_allclose(v_ent, s_ent, atol=1e-14)


def check_edge_support(distfn, args):
    # Make sure that x=self.a and self.b are handled correctly.
    x = distfn.support(*args)
    if isinstance(distfn, stats.rv_discrete):
        x = x[0]-1, x[1]

    npt.assert_equal(distfn.cdf(x, *args), [0.0, 1.0])
    npt.assert_equal(distfn.sf(x, *args), [1.0, 0.0])

    if distfn.name not in ('skellam', 'dlaplace'):
        # with a = -inf, log(0) generates warnings
        npt.assert_equal(distfn.logcdf(x, *args), [-np.inf, 0.0])
        npt.assert_equal(distfn.logsf(x, *args), [0.0, -np.inf])

    npt.assert_equal(distfn.ppf([0.0, 1.0], *args), x)
    npt.assert_equal(distfn.isf([0.0, 1.0], *args), x[::-1])

    # out-of-bounds for isf & ppf
    npt.assert_(np.isnan(distfn.isf([-1, 2], *args)).all())
    npt.assert_(np.isnan(distfn.ppf([-1, 2], *args)).all())


def check_named_args(distfn, x, shape_args, defaults, meths):
    ## Check calling w/ named arguments.

    # check consistency of shapes, numargs and _parse signature
    signature = _getfullargspec(distfn._parse_args)
    npt.assert_(signature.varargs is None)
    npt.assert_(signature.varkw is None)
    npt.assert_(not signature.kwonlyargs)
    npt.assert_(list(signature.defaults) == list(defaults))

    shape_argnames = signature.args[:-len(defaults)]  # a, b, loc=0, scale=1
    if distfn.shapes:
        shapes_ = distfn.shapes.replace(',', ' ').split()
    else:
        shapes_ = ''
    npt.assert_(len(shapes_) == distfn.numargs)
    npt.assert_(len(shapes_) == len(shape_argnames))

    # check calling w/ named arguments
    shape_args = list(shape_args)

    vals = [meth(x, *shape_args) for meth in meths]
    npt.assert_(np.all(np.isfinite(vals)))

    names, a, k = shape_argnames[:], shape_args[:], {}
    while names:
        k.update({names.pop(): a.pop()})
        v = [meth(x, *a, **k) for meth in meths]
        npt.assert_array_equal(vals, v)
        if 'n' not in k.keys():
            # `n` is first parameter of moment(), so can't be used as named arg
            npt.assert_equal(distfn.moment(1, *a, **k),
                             distfn.moment(1, *shape_args))

    # unknown arguments should not go through:
    k.update({'kaboom': 42})
    assert_raises(TypeError, distfn.cdf, x, **k)


def check_random_state_property(distfn, args):
    # check the random_state attribute of a distribution *instance*

    # This test fiddles with distfn.random_state. This breaks other tests,
    # hence need to save it and then restore.
    rndm = distfn.random_state

    # baseline: this relies on the global state
    np.random.seed(1234)
    distfn.random_state = None
    r0 = distfn.rvs(*args, size=8)

    # use an explicit instance-level random_state
    distfn.random_state = 1234
    r1 = distfn.rvs(*args, size=8)
    npt.assert_equal(r0, r1)

    distfn.random_state = np.random.RandomState(1234)
    r2 = distfn.rvs(*args, size=8)
    npt.assert_equal(r0, r2)

    # check that np.random.Generator can be used (numpy >= 1.17)
    if hasattr(np.random, 'default_rng'):
        # obtain a np.random.Generator object
        rng = np.random.default_rng(1234)
        distfn.rvs(*args, size=1, random_state=rng)

    # can override the instance-level random_state for an individual .rvs call
    distfn.random_state = 2
    orig_state = distfn.random_state.get_state()

    r3 = distfn.rvs(*args, size=8, random_state=np.random.RandomState(1234))
    npt.assert_equal(r0, r3)

    # ... and that does not alter the instance-level random_state!
    npt.assert_equal(distfn.random_state.get_state(), orig_state)

    # finally, restore the random_state
    distfn.random_state = rndm


def check_meth_dtype(distfn, arg, meths):
    q0 = [0.25, 0.5, 0.75]
    x0 = distfn.ppf(q0, *arg)
    x_cast = [x0.astype(tp) for tp in
                        (np.int_, np.float16, np.float32, np.float64)]

    for x in x_cast:
        # casting may have clipped the values, exclude those
        distfn._argcheck(*arg)
        x = x[(distfn.a < x) & (x < distfn.b)]
        for meth in meths:
            val = meth(x, *arg)
            npt.assert_(val.dtype == np.float_)


def check_ppf_dtype(distfn, arg):
    q0 = np.asarray([0.25, 0.5, 0.75])
    q_cast = [q0.astype(tp) for tp in (np.float16, np.float32, np.float64)]
    for q in q_cast:
        for meth in [distfn.ppf, distfn.isf]:
            val = meth(q, *arg)
            npt.assert_(val.dtype == np.float_)


def check_cmplx_deriv(distfn, arg):
    # Distributions allow complex arguments.
    def deriv(f, x, *arg):
        x = np.asarray(x)
        h = 1e-10
        return (f(x + h*1j, *arg)/h).imag

    x0 = distfn.ppf([0.25, 0.51, 0.75], *arg)
    x_cast = [x0.astype(tp) for tp in
                        (np.int_, np.float16, np.float32, np.float64)]

    for x in x_cast:
        # casting may have clipped the values, exclude those
        distfn._argcheck(*arg)
        x = x[(distfn.a < x) & (x < distfn.b)]

        pdf, cdf, sf = distfn.pdf(x, *arg), distfn.cdf(x, *arg), distfn.sf(x, *arg)
        assert_allclose(deriv(distfn.cdf, x, *arg), pdf, rtol=1e-5)
        assert_allclose(deriv(distfn.logcdf, x, *arg), pdf/cdf, rtol=1e-5)

        assert_allclose(deriv(distfn.sf, x, *arg), -pdf, rtol=1e-5)
        assert_allclose(deriv(distfn.logsf, x, *arg), -pdf/sf, rtol=1e-5)

        assert_allclose(deriv(distfn.logpdf, x, *arg),
                        deriv(distfn.pdf, x, *arg) / distfn.pdf(x, *arg),
                        rtol=1e-5)


def check_pickling(distfn, args):
    # check that a distribution instance pickles and unpickles
    # pay special attention to the random_state property

    # save the random_state (restore later)
    rndm = distfn.random_state

    # check unfrozen
    distfn.random_state = 1234
    distfn.rvs(*args, size=8)
    s = pickle.dumps(distfn)
    r0 = distfn.rvs(*args, size=8)

    unpickled = pickle.loads(s)
    r1 = unpickled.rvs(*args, size=8)
    npt.assert_equal(r0, r1)

    # also smoke test some methods
    medians = [distfn.ppf(0.5, *args), unpickled.ppf(0.5, *args)]
    npt.assert_equal(medians[0], medians[1])
    npt.assert_equal(distfn.cdf(medians[0], *args),
                     unpickled.cdf(medians[1], *args))

    # check frozen pickling/unpickling with rvs
    frozen_dist = distfn(*args)
    pkl = pickle.dumps(frozen_dist)
    unpickled = pickle.loads(pkl)

    r0 = frozen_dist.rvs(size=8)
    r1 = unpickled.rvs(size=8)
    npt.assert_equal(r0, r1)

    # restore the random_state
    distfn.random_state = rndm


def check_freezing(distfn, args):
    # regression test for gh-11089: freezing a distribution fails
    # if loc and/or scale are specified
    if isinstance(distfn, stats.rv_continuous):
        locscale = {'loc': 1, 'scale': 2}
    else:
        locscale = {'loc': 1}

    rv = distfn(*args, **locscale)
    assert rv.a == distfn(*args).a
    assert rv.b == distfn(*args).b


def check_rvs_broadcast(distfunc, distname, allargs, shape, shape_only, otype):
    np.random.seed(123)
    sample = distfunc.rvs(*allargs)
    assert_equal(sample.shape, shape, "%s: rvs failed to broadcast" % distname)
    if not shape_only:
        rvs = np.vectorize(lambda *allargs: distfunc.rvs(*allargs), otypes=otype)
        np.random.seed(123)
        expected = rvs(*allargs)
        assert_allclose(sample, expected, rtol=1e-13)
