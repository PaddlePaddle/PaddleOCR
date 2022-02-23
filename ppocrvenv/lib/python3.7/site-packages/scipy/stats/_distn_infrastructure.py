#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
from scipy._lib._util import getfullargspec_no_self as _getfullargspec

import sys
import keyword
import re
import types
import warnings
import inspect
from itertools import zip_longest

from scipy._lib import doccer
from scipy._lib._util import _lazywhere
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state

from scipy.special import (comb, chndtr, entr, xlogy, ive)

# for root finding for continuous distribution ppf, and max likelihood
# estimation
from scipy import optimize

# for functions of continuous distributions (e.g. moments, entropy, cdf)
from scipy import integrate

# to approximate the pdf of a continuous distribution given its cdf
from scipy.misc import derivative

# for scipy.stats.entropy. Attempts to import just that function or file
# have cause import problems
from scipy import stats

from numpy import (arange, putmask, ravel, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, isinf, NINF, empty)

import numpy as np
from ._constants import _XMAX

# These are the docstring parts used for substitution in specific
# distribution docstrings

docheaders = {'methods': """\nMethods\n-------\n""",
              'notes': """\nNotes\n-----\n""",
              'examples': """\nExamples\n--------\n"""}

_doc_rvs = """\
rvs(%(shapes)s, loc=0, scale=1, size=1, random_state=None)
    Random variates.
"""
_doc_pdf = """\
pdf(x, %(shapes)s, loc=0, scale=1)
    Probability density function.
"""
_doc_logpdf = """\
logpdf(x, %(shapes)s, loc=0, scale=1)
    Log of the probability density function.
"""
_doc_pmf = """\
pmf(k, %(shapes)s, loc=0, scale=1)
    Probability mass function.
"""
_doc_logpmf = """\
logpmf(k, %(shapes)s, loc=0, scale=1)
    Log of the probability mass function.
"""
_doc_cdf = """\
cdf(x, %(shapes)s, loc=0, scale=1)
    Cumulative distribution function.
"""
_doc_logcdf = """\
logcdf(x, %(shapes)s, loc=0, scale=1)
    Log of the cumulative distribution function.
"""
_doc_sf = """\
sf(x, %(shapes)s, loc=0, scale=1)
    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
"""
_doc_logsf = """\
logsf(x, %(shapes)s, loc=0, scale=1)
    Log of the survival function.
"""
_doc_ppf = """\
ppf(q, %(shapes)s, loc=0, scale=1)
    Percent point function (inverse of ``cdf`` --- percentiles).
"""
_doc_isf = """\
isf(q, %(shapes)s, loc=0, scale=1)
    Inverse survival function (inverse of ``sf``).
"""
_doc_moment = """\
moment(n, %(shapes)s, loc=0, scale=1)
    Non-central moment of order n
"""
_doc_stats = """\
stats(%(shapes)s, loc=0, scale=1, moments='mv')
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
"""
_doc_entropy = """\
entropy(%(shapes)s, loc=0, scale=1)
    (Differential) entropy of the RV.
"""
_doc_fit = """\
fit(data)
    Parameter estimates for generic data.
    See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
    keyword arguments.
"""
_doc_expect = """\
expect(func, args=(%(shapes_)s), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
    Expected value of a function (of one argument) with respect to the distribution.
"""
_doc_expect_discrete = """\
expect(func, args=(%(shapes_)s), loc=0, lb=None, ub=None, conditional=False)
    Expected value of a function (of one argument) with respect to the distribution.
"""
_doc_median = """\
median(%(shapes)s, loc=0, scale=1)
    Median of the distribution.
"""
_doc_mean = """\
mean(%(shapes)s, loc=0, scale=1)
    Mean of the distribution.
"""
_doc_var = """\
var(%(shapes)s, loc=0, scale=1)
    Variance of the distribution.
"""
_doc_std = """\
std(%(shapes)s, loc=0, scale=1)
    Standard deviation of the distribution.
"""
_doc_interval = """\
interval(alpha, %(shapes)s, loc=0, scale=1)
    Endpoints of the range that contains fraction alpha [0, 1] of the
    distribution
"""
_doc_allmethods = ''.join([docheaders['methods'], _doc_rvs, _doc_pdf,
                           _doc_logpdf, _doc_cdf, _doc_logcdf, _doc_sf,
                           _doc_logsf, _doc_ppf, _doc_isf, _doc_moment,
                           _doc_stats, _doc_entropy, _doc_fit,
                           _doc_expect, _doc_median,
                           _doc_mean, _doc_var, _doc_std, _doc_interval])

_doc_default_longsummary = """\
As an instance of the `rv_continuous` class, `%(name)s` object inherits from it
a collection of generic methods (see below for the full list),
and completes them with details specific for this particular distribution.
"""

_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape,
location, and scale parameters returning a "frozen" continuous RV object:

rv = %(name)s(%(shapes)s, loc=0, scale=1)
    - Frozen RV object with the same methods but holding the given shape,
      location, and scale fixed.
"""
_doc_default_example = """\
Examples
--------
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate the first four moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability density function (``pdf``):

>>> x = np.linspace(%(name)s.ppf(0.01, %(shapes)s),
...                 %(name)s.ppf(0.99, %(shapes)s), 100)
>>> ax.plot(x, %(name)s.pdf(x, %(shapes)s),
...        'r-', lw=5, alpha=0.6, label='%(name)s pdf')

Alternatively, the distribution object can be called (as a function)
to fix the shape, location and scale parameters. This returns a "frozen"
RV object holding the given parameters fixed.

Freeze the distribution and display the frozen ``pdf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

Check accuracy of ``cdf`` and ``ppf``:

>>> vals = %(name)s.ppf([0.001, 0.5, 0.999], %(shapes)s)
>>> np.allclose([0.001, 0.5, 0.999], %(name)s.cdf(vals, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)

And compare the histogram:

>>> ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

"""

_doc_default_locscale = """\
The probability density above is defined in the "standardized" form. To shift
and/or scale the distribution use the ``loc`` and ``scale`` parameters.
Specifically, ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` is identically
equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
``y = (x - loc) / scale``. Note that shifting the location of a distribution
does not make it a "noncentral" distribution; noncentral generalizations of
some distributions are available in separate classes.
"""

_doc_default = ''.join([_doc_default_longsummary,
                        _doc_allmethods,
                        '\n',
                        _doc_default_example])

_doc_default_before_notes = ''.join([_doc_default_longsummary,
                                     _doc_allmethods])

docdict = {
    'rvs': _doc_rvs,
    'pdf': _doc_pdf,
    'logpdf': _doc_logpdf,
    'cdf': _doc_cdf,
    'logcdf': _doc_logcdf,
    'sf': _doc_sf,
    'logsf': _doc_logsf,
    'ppf': _doc_ppf,
    'isf': _doc_isf,
    'stats': _doc_stats,
    'entropy': _doc_entropy,
    'fit': _doc_fit,
    'moment': _doc_moment,
    'expect': _doc_expect,
    'interval': _doc_interval,
    'mean': _doc_mean,
    'std': _doc_std,
    'var': _doc_var,
    'median': _doc_median,
    'allmethods': _doc_allmethods,
    'longsummary': _doc_default_longsummary,
    'frozennote': _doc_default_frozen_note,
    'example': _doc_default_example,
    'default': _doc_default,
    'before_notes': _doc_default_before_notes,
    'after_notes': _doc_default_locscale
}

# Reuse common content between continuous and discrete docs, change some
# minor bits.
docdict_discrete = docdict.copy()

docdict_discrete['pmf'] = _doc_pmf
docdict_discrete['logpmf'] = _doc_logpmf
docdict_discrete['expect'] = _doc_expect_discrete
_doc_disc_methods = ['rvs', 'pmf', 'logpmf', 'cdf', 'logcdf', 'sf', 'logsf',
                     'ppf', 'isf', 'stats', 'entropy', 'expect', 'median',
                     'mean', 'var', 'std', 'interval']
for obj in _doc_disc_methods:
    docdict_discrete[obj] = docdict_discrete[obj].replace(', scale=1', '')

_doc_disc_methods_err_varname = ['cdf', 'logcdf', 'sf', 'logsf']
for obj in _doc_disc_methods_err_varname:
    docdict_discrete[obj] = docdict_discrete[obj].replace('(x, ', '(k, ')

docdict_discrete.pop('pdf')
docdict_discrete.pop('logpdf')

_doc_allmethods = ''.join([docdict_discrete[obj] for obj in _doc_disc_methods])
docdict_discrete['allmethods'] = docheaders['methods'] + _doc_allmethods

docdict_discrete['longsummary'] = _doc_default_longsummary.replace(
    'rv_continuous', 'rv_discrete')

_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape and
location parameters returning a "frozen" discrete RV object:

rv = %(name)s(%(shapes)s, loc=0)
    - Frozen RV object with the same methods but holding the given shape and
      location fixed.
"""
docdict_discrete['frozennote'] = _doc_default_frozen_note

_doc_default_discrete_example = """\
Examples
--------
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate the first four moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability mass function (``pmf``):

>>> x = np.arange(%(name)s.ppf(0.01, %(shapes)s),
...               %(name)s.ppf(0.99, %(shapes)s))
>>> ax.plot(x, %(name)s.pmf(x, %(shapes)s), 'bo', ms=8, label='%(name)s pmf')
>>> ax.vlines(x, 0, %(name)s.pmf(x, %(shapes)s), colors='b', lw=5, alpha=0.5)

Alternatively, the distribution object can be called (as a function)
to fix the shape and location. This returns a "frozen" RV object holding
the given parameters fixed.

Freeze the distribution and display the frozen ``pmf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
...         label='frozen pmf')
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

Check accuracy of ``cdf`` and ``ppf``:

>>> prob = %(name)s.cdf(x, %(shapes)s)
>>> np.allclose(x, %(name)s.ppf(prob, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)
"""


_doc_default_discrete_locscale = """\
The probability mass function above is defined in the "standardized" form.
To shift distribution use the ``loc`` parameter.
Specifically, ``%(name)s.pmf(k, %(shapes)s, loc)`` is identically
equivalent to ``%(name)s.pmf(k - loc, %(shapes)s)``.
"""

docdict_discrete['example'] = _doc_default_discrete_example
docdict_discrete['after_notes'] = _doc_default_discrete_locscale

_doc_default_before_notes = ''.join([docdict_discrete['longsummary'],
                                     docdict_discrete['allmethods']])
docdict_discrete['before_notes'] = _doc_default_before_notes

_doc_default_disc = ''.join([docdict_discrete['longsummary'],
                             docdict_discrete['allmethods'],
                             docdict_discrete['frozennote'],
                             docdict_discrete['example']])
docdict_discrete['default'] = _doc_default_disc

# clean up all the separate docstring elements, we do not need them anymore
for obj in [s for s in dir() if s.startswith('_doc_')]:
    exec('del ' + obj)
del obj


def _moment(data, n, mu=None):
    if mu is None:
        mu = data.mean()
    return ((data - mu)**n).mean()


def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    if (n == 0):
        return 1.0
    elif (n == 1):
        if mu is None:
            val = moment_func(1, *args)
        else:
            val = mu
    elif (n == 2):
        if mu2 is None or mu is None:
            val = moment_func(2, *args)
        else:
            val = mu2 + mu*mu
    elif (n == 3):
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)
        else:
            mu3 = g1 * np.power(mu2, 1.5)  # 3rd central moment
            val = mu3+3*mu*mu2+mu*mu*mu  # 3rd non-central moment
    elif (n == 4):
        if g1 is None or g2 is None or mu2 is None or mu is None:
            val = moment_func(4, *args)
        else:
            mu4 = (g2+3.0)*(mu2**2.0)  # 4th central moment
            mu3 = g1*np.power(mu2, 1.5)  # 3rd central moment
            val = mu4+4*mu*mu3+6*mu*mu*mu2+mu*mu*mu*mu
    else:
        val = moment_func(n, *args)

    return val


def _skew(data):
    """
    skew is third central moment / variance**(1.5)
    """
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu)**2).mean()
    m3 = ((data - mu)**3).mean()
    return m3 / np.power(m2, 1.5)


def _kurtosis(data):
    """kurtosis is fourth central moment / variance**2 - 3."""
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu)**2).mean()
    m4 = ((data - mu)**4).mean()
    return m4 / m2**2 - 3


def _fit_determine_optimizer(optimizer):
    if not callable(optimizer) and isinstance(optimizer, str):
        if not optimizer.startswith('fmin_'):
            optimizer = "fmin_"+optimizer
        if optimizer == 'fmin_':
            optimizer = 'fmin'
        try:
            optimizer = getattr(optimize, optimizer)
        except AttributeError as e:
            raise ValueError("%s is not a valid optimizer" % optimizer) from e
    return optimizer


# Frozen RV class
class rv_frozen:

    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds

        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.a, self.b = self.dist._get_support(*shapes)

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def pdf(self, x):   # raises AttributeError in frozen discrete distribution
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)

    def cdf(self, x):
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, n):
        return self.dist.moment(n, *self.args, **self.kwds)

    def entropy(self):
        return self.dist.entropy(*self.args, **self.kwds)

    def pmf(self, k):
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):
        return self.dist.logpmf(k, *self.args, **self.kwds)

    def interval(self, alpha):
        return self.dist.interval(alpha, *self.args, **self.kwds)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        # expect method only accepts shape parameters as positional args
        # hence convert self.args, self.kwds, also loc/scale
        # See the .expect method docstrings for the meaning of
        # other parameters.
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        if isinstance(self.dist, rv_discrete):
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            return self.dist.expect(func, a, loc, scale, lb, ub,
                                    conditional, **kwds)

    def support(self):
        return self.dist.support(*self.args, **self.kwds)


def argsreduce(cond, *args):
    """Clean arguments to:

    1. Ensure all arguments are iterable (arrays of dimension at least one
    2. If cond != True and size > 1, ravel(args[i]) where ravel(condition) is
       True, in 1D.

    Return list of processed arguments.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> A = rng.random((4, 5))
    >>> B = 2
    >>> C = rng.random((1, 5))
    >>> cond = np.ones(A.shape)
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> A1.shape
    (4, 5)
    >>> B1.shape
    (1,)
    >>> C1.shape
    (1, 5)
    >>> cond[2,:] = 0
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> A1.shape
    (15,)
    >>> B1.shape
    (1,)
    >>> C1.shape
    (15,)

    """
    # some distributions assume arguments are iterable.
    newargs = np.atleast_1d(*args)

    # np.atleast_1d returns an array if only one argument, or a list of arrays
    # if more than one argument.
    if not isinstance(newargs, list):
        newargs = [newargs, ]

    if np.all(cond):
        # Nothing to do
        return newargs

    s = cond.shape
    # np.extract returns flattened arrays, which are not broadcastable together
    # unless they are either the same size or size == 1.
    return [(arg if np.size(arg) == 1
            else np.extract(cond, np.broadcast_to(arg, s)))
            for arg in newargs]


parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s

def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):
    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size)

def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""


# Both the continuous and discrete distributions depend on ncx2.
# The function name ncx2 is an abbreviation for noncentral chi squared.

def _ncx2_log_pdf(x, df, nc):
    # We use (xs**2 + ns**2)/2 = (xs - ns)**2/2  + xs*ns, and include the
    # factor of exp(-xs*ns) into the ive function to improve numerical
    # stability at large values of xs. See also `rice.pdf`.
    df2 = df/2.0 - 1.0
    xs, ns = np.sqrt(x), np.sqrt(nc)
    res = xlogy(df2/2.0, x/nc) - 0.5*(xs - ns)**2
    corr = ive(df2, xs*ns) / 2.0
    # Return res + np.log(corr) avoiding np.log(0)
    return _lazywhere(
        corr > 0,
        (res, corr),
        f=lambda r, c: r + np.log(c),
        fillvalue=-np.inf)


def _ncx2_pdf(x, df, nc):
    # Copy of _ncx2_log_pdf avoiding np.log(0) when corr = 0
    df2 = df/2.0 - 1.0
    xs, ns = np.sqrt(x), np.sqrt(nc)
    res = xlogy(df2/2.0, x/nc) - 0.5*(xs - ns)**2
    corr = ive(df2, xs*ns) / 2.0
    return np.exp(res) * corr


def _ncx2_cdf(x, df, nc):
    return chndtr(x, df, nc)


class rv_generic:
    """Class which encapsulates common functionality between rv_discrete
    and rv_continuous.

    """
    def __init__(self, seed=None):
        super().__init__()

        # figure out if _stats signature has 'moments' keyword
        sig = _getfullargspec(self._stats)
        self._stats_has_moments = ((sig.varkw is not None) or
                                   ('moments' in sig.args) or
                                   ('moments' in sig.kwonlyargs))
        self._random_state = check_random_state(seed)

        # For historical reasons, `size` was made an attribute that was read
        # inside _rvs().  The code is being changed so that 'size'
        # is an argument
        # to self._rvs(). However some external (non-SciPy) distributions
        # have not
        # been updated.  Maintain backwards compatibility by checking if
        # the self._rvs() signature has the 'size' keyword, or a **kwarg,
        # and if not set self._size inside self.rvs()
        # before calling self._rvs().
        argspec = inspect.getfullargspec(self._rvs)
        self._rvs_uses_size_attribute = (argspec.varkw is None and
                                         'size' not in argspec.args and
                                         'size' not in argspec.kwonlyargs)
        # Warn on first use only
        self._rvs_size_warned = False

    @property
    def random_state(self):
        """Get or set the generator object for generating random variates.

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def __setstate__(self, state):
        try:
            self.__dict__.update(state)
            # attaches the dynamically created methods on each instance.
            # if a subclass overrides rv_generic.__setstate__, or implements
            # it's own _attach_methods, then it must make sure that
            # _attach_argparser_methods is called.
            self._attach_methods()
        except ValueError:
            # reconstitute an old pickle scipy<1.6, that contains
            # (_ctor_param, random_state) as state
            self._ctor_param = state[0]
            self._random_state = state[1]
            self.__init__()

    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_* instance.

        This method must be overridden by subclasses, and must itself call
         _attach_argparser_methods. This method is called in __init__ in
         subclasses, and in __setstate__
        """
        raise NotImplementedError

    def _attach_argparser_methods(self):
        """
        Generates the argument-parsing functions dynamically and attaches
        them to the instance.

        Should be called from `_attach_methods`, typically in __init__ and
        during unpickling (__setstate__)
        """
        ns = {}
        exec(self._parse_arg_template, ns)
        # NB: attach to the instance, not class
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, types.MethodType(ns[name], self))

    def _construct_argparser(
            self, meths_to_inspect, locscale_in, locscale_out):
        """Construct the parser string for the shape arguments.

        This method should be called in __init__ of a class for each
        distribution. It creates the `_parse_arg_template` attribute that is
        then used by `_attach_argparser_methods` to dynamically create and
        attach the `_parse_args`, `_parse_args_stats`, `_parse_args_rvs`
        methods to the instance.

        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.

        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """

        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, str):
                raise TypeError('shapes must be a string.')

            shapes = self.shapes.replace(',', ' ').split()

            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError(
                        'shapes must be valid python identifiers')
        else:
            # find out the call signatures (_pdf, _cdf etc), deduce shape
            # arguments. Generic methods only have 'self, x', any further args
            # are shapes.
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getfullargspec(meth)  # NB does not contain self
                args = shapes_args.args[1:]       # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError(
                            '*args are not allowed w/out explicit shapes')
                    if shapes_args.varkw is not None:
                        raise TypeError(
                            '**kwds are not allowed w/out explicit shapes')
                    if shapes_args.kwonlyargs:
                        raise TypeError(
                            'kwonly args are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')

            if shapes_list:
                shapes = shapes_list[0]

                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str,
                   locscale_in=locscale_in,
                   locscale_out=locscale_out,
                   )

        # this string is used by _attach_argparser_methods
        self._parse_arg_template = parse_arg_template % dct

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            # allows more general subclassing with *args
            self.numargs = len(shapes)

    def _construct_doc(self, docdict, shapes_vals=None):
        """Construct the instance docstring with string substitutions."""
        tempdict = docdict.copy()
        tempdict['name'] = self.name or 'distname'
        tempdict['shapes'] = self.shapes or ''

        if shapes_vals is None:
            shapes_vals = ()
        vals = ', '.join('%.3g' % val for val in shapes_vals)
        tempdict['vals'] = vals

        tempdict['shapes_'] = self.shapes or ''
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','

        if self.shapes:
            tempdict['set_vals_stmt'] = '>>> %s = %s' % (self.shapes, vals)
        else:
            tempdict['set_vals_stmt'] = ''

        if self.shapes is None:
            # remove shapes from call parameters if there are none
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace(
                    "\n%(shapes)s : array_like\n    shape parameters", "")
        for i in range(2):
            if self.shapes is None:
                # necessary because we use %(shapes)s in two forms (w w/o ", ")
                self.__doc__ = self.__doc__.replace("%(shapes)s, ", "")
            try:
                self.__doc__ = doccer.docformat(self.__doc__, tempdict)
            except TypeError as e:
                raise Exception("Unable to construct docstring for "
                                "distribution \"%s\": %s" %
                                (self.name, repr(e))) from e

        # correct for empty shapes
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')

    def _construct_default_doc(self, longname=None, extradoc=None,
                               docdict=None, discrete='continuous'):
        """Construct instance docstring from the default template."""
        if longname is None:
            longname = 'A'
        if extradoc is None:
            extradoc = ''
        if extradoc.startswith('\n\n'):
            extradoc = extradoc[2:]
        self.__doc__ = ''.join(['%s %s random variable.' % (longname, discrete),
                                '\n\n%(before_notes)s\n', docheaders['notes'],
                                extradoc, '\n%(example)s'])
        self._construct_doc(docdict)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.

        """
        return rv_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)
    __call__.__doc__ = freeze.__doc__

    # The actual calculation functions (no basic checking need be done)
    # If these are defined, the others won't be looked at.
    # Otherwise, the other set can be defined.
    def _stats(self, *args, **kwds):
        return None, None, None, None

    # Noncentral moments (also known as the moment about the origin).
    # Expressed in LaTeX, munp would be $\mu'_{n}$, i.e. "mu-sub-n-prime".
    # The primed mu is a widely used notation for the noncentral moment.
    def _munp(self, n, *args):
        # Silence floating point warnings from integration.
        with np.errstate(all='ignore'):
            vals = self.generic_moment(n, *args)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        # Handle broadcasting and size validation of the rvs method.
        # Subclasses should not have to override this method.
        # The rule is that if `size` is not None, then `size` gives the
        # shape of the result (integer values of `size` are treated as
        # tuples with length 1; i.e. `size=3` is the same as `size=(3,)`.)
        #
        # `args` is expected to contain the shape parameters (if any), the
        # location and the scale in a flat tuple (e.g. if there are two
        # shape parameters `a` and `b`, `args` will be `(a, b, loc, scale)`).
        # The only keyword argument expected is 'size'.
        size = kwargs.get('size', None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a

        # Eliminate trivial leading dimensions.  In the convention
        # used by numpy's random variate generators, trivial leading
        # dimensions are effectively ignored.  In other words, when `size`
        # is given, trivial leading dimensions of the broadcast parameters
        # in excess of the number of dimensions  in size are ignored, e.g.
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]], size=3)
        #   array([ 1.00104267,  3.00422496,  4.99799278])
        # If `size` is not given, the exact broadcast shape is preserved:
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]])
        #   array([[[[ 1.00862899,  3.00061431,  4.99867122]]]])
        #
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim

        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))

        # Check compatibility of size_ with the broadcast shape of all
        # the parameters.  This check is intended to be consistent with
        # how the numpy random variate generators (e.g. np.random.normal,
        # np.random.beta) handle their arguments.   The rule is that, if size
        # is given, it determines the shape of the output.  Broadcasting
        # can't change the output size.

        # This is the standard broadcasting convention of extending the
        # shape with fewer dimensions with enough dimensions of length 1
        # so that the two shapes have the same number of dimensions.
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,)*(-ndiff) + bcast_shape
        elif ndiff > 0:
            size_ = (1,)*ndiff + size_

        # This compatibility test is not standard.  In "regular" broadcasting,
        # two shapes are compatible if for each dimension, the lengths are the
        # same or one of the lengths is 1.  Here, the length of a dimension in
        # size_ must not be less than the corresponding length in bcast_shape.
        ok = all([bcdim == 1 or bcdim == szdim
                  for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError("size does not match the broadcast shape of "
                             "the parameters. %s, %s, %s" % (size, size_,
                                                             bcast_shape))

        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]

        return param_bcast, loc_bcast, scale_bcast, size_

    # These are the methods you must define (standard form functions)
    # NB: generic _pdf, _logpdf, _cdf are different for
    # rv_continuous and rv_discrete hence are defined in there
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond

    def _get_support(self, *args, **kwargs):
        """Return the support of the (unscaled, unshifted) distribution.

        *Must* be overridden by distributions which have support dependent
        upon the shape parameters of the distribution.  Any such override
        *must not* set or change any of the class members, as these members
        are shared amongst all instances of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support for the specified
            shape parameters.
        """
        return self.a, self.b

    def _support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a <= x) & (x <= b)

    def _open_support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a < x) & (x < b)

    def _rvs(self, *args, size=None, random_state=None):
        # This method must handle size being a tuple, and it must
        # properly broadcast *args and size.  size might be
        # an empty tuple, which means a scalar random variate is to be
        # generated.

        # Use basic inverse cdf algorithm for RV generation as default.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        return 1.0-self._cdf(x, *args)

    def _logsf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        return self._ppf(1.0-q, *args)  # use correct _ppf for subclasses

    # These are actually called, and should not be overwritten if you
    # want to keep error checking.
    def rvs(self, *args, **kwds):
        """Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            raise ValueError("Domain error in arguments.")

        if np.all(scale == 0):
            return loc*ones(size, 'd')

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state

        # Maintain backwards compatibility by setting self._size
        # for distributions that still need it.
        if self._rvs_uses_size_attribute:
            if not self._rvs_size_warned:
                warnings.warn(
                    f'The signature of {self._rvs} does not contain '
                    f'a "size" keyword.  Such signatures are deprecated.',
                    np.VisibleDeprecationWarning)
                self._rvs_size_warned = True
            self._size = size
            self._random_state = random_state
            vals = self._rvs(*args)
        else:
            vals = self._rvs(*args, size=size, random_state=random_state)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # Cast to int if discrete
        if discrete:
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(int)

        return vals

    def stats(self, *args, **kwds):
        """Some statistics of the given RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional (continuous RVs only)
            scale parameter (default=1)
        moments : str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean,
            'v' = variance,
            's' = (Fisher's) skew,
            'k' = (Fisher's) kurtosis.
            (default is 'mv')

        Returns
        -------
        stats : sequence
            of requested moments.

        """
        args, loc, scale, moments = self._parse_args_stats(*args, **kwds)
        # scale = 1 by construction for discrete RVs
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = np.full(shape(cond), fill_value=self.badvalue)

        # Use only entries that are valid in calculation
        if np.any(cond):
            goodargs = argsreduce(cond, *(args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]

            if self._stats_has_moments:
                mu, mu2, g1, g2 = self._stats(*goodargs,
                                              **{'moments': moments})
            else:
                mu, mu2, g1, g2 = self._stats(*goodargs)
            if g1 is None:
                mu3 = None
            else:
                if mu2 is None:
                    mu2 = self._munp(2, *goodargs)
                if g2 is None:
                    # (mu2**1.5) breaks down for nan and inf
                    mu3 = g1 * np.power(mu2, 1.5)

            if 'm' in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)

            if 'v' in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    # if mean is inf then var is also inf
                    with np.errstate(invalid='ignore'):
                        mu2 = np.where(np.isfinite(mu), mu2p - mu**2, np.inf)
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)

            if 's' in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    with np.errstate(invalid='ignore'):
                        mu3 = (-mu*mu - 3*mu2)*mu + mu3p
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)

            if 'k' in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                    with np.errstate(invalid='ignore'):
                        mu4 = ((-mu**2 - 6*mu2) * mu - 4*mu3)*mu + mu4p
                        g2 = mu4 / mu2**2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:  # no valid args
            output = [default.copy() for _ in moments]

        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def entropy(self, *args, **kwds):
        """Differential entropy of the RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional  (continuous distributions only).
            Scale parameter (default=1).

        Notes
        -----
        Entropy is defined base `e`:

        >>> drv = rv_discrete(values=((0, 1), (0.5, 0.5)))
        >>> np.allclose(drv.entropy(), np.log(2.0))
        True

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        # NB: for discrete distributions scale=1 by construction in _parse_args
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), 'd')
        place(output, (1-cond0), self.badvalue)
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        return output

    def moment(self, n, *args, **kwds):
        """n-th order non-central moment of distribution.

        Parameters
        ----------
        n : int, n >= 1
            Order of moment.
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        if not (self._argcheck(*args) and (scale > 0)):
            return nan
        if (floor(n) != n):
            raise ValueError("Moment must be an integer.")
        if (n < 0):
            raise ValueError("Moment must be positive.")
        mu, mu2, g1, g2 = None, None, None, None
        if (n > 0) and (n < 5):
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'vk'}[n]}
            else:
                mdict = {}
            mu, mu2, g1, g2 = self._stats(*args, **mdict)
        val = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, args)

        # Convert to transformed  X = L + S*Y
        # E[X^n] = E[(L+S*Y)^n] = L^n sum(comb(n, k)*(S/L)^k E[Y^k], k=0...n)
        if loc == 0:
            return scale**n * val
        else:
            result = 0
            fac = float(scale) / float(loc)
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp, args)
                result += comb(n, k, exact=True)*(fac**k) * valk
            result += fac**n * val
            return result * loc**n

    def median(self, *args, **kwds):
        """Median of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter, Default is 0.
        scale : array_like, optional
            Scale parameter, Default is 1.

        Returns
        -------
        median : float
            The median of the distribution.

        See Also
        --------
        rv_discrete.ppf
            Inverse of the CDF

        """
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        """Mean of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        mean : float
            the mean of the distribution

        """
        kwds['moments'] = 'm'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        """Variance of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        var : float
            the variance of the distribution

        """
        kwds['moments'] = 'v'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        """Standard deviation of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        std : float
            standard deviation of the distribution

        """
        kwds['moments'] = 'v'
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, alpha, *args, **kwds):
        """Confidence interval with equal areas around the median.

        Parameters
        ----------
        alpha : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : ndarray of float
            end-points of range that contain ``100 * alpha %`` of the rv's
            possible values.

        """
        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")
        q1 = (1.0-alpha)/2
        q2 = (1.0+alpha)/2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return a, b

    def support(self, *args, **kwargs):
        """Support of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : array_like
            end-points of the distribution's support.

        """
        args, loc, scale = self._parse_args(*args, **kwargs)
        arrs = np.broadcast_arrays(*args, loc, scale)
        args, loc, scale = arrs[:-2], arrs[-2], arrs[-1]
        cond = self._argcheck(*args) & (scale > 0)
        _a, _b = self._get_support(*args)
        if cond.all():
            return _a * scale + loc, _b * scale + loc
        elif cond.ndim == 0:
            return self.badvalue, self.badvalue
        # promote bounds to at least float to fill in the badvalue
        _a, _b = np.asarray(_a).astype('d'), np.asarray(_b).astype('d')
        out_a, out_b = _a * scale + loc, _b * scale + loc
        place(out_a, 1-cond, self.badvalue)
        place(out_b, 1-cond, self.badvalue)
        return out_a, out_b


def _get_fixed_fit_value(kwds, names):
    """
    Given names such as `['f0', 'fa', 'fix_a']`, check that there is
    at most one non-None value in `kwds` associaed with those names.
    Return that value, or None if none of the names occur in `kwds`.
    As a side effect, all occurrences of those names in `kwds` are
    removed.
    """
    vals = [(name, kwds.pop(name)) for name in names if name in kwds]
    if len(vals) > 1:
        repeated = [name for name, val in vals]
        raise ValueError("fit method got multiple keyword arguments to "
                         "specify the same fixed parameter: " +
                         ', '.join(repeated))
    return vals[0][1] if vals else None

#  continuous random variables: implement maybe later
#
#  hf  --- Hazard Function (PDF / SF)
#  chf  --- Cumulative hazard function (-log(SF))
#  psf --- Probability sparsity function (reciprocal of the pdf) in
#                units of percent-point-function (as a function of q).
#                Also, the derivative of the percent-point function.


class rv_continuous(rv_generic):
    """A generic continuous random variable class meant for subclassing.

    `rv_continuous` is a base class to construct specific distribution classes
    and instances for continuous random variables. It cannot be used
    directly as a distribution.

    Parameters
    ----------
    momtype : int, optional
        The type of generic moment calculation to use: 0 for pdf, 1 (default)
        for ppf.
    a : float, optional
        Lower bound of the support of the distribution, default is minus
        infinity.
    b : float, optional
        Upper bound of the support of the distribution, default is plus
        infinity.
    xtol : float, optional
        The tolerance for fixed point calculation for generic ppf.
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    extradoc :  str, optional, deprecated
        This string is used as the last part of the docstring returned when a
        subclass has no docstring of its own. Note: `extradoc` exists for
        backwards compatibility, do not use for new subclasses.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pdf
    logpdf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    fit
    fit_loc_scale
    nnlf
    support

    Notes
    -----
    Public methods of an instance of a distribution class (e.g., ``pdf``,
    ``cdf``) check their arguments and pass valid arguments to private,
    computational methods (``_pdf``, ``_cdf``). For ``pdf(x)``, ``x`` is valid
    if it is within the support of the distribution.
    Whether a shape parameter is valid is decided by an ``_argcheck`` method
    (which defaults to checking that its arguments are strictly positive.)

    **Subclassing**

    New random variables can be defined by subclassing the `rv_continuous` class
    and re-defining at least the ``_pdf`` or the ``_cdf`` method (normalized
    to location 0 and scale 1).

    If positive argument checking is not correct for your RV
    then you will also need to re-define the ``_argcheck`` method.

    For most of the scipy.stats distributions, the support interval doesn't
    depend on the shape parameters. ``x`` being in the support interval is
    equivalent to ``self.a <= x <= self.b``.  If either of the endpoints of
    the support do depend on the shape parameters, then
    i) the distribution must implement the ``_get_support`` method; and
    ii) those dependent endpoints must be omitted from the distribution's
    call to the ``rv_continuous`` initializer.

    Correct, but potentially slow defaults exist for the remaining
    methods but for speed and/or accuracy you can over-ride::

      _logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf

    The default method ``_rvs`` relies on the inverse of the cdf, ``_ppf``,
    applied to a uniform random variate. In order to generate random variates
    efficiently, either the default ``_ppf`` needs to be overwritten (e.g.
    if the inverse cdf can expressed in an explicit form) or a sampling
    method needs to be implemented in a custom ``_rvs`` method.

    If possible, you should override ``_isf``, ``_sf`` or ``_logsf``.
    The main reason would be to improve numerical accuracy: for example,
    the survival function ``_sf`` is computed as ``1 - _cdf`` which can
    result in loss of precision if ``_cdf(x)`` is close to one.

    **Methods that can be overwritten by subclasses**
    ::

      _rvs
      _pdf
      _cdf
      _sf
      _ppf
      _isf
      _stats
      _munp
      _entropy
      _argcheck
      _get_support

    There are additional (internal and private) generic methods that can
    be useful for cross-checking and for debugging, but might work in all
    cases when directly called.

    A note on ``shapes``: subclasses need not specify them explicitly. In this
    case, `shapes` will be automatically deduced from the signatures of the
    overridden methods (`pdf`, `cdf` etc).
    If, for some reason, you prefer to avoid relying on introspection, you can
    specify ``shapes`` explicitly as an argument to the instance constructor.


    **Frozen Distributions**

    Normally, you must provide shape parameters (and, optionally, location and
    scale parameters to each call of a method of a distribution.

    Alternatively, the object may be called (as a function) to fix the shape,
    location, and scale parameters returning a "frozen" continuous RV object:

    rv = generic(<shape(s)>, loc=0, scale=1)
        `rv_frozen` object with the same methods but holding the given shape,
        location, and scale fixed

    **Statistics**

    Statistics are computed using numerical integration by default.
    For speed you can redefine this using ``_stats``:

     - take shape parameters and return mu, mu2, g1, g2
     - If you can't compute one of these, return it as None
     - Can also be defined with a keyword argument ``moments``, which is a
       string composed of "m", "v", "s", and/or "k".
       Only the components appearing in string should be computed and
       returned in the order "m", "v", "s", or "k"  with missing values
       returned as None.

    Alternatively, you can override ``_munp``, which takes ``n`` and shape
    parameters and returns the n-th non-central moment of the distribution.

    Examples
    --------
    To create a new Gaussian distribution, we would do the following:

    >>> from scipy.stats import rv_continuous
    >>> class gaussian_gen(rv_continuous):
    ...     "Gaussian distribution"
    ...     def _pdf(self, x):
    ...         return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
    >>> gaussian = gaussian_gen(name='gaussian')

    ``scipy.stats`` distributions are *instances*, so here we subclass
    `rv_continuous` and create an instance. With this, we now have
    a fully functional distribution with all relevant methods automagically
    generated by the framework.

    Note that above we defined a standard normal distribution, with zero mean
    and unit variance. Shifting and scaling of the distribution can be done
    by using ``loc`` and ``scale`` parameters: ``gaussian.pdf(x, loc, scale)``
    essentially computes ``y = (x - loc) / scale`` and
    ``gaussian._pdf(y) / scale``.

    """
    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, extradoc=None, seed=None):

        super().__init__(seed)

        # save the ctor parameters, cf generic freeze
        self._ctor_param = dict(
            momtype=momtype, a=a, b=b, xtol=xtol,
            badvalue=badvalue, name=name, longname=longname,
            shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        if name is None:
            name = 'Distribution'
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = -inf
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes
        self.extradoc = extradoc

        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf],
                                  locscale_in='loc=0, scale=1',
                                  locscale_out='loc, scale')
        self._attach_methods()

        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        if sys.flags.optimize < 2:
            # Skip adding docstrings if interpreter is run with -OO
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            extradoc=extradoc,
                                            docdict=docdict,
                                            discrete='continuous')
            else:
                dct = dict(distcont)
                self._construct_doc(docdict, dct.get(self.name))

    def __getstate__(self):
        dct = self.__dict__.copy()

        # these methods will be remade in __setstate__
        # _random_state attribute is taken care of by rv_generic
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs",
                 "_cdfvec", "_ppfvec", "vecentropy", "generic_moment"]
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        """
        Attaches dynamically created methods to the rv_continuous instance.
        """
        # _attach_methods is responsible for calling _attach_argparser_methods
        self._attach_argparser_methods()

        # nin correction
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = vectorize(self._entropy, otypes='d')
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1

        if self.moment_type == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes='d')
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes='d')
        # Because of the *args argument of _mom0_sc, vectorize cannot count the
        # number of arguments correctly.
        self.generic_moment.nin = self.numargs + 1

    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['xtol'] = self.xtol
        dct['badvalue'] = self.badvalue
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        dct['extradoc'] = self.extradoc
        return dct

    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x, )+args)-q

    def _ppf_single(self, q, *args):
        factor = 10.
        left, right = self._get_support(*args)

        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q, *args) > 0.:
                left, right = left * factor, left
            # left is now such that cdf(left) <= q
            # if right has changed, then cdf(right) > q

        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q, *args) < 0.:
                left, right = right, right * factor
            # right is now such that cdf(right) >= q

        return optimize.brentq(self._ppf_to_solve,
                               left, right, args=(q,)+args, xtol=self.xtol)

    # moment from definition
    def _mom_integ0(self, x, m, *args):
        return x**m * self.pdf(x, *args)

    def _mom0_sc(self, m, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._mom_integ0, _a, _b,
                              args=(m,)+args)[0]

    # moment calculated using ppf
    def _mom_integ1(self, q, m, *args):
        return (self.ppf(q, *args))**m

    def _mom1_sc(self, m, *args):
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,)+args)[0]

    def _pdf(self, x, *args):
        return derivative(self._cdf, x, dx=1e-5, args=args, order=5)

    # Could also define any of these
    def _logpdf(self, x, *args):
        return log(self._pdf(x, *args))

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._pdf, _a, x, args=args)[0]

    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)

    # generic _argcheck, _logcdf, _sf, _logsf, _ppf, _isf, _rvs are defined
    # in rv_generic

    def pdf(self, x, *args, **kwds):
        """Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output

    def logpdf(self, x, *args, **kwds):
        """Log of the probability density function at x of the given RV.

        This uses a more numerically accurate calculation if available.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logpdf : array_like
            Log of the probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(NINF)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, x, *args, **kwds):
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= np.asarray(_b)) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, x, *args, **kwds):
        """Log of the cumulative distribution function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(NINF)
        place(output, (1-cond0)*(cond1 == cond1)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, x, *args, **kwds):
        """Survival function (1 - `cdf`) at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        sf : array_like
            Survival function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._sf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, x, *args, **kwds):
        """Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as (1 - `cdf`),
        evaluated at `x`.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `x`.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.find_common_type([x.dtype, np.float64], [])
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(NINF)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            lower tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            upper tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : ndarray or scalar
            Quantile corresponding to the upper tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 1)
        cond3 = cond0 & (q == 0)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._isf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _nnlf(self, x, *args):
        return -np.sum(self._logpdf(x, *args), axis=0)

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError as e:
            raise ValueError("Not enough input arguments.") from e
        return loc, scale, args

    def nnlf(self, theta, x):
        """Negative loglikelihood function.

        Notes
        -----
        This is ``-sum(log pdf(x, theta), axis=0)`` where `theta` are the
        parameters (including loc and scale).
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x-loc) / scale)
        n_log_scale = len(x) * log(scale)
        if np.any(~self._support_mask(x, *args)):
            return inf
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf_and_penalty(self, x, args):
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        logpdf = self._logpdf(x, *args)
        finite_logpdf = np.isfinite(logpdf)
        n_bad += np.sum(~finite_logpdf, axis=0)
        if n_bad > 0:
            penalty = n_bad * log(_XMAX) * 100
            return -np.sum(logpdf[finite_logpdf], axis=0) + penalty
        return -np.sum(logpdf, axis=0)

    def _penalized_nnlf(self, theta, x):
        """Penalized negative loglikelihood function.

        i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x-loc) / scale)
        n_log_scale = len(x) * log(scale)
        return self._nnlf_and_penalty(x, args) + n_log_scale

    def _fitstart(self, data, args=None):
        """Starting point for fit (shape arguments + loc + scale)."""
        if args is None:
            args = (1.0,)*self.numargs
        loc, scale = self._fit_loc_scale_support(data, *args)
        return args + (loc, scale)

    def _reduce_func(self, args, kwds, data=None):
        """
        Return the (possibly reduced) function to optimize in order to find MLE
        estimates for the .fit method.
        """
        # Convert fixed shape parameters to the standard numeric form: e.g. for
        # stats.beta, shapes='a, b'. To fix `a`, the caller can give a value
        # for `f0`, `fa` or 'fix_a'.  The following converts the latter two
        # into the first (numeric) form.
        shapes = []
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                key = 'f' + str(j)
                names = [key, 'f' + s, 'fix_' + s]
                val = _get_fixed_fit_value(kwds, names)
                if val is not None:
                    kwds[key] = val

        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        methods = {"mle", "mm"}
        method = kwds.pop('method', "mle").lower()
        if method == "mm":
            n_params = len(shapes) + 2 - len(fixedn)
            exponents = (np.arange(1, n_params+1))[:, np.newaxis]
            data_moments = np.sum(data[None, :]**exponents/len(data), axis=1)

            def objective(theta, x):
                return self._moment_error(theta, x, data_moments)
        elif method == "mle":
            objective = self._penalized_nnlf
        else:
            raise ValueError("Method '{0}' not available; must be one of {1}"
                             .format(method, methods))

        if len(fixedn) == 0:
            func = objective
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError(
                    "All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                # Replace with theta for all numbers not in fixedn
                # This allows the non-fixed values to vary, but
                #  we still call self.nnlf with all parameters.
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return objective(newtheta, x)

        return x0, func, restore, args

    def _moment_error(self, theta, x, data_moments):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf

        dist_moments = np.array([self.moment(i+1, *args, loc=loc, scale=scale)
                                 for i in range(len(data_moments))])
        if np.any(np.isnan(dist_moments)):
            raise ValueError("Method of moments encountered a non-finite "
                             "distribution moment and cannot continue. "
                             "Consider trying method='MLE'.")

        return (((data_moments - dist_moments) /
                 np.maximum(np.abs(data_moments), 1e-8))**2).sum()

    def fit(self, data, *args, **kwds):
        """
        Return estimates of shape (if applicable), location, and scale
        parameters from data. The default estimation method is Maximum
        Likelihood Estimation (MLE), but Method of Moments (MM)
        is also available.

        Starting estimates for
        the fit are given by input arguments; for any arguments not provided
        with starting estimates, ``self._fitstart(data)`` is called to generate
        such.

        One can hold some parameters fixed to specific values by passing in
        keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)
        and ``floc`` and ``fscale`` (for location and scale parameters,
        respectively).

        Parameters
        ----------
        data : array_like
            Data to use in estimating the distribution parameters.
        arg1, arg2, arg3,... : floats, optional
            Starting value(s) for any shape-characterizing arguments (those not
            provided will be determined by a call to ``_fitstart(data)``).
            No default value.
        kwds : floats, optional
            - `loc`: initial guess of the distribution's location parameter.
            - `scale`: initial guess of the distribution's scale parameter.

            Special keyword arguments are recognized as holding certain
            parameters fixed:

            - f0...fn : hold respective shape parameters fixed.
              Alternatively, shape parameters to fix can be specified by name.
              For example, if ``self.shapes == "a, b"``, ``fa`` and ``fix_a``
              are equivalent to ``f0``, and ``fb`` and ``fix_b`` are
              equivalent to ``f1``.

            - floc : hold location parameter fixed to specified value.

            - fscale : hold scale parameter fixed to specified value.

            - optimizer : The optimizer to use.
              The optimizer must take ``func``,
              and starting position as the first two arguments,
              plus ``args`` (for extra arguments to pass to the
              function to be optimized) and ``disp=0`` to suppress
              output as keyword arguments.

            - method : The method to use. The default is "MLE" (Maximum
              Likelihood Estimate); "MM" (Method of Moments)
              is also available.


        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable),
            followed by those for location and scale.
            For most random variables, shape statistics
            will be returned, but there are exceptions (e.g. ``norm``).

        Notes
        -----
        With ``method="MLE"`` (default), the fit is computed by minimizing
        the negative log-likelihood function. A large, finite penalty
        (rather than infinite negative log-likelihood) is applied for
        observations beyond the support of the distribution.

        With ``method="MM"``, the fit is computed by minimizing the L2 norm
        of the relative errors between the first *k* raw (about zero) data
        moments and the corresponding distribution moments, where *k* is the
        number of non-fixed parameters.
        More precisely, the objective function is::

            (((data_moments - dist_moments)
              / np.maximum(np.abs(data_moments), 1e-8))**2).sum()

        where the constant ``1e-8`` avoids division by zero in case of
        vanishing data moments. Typically, this error norm can be reduced to
        zero.
        Note that the standard method of moments can produce parameters for
        which some data are outside the support of the fitted distribution;
        this implementation does nothing to prevent this.

        For either method,
        the returned answer is not guaranteed to be globally optimal; it
        may only be locally optimal, or the optimization may fail altogether.
        If the data contain any of ``np.nan``, ``np.inf``, or ``-np.inf``,
        the `fit` method will raise a ``RuntimeError``.

        Examples
        --------

        Generate some data to fit: draw random variates from the `beta`
        distribution

        >>> from scipy.stats import beta
        >>> a, b = 1., 2.
        >>> x = beta.rvs(a, b, size=1000)

        Now we can fit all four parameters (``a``, ``b``, ``loc``
        and ``scale``):

        >>> a1, b1, loc1, scale1 = beta.fit(x)

        We can also use some prior knowledge about the dataset: let's keep
        ``loc`` and ``scale`` fixed:

        >>> a1, b1, loc1, scale1 = beta.fit(x, floc=0, fscale=1)
        >>> loc1, scale1
        (0, 1)

        We can also keep shape parameters fixed by using ``f``-keywords. To
        keep the zero-th shape parameter ``a`` equal 1, use ``f0=1`` or,
        equivalently, ``fa=1``:

        >>> a1, b1, loc1, scale1 = beta.fit(x, fa=1, floc=0, fscale=1)
        >>> a1
        1

        Not all distributions return estimates for the shape parameters.
        ``norm`` for example just returns estimates for location and scale:

        >>> from scipy.stats import norm
        >>> x = norm.rvs(a, b, size=1000, random_state=123)
        >>> loc1, scale1 = norm.fit(x)
        >>> loc1, scale1
        (0.92087172783841631, 2.0015750750324668)
        """
        data = np.asarray(data)
        method = kwds.get('method', "mle").lower()

        # memory for method of moments
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        if not np.isfinite(data).all():
            raise RuntimeError("The data contains non-finite values.")

        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds, data=data)
        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        optimizer = _fit_determine_optimizer(optimizer)
        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError("Unknown arguments: %s." % kwds)

        # In some cases, method of moments can be done with fsolve/root
        # instead of an optimizer, but sometimes no solution exists,
        # especially when the user fixes parameters. Minimizing the sum
        # of squares of the error generalizes to these cases.
        vals = optimizer(func, x0, args=(ravel(data),), disp=0)
        obj = func(vals, data)

        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)

        loc, scale, shapes = self._unpack_loc_scale(vals)
        if not (np.all(self._argcheck(*shapes)) and scale > 0):
            raise Exception("Optimization converged to parameters that are "
                            "outside the range allowed by the distribution.")

        if method == 'mm':
            if not np.isfinite(obj):
                raise Exception("Optimization failed: either a data moment "
                                "or fitted distribution moment is "
                                "non-finite.")

        return vals

    def _fit_loc_scale_support(self, data, *args):
        """Estimate loc and scale parameters from data accounting for support.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        data = np.asarray(data)

        # Estimate location and scale according to the method of moments.
        loc_hat, scale_hat = self.fit_loc_scale(data, *args)

        # Compute the support according to the shape parameters.
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        a, b = _a, _b
        support_width = b - a

        # If the support is empty then return the moment-based estimates.
        if support_width <= 0:
            return loc_hat, scale_hat

        # Compute the proposed support according to the loc and scale
        # estimates.
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat

        # Use the moment-based estimates if they are compatible with the data.
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return loc_hat, scale_hat

        # Otherwise find other estimates that are compatible with the data.
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin

        # For a finite interval, both the location and scale
        # should have interesting values.
        if support_width < np.inf:
            loc_hat = (data_a - a) - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return loc_hat, scale_hat

        # For a one-sided interval, use only an interesting location parameter.
        if a > -np.inf:
            return (data_a - a) - margin, 1
        elif b < np.inf:
            return (data_b - b) + margin, 1
        else:
            raise RuntimeError

    def fit_loc_scale(self, data, *args):
        """
        Estimate loc and scale parameters from data using 1st and 2nd moments.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        mu, mu2 = self.stats(*args, **{'moments': 'mv'})
        tmp = asarray(data)
        muhat = tmp.mean()
        mu2hat = tmp.var()
        Shat = sqrt(mu2hat / mu2)
        Lhat = muhat - Shat*mu
        if not np.isfinite(Lhat):
            Lhat = 0
        if not (np.isfinite(Shat) and (0 < Shat)):
            Shat = 1
        return Lhat, Shat

    def _entropy(self, *args):
        def integ(x):
            val = self._pdf(x, *args)
            return entr(val)

        # upper limit is often inf, so suppress warnings when integrating
        _a, _b = self._get_support(*args)
        with np.errstate(over='ignore'):
            h = integrate.quad(integ, _a, _b)[0]

        if not np.isnan(h):
            return h
        else:
            # try with different limits if integration problems
            low, upp = self.ppf([1e-10, 1. - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            return integrate.quad(integ, lower, upper)[0]

    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None,
               conditional=False, **kwds):
        """Calculate expected value of a function with respect to the
        distribution by numerical integration.

        The expected value of a function ``f(x)`` with respect to a
        distribution ``dist`` is defined as::

                    ub
            E[f(x)] = Integral(f(x) * dist.pdf(x)),
                    lb

        where ``ub`` and ``lb`` are arguments and ``x`` has the ``dist.pdf(x)``
        distribution. If the bounds ``lb`` and ``ub`` correspond to the
        support of the distribution, e.g. ``[-inf, inf]`` in the default
        case, then the integral is the unrestricted expectation of ``f(x)``.
        Also, the function ``f(x)`` may be defined such that ``f(x)`` is ``0``
        outside a finite interval in which case the expectation is
        calculated within the finite range ``[lb, ub]``.

        Parameters
        ----------
        func : callable, optional
            Function for which integral is calculated. Takes only one argument.
            The default is the identity mapping f(x) = x.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter (default=0).
        scale : float, optional
            Scale parameter (default=1).
        lb, ub : scalar, optional
            Lower and upper bound for integration. Default is set to the
            support of the distribution.
        conditional : bool, optional
            If True, the integral is corrected by the conditional probability
            of the integration interval.  The return value is the expectation
            of the function, conditional on being in the given interval.
            Default is False.

        Additional keyword arguments are passed to the integration routine.

        Returns
        -------
        expect : float
            The calculated expected value.

        Notes
        -----
        The integration behavior of this function is inherited from
        `scipy.integrate.quad`. Neither this function nor
        `scipy.integrate.quad` can verify whether the integral exists or is
        finite. For example ``cauchy(0).mean()`` returns ``np.nan`` and
        ``cauchy(0).expect()`` returns ``0.0``.

        The function is not vectorized.

        Examples
        --------

        To understand the effect of the bounds of integration consider

        >>> from scipy.stats import expon
        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0)
        0.6321205588285578

        This is close to

        >>> expon(1).cdf(2.0) - expon(1).cdf(0.0)
        0.6321205588285577

        If ``conditional=True``

        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0, conditional=True)
        1.0000000000000002

        The slight deviation from 1 is due to numerical integration.
        """
        lockwds = {'loc': loc,
                   'scale': scale}
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        if func is None:
            def fun(x, *args):
                return x * self.pdf(x, *args, **lockwds)
        else:
            def fun(x, *args):
                return func(x) * self.pdf(x, *args, **lockwds)
        if lb is None:
            lb = loc + _a * scale
        if ub is None:
            ub = loc + _b * scale
        if conditional:
            invfac = (self.sf(lb, *args, **lockwds)
                      - self.sf(ub, *args, **lockwds))
        else:
            invfac = 1.0
        kwds['args'] = args
        # Silence floating point warnings from integration.
        with np.errstate(all='ignore'):
            vals = integrate.quad(fun, lb, ub, **kwds)[0] / invfac
        return vals


# Helpers for the discrete distributions
def _drv2_moment(self, n, *args):
    """Non-central moment of discrete distribution."""
    def fun(x):
        return np.power(x, n) * self._pmf(x, *args)

    _a, _b = self._get_support(*args)
    return _expect(fun, _a, _b, self.ppf(0.5, *args), self.inc)


def _drv2_ppfsingle(self, q, *args):  # Use basic bisection algorithm
    _a, _b = self._get_support(*args)
    b = _b
    a = _a
    if isinf(b):            # Be sure ending point is > q
        b = int(max(100*q, 10))
        while 1:
            if b >= _b:
                qb = 1.0
                break
            qb = self._cdf(b, *args)
            if (qb < q):
                b += 10
            else:
                break
    else:
        qb = 1.0
    if isinf(a):    # be sure starting point < q
        a = int(min(-100*q, -10))
        while 1:
            if a <= _a:
                qb = 0.0
                break
            qa = self._cdf(a, *args)
            if (qa > q):
                a -= 10
            else:
                break
    else:
        qa = self._cdf(a, *args)

    while 1:
        if (qa == q):
            return a
        if (qb == q):
            return b
        if b <= a+1:
            if qa > q:
                return a
            else:
                return b
        c = int((a+b)/2.0)
        qc = self._cdf(c, *args)
        if (qc < q):
            if a != c:
                a = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qa = qc
        elif (qc > q):
            if b != c:
                b = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qb = qc
        else:
            return c


# Must over-ride one of _pmf or _cdf or pass in
#  x_k, p(x_k) lists in initialization


class rv_discrete(rv_generic):
    """A generic discrete random variable class meant for subclassing.

    `rv_discrete` is a base class to construct specific distribution classes
    and instances for discrete random variables. It can also be used
    to construct an arbitrary distribution defined by a list of support
    points and corresponding probabilities.

    Parameters
    ----------
    a : float, optional
        Lower bound of the support of the distribution, default: 0
    b : float, optional
        Upper bound of the support of the distribution, default: plus infinity
    moment_tol : float, optional
        The tolerance for the generic calculation of moments.
    values : tuple of two array_like, optional
        ``(xk, pk)`` where ``xk`` are integers and ``pk`` are the non-zero
        probabilities between 0 and 1 with ``sum(pk) = 1``. ``xk``
        and ``pk`` must have the same shape.
    inc : integer, optional
        Increment for the support of the distribution.
        Default is 1. (other values have not been tested)
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example "m, n" for a distribution
        that takes two integers as the two shape arguments for all its methods
        If not provided, shape parameters will be inferred from
        the signatures of the private methods, ``_pmf`` and ``_cdf`` of
        the instance.
    extradoc :  str, optional
        This string is used as the last part of the docstring returned when a
        subclass has no docstring of its own. Note: `extradoc` exists for
        backwards compatibility, do not use for new subclasses.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pmf
    logpmf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    support

    Notes
    -----
    This class is similar to `rv_continuous`. Whether a shape parameter is
    valid is decided by an ``_argcheck`` method (which defaults to checking
    that its arguments are strictly positive.)
    The main differences are:

    - the support of the distribution is a set of integers
    - instead of the probability density function, ``pdf`` (and the
      corresponding private ``_pdf``), this class defines the
      *probability mass function*, `pmf` (and the corresponding
      private ``_pmf``.)
    - scale parameter is not defined.

    To create a new discrete distribution, we would do the following:

    >>> from scipy.stats import rv_discrete
    >>> class poisson_gen(rv_discrete):
    ...     "Poisson distribution"
    ...     def _pmf(self, k, mu):
    ...         return exp(-mu) * mu**k / factorial(k)

    and create an instance::

    >>> poisson = poisson_gen(name="poisson")

    Note that above we defined the Poisson distribution in the standard form.
    Shifting the distribution can be done by providing the ``loc`` parameter
    to the methods of the instance. For example, ``poisson.pmf(x, mu, loc)``
    delegates the work to ``poisson._pmf(x-loc, mu)``.

    **Discrete distributions from a list of probabilities**

    Alternatively, you can construct an arbitrary discrete rv defined
    on a finite set of values ``xk`` with ``Prob{X=xk} = pk`` by using the
    ``values`` keyword argument to the `rv_discrete` constructor.

    Examples
    --------
    Custom made discrete distribution:

    >>> from scipy import stats
    >>> xk = np.arange(7)
    >>> pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
    >>> custm = stats.rv_discrete(name='custm', values=(xk, pk))
    >>>
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
    >>> ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
    >>> plt.show()

    Random number generation:

    >>> R = custm.rvs(size=100)

    """
    def __new__(cls, a=0, b=inf, name=None, badvalue=None,
                moment_tol=1e-8, values=None, inc=1, longname=None,
                shapes=None, extradoc=None, seed=None):

        if values is not None:
            # dispatch to a subclass
            return super(rv_discrete, cls).__new__(rv_sample)
        else:
            # business as usual
            return super(rv_discrete, cls).__new__(cls)

    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, extradoc=None, seed=None):

        super().__init__(seed)

        # cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.a = a
        self.b = b
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes

        if values is not None:
            raise ValueError("rv_discrete.__init__(..., values != None, ...)")

        self._construct_argparser(meths_to_inspect=[self._pmf, self._cdf],
                                  locscale_in='loc=0',
                                  # scale=1 for discrete RVs
                                  locscale_out='loc, 1')
        self._attach_methods()
        self._construct_docstrings(name, longname, extradoc)

    def __getstate__(self):
        dct = self.__dict__.copy()
        # these methods will be remade in __setstate__
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs",
                 "_cdfvec", "_ppfvec", "generic_moment"]
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_discrete instance."""
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self.vecentropy = vectorize(self._entropy)

        # _attach_methods is responsible for calling _attach_argparser_methods
        self._attach_argparser_methods()

        # nin correction needs to be after we know numargs
        # correct nin for generic moment vectorization
        _vec_generic_moment = vectorize(_drv2_moment, otypes='d')
        _vec_generic_moment.nin = self.numargs + 2
        self.generic_moment = types.MethodType(_vec_generic_moment, self)

        # correct nin for ppf vectorization
        _vppf = vectorize(_drv2_ppfsingle, otypes='d')
        _vppf.nin = self.numargs + 2
        self._ppfvec = types.MethodType(_vppf, self)

        # now that self.numargs is defined, we can adjust nin
        self._cdfvec.nin = self.numargs + 1

    def _construct_docstrings(self, name, longname, extradoc):
        if name is None:
            name = 'Distribution'
        self.name = name
        self.extradoc = extradoc

        # generate docstring for subclass instances
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name

        if sys.flags.optimize < 2:
            # Skip adding docstrings if interpreter is run with -OO
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname,
                                            extradoc=extradoc,
                                            docdict=docdict_discrete,
                                            discrete='discrete')
            else:
                dct = dict(distdiscrete)
                self._construct_doc(docdict_discrete, dct.get(self.name))

            # discrete RV do not have the scale parameter, remove it
            self.__doc__ = self.__doc__.replace(
                '\n    scale : array_like, '
                'optional\n        scale parameter (default=1)', '')

    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['badvalue'] = self.badvalue
        dct['moment_tol'] = self.moment_tol
        dct['inc'] = self.inc
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        dct['extradoc'] = self.extradoc
        return dct

    def _nonzero(self, k, *args):
        return floor(k) == k

    def _pmf(self, k, *args):
        return self._cdf(k, *args) - self._cdf(k-1, *args)

    def _logpmf(self, k, *args):
        return log(self._pmf(k, *args))

    def _cdf_single(self, k, *args):
        _a, _b = self._get_support(*args)
        m = arange(int(_a), k+1)
        return np.sum(self._pmf(m, *args), axis=0)

    def _cdf(self, x, *args):
        k = floor(x)
        return self._cdfvec(k, *args)

    # generic _logcdf, _sf, _logsf, _ppf, _isf, _rvs defined in rv_generic

    def rvs(self, *args, **kwargs):
        """Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        size : int or tuple of ints, optional
            Defining number of random variates (Default is 1). Note that `size`
            has to be given as keyword, not as positional argument.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        kwargs['discrete'] = True
        return super().rvs(*args, **kwargs)

    def pmf(self, k, *args, **kwds):
        """Probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        pmf : array_like
            Probability mass function evaluated at k

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b) & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        """Log of the probability mass function at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter. Default is 0.

        Returns
        -------
        logpmf : array_like
            Log of the probability mass function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b) & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(NINF)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, k, *args, **kwds):
        """Cumulative distribution function of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k >= _b)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, cond2*(cond0 == cond0), 1.0)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, k, *args, **kwds):
        """Log of the cumulative distribution function at k of the given RV.

        Parameters
        ----------
        k : array_like, int
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray((k-loc))
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k >= _b)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(NINF)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2*(cond0 == cond0), 0.0)

        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, k, *args, **kwds):
        """Survival function (1 - `cdf`) at k of the given RV.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        sf : array_like
            Survival function evaluated at k.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, k, *args, **kwds):
        """Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as 1 - `cdf`,
        evaluated at `k`.

        Parameters
        ----------
        k : array_like
            Quantiles.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `k`.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        k, loc = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        k = asarray(k-loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(NINF)
        place(output, (1-cond0) + np.isnan(k), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((k,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Lower tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : array_like
            Quantile corresponding to the lower tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        # output type 'd' to handle nin and inf
        place(output, (q == 0)*(cond == cond), _a-1 + loc)
        place(output, cond2, _b + loc)
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._ppf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            Upper tail probability.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).

        Returns
        -------
        k : ndarray or scalar
            Quantile corresponding to the upper tail probability, q.

        """
        args, loc, _ = self._parse_args(*args, **kwds)
        q, loc = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1

        # same problem as with ppf; copied from ppf and changed
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        # output type 'd' to handle nin and inf
        place(output, (q == 0)*(cond == cond), _b)
        place(output, cond2, _a-1)

        # call place only if at least 1 valid argument
        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(loc,)))
            loc, goodargs = goodargs[-1], goodargs[:-1]
            # PB same as ticket 766
            place(output, cond, self._isf(*goodargs) + loc)

        if output.ndim == 0:
            return output[()]
        return output

    def _entropy(self, *args):
        if hasattr(self, 'pk'):
            return stats.entropy(self.pk)
        else:
            _a, _b = self._get_support(*args)
            return _expect(lambda x: entr(self.pmf(x, *args)),
                           _a, _b, self.ppf(0.5, *args), self.inc)

    def expect(self, func=None, args=(), loc=0, lb=None, ub=None,
               conditional=False, maxcount=1000, tolerance=1e-10, chunksize=32):
        """
        Calculate expected value of a function with respect to the distribution
        for discrete distribution by numerical summation.

        Parameters
        ----------
        func : callable, optional
            Function for which the expectation value is calculated.
            Takes only one argument.
            The default is the identity mapping f(k) = k.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter.
            Default is 0.
        lb, ub : int, optional
            Lower and upper bound for the summation, default is set to the
            support of the distribution, inclusive (``lb <= k <= ub``).
        conditional : bool, optional
            If true then the expectation is corrected by the conditional
            probability of the summation interval. The return value is the
            expectation of the function, `func`, conditional on being in
            the given interval (k such that ``lb <= k <= ub``).
            Default is False.
        maxcount : int, optional
            Maximal number of terms to evaluate (to avoid an endless loop for
            an infinite sum). Default is 1000.
        tolerance : float, optional
            Absolute tolerance for the summation. Default is 1e-10.
        chunksize : int, optional
            Iterate over the support of a distributions in chunks of this size.
            Default is 32.

        Returns
        -------
        expect : float
            Expected value.

        Notes
        -----
        For heavy-tailed distributions, the expected value may or
        may not exist,
        depending on the function, `func`. If it does exist, but the
        sum converges
        slowly, the accuracy of the result may be rather low. For instance, for
        ``zipf(4)``, accuracy for mean, variance in example is only 1e-5.
        increasing `maxcount` and/or `chunksize` may improve the result,
        but may also make zipf very slow.

        The function is not vectorized.

        """
        if func is None:
            def fun(x):
                # loc and args from outer scope
                return (x+loc)*self._pmf(x, *args)
        else:
            def fun(x):
                # loc and args from outer scope
                return func(x+loc)*self._pmf(x, *args)
        # used pmf because _pmf does not check support in randint and there
        # might be problems(?) with correct self.a, self.b at this stage maybe
        # not anymore, seems to work now with _pmf

        _a, _b = self._get_support(*args)
        if lb is None:
            lb = _a
        else:
            lb = lb - loc   # convert bound for standardized distribution
        if ub is None:
            ub = _b
        else:
            ub = ub - loc   # convert bound for standardized distribution
        if conditional:
            invfac = self.sf(lb-1, *args) - self.sf(ub, *args)
        else:
            invfac = 1.0

        if isinstance(self, rv_sample):
            res = self._expect(fun, lb, ub)
            return res / invfac

        # iterate over the support, starting from the median
        x0 = self.ppf(0.5, *args)
        res = _expect(fun, lb, ub, x0, self.inc, maxcount, tolerance, chunksize)
        return res / invfac


def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10,
            chunksize=32):
    """Helper for computing the expectation value of `fun`."""
    # short-circuit if the support size is small enough
    if (ub - lb) <= chunksize:
        supp = np.arange(lb, ub+1, inc)
        vals = fun(supp)
        return np.sum(vals)

    # otherwise, iterate starting from x0
    if x0 < lb:
        x0 = lb
    if x0 > ub:
        x0 = ub

    count, tot = 0, 0.
    # iterate over [x0, ub] inclusive
    for x in _iter_chunked(x0, ub+1, chunksize=chunksize, inc=inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge', RuntimeWarning)
            return tot

    # iterate over [lb, x0)
    for x in _iter_chunked(x0-1, lb-1, chunksize=chunksize, inc=-inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge', RuntimeWarning)
            break

    return tot


def _iter_chunked(x0, x1, chunksize=4, inc=1):
    """Iterate from x0 to x1 in chunks of chunksize and steps inc.

    x0 must be finite, x1 need not be. In the latter case, the iterator is
    infinite.
    Handles both x0 < x1 and x0 > x1. In the latter case, iterates downwards
    (make sure to set inc < 0.)

    >>> [x for x in _iter_chunked(2, 5, inc=2)]
    [array([2, 4])]
    >>> [x for x in _iter_chunked(2, 11, inc=2)]
    [array([2, 4, 6, 8]), array([10])]
    >>> [x for x in _iter_chunked(2, -5, inc=-2)]
    [array([ 2,  0, -2, -4])]
    >>> [x for x in _iter_chunked(2, -9, inc=-2)]
    [array([ 2,  0, -2, -4]), array([-6, -8])]

    """
    if inc == 0:
        raise ValueError('Cannot increment by zero.')
    if chunksize <= 0:
        raise ValueError('Chunk size must be positive; got %s.' % chunksize)

    s = 1 if inc > 0 else -1
    stepsize = abs(chunksize * inc)

    x = x0
    while (x - x1) * inc < 0:
        delta = min(stepsize, abs(x - x1))
        step = delta * s
        supp = np.arange(x, x + step, inc)
        x += step
        yield supp


class rv_sample(rv_discrete):
    """A 'sample' discrete distribution defined by the support and values.

    The ctor ignores most of the arguments, only needs the `values` argument.
    """
    def __init__(self, a=0, b=inf, name=None, badvalue=None,
                 moment_tol=1e-8, values=None, inc=1, longname=None,
                 shapes=None, extradoc=None, seed=None):

        super(rv_discrete, self).__init__(seed)

        if values is None:
            raise ValueError("rv_sample.__init__(..., values=None,...)")

        # cf generic freeze
        self._ctor_param = dict(
            a=a, b=b, name=name, badvalue=badvalue,
            moment_tol=moment_tol, values=values, inc=inc,
            longname=longname, shapes=shapes, extradoc=extradoc, seed=seed)

        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        self.vecentropy = self._entropy

        xk, pk = values

        if np.shape(xk) != np.shape(pk):
            raise ValueError("xk and pk must have the same shape.")
        if np.less(pk, 0.0).any():
            raise ValueError("All elements of pk must be non-negative.")
        if not np.allclose(np.sum(pk), 1):
            raise ValueError("The sum of provided pk is not 1.")

        indx = np.argsort(np.ravel(xk))
        self.xk = np.take(np.ravel(xk), indx, 0)
        self.pk = np.take(np.ravel(pk), indx, 0)
        self.a = self.xk[0]
        self.b = self.xk[-1]

        self.qvals = np.cumsum(self.pk, axis=0)

        self.shapes = ' '   # bypass inspection

        self._construct_argparser(meths_to_inspect=[self._pmf],
                                  locscale_in='loc=0',
                                  # scale=1 for discrete RVs
                                  locscale_out='loc, 1')

        self._attach_methods()

        self._construct_docstrings(name, longname, extradoc)

    def __getstate__(self):
        dct = self.__dict__.copy()

        # these methods will be remade in rv_generic.__setstate__,
        # which calls rv_generic._attach_methods
        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs"]
        [dct.pop(attr, None) for attr in attrs]

        return dct

    def _attach_methods(self):
        """Attaches dynamically created argparser methods."""
        self._attach_argparser_methods()

    def _get_support(self, *args):
        """Return the support of the (unscaled, unshifted) distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support.
        """
        return self.a, self.b

    def _pmf(self, x):
        return np.select([x == k for k in self.xk],
                         [np.broadcast_arrays(p, x)[0] for p in self.pk], 0)

    def _cdf(self, x):
        xx, xxk = np.broadcast_arrays(x[:, None], self.xk)
        indx = np.argmax(xxk > xx, axis=-1) - 1
        return self.qvals[indx]

    def _ppf(self, q):
        qq, sqq = np.broadcast_arrays(q[..., None], self.qvals)
        indx = argmax(sqq >= qq, axis=-1)
        return self.xk[indx]

    def _rvs(self, size=None, random_state=None):
        # Need to define it explicitly, otherwise .rvs() with size=None
        # fails due to explicit broadcasting in _ppf
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self._ppf(U)[0]
        else:
            Y = self._ppf(U)
        return Y

    def _entropy(self):
        return stats.entropy(self.pk)

    def generic_moment(self, n):
        n = asarray(n)
        return np.sum(self.xk**n[np.newaxis, ...] * self.pk, axis=0)

    def _expect(self, fun, lb, ub, *args, **kwds):
        # ignore all args, just do a brute force summation
        supp = self.xk[(lb <= self.xk) & (self.xk <= ub)]
        vals = fun(supp)
        return np.sum(vals)


def _check_shape(argshape, size):
    """
    This is a utility function used by `_rvs()` in the class geninvgauss_gen.
    It compares the tuple argshape to the tuple size.

    Parameters
    ----------
    argshape : tuple of integers
        Shape of the arguments.
    size : tuple of integers or integer
        Size argument of rvs().

    Returns
    -------
    The function returns two tuples, scalar_shape and bc.

    scalar_shape : tuple
        Shape to which the 1-d array of random variates returned by
        _rvs_scalar() is converted when it is copied into the
        output array of _rvs().

    bc : tuple of booleans
        bc is an tuple the same length as size. bc[j] is True if the data
        associated with that index is generated in one call of _rvs_scalar().

    """
    scalar_shape = []
    bc = []
    for argdim, sizedim in zip_longest(argshape[::-1], size[::-1],
                                       fillvalue=1):
        if sizedim > argdim or (argdim == sizedim == 1):
            scalar_shape.append(sizedim)
            bc.append(True)
        else:
            bc.append(False)
    return tuple(scalar_shape[::-1]), tuple(bc[::-1])


def get_distribution_names(namespace_pairs, rv_base_class):
    """Collect names of statistical distributions and their generators.

    Parameters
    ----------
    namespace_pairs : sequence
        A snapshot of (name, value) pairs in the namespace of a module.
    rv_base_class : class
        The base class of random variable generator classes in a module.

    Returns
    -------
    distn_names : list of strings
        Names of the statistical distributions.
    distn_gen_names : list of strings
        Names of the generators of the statistical distributions.
        Note that these are not simply the names of the statistical
        distributions, with a _gen suffix added.

    """
    distn_names = []
    distn_gen_names = []
    for name, value in namespace_pairs:
        if name.startswith('_'):
            continue
        if name.endswith('_gen') and issubclass(value, rv_base_class):
            distn_gen_names.append(name)
        if isinstance(value, rv_base_class):
            distn_names.append(name)
    return distn_names, distn_gen_names
