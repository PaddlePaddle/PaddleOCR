"""
Python wrappers for Orthogonal Distance Regression (ODRPACK).

Notes
=====

* Array formats -- FORTRAN stores its arrays in memory column first, i.e., an
  array element A(i, j, k) will be next to A(i+1, j, k). In C and, consequently,
  NumPy, arrays are stored row first: A[i, j, k] is next to A[i, j, k+1]. For
  efficiency and convenience, the input and output arrays of the fitting
  function (and its Jacobians) are passed to FORTRAN without transposition.
  Therefore, where the ODRPACK documentation says that the X array is of shape
  (N, M), it will be passed to the Python function as an array of shape (M, N).
  If M==1, the 1-D case, then nothing matters; if M>1, then your
  Python functions will be dealing with arrays that are indexed in reverse of
  the ODRPACK documentation. No real issue, but watch out for your indexing of
  the Jacobians: the i,jth elements (@f_i/@x_j) evaluated at the nth
  observation will be returned as jacd[j, i, n]. Except for the Jacobians, it
  really is easier to deal with x[0] and x[1] than x[:,0] and x[:,1]. Of course,
  you can always use the transpose() function from SciPy explicitly.

* Examples -- See the accompanying file test/test.py for examples of how to set
  up fits of your own. Some are taken from the User's Guide; some are from
  other sources.

* Models -- Some common models are instantiated in the accompanying module
  models.py . Contributions are welcome.

Credits
=======

* Thanks to Arnold Moene and Gerard Vermeulen for fixing some killer bugs.

Robert Kern
robert.kern@gmail.com

"""
import os

import numpy
from warnings import warn
from scipy.odr import __odrpack

__all__ = ['odr', 'OdrWarning', 'OdrError', 'OdrStop',
           'Data', 'RealData', 'Model', 'Output', 'ODR',
           'odr_error', 'odr_stop']

odr = __odrpack.odr


class OdrWarning(UserWarning):
    """
    Warning indicating that the data passed into
    ODR will cause problems when passed into 'odr'
    that the user should be aware of.
    """
    pass


class OdrError(Exception):
    """
    Exception indicating an error in fitting.

    This is raised by `~scipy.odr.odr` if an error occurs during fitting.
    """
    pass


class OdrStop(Exception):
    """
    Exception stopping fitting.

    You can raise this exception in your objective function to tell
    `~scipy.odr.odr` to stop fitting.
    """
    pass


# Backwards compatibility
odr_error = OdrError
odr_stop = OdrStop

__odrpack._set_exceptions(OdrError, OdrStop)


def _conv(obj, dtype=None):
    """ Convert an object to the preferred form for input to the odr routine.
    """

    if obj is None:
        return obj
    else:
        if dtype is None:
            obj = numpy.asarray(obj)
        else:
            obj = numpy.asarray(obj, dtype)
        if obj.shape == ():
            # Scalar.
            return obj.dtype.type(obj)
        else:
            return obj


def _report_error(info):
    """ Interprets the return code of the odr routine.

    Parameters
    ----------
    info : int
        The return code of the odr routine.

    Returns
    -------
    problems : list(str)
        A list of messages about why the odr() routine stopped.
    """

    stopreason = ('Blank',
                  'Sum of squares convergence',
                  'Parameter convergence',
                  'Both sum of squares and parameter convergence',
                  'Iteration limit reached')[info % 5]

    if info >= 5:
        # questionable results or fatal error

        I = (info//10000 % 10,
             info//1000 % 10,
             info//100 % 10,
             info//10 % 10,
             info % 10)
        problems = []

        if I[0] == 0:
            if I[1] != 0:
                problems.append('Derivatives possibly not correct')
            if I[2] != 0:
                problems.append('Error occurred in callback')
            if I[3] != 0:
                problems.append('Problem is not full rank at solution')
            problems.append(stopreason)
        elif I[0] == 1:
            if I[1] != 0:
                problems.append('N < 1')
            if I[2] != 0:
                problems.append('M < 1')
            if I[3] != 0:
                problems.append('NP < 1 or NP > N')
            if I[4] != 0:
                problems.append('NQ < 1')
        elif I[0] == 2:
            if I[1] != 0:
                problems.append('LDY and/or LDX incorrect')
            if I[2] != 0:
                problems.append('LDWE, LD2WE, LDWD, and/or LD2WD incorrect')
            if I[3] != 0:
                problems.append('LDIFX, LDSTPD, and/or LDSCLD incorrect')
            if I[4] != 0:
                problems.append('LWORK and/or LIWORK too small')
        elif I[0] == 3:
            if I[1] != 0:
                problems.append('STPB and/or STPD incorrect')
            if I[2] != 0:
                problems.append('SCLB and/or SCLD incorrect')
            if I[3] != 0:
                problems.append('WE incorrect')
            if I[4] != 0:
                problems.append('WD incorrect')
        elif I[0] == 4:
            problems.append('Error in derivatives')
        elif I[0] == 5:
            problems.append('Error occurred in callback')
        elif I[0] == 6:
            problems.append('Numerical error detected')

        return problems

    else:
        return [stopreason]


class Data:
    """
    The data to fit.

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like, optional
        If array-like, observed data for the dependent variable of the
        regression. A scalar input implies that the model to be used on
        the data is implicit.
    we : array_like, optional
        If `we` is a scalar, then that value is used for all data points (and
        all dimensions of the response variable).
        If `we` is a rank-1 array of length q (the dimensionality of the
        response variable), then this vector is the diagonal of the covariant
        weighting matrix for all data points.
        If `we` is a rank-1 array of length n (the number of data points), then
        the i'th element is the weight for the i'th response variable
        observation (single-dimensional only).
        If `we` is a rank-2 array of shape (q, q), then this is the full
        covariant weighting matrix broadcast to each observation.
        If `we` is a rank-2 array of shape (q, n), then `we[:,i]` is the
        diagonal of the covariant weighting matrix for the i'th observation.
        If `we` is a rank-3 array of shape (q, q, n), then `we[:,:,i]` is the
        full specification of the covariant weighting matrix for each
        observation.
        If the fit is implicit, then only a positive scalar value is used.
    wd : array_like, optional
        If `wd` is a scalar, then that value is used for all data points
        (and all dimensions of the input variable). If `wd` = 0, then the
        covariant weighting matrix for each observation is set to the identity
        matrix (so each dimension of each observation has the same weight).
        If `wd` is a rank-1 array of length m (the dimensionality of the input
        variable), then this vector is the diagonal of the covariant weighting
        matrix for all data points.
        If `wd` is a rank-1 array of length n (the number of data points), then
        the i'th element is the weight for the ith input variable observation
        (single-dimensional only).
        If `wd` is a rank-2 array of shape (m, m), then this is the full
        covariant weighting matrix broadcast to each observation.
        If `wd` is a rank-2 array of shape (m, n), then `wd[:,i]` is the
        diagonal of the covariant weighting matrix for the ith observation.
        If `wd` is a rank-3 array of shape (m, m, n), then `wd[:,:,i]` is the
        full specification of the covariant weighting matrix for each
        observation.
    fix : array_like of ints, optional
        The `fix` argument is the same as ifixx in the class ODR. It is an
        array of integers with the same shape as data.x that determines which
        input observations are treated as fixed. One can use a sequence of
        length m (the dimensionality of the input observations) to fix some
        dimensions for all observations. A value of 0 fixes the observation,
        a value > 0 makes it free.
    meta : dict, optional
        Free-form dictionary for metadata.

    Notes
    -----
    Each argument is attached to the member of the instance of the same name.
    The structures of `x` and `y` are described in the Model class docstring.
    If `y` is an integer, then the Data instance can only be used to fit with
    implicit models where the dimensionality of the response is equal to the
    specified value of `y`.

    The `we` argument weights the effect a deviation in the response variable
    has on the fit. The `wd` argument weights the effect a deviation in the
    input variable has on the fit. To handle multidimensional inputs and
    responses easily, the structure of these arguments has the n'th
    dimensional axis first. These arguments heavily use the structured
    arguments feature of ODRPACK to conveniently and flexibly support all
    options. See the ODRPACK User's Guide for a full explanation of how these
    weights are used in the algorithm. Basically, a higher value of the weight
    for a particular data point makes a deviation at that point more
    detrimental to the fit.

    """

    def __init__(self, x, y=None, we=None, wd=None, fix=None, meta={}):
        self.x = _conv(x)

        if not isinstance(self.x, numpy.ndarray):
            raise ValueError(("Expected an 'ndarray' of data for 'x', "
                              "but instead got data of type '{name}'").format(
                    name=type(self.x).__name__))

        self.y = _conv(y)
        self.we = _conv(we)
        self.wd = _conv(wd)
        self.fix = _conv(fix)
        self.meta = meta

    def set_meta(self, **kwds):
        """ Update the metadata dictionary with the keywords and data provided
        by keywords.

        Examples
        --------
        ::

            data.set_meta(lab="Ph 7; Lab 26", title="Ag110 + Ag108 Decay")
        """

        self.meta.update(kwds)

    def __getattr__(self, attr):
        """ Dispatch attribute access to the metadata dictionary.
        """
        if attr in self.meta:
            return self.meta[attr]
        else:
            raise AttributeError("'%s' not in metadata" % attr)


class RealData(Data):
    """
    The data, with weightings as actual standard deviations and/or
    covariances.

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like, optional
        If array-like, observed data for the dependent variable of the
        regression. A scalar input implies that the model to be used on
        the data is implicit.
    sx : array_like, optional
        Standard deviations of `x`.
        `sx` are standard deviations of `x` and are converted to weights by
        dividing 1.0 by their squares.
    sy : array_like, optional
        Standard deviations of `y`.
        `sy` are standard deviations of `y` and are converted to weights by
        dividing 1.0 by their squares.
    covx : array_like, optional
        Covariance of `x`
        `covx` is an array of covariance matrices of `x` and are converted to
        weights by performing a matrix inversion on each observation's
        covariance matrix.
    covy : array_like, optional
        Covariance of `y`
        `covy` is an array of covariance matrices and are converted to
        weights by performing a matrix inversion on each observation's
        covariance matrix.
    fix : array_like, optional
        The argument and member fix is the same as Data.fix and ODR.ifixx:
        It is an array of integers with the same shape as `x` that
        determines which input observations are treated as fixed. One can
        use a sequence of length m (the dimensionality of the input
        observations) to fix some dimensions for all observations. A value
        of 0 fixes the observation, a value > 0 makes it free.
    meta : dict, optional
        Free-form dictionary for metadata.

    Notes
    -----
    The weights `wd` and `we` are computed from provided values as follows:

    `sx` and `sy` are converted to weights by dividing 1.0 by their squares.
    For example, ``wd = 1./numpy.power(`sx`, 2)``.

    `covx` and `covy` are arrays of covariance matrices and are converted to
    weights by performing a matrix inversion on each observation's covariance
    matrix. For example, ``we[i] = numpy.linalg.inv(covy[i])``.

    These arguments follow the same structured argument conventions as wd and
    we only restricted by their natures: `sx` and `sy` can't be rank-3, but
    `covx` and `covy` can be.

    Only set *either* `sx` or `covx` (not both). Setting both will raise an
    exception. Same with `sy` and `covy`.

    """

    def __init__(self, x, y=None, sx=None, sy=None, covx=None, covy=None,
                 fix=None, meta={}):
        if (sx is not None) and (covx is not None):
            raise ValueError("cannot set both sx and covx")
        if (sy is not None) and (covy is not None):
            raise ValueError("cannot set both sy and covy")

        # Set flags for __getattr__
        self._ga_flags = {}
        if sx is not None:
            self._ga_flags['wd'] = 'sx'
        else:
            self._ga_flags['wd'] = 'covx'
        if sy is not None:
            self._ga_flags['we'] = 'sy'
        else:
            self._ga_flags['we'] = 'covy'

        self.x = _conv(x)

        if not isinstance(self.x, numpy.ndarray):
            raise ValueError(("Expected an 'ndarray' of data for 'x', "
                              "but instead got data of type '{name}'").format(
                    name=type(self.x).__name__))

        self.y = _conv(y)
        self.sx = _conv(sx)
        self.sy = _conv(sy)
        self.covx = _conv(covx)
        self.covy = _conv(covy)
        self.fix = _conv(fix)
        self.meta = meta

    def _sd2wt(self, sd):
        """ Convert standard deviation to weights.
        """

        return 1./numpy.power(sd, 2)

    def _cov2wt(self, cov):
        """ Convert covariance matrix(-ices) to weights.
        """

        from scipy.linalg import inv

        if len(cov.shape) == 2:
            return inv(cov)
        else:
            weights = numpy.zeros(cov.shape, float)

            for i in range(cov.shape[-1]):  # n
                weights[:,:,i] = inv(cov[:,:,i])

            return weights

    def __getattr__(self, attr):
        lookup_tbl = {('wd', 'sx'): (self._sd2wt, self.sx),
                      ('wd', 'covx'): (self._cov2wt, self.covx),
                      ('we', 'sy'): (self._sd2wt, self.sy),
                      ('we', 'covy'): (self._cov2wt, self.covy)}

        if attr not in ('wd', 'we'):
            if attr in self.meta:
                return self.meta[attr]
            else:
                raise AttributeError("'%s' not in metadata" % attr)
        else:
            func, arg = lookup_tbl[(attr, self._ga_flags[attr])]

            if arg is not None:
                return func(*(arg,))
            else:
                return None


class Model:
    """
    The Model class stores information about the function you wish to fit.

    It stores the function itself, at the least, and optionally stores
    functions which compute the Jacobians used during fitting. Also, one
    can provide a function that will provide reasonable starting values
    for the fit parameters possibly given the set of data.

    Parameters
    ----------
    fcn : function
          fcn(beta, x) --> y
    fjacb : function
          Jacobian of fcn wrt the fit parameters beta.

          fjacb(beta, x) --> @f_i(x,B)/@B_j
    fjacd : function
          Jacobian of fcn wrt the (possibly multidimensional) input
          variable.

          fjacd(beta, x) --> @f_i(x,B)/@x_j
    extra_args : tuple, optional
          If specified, `extra_args` should be a tuple of extra
          arguments to pass to `fcn`, `fjacb`, and `fjacd`. Each will be called
          by `apply(fcn, (beta, x) + extra_args)`
    estimate : array_like of rank-1
          Provides estimates of the fit parameters from the data

          estimate(data) --> estbeta
    implicit : boolean
          If TRUE, specifies that the model
          is implicit; i.e `fcn(beta, x)` ~= 0 and there is no y data to fit
          against
    meta : dict, optional
          freeform dictionary of metadata for the model

    Notes
    -----
    Note that the `fcn`, `fjacb`, and `fjacd` operate on NumPy arrays and
    return a NumPy array. The `estimate` object takes an instance of the
    Data class.

    Here are the rules for the shapes of the argument and return
    arrays of the callback functions:

    `x`
        if the input data is single-dimensional, then `x` is rank-1
        array; i.e., ``x = array([1, 2, 3, ...]); x.shape = (n,)``
        If the input data is multi-dimensional, then `x` is a rank-2 array;
        i.e., ``x = array([[1, 2, ...], [2, 4, ...]]); x.shape = (m, n)``.
        In all cases, it has the same shape as the input data array passed to
        `~scipy.odr.odr`. `m` is the dimensionality of the input data,
        `n` is the number of observations.
    `y`
        if the response variable is single-dimensional, then `y` is a
        rank-1 array, i.e., ``y = array([2, 4, ...]); y.shape = (n,)``.
        If the response variable is multi-dimensional, then `y` is a rank-2
        array, i.e., ``y = array([[2, 4, ...], [3, 6, ...]]); y.shape =
        (q, n)`` where `q` is the dimensionality of the response variable.
    `beta`
        rank-1 array of length `p` where `p` is the number of parameters;
        i.e. ``beta = array([B_1, B_2, ..., B_p])``
    `fjacb`
        if the response variable is multi-dimensional, then the
        return array's shape is `(q, p, n)` such that ``fjacb(x,beta)[l,k,i] =
        d f_l(X,B)/d B_k`` evaluated at the ith data point.  If `q == 1`, then
        the return array is only rank-2 and with shape `(p, n)`.
    `fjacd`
        as with fjacb, only the return array's shape is `(q, m, n)`
        such that ``fjacd(x,beta)[l,j,i] = d f_l(X,B)/d X_j`` at the ith data
        point.  If `q == 1`, then the return array's shape is `(m, n)`. If
        `m == 1`, the shape is (q, n). If `m == q == 1`, the shape is `(n,)`.

    """

    def __init__(self, fcn, fjacb=None, fjacd=None,
        extra_args=None, estimate=None, implicit=0, meta=None):

        self.fcn = fcn
        self.fjacb = fjacb
        self.fjacd = fjacd

        if extra_args is not None:
            extra_args = tuple(extra_args)

        self.extra_args = extra_args
        self.estimate = estimate
        self.implicit = implicit
        self.meta = meta

    def set_meta(self, **kwds):
        """ Update the metadata dictionary with the keywords and data provided
        here.

        Examples
        --------
        set_meta(name="Exponential", equation="y = a exp(b x) + c")
        """

        self.meta.update(kwds)

    def __getattr__(self, attr):
        """ Dispatch attribute access to the metadata.
        """

        if attr in self.meta:
            return self.meta[attr]
        else:
            raise AttributeError("'%s' not in metadata" % attr)


class Output:
    """
    The Output class stores the output of an ODR run.

    Attributes
    ----------
    beta : ndarray
        Estimated parameter values, of shape (q,).
    sd_beta : ndarray
        Standard deviations of the estimated parameters, of shape (p,).
    cov_beta : ndarray
        Covariance matrix of the estimated parameters, of shape (p,p).
    delta : ndarray, optional
        Array of estimated errors in input variables, of same shape as `x`.
    eps : ndarray, optional
        Array of estimated errors in response variables, of same shape as `y`.
    xplus : ndarray, optional
        Array of ``x + delta``.
    y : ndarray, optional
        Array ``y = fcn(x + delta)``.
    res_var : float, optional
        Residual variance.
    sum_square : float, optional
        Sum of squares error.
    sum_square_delta : float, optional
        Sum of squares of delta error.
    sum_square_eps : float, optional
        Sum of squares of eps error.
    inv_condnum : float, optional
        Inverse condition number (cf. ODRPACK UG p. 77).
    rel_error : float, optional
        Relative error in function values computed within fcn.
    work : ndarray, optional
        Final work array.
    work_ind : dict, optional
        Indices into work for drawing out values (cf. ODRPACK UG p. 83).
    info : int, optional
        Reason for returning, as output by ODRPACK (cf. ODRPACK UG p. 38).
    stopreason : list of str, optional
        `info` interpreted into English.

    Notes
    -----
    Takes one argument for initialization, the return value from the
    function `~scipy.odr.odr`. The attributes listed as "optional" above are
    only present if `~scipy.odr.odr` was run with ``full_output=1``.

    """

    def __init__(self, output):
        self.beta = output[0]
        self.sd_beta = output[1]
        self.cov_beta = output[2]

        if len(output) == 4:
            # full output
            self.__dict__.update(output[3])
            self.stopreason = _report_error(self.info)

    def pprint(self):
        """ Pretty-print important results.
        """

        print('Beta:', self.beta)
        print('Beta Std Error:', self.sd_beta)
        print('Beta Covariance:', self.cov_beta)
        if hasattr(self, 'info'):
            print('Residual Variance:',self.res_var)
            print('Inverse Condition #:', self.inv_condnum)
            print('Reason(s) for Halting:')
            for r in self.stopreason:
                print('  %s' % r)


class ODR:
    """
    The ODR class gathers all information and coordinates the running of the
    main fitting routine.

    Members of instances of the ODR class have the same names as the arguments
    to the initialization routine.

    Parameters
    ----------
    data : Data class instance
        instance of the Data class
    model : Model class instance
        instance of the Model class

    Other Parameters
    ----------------
    beta0 : array_like of rank-1
        a rank-1 sequence of initial parameter values. Optional if
        model provides an "estimate" function to estimate these values.
    delta0 : array_like of floats of rank-1, optional
        a (double-precision) float array to hold the initial values of
        the errors in the input variables. Must be same shape as data.x
    ifixb : array_like of ints of rank-1, optional
        sequence of integers with the same length as beta0 that determines
        which parameters are held fixed. A value of 0 fixes the parameter,
        a value > 0 makes the parameter free.
    ifixx : array_like of ints with same shape as data.x, optional
        an array of integers with the same shape as data.x that determines
        which input observations are treated as fixed. One can use a sequence
        of length m (the dimensionality of the input observations) to fix some
        dimensions for all observations. A value of 0 fixes the observation,
        a value > 0 makes it free.
    job : int, optional
        an integer telling ODRPACK what tasks to perform. See p. 31 of the
        ODRPACK User's Guide if you absolutely must set the value here. Use the
        method set_job post-initialization for a more readable interface.
    iprint : int, optional
        an integer telling ODRPACK what to print. See pp. 33-34 of the
        ODRPACK User's Guide if you absolutely must set the value here. Use the
        method set_iprint post-initialization for a more readable interface.
    errfile : str, optional
        string with the filename to print ODRPACK errors to. If the file already
        exists, an error will be thrown. The `overwrite` argument can be used to
        prevent this. *Do Not Open This File Yourself!*
    rptfile : str, optional
        string with the filename to print ODRPACK summaries to. If the file
        already exists, an error will be thrown. The `overwrite` argument can be
        used to prevent this. *Do Not Open This File Yourself!*
    ndigit : int, optional
        integer specifying the number of reliable digits in the computation
        of the function.
    taufac : float, optional
        float specifying the initial trust region. The default value is 1.
        The initial trust region is equal to taufac times the length of the
        first computed Gauss-Newton step. taufac must be less than 1.
    sstol : float, optional
        float specifying the tolerance for convergence based on the relative
        change in the sum-of-squares. The default value is eps**(1/2) where eps
        is the smallest value such that 1 + eps > 1 for double precision
        computation on the machine. sstol must be less than 1.
    partol : float, optional
        float specifying the tolerance for convergence based on the relative
        change in the estimated parameters. The default value is eps**(2/3) for
        explicit models and ``eps**(1/3)`` for implicit models. partol must be less
        than 1.
    maxit : int, optional
        integer specifying the maximum number of iterations to perform. For
        first runs, maxit is the total number of iterations performed and
        defaults to 50. For restarts, maxit is the number of additional
        iterations to perform and defaults to 10.
    stpb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of relative step sizes to compute
        finite difference derivatives wrt the parameters.
    stpd : optional
        array (``stpd.shape == data.x.shape`` or ``stpd.shape == (m,)``) of relative
        step sizes to compute finite difference derivatives wrt the input
        variable errors. If stpd is a rank-1 array with length m (the
        dimensionality of the input variable), then the values are broadcast to
        all observations.
    sclb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of scaling factors for the
        parameters. The purpose of these scaling factors are to scale all of
        the parameters to around unity. Normally appropriate scaling factors
        are computed if this argument is not specified. Specify them yourself
        if the automatic procedure goes awry.
    scld : array_like, optional
        array (scld.shape == data.x.shape or scld.shape == (m,)) of scaling
        factors for the *errors* in the input variables. Again, these factors
        are automatically computed if you do not provide them. If scld.shape ==
        (m,), then the scaling factors are broadcast to all observations.
    work : ndarray, optional
        array to hold the double-valued working data for ODRPACK. When
        restarting, takes the value of self.output.work.
    iwork : ndarray, optional
        array to hold the integer-valued working data for ODRPACK. When
        restarting, takes the value of self.output.iwork.
    overwrite : bool, optional
        If it is True, output files defined by `errfile` and `rptfile` are
        overwritten. The default is False.

    Attributes
    ----------
    data : Data
        The data for this fit
    model : Model
        The model used in fit
    output : Output
        An instance if the Output class containing all of the returned
        data from an invocation of ODR.run() or ODR.restart()

    """

    def __init__(self, data, model, beta0=None, delta0=None, ifixb=None,
        ifixx=None, job=None, iprint=None, errfile=None, rptfile=None,
        ndigit=None, taufac=None, sstol=None, partol=None, maxit=None,
        stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None,
        overwrite=False):

        self.data = data
        self.model = model

        if beta0 is None:
            if self.model.estimate is not None:
                self.beta0 = _conv(self.model.estimate(self.data))
            else:
                raise ValueError(
                  "must specify beta0 or provide an estimater with the model"
                )
        else:
            self.beta0 = _conv(beta0)

        if ifixx is None and data.fix is not None:
            ifixx = data.fix

        if overwrite:
            # remove output files for overwriting.
            if rptfile is not None and os.path.exists(rptfile):
                os.remove(rptfile)
            if errfile is not None and os.path.exists(errfile):
                os.remove(errfile)

        self.delta0 = _conv(delta0)
        # These really are 32-bit integers in FORTRAN (gfortran), even on 64-bit
        # platforms.
        # XXX: some other FORTRAN compilers may not agree.
        self.ifixx = _conv(ifixx, dtype=numpy.int32)
        self.ifixb = _conv(ifixb, dtype=numpy.int32)
        self.job = job
        self.iprint = iprint
        self.errfile = errfile
        self.rptfile = rptfile
        self.ndigit = ndigit
        self.taufac = taufac
        self.sstol = sstol
        self.partol = partol
        self.maxit = maxit
        self.stpb = _conv(stpb)
        self.stpd = _conv(stpd)
        self.sclb = _conv(sclb)
        self.scld = _conv(scld)
        self.work = _conv(work)
        self.iwork = _conv(iwork)

        self.output = None

        self._check()

    def _check(self):
        """ Check the inputs for consistency, but don't bother checking things
        that the builtin function odr will check.
        """

        x_s = list(self.data.x.shape)

        if isinstance(self.data.y, numpy.ndarray):
            y_s = list(self.data.y.shape)
            if self.model.implicit:
                raise OdrError("an implicit model cannot use response data")
        else:
            # implicit model with q == self.data.y
            y_s = [self.data.y, x_s[-1]]
            if not self.model.implicit:
                raise OdrError("an explicit model needs response data")
            self.set_job(fit_type=1)

        if x_s[-1] != y_s[-1]:
            raise OdrError("number of observations do not match")

        n = x_s[-1]

        if len(x_s) == 2:
            m = x_s[0]
        else:
            m = 1
        if len(y_s) == 2:
            q = y_s[0]
        else:
            q = 1

        p = len(self.beta0)

        # permissible output array shapes

        fcn_perms = [(q, n)]
        fjacd_perms = [(q, m, n)]
        fjacb_perms = [(q, p, n)]

        if q == 1:
            fcn_perms.append((n,))
            fjacd_perms.append((m, n))
            fjacb_perms.append((p, n))
        if m == 1:
            fjacd_perms.append((q, n))
        if p == 1:
            fjacb_perms.append((q, n))
        if m == q == 1:
            fjacd_perms.append((n,))
        if p == q == 1:
            fjacb_perms.append((n,))

        # try evaluating the supplied functions to make sure they provide
        # sensible outputs

        arglist = (self.beta0, self.data.x)
        if self.model.extra_args is not None:
            arglist = arglist + self.model.extra_args
        res = self.model.fcn(*arglist)

        if res.shape not in fcn_perms:
            print(res.shape)
            print(fcn_perms)
            raise OdrError("fcn does not output %s-shaped array" % y_s)

        if self.model.fjacd is not None:
            res = self.model.fjacd(*arglist)
            if res.shape not in fjacd_perms:
                raise OdrError(
                    "fjacd does not output %s-shaped array" % repr((q, m, n)))
        if self.model.fjacb is not None:
            res = self.model.fjacb(*arglist)
            if res.shape not in fjacb_perms:
                raise OdrError(
                    "fjacb does not output %s-shaped array" % repr((q, p, n)))

        # check shape of delta0

        if self.delta0 is not None and self.delta0.shape != self.data.x.shape:
            raise OdrError(
                "delta0 is not a %s-shaped array" % repr(self.data.x.shape))

        if self.data.x.size == 0:
            warn(("Empty data detected for ODR instance. "
                  "Do not expect any fitting to occur"),
                 OdrWarning)

    def _gen_work(self):
        """ Generate a suitable work array if one does not already exist.
        """

        n = self.data.x.shape[-1]
        p = self.beta0.shape[0]

        if len(self.data.x.shape) == 2:
            m = self.data.x.shape[0]
        else:
            m = 1

        if self.model.implicit:
            q = self.data.y
        elif len(self.data.y.shape) == 2:
            q = self.data.y.shape[0]
        else:
            q = 1

        if self.data.we is None:
            ldwe = ld2we = 1
        elif len(self.data.we.shape) == 3:
            ld2we, ldwe = self.data.we.shape[1:]
        else:
            # Okay, this isn't precisely right, but for this calculation,
            # it's fine
            ldwe = 1
            ld2we = self.data.we.shape[1]

        if self.job % 10 < 2:
            # ODR not OLS
            lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 6*n*m + 2*n*q*p +
                     2*n*q*m + q*q + 5*q + q*(p+m) + ldwe*ld2we*q)
        else:
            # OLS not ODR
            lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 2*n*m + 2*n*q*p +
                     5*q + q*(p+m) + ldwe*ld2we*q)

        if isinstance(self.work, numpy.ndarray) and self.work.shape == (lwork,)\
                and self.work.dtype.str.endswith('f8'):
            # the existing array is fine
            return
        else:
            self.work = numpy.zeros((lwork,), float)

    def set_job(self, fit_type=None, deriv=None, var_calc=None,
        del_init=None, restart=None):
        """
        Sets the "job" parameter is a hopefully comprehensible way.

        If an argument is not specified, then the value is left as is. The
        default value from class initialization is for all of these options set
        to 0.

        Parameters
        ----------
        fit_type : {0, 1, 2} int
            0 -> explicit ODR

            1 -> implicit ODR

            2 -> ordinary least-squares
        deriv : {0, 1, 2, 3} int
            0 -> forward finite differences

            1 -> central finite differences

            2 -> user-supplied derivatives (Jacobians) with results
              checked by ODRPACK

            3 -> user-supplied derivatives, no checking
        var_calc : {0, 1, 2} int
            0 -> calculate asymptotic covariance matrix and fit
                 parameter uncertainties (V_B, s_B) using derivatives
                 recomputed at the final solution

            1 -> calculate V_B and s_B using derivatives from last iteration

            2 -> do not calculate V_B and s_B
        del_init : {0, 1} int
            0 -> initial input variable offsets set to 0

            1 -> initial offsets provided by user in variable "work"
        restart : {0, 1} int
            0 -> fit is not a restart

            1 -> fit is a restart

        Notes
        -----
        The permissible values are different from those given on pg. 31 of the
        ODRPACK User's Guide only in that one cannot specify numbers greater than
        the last value for each variable.

        If one does not supply functions to compute the Jacobians, the fitting
        procedure will change deriv to 0, finite differences, as a default. To
        initialize the input variable offsets by yourself, set del_init to 1 and
        put the offsets into the "work" variable correctly.

        """

        if self.job is None:
            job_l = [0, 0, 0, 0, 0]
        else:
            job_l = [self.job // 10000 % 10,
                     self.job // 1000 % 10,
                     self.job // 100 % 10,
                     self.job // 10 % 10,
                     self.job % 10]

        if fit_type in (0, 1, 2):
            job_l[4] = fit_type
        if deriv in (0, 1, 2, 3):
            job_l[3] = deriv
        if var_calc in (0, 1, 2):
            job_l[2] = var_calc
        if del_init in (0, 1):
            job_l[1] = del_init
        if restart in (0, 1):
            job_l[0] = restart

        self.job = (job_l[0]*10000 + job_l[1]*1000 +
                    job_l[2]*100 + job_l[3]*10 + job_l[4])

    def set_iprint(self, init=None, so_init=None,
        iter=None, so_iter=None, iter_step=None, final=None, so_final=None):
        """ Set the iprint parameter for the printing of computation reports.

        If any of the arguments are specified here, then they are set in the
        iprint member. If iprint is not set manually or with this method, then
        ODRPACK defaults to no printing. If no filename is specified with the
        member rptfile, then ODRPACK prints to stdout. One can tell ODRPACK to
        print to stdout in addition to the specified filename by setting the
        so_* arguments to this function, but one cannot specify to print to
        stdout but not a file since one can do that by not specifying a rptfile
        filename.

        There are three reports: initialization, iteration, and final reports.
        They are represented by the arguments init, iter, and final
        respectively.  The permissible values are 0, 1, and 2 representing "no
        report", "short report", and "long report" respectively.

        The argument iter_step (0 <= iter_step <= 9) specifies how often to make
        the iteration report; the report will be made for every iter_step'th
        iteration starting with iteration one. If iter_step == 0, then no
        iteration report is made, regardless of the other arguments.

        If the rptfile is None, then any so_* arguments supplied will raise an
        exception.
        """
        if self.iprint is None:
            self.iprint = 0

        ip = [self.iprint // 1000 % 10,
              self.iprint // 100 % 10,
              self.iprint // 10 % 10,
              self.iprint % 10]

        # make a list to convert iprint digits to/from argument inputs
        #                   rptfile, stdout
        ip2arg = [[0, 0],  # none,  none
                  [1, 0],  # short, none
                  [2, 0],  # long,  none
                  [1, 1],  # short, short
                  [2, 1],  # long,  short
                  [1, 2],  # short, long
                  [2, 2]]  # long,  long

        if (self.rptfile is None and
            (so_init is not None or
             so_iter is not None or
             so_final is not None)):
            raise OdrError(
                "no rptfile specified, cannot output to stdout twice")

        iprint_l = ip2arg[ip[0]] + ip2arg[ip[1]] + ip2arg[ip[3]]

        if init is not None:
            iprint_l[0] = init
        if so_init is not None:
            iprint_l[1] = so_init
        if iter is not None:
            iprint_l[2] = iter
        if so_iter is not None:
            iprint_l[3] = so_iter
        if final is not None:
            iprint_l[4] = final
        if so_final is not None:
            iprint_l[5] = so_final

        if iter_step in range(10):
            # 0..9
            ip[2] = iter_step

        ip[0] = ip2arg.index(iprint_l[0:2])
        ip[1] = ip2arg.index(iprint_l[2:4])
        ip[3] = ip2arg.index(iprint_l[4:6])

        self.iprint = ip[0]*1000 + ip[1]*100 + ip[2]*10 + ip[3]

    def run(self):
        """ Run the fitting routine with all of the information given and with ``full_output=1``.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """

        args = (self.model.fcn, self.beta0, self.data.y, self.data.x)
        kwds = {'full_output': 1}
        kwd_l = ['ifixx', 'ifixb', 'job', 'iprint', 'errfile', 'rptfile',
                 'ndigit', 'taufac', 'sstol', 'partol', 'maxit', 'stpb',
                 'stpd', 'sclb', 'scld', 'work', 'iwork']

        if self.delta0 is not None and (self.job // 10000) % 10 == 0:
            # delta0 provided and fit is not a restart
            self._gen_work()

            d0 = numpy.ravel(self.delta0)

            self.work[:len(d0)] = d0

        # set the kwds from other objects explicitly
        if self.model.fjacb is not None:
            kwds['fjacb'] = self.model.fjacb
        if self.model.fjacd is not None:
            kwds['fjacd'] = self.model.fjacd
        if self.data.we is not None:
            kwds['we'] = self.data.we
        if self.data.wd is not None:
            kwds['wd'] = self.data.wd
        if self.model.extra_args is not None:
            kwds['extra_args'] = self.model.extra_args

        # implicitly set kwds from self's members
        for attr in kwd_l:
            obj = getattr(self, attr)
            if obj is not None:
                kwds[attr] = obj

        self.output = Output(odr(*args, **kwds))

        return self.output

    def restart(self, iter=None):
        """ Restarts the run with iter more iterations.

        Parameters
        ----------
        iter : int, optional
            ODRPACK's default for the number of new iterations is 10.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """

        if self.output is None:
            raise OdrError("cannot restart: run() has not been called before")

        self.set_job(restart=1)
        self.work = self.output.work
        self.iwork = self.output.iwork

        self.maxit = iter

        return self.run()
