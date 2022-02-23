"""
shgo: The simplicial homology global optimisation algorithm
"""

import numpy as np
import time
import logging
import warnings
from scipy import spatial
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._shgo_lib.triangulation import Complex


__all__ = ['shgo']


def shgo(func, bounds, args=(), constraints=None, n=None, iters=1,
         callback=None,
         minimizer_kwargs=None, options=None, sampling_method='simplicial'):
    """
    Finds the global minimum of a function using SHG optimization.

    SHGO stands for "simplicial homology global optimization".

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
        Use ``None`` for one of min or max when there is no bound in that
        direction. By default bounds are ``(None, None)``.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.
    constraints : dict or sequence of dict, optional
        Constraints definition.
        Function(s) ``R**n`` in the form::

            g(x) >= 0 applied as g : R^n -> R^m
            h(x) == 0 applied as h : R^n -> R^p

        Each constraint is defined in a dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
        Note that COBYLA only supports inequality constraints.

        .. note::

           Only the COBYLA and SLSQP local minimize methods currently
           support constraint arguments. If the ``constraints`` sequence
           used in the local optimization problem is not defined in
           ``minimizer_kwargs`` and a constrained method is used then the
           global ``constraints`` will be used.
           (Defining a ``constraints`` sequence in ``minimizer_kwargs``
           means that ``constraints`` will not be added so if equality
           constraints and so forth need to be added then the inequality
           functions in ``constraints`` need to be added to
           ``minimizer_kwargs`` too).

    n : int, optional
        Number of sampling points used in the construction of the simplicial
        complex. Note that this argument is only used for ``sobol`` and other
        arbitrary `sampling_methods`. In case of ``sobol``, it must be a
        power of 2: ``n=2**m``, and the argument will automatically be
        converted to the next higher power of 2. Default is 100 for
        ``sampling_method='simplicial'`` and 128 for
        ``sampling_method='sobol'``.
    iters : int, optional
        Number of iterations used in the construction of the simplicial
        complex. Default is 1.
    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the minimizer
        ``scipy.optimize.minimize`` Some important options could be:

            * method : str
                The minimization method, the default is ``SLSQP``.
            * args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).
            * options : dict, optional
                Note that by default the tolerance is specified as
                ``{ftol: 1e-12}``

    options : dict, optional
        A dictionary of solver options. Many of the options specified for the
        global routine are also passed to the scipy.optimize.minimize routine.
        The options that are also passed to the local routine are marked with
        "(L)".

        Stopping criteria, the algorithm will terminate if any of the specified
        criteria are met. However, the default algorithm does not require any to
        be specified:

        * maxfev : int (L)
            Maximum number of function evaluations in the feasible domain.
            (Note only methods that support this option will terminate
            the routine at precisely exact specified value. Otherwise the
            criterion will only terminate during a global iteration)
        * f_min
            Specify the minimum objective function value, if it is known.
        * f_tol : float
            Precision goal for the value of f in the stopping
            criterion. Note that the global routine will also
            terminate if a sampling point in the global routine is
            within this tolerance.
        * maxiter : int
            Maximum number of iterations to perform.
        * maxev : int
            Maximum number of sampling evaluations to perform (includes
            searching in infeasible points).
        * maxtime : float
            Maximum processing runtime allowed
        * minhgrd : int
            Minimum homology group rank differential. The homology group of the
            objective function is calculated (approximately) during every
            iteration. The rank of this group has a one-to-one correspondence
            with the number of locally convex subdomains in the objective
            function (after adequate sampling points each of these subdomains
            contain a unique global minimum). If the difference in the hgr is 0
            between iterations for ``maxhgrd`` specified iterations the
            algorithm will terminate.

        Objective function knowledge:

        * symmetry : bool
            Specify True if the objective function contains symmetric variables.
            The search space (and therefore performance) is decreased by O(n!).

        * jac : bool or callable, optional
            Jacobian (gradient) of objective function. Only for CG, BFGS,
            Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. If ``jac`` is a
            boolean and is True, ``fun`` is assumed to return the gradient along
            with the objective function. If False, the gradient will be
            estimated numerically. ``jac`` can also be a callable returning the
            gradient of the objective. In this case, it must accept the same
            arguments as ``fun``. (Passed to `scipy.optimize.minmize` automatically)

        * hess, hessp : callable, optional
            Hessian (matrix of second-order derivatives) of objective function
            or Hessian of objective function times an arbitrary vector p.
            Only for Newton-CG, dogleg, trust-ncg. Only one of ``hessp`` or
            ``hess`` needs to be given. If ``hess`` is provided, then
            ``hessp`` will be ignored. If neither ``hess`` nor ``hessp`` is
            provided, then the Hessian product will be approximated using
            finite differences on ``jac``. ``hessp`` must compute the Hessian
            times an arbitrary vector. (Passed to `scipy.optimize.minmize`
            automatically)

        Algorithm settings:

        * minimize_every_iter : bool
            If True then promising global sampling points will be passed to a
            local minimization routine every iteration. If False then only the
            final minimizer pool will be run. Defaults to False.
        * local_iter : int
            Only evaluate a few of the best minimizer pool candidates every
            iteration. If False all potential points are passed to the local
            minimization routine.
        * infty_constraints: bool
            If True then any sampling points generated which are outside will
            the feasible domain will be saved and given an objective function
            value of ``inf``. If False then these points will be discarded.
            Using this functionality could lead to higher performance with
            respect to function evaluations before the global minimum is found,
            specifying False will use less memory at the cost of a slight
            decrease in performance. Defaults to True.

        Feedback:

        * disp : bool (L)
            Set to True to print convergence messages.

    sampling_method : str or function, optional
        Current built in sampling method options are ``halton``, ``sobol`` and
        ``simplicial``. The default ``simplicial`` provides
        the theoretical guarantee of convergence to the global minimum in finite
        time. ``halton`` and ``sobol`` method are faster in terms of sampling
        point generation at the cost of the loss of
        guaranteed convergence. It is more appropriate for most "easier"
        problems where the convergence is relatively fast.
        User defined sampling functions must accept two arguments of ``n``
        sampling points of dimension ``dim`` per call and output an array of
        sampling points with shape `n x dim`.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are:
        ``x`` the solution array corresponding to the global minimum,
        ``fun`` the function output at the global solution,
        ``xl`` an ordered list of local minima solutions,
        ``funl`` the function output at the corresponding local solutions,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully,
        ``message`` which describes the cause of the termination,
        ``nfev`` the total number of objective function evaluations including
        the sampling calls,
        ``nlfev`` the total number of objective function evaluations
        culminating from all local search optimizations,
        ``nit`` number of iterations performed by the global routine.

    Notes
    -----
    Global optimization using simplicial homology global optimization [1]_.
    Appropriate for solving general purpose NLP and blackbox optimization
    problems to global optimality (low-dimensional problems).

    In general, the optimization problems are of the form::

        minimize f(x) subject to

        g_i(x) >= 0,  i = 1,...,m
        h_j(x)  = 0,  j = 1,...,p

    where x is a vector of one or more variables. ``f(x)`` is the objective
    function ``R^n -> R``, ``g_i(x)`` are the inequality constraints, and
    ``h_j(x)`` are the equality constraints.

    Optionally, the lower and upper bounds for each element in x can also be
    specified using the `bounds` argument.

    While most of the theoretical advantages of SHGO are only proven for when
    ``f(x)`` is a Lipschitz smooth function, the algorithm is also proven to
    converge to the global optimum for the more general case where ``f(x)`` is
    non-continuous, non-convex and non-smooth, if the default sampling method
    is used [1]_.

    The local search method may be specified using the ``minimizer_kwargs``
    parameter which is passed on to ``scipy.optimize.minimize``. By default,
    the ``SLSQP`` method is used. In general, it is recommended to use the
    ``SLSQP`` or ``COBYLA`` local minimization if inequality constraints
    are defined for the problem since the other methods do not use constraints.

    The ``halton`` and ``sobol`` method points are generated using
    `scipy.stats.qmc`. Any other QMC method could be used.

    References
    ----------
    .. [1] Endres, SC, Sandrock, C, Focke, WW (2018) "A simplicial homology
           algorithm for lipschitz optimisation", Journal of Global Optimization.
    .. [2] Joe, SW and Kuo, FY (2008) "Constructing Sobol' sequences with
           better  two-dimensional projections", SIAM J. Sci. Comput. 30,
           2635-2654.
    .. [3] Hoch, W and Schittkowski, K (1981) "Test examples for nonlinear
           programming codes", Lecture Notes in Economics and Mathematical
           Systems, 187. Springer-Verlag, New York.
           http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    .. [4] Wales, DJ (2015) "Perspective: Insight into reaction coordinates and
           dynamics from the potential energy landscape",
           Journal of Chemical Physics, 142(13), 2015.

    Examples
    --------
    First consider the problem of minimizing the Rosenbrock function, `rosen`:

    >>> from scipy.optimize import rosen, shgo
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = shgo(rosen, bounds)
    >>> result.x, result.fun
    (array([1., 1., 1., 1., 1.]), 2.920392374190081e-18)

    Note that bounds determine the dimensionality of the objective
    function and is therefore a required input, however you can specify
    empty bounds using ``None`` or objects like ``np.inf`` which will be
    converted to large float numbers.

    >>> bounds = [(None, None), ]*4
    >>> result = shgo(rosen, bounds)
    >>> result.x
    array([0.99999851, 0.99999704, 0.99999411, 0.9999882 ])

    Next, we consider the Eggholder function, a problem with several local
    minima and one global minimum. We will demonstrate the use of arguments and
    the capabilities of `shgo`.
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization)

    >>> def eggholder(x):
    ...     return (-(x[1] + 47.0)
    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    ...             )
    ...
    >>> bounds = [(-512, 512), (-512, 512)]

    `shgo` has built-in low discrepancy sampling sequences. First, we will
    input 64 initial sampling points of the *Sobol'* sequence:

    >>> result = shgo(eggholder, bounds, n=64, sampling_method='sobol')
    >>> result.x, result.fun
    (array([512.        , 404.23180824]), -959.6406627208397)

    `shgo` also has a return for any other local minima that was found, these
    can be called using:

    >>> result.xl
    array([[ 512.        ,  404.23180824],
           [ 283.0759062 , -487.12565635],
           [-294.66820039, -462.01964031],
           [-105.87688911,  423.15323845],
           [-242.97926   ,  274.38030925],
           [-506.25823477,    6.3131022 ],
           [-408.71980731, -156.10116949],
           [ 150.23207937,  301.31376595],
           [  91.00920901, -391.283763  ],
           [ 202.89662724, -269.38043241],
           [ 361.66623976, -106.96493868],
           [-219.40612786, -244.06020508]])

    >>> result.funl
    array([-959.64066272, -718.16745962, -704.80659592, -565.99778097,
           -559.78685655, -557.36868733, -507.87385942, -493.9605115 ,
           -426.48799655, -421.15571437, -419.31194957, -410.98477763])

    These results are useful in applications where there are many global minima
    and the values of other global minima are desired or where the local minima
    can provide insight into the system (for example morphologies
    in physical chemistry [4]_).

    If we want to find a larger number of local minima, we can increase the
    number of sampling points or the number of iterations. We'll increase the
    number of sampling points to 64 and the number of iterations from the
    default of 1 to 3. Using ``simplicial`` this would have given us
    64 x 3 = 192 initial sampling points.

    >>> result_2 = shgo(eggholder, bounds, n=64, iters=3, sampling_method='sobol')
    >>> len(result.xl), len(result_2.xl)
    (12, 20)

    Note the difference between, e.g., ``n=192, iters=1`` and ``n=64,
    iters=3``.
    In the first case the promising points contained in the minimiser pool
    are processed only once. In the latter case it is processed every 64
    sampling points for a total of 3 times.

    To demonstrate solving problems with non-linear constraints consider the
    following example from Hock and Schittkowski problem 73 (cattle-feed) [3]_::

        minimize: f = 24.55 * x_1 + 26.75 * x_2 + 39 * x_3 + 40.50 * x_4

        subject to: 2.3 * x_1 + 5.6 * x_2 + 11.1 * x_3 + 1.3 * x_4 - 5     >= 0,

                    12 * x_1 + 11.9 * x_2 + 41.8 * x_3 + 52.1 * x_4 - 21
                        -1.645 * sqrt(0.28 * x_1**2 + 0.19 * x_2**2 +
                                      20.5 * x_3**2 + 0.62 * x_4**2)       >= 0,

                    x_1 + x_2 + x_3 + x_4 - 1                              == 0,

                    1 >= x_i >= 0 for all i

    The approximate answer given in [3]_ is::

        f([0.6355216, -0.12e-11, 0.3127019, 0.05177655]) = 29.894378

    >>> def f(x):  # (cattle-feed)
    ...     return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]
    ...
    >>> def g1(x):
    ...     return 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5  # >=0
    ...
    >>> def g2(x):
    ...     return (12*x[0] + 11.9*x[1] +41.8*x[2] + 52.1*x[3] - 21
    ...             - 1.645 * np.sqrt(0.28*x[0]**2 + 0.19*x[1]**2
    ...                             + 20.5*x[2]**2 + 0.62*x[3]**2)
    ...             ) # >=0
    ...
    >>> def h1(x):
    ...     return x[0] + x[1] + x[2] + x[3] - 1  # == 0
    ...
    >>> cons = ({'type': 'ineq', 'fun': g1},
    ...         {'type': 'ineq', 'fun': g2},
    ...         {'type': 'eq', 'fun': h1})
    >>> bounds = [(0, 1.0),]*4
    >>> res = shgo(f, bounds, iters=3, constraints=cons)
    >>> res
         fun: 29.894378159142136
        funl: array([29.89437816])
     message: 'Optimization terminated successfully.'
        nfev: 114
         nit: 3
       nlfev: 35
       nlhev: 0
       nljev: 5
     success: True
           x: array([6.35521569e-01, 1.13700270e-13, 3.12701881e-01, 5.17765506e-02])
          xl: array([[6.35521569e-01, 1.13700270e-13, 3.12701881e-01, 5.17765506e-02]])

    >>> g1(res.x), g2(res.x), h1(res.x)
    (-5.062616992290714e-14, -2.9594104944408173e-12, 0.0)

    """
    # Initiate SHGO class
    shc = SHGO(func, bounds, args=args, constraints=constraints, n=n,
               iters=iters, callback=callback,
               minimizer_kwargs=minimizer_kwargs,
               options=options, sampling_method=sampling_method)

    # Run the algorithm, process results and test success
    shc.construct_complex()

    if not shc.break_routine:
        if shc.disp:
            print("Successfully completed construction of complex.")

    # Test post iterations success
    if len(shc.LMC.xl_maps) == 0:
        # If sampling failed to find pool, return lowest sampled point
        # with a warning
        shc.find_lowest_vertex()
        shc.break_routine = True
        shc.fail_routine(mes="Failed to find a feasible minimizer point. "
                             "Lowest sampling point = {}".format(shc.f_lowest))
        shc.res.fun = shc.f_lowest
        shc.res.x = shc.x_lowest
        shc.res.nfev = shc.fn

    # Confirm the routine ran successfully
    if not shc.break_routine:
        shc.res.message = 'Optimization terminated successfully.'
        shc.res.success = True

    # Return the final results
    return shc.res


class SHGO:
    def __init__(self, func, bounds, args=(), constraints=None, n=None,
                 iters=None, callback=None, minimizer_kwargs=None,
                 options=None, sampling_method='sobol'):

        from scipy.stats import qmc

        # Input checks
        methods = ['halton', 'sobol', 'simplicial']
        if isinstance(sampling_method, str) and sampling_method not in methods:
            raise ValueError(("Unknown sampling_method specified."
                              " Valid methods: {}").format(', '.join(methods)))

        # Initiate class
        self.func = func
        self.bounds = bounds
        self.args = args
        self.callback = callback

        # Bounds
        abound = np.array(bounds, float)
        self.dim = np.shape(abound)[0]  # Dimensionality of problem

        # Set none finite values to large floats
        infind = ~np.isfinite(abound)
        abound[infind[:, 0], 0] = -1e50
        abound[infind[:, 1], 1] = 1e50

        # Check if bounds are correctly specified
        bnderr = abound[:, 0] > abound[:, 1]
        if bnderr.any():
            raise ValueError('Error: lb > ub in bounds {}.'
                             .format(', '.join(str(b) for b in bnderr)))

        self.bounds = abound

        # Constraints
        # Process constraint dict sequence:
        if constraints is not None:
            self.min_cons = constraints
            self.g_cons = []
            self.g_args = []
            if (type(constraints) is not tuple) and (type(constraints)
                                                     is not list):
                constraints = (constraints,)

            for cons in constraints:
                if cons['type'] == 'ineq':
                    self.g_cons.append(cons['fun'])
                    try:
                        self.g_args.append(cons['args'])
                    except KeyError:
                        self.g_args.append(())
            self.g_cons = tuple(self.g_cons)
            self.g_args = tuple(self.g_args)
        else:
            self.g_cons = None
            self.g_args = None

        # Define local minimization keyword arguments
        # Start with defaults
        self.minimizer_kwargs = {'args': self.args,
                                 'method': 'SLSQP',
                                 'bounds': self.bounds,
                                 'options': {},
                                 'callback': self.callback
                                 }
        if minimizer_kwargs is not None:
            # Overwrite with supplied values
            self.minimizer_kwargs.update(minimizer_kwargs)

        else:
            self.minimizer_kwargs['options'] = {'ftol': 1e-12}

        if (self.minimizer_kwargs['method'] in ('SLSQP', 'COBYLA') and
                (minimizer_kwargs is not None and
                 'constraints' not in minimizer_kwargs and
                 constraints is not None) or
                (self.g_cons is not None)):
            self.minimizer_kwargs['constraints'] = self.min_cons

        # Process options dict
        if options is not None:
            self.init_options(options)
        else:  # Default settings:
            self.f_min_true = None
            self.minimize_every_iter = False

            # Algorithm limits
            self.maxiter = None
            self.maxfev = None
            self.maxev = None
            self.maxtime = None
            self.f_min_true = None
            self.minhgrd = None

            # Objective function knowledge
            self.symmetry = False

            # Algorithm functionality
            self.local_iter = False
            self.infty_cons_sampl = True

            # Feedback
            self.disp = False

        # Remove unknown arguments in self.minimizer_kwargs
        # Start with arguments all the solvers have in common
        self.min_solver_args = ['fun', 'x0', 'args',
                                'callback', 'options', 'method']
        # then add the ones unique to specific solvers
        solver_args = {
            '_custom': ['jac', 'hess', 'hessp', 'bounds', 'constraints'],
            'nelder-mead': [],
            'powell': [],
            'cg': ['jac'],
            'bfgs': ['jac'],
            'newton-cg': ['jac', 'hess', 'hessp'],
            'l-bfgs-b': ['jac', 'bounds'],
            'tnc': ['jac', 'bounds'],
            'cobyla': ['constraints'],
            'slsqp': ['jac', 'bounds', 'constraints'],
            'dogleg': ['jac', 'hess'],
            'trust-ncg': ['jac', 'hess', 'hessp'],
            'trust-krylov': ['jac', 'hess', 'hessp'],
            'trust-exact': ['jac', 'hess'],
        }
        method = self.minimizer_kwargs['method']
        self.min_solver_args += solver_args[method.lower()]

        # Only retain the known arguments
        def _restrict_to_keys(dictionary, goodkeys):
            """Remove keys from dictionary if not in goodkeys - inplace"""
            existingkeys = set(dictionary)
            for key in existingkeys - set(goodkeys):
                dictionary.pop(key, None)

        _restrict_to_keys(self.minimizer_kwargs, self.min_solver_args)
        _restrict_to_keys(self.minimizer_kwargs['options'],
                          self.min_solver_args + ['ftol'])

        # Algorithm controls
        # Global controls
        self.stop_global = False  # Used in the stopping_criteria method
        self.break_routine = False  # Break the algorithm globally
        self.iters = iters  # Iterations to be ran
        self.iters_done = 0  # Iterations to be ran
        self.n = n  # Sampling points per iteration
        self.nc = n  # Sampling points to sample in current iteration
        self.n_prc = 0  # Processed points (used to track Delaunay iters)
        self.n_sampled = 0  # To track number of sampling points already generated
        self.fn = 0  # Number of feasible sampling points evaluations performed
        self.hgr = 0  # Homology group rank

        # Default settings if no sampling criteria.
        if self.iters is None:
            self.iters = 1
        if self.n is None:
            self.n = 100
            if sampling_method == 'sobol':
                self.n = 128
            self.nc = self.n

        if not ((self.maxiter is None) and (self.maxfev is None) and (
                    self.maxev is None)
                and (self.minhgrd is None) and (self.f_min_true is None)):
            self.iters = None

        # Set complex construction mode based on a provided stopping criteria:
        # Choose complex constructor
        if sampling_method == 'simplicial':
            self.iterate_complex = self.iterate_hypercube
            self.minimizers = self.simplex_minimizers
            self.sampling_method = sampling_method

        elif sampling_method in ['halton', 'sobol'] or \
                not isinstance(sampling_method, str):
            self.iterate_complex = self.iterate_delaunay
            self.minimizers = self.delaunay_complex_minimisers
            # Sampling method used
            if sampling_method in ['halton', 'sobol']:
                if sampling_method == 'sobol':
                    self.n = int(2 ** np.ceil(np.log2(self.n)))
                    self.nc = self.n
                    self.sampling_method = 'sobol'
                    self.qmc_engine = qmc.Sobol(d=self.dim, scramble=False,
                                                seed=np.random.RandomState())
                else:
                    self.sampling_method = 'halton'
                    self.qmc_engine = qmc.Halton(d=self.dim, scramble=True,
                                                 seed=np.random.RandomState())
                sampling_method = lambda n, d: self.qmc_engine.random(n)
            else:
                # A user defined sampling method:
                self.sampling_method = 'custom'

            self.sampling = self.sampling_custom
            self.sampling_function = sampling_method  # F(n, d)

        # Local controls
        self.stop_l_iter = False  # Local minimisation iterations
        self.stop_complex_iter = False  # Sampling iterations

        # Initiate storage objects used in algorithm classes
        self.minimizer_pool = []

        # Cache of local minimizers mapped
        self.LMC = LMapCache()

        # Initialize return object
        self.res = OptimizeResult()  # scipy.optimize.OptimizeResult object
        self.res.nfev = 0  # Includes each sampling point as func evaluation
        self.res.nlfev = 0  # Local function evals for all minimisers
        self.res.nljev = 0  # Local Jacobian evals for all minimisers
        self.res.nlhev = 0  # Local Hessian evals for all minimisers

    # Initiation aids
    def init_options(self, options):
        """
        Initiates the options.

        Can also be useful to change parameters after class initiation.

        Parameters
        ----------
        options : dict

        Returns
        -------
        None

        """
        self.minimizer_kwargs['options'].update(options)
        # Default settings:
        self.minimize_every_iter = options.get('minimize_every_iter', False)

        # Algorithm limits
        # Maximum number of iterations to perform.
        self.maxiter = options.get('maxiter', None)
        # Maximum number of function evaluations in the feasible domain
        self.maxfev = options.get('maxfev', None)
        # Maximum number of sampling evaluations (includes searching in
        # infeasible points
        self.maxev = options.get('maxev', None)
        # Maximum processing runtime allowed
        self.init = time.time()
        self.maxtime = options.get('maxtime', None)
        if 'f_min' in options:
            # Specify the minimum objective function value, if it is known.
            self.f_min_true = options['f_min']
            self.f_tol = options.get('f_tol', 1e-4)
        else:
            self.f_min_true = None

        self.minhgrd = options.get('minhgrd', None)

        # Objective function knowledge
        self.symmetry = 'symmetry' in options

        # Algorithm functionality
        # Only evaluate a few of the best candiates
        self.local_iter = options.get('local_iter', False)

        self.infty_cons_sampl = options.get('infty_constraints', True)

        # Feedback
        self.disp = options.get('disp', False)

    # Iteration properties
    # Main construction loop:
    def construct_complex(self):
        """
        Construct for `iters` iterations.

        If uniform sampling is used, every iteration adds 'n' sampling points.

        Iterations if a stopping criteria (e.g., sampling points or
        processing time) has been met.

        """
        if self.disp:
            print('Splitting first generation')

        while not self.stop_global:
            if self.break_routine:
                break
            # Iterate complex, process minimisers
            self.iterate()
            self.stopping_criteria()

        # Build minimiser pool
        # Final iteration only needed if pools weren't minimised every iteration
        if not self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()

        self.res.nit = self.iters_done + 1

    def find_minima(self):
        """
        Construct the minimizer pool, map the minimizers to local minima
        and sort the results into a global return object.
        """
        self.minimizers()
        if len(self.X_min) != 0:
            # Minimize the pool of minimizers with local minimization methods
            # Note that if Options['local_iter'] is an `int` instead of default
            # value False then only that number of candidates will be minimized
            self.minimise_pool(self.local_iter)
            # Sort results and build the global return object
            self.sort_result()

            # Lowest values used to report in case of failures
            self.f_lowest = self.res.fun
            self.x_lowest = self.res.x
        else:
            self.find_lowest_vertex()

    def find_lowest_vertex(self):
        # Find the lowest objective function value on one of
        # the vertices of the simplicial complex
        if self.sampling_method == 'simplicial':
            self.f_lowest = np.inf
            for x in self.HC.V.cache:
                if self.HC.V[x].f < self.f_lowest:
                    self.f_lowest = self.HC.V[x].f
                    self.x_lowest = self.HC.V[x].x_a
            if self.f_lowest == np.inf:  # no feasible point
                self.f_lowest = None
                self.x_lowest = None
        else:
            if self.fn == 0:
                self.f_lowest = None
                self.x_lowest = None
            else:
                self.f_I = np.argsort(self.F, axis=-1)
                self.f_lowest = self.F[self.f_I[0]]
                self.x_lowest = self.C[self.f_I[0]]

    # Stopping criteria functions:
    def finite_iterations(self):
        if self.iters is not None:
            if self.iters_done >= (self.iters - 1):
                self.stop_global = True

        if self.maxiter is not None:  # Stop for infeasible sampling
            if self.iters_done >= (self.maxiter - 1):
                self.stop_global = True
        return self.stop_global

    def finite_fev(self):
        # Finite function evals in the feasible domain
        if self.fn >= self.maxfev:
            self.stop_global = True
        return self.stop_global

    def finite_ev(self):
        # Finite evaluations including infeasible sampling points
        if self.n_sampled >= self.maxev:
            self.stop_global = True

    def finite_time(self):
        if (time.time() - self.init) >= self.maxtime:
            self.stop_global = True

    def finite_precision(self):
        """
        Stop the algorithm if the final function value is known

        Specify in options (with ``self.f_min_true = options['f_min']``)
        and the tolerance with ``f_tol = options['f_tol']``
        """
        # If no minimizer has been found use the lowest sampling value
        if len(self.LMC.xl_maps) == 0:
            self.find_lowest_vertex()

        # Function to stop algorithm at specified percentage error:
        if self.f_lowest == 0.0:
            if self.f_min_true == 0.0:
                if self.f_lowest <= self.f_tol:
                    self.stop_global = True
        else:
            pe = (self.f_lowest - self.f_min_true) / abs(self.f_min_true)
            if self.f_lowest <= self.f_min_true:
                self.stop_global = True
                # 2if (pe - self.f_tol) <= abs(1.0 / abs(self.f_min_true)):
                if abs(pe) >= 2 * self.f_tol:
                    warnings.warn("A much lower value than expected f* =" +
                                  " {} than".format(self.f_min_true) +
                                  " the was found f_lowest =" +
                                  "{} ".format(self.f_lowest))
            if pe <= self.f_tol:
                self.stop_global = True

        return self.stop_global

    def finite_homology_growth(self):
        if self.LMC.size == 0:
            return  # pass on no reason to stop yet.
        self.hgrd = self.LMC.size - self.hgr

        self.hgr = self.LMC.size
        if self.hgrd <= self.minhgrd:
            self.stop_global = True
        return self.stop_global

    def stopping_criteria(self):
        """
        Various stopping criteria ran every iteration

        Returns
        -------
        stop : bool
        """
        if self.maxiter is not None:
            self.finite_iterations()
        if self.iters is not None:
            self.finite_iterations()
        if self.maxfev is not None:
            self.finite_fev()
        if self.maxev is not None:
            self.finite_ev()
        if self.maxtime is not None:
            self.finite_time()
        if self.f_min_true is not None:
            self.finite_precision()
        if self.minhgrd is not None:
            self.finite_homology_growth()

    def iterate(self):
        self.iterate_complex()

        # Build minimizer pool
        if self.minimize_every_iter:
            if not self.break_routine:
                self.find_minima()  # Process minimizer pool

        # Algorithm updates
        self.iters_done += 1

    def iterate_hypercube(self):
        """
        Iterate a subdivision of the complex

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        # Iterate the complex
        if self.n_sampled == 0:
            # Initial triangulation of the hyper-rectangle
            self.HC = Complex(self.dim, self.func, self.args,
                              self.symmetry, self.bounds, self.g_cons,
                              self.g_args)
        else:
            self.HC.split_generation()

        # feasible sampling points counted by the triangulation.py routines
        self.fn = self.HC.V.nfev
        self.n_sampled = self.HC.V.size  # nevs counted in triangulation.py
        return

    def iterate_delaunay(self):
        """
        Build a complex of Delaunay triangulated points

        Note: called with ``self.iterate_complex()`` after class initiation
        """
        self.sampled_surface(infty_cons_sampl=self.infty_cons_sampl)
        self.nc += self.n
        self.n_sampled = self.nc

    # Hypercube minimizers
    def simplex_minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        self.minimizer_pool = []
        # Note: Can implement parallelization here
        for x in self.HC.V.cache:
            if self.HC.V[x].minimiser():
                if self.disp:
                    logging.info('=' * 60)
                    logging.info(
                        'v.x = {} is minimizer'.format(self.HC.V[x].x_a))
                    logging.info('v.f = {} is minimizer'.format(self.HC.V[x].f))
                    logging.info('=' * 30)

                if self.HC.V[x] not in self.minimizer_pool:
                    self.minimizer_pool.append(self.HC.V[x])

                if self.disp:
                    logging.info('Neighbors:')
                    logging.info('=' * 30)
                    for vn in self.HC.V[x].nn:
                        logging.info('x = {} || f = {}'.format(vn.x, vn.f))

                    logging.info('=' * 60)

        self.minimizer_pool_F = []
        self.X_min = []
        # normalized tuple in the Vertex cache
        self.X_min_cache = {}  # Cache used in hypercube sampling

        for v in self.minimizer_pool:
            self.X_min.append(v.x_a)
            self.minimizer_pool_F.append(v.f)
            self.X_min_cache[tuple(v.x_a)] = v.x

        self.minimizer_pool_F = np.array(self.minimizer_pool_F)
        self.X_min = np.array(self.X_min)

        # TODO: Only do this if global mode
        self.sort_min_pool()

        return self.X_min

    # Local minimization
    # Minimizer pool processing
    def minimise_pool(self, force_iter=False):
        """
        This processing method can optionally minimise only the best candidate
        solutions in the minimizer pool

        Parameters
        ----------
        force_iter : int
                     Number of starting minimizers to process (can be sepcified
                     globally or locally)

        """
        # Find first local minimum
        # NOTE: Since we always minimize this value regardless it is a waste to
        # build the topograph first before minimizing
        lres_f_min = self.minimize(self.X_min[0], ind=self.minimizer_pool[0])

        # Trim minimized point from current minimizer set
        self.trim_min_pool(0)

        # Force processing to only
        if force_iter:
            self.local_iter = force_iter

        while not self.stop_l_iter:
            # Global stopping criteria:
            if self.f_min_true is not None:
                if (lres_f_min.fun - self.f_min_true) / abs(
                        self.f_min_true) <= self.f_tol:
                    self.stop_l_iter = True
                    break
            # Note first iteration is outside loop:
            if self.local_iter is not None:
                if self.disp:
                    logging.info(
                        'SHGO.iters in function minimise_pool = {}'.format(
                            self.local_iter))
                self.local_iter -= 1
                if self.local_iter == 0:
                    self.stop_l_iter = True
                    break

            if np.shape(self.X_min)[0] == 0:
                self.stop_l_iter = True
                break

            # Construct topograph from current minimizer set
            # (NOTE: This is a very small topograph using only the minizer pool
            #        , it might be worth using some graph theory tools instead.
            self.g_topograph(lres_f_min.x, self.X_min)

            # Find local minimum at the miniser with the greatest Euclidean
            # distance from the current solution
            ind_xmin_l = self.Z[:, -1]
            lres_f_min = self.minimize(self.Ss[-1, :], self.minimizer_pool[-1])

            # Trim minimised point from current minimizer set
            self.trim_min_pool(ind_xmin_l)

        # Reset controls
        self.stop_l_iter = False
        return

    def sort_min_pool(self):
        # Sort to find minimum func value in min_pool
        self.ind_f_min = np.argsort(self.minimizer_pool_F)
        self.minimizer_pool = np.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = np.array(self.minimizer_pool_F)[
            self.ind_f_min]
        return

    def trim_min_pool(self, trim_ind):
        self.X_min = np.delete(self.X_min, trim_ind, axis=0)
        self.minimizer_pool_F = np.delete(self.minimizer_pool_F, trim_ind)
        self.minimizer_pool = np.delete(self.minimizer_pool, trim_ind)
        return

    def g_topograph(self, x_min, X_min):
        """
        Returns the topographical vector stemming from the specified value
        ``x_min`` for the current feasible set ``X_min`` with True boolean
        values indicating positive entries and False values indicating
        negative entries.

        """
        x_min = np.array([x_min])
        self.Y = spatial.distance.cdist(x_min, X_min, 'euclidean')
        # Find sorted indexes of spatial distances:
        self.Z = np.argsort(self.Y, axis=-1)

        self.Ss = X_min[self.Z][0]
        self.minimizer_pool = self.minimizer_pool[self.Z]
        self.minimizer_pool = self.minimizer_pool[0]
        return self.Ss

    # Local bound functions
    def construct_lcb_simplicial(self, v_min):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each dimension

        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]
        # Loop over all bounds
        for vn in v_min.nn:
            for i, x_i in enumerate(vn.x_a):
                # Lower bound
                if (x_i < v_min.x_a[i]) and (x_i > cbounds[i][0]):
                    cbounds[i][0] = x_i

                # Upper bound
                if (x_i > v_min.x_a[i]) and (x_i < cbounds[i][1]):
                    cbounds[i][1] = x_i

        if self.disp:
            logging.info('cbounds found for v_min.x_a = {}'.format(v_min.x_a))
            logging.info('cbounds = {}'.format(cbounds))

        return cbounds

    def construct_lcb_delaunay(self, v_min, ind=None):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimizer vertex

        Returns
        -------
        cbounds : list of lists
            List of size dimension with length-2 list of bounds for each dimension
        """
        cbounds = [[x_b_i[0], x_b_i[1]] for x_b_i in self.bounds]

        return cbounds

    # Minimize a starting point locally
    def minimize(self, x_min, ind=None):
        """
        This function is used to calculate the local minima using the specified
        sampling point as a starting value.

        Parameters
        ----------
        x_min : vector of floats
            Current starting point to minimize.

        Returns
        -------
        lres : OptimizeResult
            The local optimization result represented as a `OptimizeResult`
            object.
        """
        # Use minima maps if vertex was already run
        if self.disp:
            logging.info('Vertex minimiser maps = {}'.format(self.LMC.v_maps))

        if self.LMC[x_min].lres is not None:
            return self.LMC[x_min].lres

        # TODO: Check discarded bound rules

        if self.callback is not None:
            print('Callback for '
                  'minimizer starting at {}:'.format(x_min))

        if self.disp:
            print('Starting '
                  'minimization at {}...'.format(x_min))

        if self.sampling_method == 'simplicial':
            x_min_t = tuple(x_min)
            # Find the normalized tuple in the Vertex cache:
            x_min_t_norm = self.X_min_cache[tuple(x_min_t)]

            x_min_t_norm = tuple(x_min_t_norm)

            g_bounds = self.construct_lcb_simplicial(self.HC.V[x_min_t_norm])
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds

        else:
            g_bounds = self.construct_lcb_delaunay(x_min, ind=ind)
            if 'bounds' in self.min_solver_args:
                self.minimizer_kwargs['bounds'] = g_bounds

        if self.disp and 'bounds' in self.minimizer_kwargs:
            print('bounds in kwarg:')
            print(self.minimizer_kwargs['bounds'])

        # Local minimization using scipy.optimize.minimize:
        lres = minimize(self.func, x_min, **self.minimizer_kwargs)

        if self.disp:
            print('lres = {}'.format(lres))

        # Local function evals for all minimizers
        self.res.nlfev += lres.nfev
        if 'njev' in lres:
            self.res.nljev += lres.njev
        if 'nhev' in lres:
            self.res.nlhev += lres.nhev

        try:  # Needed because of the brain dead 1x1 NumPy arrays
            lres.fun = lres.fun[0]
        except (IndexError, TypeError):
            lres.fun

        # Append minima maps
        self.LMC[x_min]
        self.LMC.add_res(x_min, lres, bounds=g_bounds)

        return lres

    # Post local minimization processing
    def sort_result(self):
        """
        Sort results and build the global return object
        """
        # Sort results in local minima cache
        results = self.LMC.sort_cache_result()
        self.res.xl = results['xl']
        self.res.funl = results['funl']
        self.res.x = results['x']
        self.res.fun = results['fun']

        # Add local func evals to sampling func evals
        # Count the number of feasible vertices and add to local func evals:
        self.res.nfev = self.fn + self.res.nlfev
        return self.res

    # Algorithm controls
    def fail_routine(self, mes=("Failed to converge")):
        self.break_routine = True
        self.res.success = False
        self.X_min = [None]
        self.res.message = mes

    def sampled_surface(self, infty_cons_sampl=False):
        """
        Sample the function surface.

        There are 2 modes, if ``infty_cons_sampl`` is True then the sampled
        points that are generated outside the feasible domain will be
        assigned an ``inf`` value in accordance with SHGO rules.
        This guarantees convergence and usually requires less objective function
        evaluations at the computational costs of more Delaunay triangulation
        points.

        If ``infty_cons_sampl`` is False, then the infeasible points are discarded
        and only a subspace of the sampled points are used. This comes at the
        cost of the loss of guaranteed convergence and usually requires more
        objective function evaluations.
        """
        # Generate sampling points
        if self.disp:
            print('Generating sampling points')
        self.sampling(self.nc, self.dim)
        self.n = self.nc

        if not infty_cons_sampl:
            # Find subspace of feasible points
            if self.g_cons is not None:
                self.sampling_subspace()

        # Sort remaining samples
        self.sorted_samples()

        # Find objective function references
        self.fun_ref()

        self.n_sampled = self.nc

    def delaunay_complex_minimisers(self):
        # Construct complex minimizers on the current sampling set.
        # if self.fn >= (self.dim + 1):
        if self.fn >= (self.dim + 2):
            # TODO: Check on strange Qhull error where the number of vertices
            # required for an initial simplex is higher than n + 1?
            if self.dim < 2:  # Scalar objective functions
                if self.disp:
                    print('Constructing 1-D minimizer pool')

                self.ax_subspace()
                self.surface_topo_ref()
                self.minimizers_1D()

            else:  # Multivariate functions.
                if self.disp:
                    print('Constructing Gabrial graph and minimizer pool')

                if self.iters == 1:
                    self.delaunay_triangulation(grow=False)
                else:
                    self.delaunay_triangulation(grow=True, n_prc=self.n_prc)
                    self.n_prc = self.C.shape[0]

                if self.disp:
                    print('Triangulation completed, building minimizer pool')

                self.delaunay_minimizers()

            if self.disp:
                logging.info(
                    "Minimizer pool = SHGO.X_min = {}".format(self.X_min))
        else:
            if self.disp:
                print(
                    'Not enough sampling points found in the feasible domain.')
            self.minimizer_pool = [None]
            try:
                self.X_min
            except AttributeError:
                self.X_min = []

    def sampling_custom(self, n, dim):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        # Generate sampling points.
        # Generate uniform sample points in [0, 1]^m \subset R^m
        self.C = self.sampling_function(n, dim)
        # Distribute over bounds
        for i in range(len(self.bounds)):
            self.C[:, i] = (self.C[:, i] *
                            (self.bounds[i][1] - self.bounds[i][0])
                            + self.bounds[i][0])
        return self.C

    def sampling_subspace(self):
        """Find subspace of feasible points from g_func definition"""
        # Subspace of feasible points.
        for ind, g in enumerate(self.g_cons):
            self.C = self.C[g(self.C.T, *self.g_args[ind]) >= 0.0]
            if self.C.size == 0:
                self.res.message = ('No sampling point found within the '
                                    + 'feasible set. Increasing sampling '
                                    + 'size.')
                # sampling correctly for both 1-D and >1-D cases
                if self.disp:
                    print(self.res.message)

    def sorted_samples(self):  # Validated
        """Find indexes of the sorted sampling points"""
        self.Ind_sorted = np.argsort(self.C, axis=0)
        self.Xs = self.C[self.Ind_sorted]
        return self.Ind_sorted, self.Xs

    def ax_subspace(self):  # Validated
        """
        Finds the subspace vectors along each component axis.
        """
        self.Ci = []
        self.Xs_i = []
        self.Ii = []
        for i in range(self.dim):
            self.Ci.append(self.C[:, i])
            self.Ii.append(self.Ind_sorted[:, i])
            self.Xs_i.append(self.Xs[:, i])

    def fun_ref(self):
        """
        Find the objective function output reference table
        """
        # TODO: Replace with cached wrapper

        # Note: This process can be pooled easily
        # Obj. function returns to be used as reference table.:
        f_cache_bool = False
        if self.fn > 0:  # Store old function evaluations
            Ftemp = self.F
            fn_old = self.fn
            f_cache_bool = True

        self.F = np.zeros(np.shape(self.C)[0])
        # NOTE: It might be easier to replace this with a cached
        #      objective function
        for i in range(self.fn, np.shape(self.C)[0]):
            eval_f = True
            if self.g_cons is not None:
                for g in self.g_cons:
                    if g(self.C[i, :], *self.args) < 0.0:
                        eval_f = False
                        break  # Breaks the g loop

            if eval_f:
                self.F[i] = self.func(self.C[i, :], *self.args)
                self.fn += 1
            elif self.infty_cons_sampl:
                self.F[i] = np.inf
                self.fn += 1
        if f_cache_bool:
            if fn_old > 0:  # Restore saved function evaluations
                self.F[0:fn_old] = Ftemp

        return self.F

    def surface_topo_ref(self):  # Validated
        """
        Find the BD and FD finite differences along each component vector.
        """
        # Replace numpy inf, -inf and nan objects with floating point numbers
        # nan --> float
        self.F[np.isnan(self.F)] = np.inf
        # inf, -inf  --> floats
        self.F = np.nan_to_num(self.F)

        self.Ft = self.F[self.Ind_sorted]
        self.Ftp = np.diff(self.Ft, axis=0)  # FD
        self.Ftm = np.diff(self.Ft[::-1], axis=0)[::-1]  # BD

    def sample_topo(self, ind):
        # Find the position of the sample in the component axial directions
        self.Xi_ind_pos = []
        self.Xi_ind_topo_i = []

        for i in range(self.dim):
            for x, I_ind in zip(self.Ii[i], range(len(self.Ii[i]))):
                if x == ind:
                    self.Xi_ind_pos.append(I_ind)

            # Use the topo reference tables to find if point is a minimizer on
            # the current axis

            # First check if index is on the boundary of the sampling points:
            if self.Xi_ind_pos[i] == 0:
                # if boundary is in basin
                self.Xi_ind_topo_i.append(self.Ftp[:, i][0] > 0)

            elif self.Xi_ind_pos[i] == self.fn - 1:
                # Largest value at sample size
                self.Xi_ind_topo_i.append(self.Ftp[:, i][self.fn - 2] < 0)

            # Find axial reference for other points
            else:
                Xi_ind_top_p = self.Ftp[:, i][self.Xi_ind_pos[i]] > 0
                Xi_ind_top_m = self.Ftm[:, i][self.Xi_ind_pos[i] - 1] > 0
                self.Xi_ind_topo_i.append(Xi_ind_top_p and Xi_ind_top_m)

        if np.array(self.Xi_ind_topo_i).all():
            self.Xi_ind_topo = True
        else:
            self.Xi_ind_topo = False
        self.Xi_ind_topo = np.array(self.Xi_ind_topo_i).all()

        return self.Xi_ind_topo

    def minimizers_1D(self):
        """
        Returns the indices of all minimizers
        """
        self.minimizer_pool = []
        # Note: Can implement parallelization here
        for ind in range(self.fn):
            min_bool = self.sample_topo(ind)
            if min_bool:
                self.minimizer_pool.append(ind)

        self.minimizer_pool_F = self.F[self.minimizer_pool]

        # Sort to find minimum func value in min_pool
        self.sort_min_pool()
        if not len(self.minimizer_pool) == 0:
            self.X_min = self.C[self.minimizer_pool]
            # If function is called again and pool is found unbreak:
        else:
            self.X_min = []

        return self.X_min

    def delaunay_triangulation(self, grow=False, n_prc=0):
        if not grow:
            self.Tri = spatial.Delaunay(self.C)
        else:
            if hasattr(self, 'Tri'):
                self.Tri.add_points(self.C[n_prc:, :])
            else:
                self.Tri = spatial.Delaunay(self.C, incremental=True)

        return self.Tri

    @staticmethod
    def find_neighbors_delaunay(pindex, triang):
        """
        Returns the indices of points connected to ``pindex`` on the Gabriel
        chain subgraph of the Delaunay triangulation.
        """
        return triang.vertex_neighbor_vertices[1][
               triang.vertex_neighbor_vertices[0][pindex]:
               triang.vertex_neighbor_vertices[0][pindex + 1]]

    def sample_delaunay_topo(self, ind):
        self.Xi_ind_topo_i = []

        # Find the position of the sample in the component Gabrial chain
        G_ind = self.find_neighbors_delaunay(ind, self.Tri)

        # Find finite deference between each point
        for g_i in G_ind:
            rel_topo_bool = self.F[ind] < self.F[g_i]
            self.Xi_ind_topo_i.append(rel_topo_bool)

        # Check if minimizer
        self.Xi_ind_topo = np.array(self.Xi_ind_topo_i).all()

        return self.Xi_ind_topo

    def delaunay_minimizers(self):
        """
        Returns the indices of all minimizers
        """
        self.minimizer_pool = []
        # Note: Can easily be parralized
        if self.disp:
            logging.info('self.fn = {}'.format(self.fn))
            logging.info('self.nc = {}'.format(self.nc))
            logging.info('np.shape(self.C)'
                         ' = {}'.format(np.shape(self.C)))
        for ind in range(self.fn):
            min_bool = self.sample_delaunay_topo(ind)
            if min_bool:
                self.minimizer_pool.append(ind)

        self.minimizer_pool_F = self.F[self.minimizer_pool]

        # Sort to find minimum func value in min_pool
        self.sort_min_pool()
        if self.disp:
            logging.info('self.minimizer_pool = {}'.format(self.minimizer_pool))
        if not len(self.minimizer_pool) == 0:
            self.X_min = self.C[self.minimizer_pool]
        else:
            self.X_min = []  # Empty pool breaks main routine
        return self.X_min


class LMap:
    def __init__(self, v):
        self.v = v
        self.x_l = None
        self.lres = None
        self.f_min = None
        self.lbounds = []


class LMapCache:
    def __init__(self):
        self.cache = {}

        # Lists for search queries
        self.v_maps = []
        self.xl_maps = []
        self.f_maps = []
        self.lbound_maps = []
        self.size = 0

    def __getitem__(self, v):
        v = np.ndarray.tolist(v)
        v = tuple(v)
        try:
            return self.cache[v]
        except KeyError:
            xval = LMap(v)
            self.cache[v] = xval

            return self.cache[v]

    def add_res(self, v, lres, bounds=None):
        v = np.ndarray.tolist(v)
        v = tuple(v)
        self.cache[v].x_l = lres.x
        self.cache[v].lres = lres
        self.cache[v].f_min = lres.fun
        self.cache[v].lbounds = bounds

        # Update cache size
        self.size += 1

        # Cache lists for search queries
        self.v_maps.append(v)
        self.xl_maps.append(lres.x)
        self.f_maps.append(lres.fun)
        self.lbound_maps.append(bounds)

    def sort_cache_result(self):
        """
        Sort results and build the global return object
        """
        results = {}
        # Sort results and save
        self.xl_maps = np.array(self.xl_maps)
        self.f_maps = np.array(self.f_maps)

        # Sorted indexes in Func_min
        ind_sorted = np.argsort(self.f_maps)

        # Save ordered list of minima
        results['xl'] = self.xl_maps[ind_sorted]  # Ordered x vals
        self.f_maps = np.array(self.f_maps)
        results['funl'] = self.f_maps[ind_sorted]
        results['funl'] = results['funl'].T

        # Find global of all minimizers
        results['x'] = self.xl_maps[ind_sorted[0]]  # Save global minima
        results['fun'] = self.f_maps[ind_sorted[0]]  # Save global fun value

        self.xl_maps = np.ndarray.tolist(self.xl_maps)
        self.f_maps = np.ndarray.tolist(self.f_maps)
        return results
