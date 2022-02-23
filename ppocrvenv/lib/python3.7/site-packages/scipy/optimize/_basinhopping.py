"""
basinhopping: The basinhopping global optimization algorithm
"""
import numpy as np
import math
from numpy import cos, sin
import scipy.optimize
from scipy._lib._util import check_random_state

__all__ = ['basinhopping']


class Storage:
    """
    Class used to store the lowest energy structure
    """
    def __init__(self, minres):
        self._add(minres)

    def _add(self, minres):
        self.minres = minres
        self.minres.x = np.copy(minres.x)

    def update(self, minres):
        if minres.fun < self.minres.fun:
            self._add(minres)
            return True
        else:
            return False

    def get_lowest(self):
        return self.minres


class BasinHoppingRunner:
    """This class implements the core of the basinhopping algorithm.

    x0 : ndarray
        The starting coordinates.
    minimizer : callable
        The local minimizer, with signature ``result = minimizer(x)``.
        The return value is an `optimize.OptimizeResult` object.
    step_taking : callable
        This function displaces the coordinates randomly. Signature should
        be ``x_new = step_taking(x)``. Note that `x` may be modified in-place.
    accept_tests : list of callables
        Each test is passed the kwargs `f_new`, `x_new`, `f_old` and
        `x_old`. These tests will be used to judge whether or not to accept
        the step. The acceptable return values are True, False, or ``"force
        accept"``. If any of the tests return False then the step is rejected.
        If ``"force accept"``, then this will override any other tests in
        order to accept the step. This can be used, for example, to forcefully
        escape from a local minimum that ``basinhopping`` is trapped in.
    disp : bool, optional
        Display status messages.

    """
    def __init__(self, x0, minimizer, step_taking, accept_tests, disp=False):
        self.x = np.copy(x0)
        self.minimizer = minimizer
        self.step_taking = step_taking
        self.accept_tests = accept_tests
        self.disp = disp

        self.nstep = 0

        # initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.minimization_failures = 0

        # do initial minimization
        minres = minimizer(self.x)
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: basinhopping: local minimization failure")
        self.x = np.copy(minres.x)
        self.energy = minres.fun
        if self.disp:
            print("basinhopping step %d: f %g" % (self.nstep, self.energy))

        # initialize storage class
        self.storage = Storage(minres)

        if hasattr(minres, "nfev"):
            self.res.nfev = minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev = minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev = minres.nhev

    def _monte_carlo_step(self):
        """Do one Monte Carlo iteration

        Randomly displace the coordinates, minimize, and decide whether
        or not to accept the new coordinates.
        """
        # Take a random step.  Make a copy of x because the step_taking
        # algorithm might change x in place
        x_after_step = np.copy(self.x)
        x_after_step = self.step_taking(x_after_step)

        # do a local minimization
        minres = self.minimizer(x_after_step)
        x_after_quench = minres.x
        energy_after_quench = minres.fun
        if not minres.success:
            self.res.minimization_failures += 1
            if self.disp:
                print("warning: basinhopping: local minimization failure")

        if hasattr(minres, "nfev"):
            self.res.nfev += minres.nfev
        if hasattr(minres, "njev"):
            self.res.njev += minres.njev
        if hasattr(minres, "nhev"):
            self.res.nhev += minres.nhev

        # accept the move based on self.accept_tests. If any test is False,
        # then reject the step.  If any test returns the special string
        # 'force accept', then accept the step regardless. This can be used
        # to forcefully escape from a local minimum if normal basin hopping
        # steps are not sufficient.
        accept = True
        for test in self.accept_tests:
            testres = test(f_new=energy_after_quench, x_new=x_after_quench,
                           f_old=self.energy, x_old=self.x)
            if testres == 'force accept':
                accept = True
                break
            elif testres is None:
                raise ValueError("accept_tests must return True, False, or "
                                 "'force accept'")
            elif not testres:
                accept = False

        # Report the result of the acceptance test to the take step class.
        # This is for adaptive step taking
        if hasattr(self.step_taking, "report"):
            self.step_taking.report(accept, f_new=energy_after_quench,
                                    x_new=x_after_quench, f_old=self.energy,
                                    x_old=self.x)

        return accept, minres

    def one_cycle(self):
        """Do one cycle of the basinhopping algorithm
        """
        self.nstep += 1
        new_global_min = False

        accept, minres = self._monte_carlo_step()

        if accept:
            self.energy = minres.fun
            self.x = np.copy(minres.x)
            new_global_min = self.storage.update(minres)

        # print some information
        if self.disp:
            self.print_report(minres.fun, accept)
            if new_global_min:
                print("found new global minimum on step %d with function"
                      " value %g" % (self.nstep, self.energy))

        # save some variables as BasinHoppingRunner attributes
        self.xtrial = minres.x
        self.energy_trial = minres.fun
        self.accept = accept

        return new_global_min

    def print_report(self, energy_trial, accept):
        """print a status update"""
        minres = self.storage.get_lowest()
        print("basinhopping step %d: f %g trial_f %g accepted %d "
              " lowest_f %g" % (self.nstep, self.energy, energy_trial,
                                accept, minres.fun))


class AdaptiveStepsize:
    """
    Class to implement adaptive stepsize.

    This class wraps the step taking class and modifies the stepsize to
    ensure the true acceptance rate is as close as possible to the target.

    Parameters
    ----------
    takestep : callable
        The step taking routine.  Must contain modifiable attribute
        takestep.stepsize
    accept_rate : float, optional
        The target step acceptance rate
    interval : int, optional
        Interval for how often to update the stepsize
    factor : float, optional
        The step size is multiplied or divided by this factor upon each
        update.
    verbose : bool, optional
        Print information about each update

    """
    def __init__(self, takestep, accept_rate=0.5, interval=50, factor=0.9,
                 verbose=True):
        self.takestep = takestep
        self.target_accept_rate = accept_rate
        self.interval = interval
        self.factor = factor
        self.verbose = verbose

        self.nstep = 0
        self.nstep_tot = 0
        self.naccept = 0

    def __call__(self, x):
        return self.take_step(x)

    def _adjust_step_size(self):
        old_stepsize = self.takestep.stepsize
        accept_rate = float(self.naccept) / self.nstep
        if accept_rate > self.target_accept_rate:
            # We're accepting too many steps. This generally means we're
            # trapped in a basin. Take bigger steps.
            self.takestep.stepsize /= self.factor
        else:
            # We're not accepting enough steps. Take smaller steps.
            self.takestep.stepsize *= self.factor
        if self.verbose:
            print("adaptive stepsize: acceptance rate %f target %f new "
                  "stepsize %g old stepsize %g" % (accept_rate,
                  self.target_accept_rate, self.takestep.stepsize,
                  old_stepsize))

    def take_step(self, x):
        self.nstep += 1
        self.nstep_tot += 1
        if self.nstep % self.interval == 0:
            self._adjust_step_size()
        return self.takestep(x)

    def report(self, accept, **kwargs):
        "called by basinhopping to report the result of the step"
        if accept:
            self.naccept += 1


class RandomDisplacement:
    """Add a random displacement of maximum size `stepsize` to each coordinate.

    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    """

    def __init__(self, stepsize=0.5, random_gen=None):
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)

    def __call__(self, x):
        x += self.random_gen.uniform(-self.stepsize, self.stepsize,
                                     np.shape(x))
        return x


class MinimizerWrapper:
    """
    wrap a minimizer function as a minimizer class
    """
    def __init__(self, minimizer, func=None, **kwargs):
        self.minimizer = minimizer
        self.func = func
        self.kwargs = kwargs

    def __call__(self, x0):
        if self.func is None:
            return self.minimizer(x0, **self.kwargs)
        else:
            return self.minimizer(self.func, x0, **self.kwargs)


class Metropolis:
    """Metropolis acceptance criterion.

    Parameters
    ----------
    T : float
        The "temperature" parameter for the accept or reject criterion.
    random_gen : {None, int, `numpy.random.Generator`,
                  `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Random number generator used for acceptance test.

    """

    def __init__(self, T, random_gen=None):
        # Avoid ZeroDivisionError since "MBH can be regarded as a special case
        # of the BH framework with the Metropolis criterion, where temperature
        # T = 0." (Reject all steps that increase energy.)
        self.beta = 1.0 / T if T != 0 else float('inf')
        self.random_gen = check_random_state(random_gen)

    def accept_reject(self, energy_new, energy_old):
        """
        If new energy is lower than old, it will always be accepted.
        If new is higher than old, there is a chance it will be accepted,
        less likely for larger differences.
        """
        with np.errstate(invalid='ignore'):
            # The energy values being fed to Metropolis are 1-length arrays, and if
            # they are equal, their difference is 0, which gets multiplied by beta,
            # which is inf, and array([0]) * float('inf') causes
            #
            # RuntimeWarning: invalid value encountered in multiply
            #
            # Ignore this warning so so when the algorithm is on a flat plane, it always
            # accepts the step, to try to move off the plane.
            prod = -(energy_new - energy_old) * self.beta
            w = math.exp(min(0, prod))

        rand = self.random_gen.uniform()
        return w >= rand

    def __call__(self, **kwargs):
        """
        f_new and f_old are mandatory in kwargs
        """
        return bool(self.accept_reject(kwargs["f_new"],
                    kwargs["f_old"]))


def basinhopping(func, x0, niter=100, T=1.0, stepsize=0.5,
                 minimizer_kwargs=None, take_step=None, accept_test=None,
                 callback=None, interval=50, disp=False, niter_success=None,
                 seed=None):
    """Find the global minimum of a function using the basin-hopping algorithm.

    Basin-hopping is a two-phase method that combines a global stepping
    algorithm with local minimization at each step. Designed to mimic
    the natural process of energy minimization of clusters of atoms, it works
    well for similar problems with "funnel-like, but rugged" energy landscapes
    [5]_.

    As the step-taking, step acceptance, and minimization methods are all
    customizable, this function can also be used to implement other two-phase
    methods.

    Parameters
    ----------
    func : callable ``f(x, *args)``
        Function to be optimized.  ``args`` can be passed as an optional item
        in the dict ``minimizer_kwargs``
    x0 : array_like
        Initial guess.
    niter : integer, optional
        The number of basin-hopping iterations. There will be a total of
        ``niter + 1`` runs of the local minimizer.
    T : float, optional
        The "temperature" parameter for the accept or reject criterion. Higher
        "temperatures" mean that larger jumps in function value will be
        accepted.  For best results ``T`` should be comparable to the
        separation (in function value) between local minima.
    stepsize : float, optional
        Maximum step size for use in the random displacement.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        ``scipy.optimize.minimize()`` Some important options could be:

            method : str
                The minimization method (e.g. ``"L-BFGS-B"``)
            args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).

    take_step : callable ``take_step(x)``, optional
        Replace the default step-taking routine with this routine. The default
        step-taking routine is a random displacement of the coordinates, but
        other step-taking algorithms may be better for some systems.
        ``take_step`` can optionally have the attribute ``take_step.stepsize``.
        If this attribute exists, then ``basinhopping`` will adjust
        ``take_step.stepsize`` in order to try to optimize the global minimum
        search.
    accept_test : callable, ``accept_test(f_new=f_new, x_new=x_new, f_old=fold, x_old=x_old)``, optional
        Define a test which will be used to judge whether or not to accept the
        step.  This will be used in addition to the Metropolis test based on
        "temperature" ``T``.  The acceptable return values are True,
        False, or ``"force accept"``. If any of the tests return False
        then the step is rejected. If the latter, then this will override any
        other tests in order to accept the step. This can be used, for example,
        to forcefully escape from a local minimum that ``basinhopping`` is
        trapped in.
    callback : callable, ``callback(x, f, accept)``, optional
        A callback function which will be called for all minima found. ``x``
        and ``f`` are the coordinates and function value of the trial minimum,
        and ``accept`` is whether or not that minimum was accepted. This can
        be used, for example, to save the lowest N minima found. Also,
        ``callback`` can be used to specify a user defined stop criterion by
        optionally returning True to stop the ``basinhopping`` routine.
    interval : integer, optional
        interval for how often to update the ``stepsize``
    disp : bool, optional
        Set to True to print status messages
    niter_success : integer, optional
        Stop the run if the global minimum candidate remains the same for this
        number of iterations.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations. The random numbers
        generated with this seed only affect the default Metropolis
        `accept_test` and the default `take_step`. If you supply your own
        `take_step` and `accept_test`, and these functions use random
        number generation, then those functions are responsible for the state
        of their random number generator.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination. The ``OptimizeResult`` object returned by the
        selected minimizer at the lowest minimum is also contained within this
        object and can be accessed through the ``lowest_optimization_result``
        attribute.  See `OptimizeResult` for a description of other attributes.

    See Also
    --------
    minimize :
        The local minimization function called once for each basinhopping step.
        ``minimizer_kwargs`` is passed to this routine.

    Notes
    -----
    Basin-hopping is a stochastic algorithm which attempts to find the global
    minimum of a smooth scalar function of one or more variables [1]_ [2]_ [3]_
    [4]_. The algorithm in its current form was described by David Wales and
    Jonathan Doye [2]_ http://www-wales.ch.cam.ac.uk/.

    The algorithm is iterative with each cycle composed of the following
    features

    1) random perturbation of the coordinates

    2) local minimization

    3) accept or reject the new coordinates based on the minimized function
       value

    The acceptance test used here is the Metropolis criterion of standard Monte
    Carlo algorithms, although there are many other possibilities [3]_.

    This global minimization method has been shown to be extremely efficient
    for a wide variety of problems in physics and chemistry. It is
    particularly useful when the function has many minima separated by large
    barriers. See the Cambridge Cluster Database
    http://www-wales.ch.cam.ac.uk/CCD.html for databases of molecular systems
    that have been optimized primarily using basin-hopping. This database
    includes minimization problems exceeding 300 degrees of freedom.

    See the free software program GMIN (http://www-wales.ch.cam.ac.uk/GMIN) for
    a Fortran implementation of basin-hopping. This implementation has many
    different variations of the procedure described above, including more
    advanced step taking algorithms and alternate acceptance criterion.

    For stochastic global optimization there is no way to determine if the true
    global minimum has actually been found. Instead, as a consistency check,
    the algorithm can be run from a number of different random starting points
    to ensure the lowest minimum found in each example has converged to the
    global minimum. For this reason, ``basinhopping`` will by default simply
    run for the number of iterations ``niter`` and return the lowest minimum
    found. It is left to the user to ensure that this is in fact the global
    minimum.

    Choosing ``stepsize``:  This is a crucial parameter in ``basinhopping`` and
    depends on the problem being solved. The step is chosen uniformly in the
    region from x0-stepsize to x0+stepsize, in each dimension. Ideally, it
    should be comparable to the typical separation (in argument values) between
    local minima of the function being optimized. ``basinhopping`` will, by
    default, adjust ``stepsize`` to find an optimal value, but this may take
    many iterations. You will get quicker results if you set a sensible
    initial value for ``stepsize``.

    Choosing ``T``: The parameter ``T`` is the "temperature" used in the
    Metropolis criterion. Basinhopping steps are always accepted if
    ``func(xnew) < func(xold)``. Otherwise, they are accepted with
    probability::

        exp( -(func(xnew) - func(xold)) / T )

    So, for best results, ``T`` should to be comparable to the typical
    difference (in function values) between local minima. (The height of
    "walls" between local minima is irrelevant.)

    If ``T`` is 0, the algorithm becomes Monotonic Basin-Hopping, in which all
    steps that increase energy are rejected.

    .. versionadded:: 0.12.0

    References
    ----------
    .. [1] Wales, David J. 2003, Energy Landscapes, Cambridge University Press,
        Cambridge, UK.
    .. [2] Wales, D J, and Doye J P K, Global Optimization by Basin-Hopping and
        the Lowest Energy Structures of Lennard-Jones Clusters Containing up to
        110 Atoms.  Journal of Physical Chemistry A, 1997, 101, 5111.
    .. [3] Li, Z. and Scheraga, H. A., Monte Carlo-minimization approach to the
        multiple-minima problem in protein folding, Proc. Natl. Acad. Sci. USA,
        1987, 84, 6611.
    .. [4] Wales, D. J. and Scheraga, H. A., Global optimization of clusters,
        crystals, and biomolecules, Science, 1999, 285, 1368.
    .. [5] Olson, B., Hashmi, I., Molloy, K., and Shehu1, A., Basin Hopping as
        a General and Versatile Optimization Framework for the Characterization
        of Biological Macromolecules, Advances in Artificial Intelligence,
        Volume 2012 (2012), Article ID 674832, :doi:`10.1155/2012/674832`

    Examples
    --------
    The following example is a 1-D minimization problem, with many
    local minima superimposed on a parabola.

    >>> from scipy.optimize import basinhopping
    >>> func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    >>> x0=[1.]

    Basinhopping, internally, uses a local minimization algorithm. We will use
    the parameter ``minimizer_kwargs`` to tell basinhopping which algorithm to
    use and how to set up that minimizer. This parameter will be passed to
    ``scipy.optimize.minimize()``.

    >>> minimizer_kwargs = {"method": "BFGS"}
    >>> ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200)
    >>> print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
    global minimum: x = -0.1951, f(x0) = -1.0009

    Next consider a 2-D minimization problem. Also, this time, we
    will use gradient information to significantly speed up the search.

    >>> def func2d(x):
    ...     f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] +
    ...                                                            0.2) * x[0]
    ...     df = np.zeros(2)
    ...     df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
    ...     df[1] = 2. * x[1] + 0.2
    ...     return f, df

    We'll also use a different local minimization algorithm. Also, we must tell
    the minimizer that our function returns both energy and gradient (Jacobian).

    >>> minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
    >>> x0 = [1.0, 1.0]
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200)
    >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
    ...                                                           ret.x[1],
    ...                                                           ret.fun))
    global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109


    Here is an example using a custom step-taking routine. Imagine you want
    the first coordinate to take larger steps than the rest of the coordinates.
    This can be implemented like so:

    >>> class MyTakeStep:
    ...    def __init__(self, stepsize=0.5):
    ...        self.stepsize = stepsize
    ...        self.rng = np.random.default_rng()
    ...    def __call__(self, x):
    ...        s = self.stepsize
    ...        x[0] += self.rng.uniform(-2.*s, 2.*s)
    ...        x[1:] += self.rng.uniform(-s, s, x[1:].shape)
    ...        return x

    Since ``MyTakeStep.stepsize`` exists basinhopping will adjust the magnitude
    of ``stepsize`` to optimize the search. We'll use the same 2-D function as
    before

    >>> mytakestep = MyTakeStep()
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=200, take_step=mytakestep)
    >>> print("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0],
    ...                                                           ret.x[1],
    ...                                                           ret.fun))
    global minimum: x = [-0.1951, -0.1000], f(x0) = -1.0109


    Now, let's do an example using a custom callback function which prints the
    value of every minimum found

    >>> def print_fun(x, f, accepted):
    ...         print("at minimum %.4f accepted %d" % (f, int(accepted)))

    We'll run it for only 10 basinhopping steps this time.

    >>> rng = np.random.default_rng()
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=10, callback=print_fun, seed=rng)
    at minimum 0.4159 accepted 1
    at minimum -0.4317 accepted 1
    at minimum -1.0109 accepted 1
    at minimum -0.9073 accepted 1
    at minimum -0.4317 accepted 0
    at minimum -0.1021 accepted 1
    at minimum -0.7425 accepted 1
    at minimum -0.9073 accepted 1
    at minimum -0.4317 accepted 0
    at minimum -0.7425 accepted 1
    at minimum -0.9073 accepted 1


    The minimum at -1.0109 is actually the global minimum, found already on the
    8th iteration.

    Now let's implement bounds on the problem using a custom ``accept_test``:

    >>> class MyBounds:
    ...     def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
    ...         self.xmax = np.array(xmax)
    ...         self.xmin = np.array(xmin)
    ...     def __call__(self, **kwargs):
    ...         x = kwargs["x_new"]
    ...         tmax = bool(np.all(x <= self.xmax))
    ...         tmin = bool(np.all(x >= self.xmin))
    ...         return tmax and tmin

    >>> mybounds = MyBounds()
    >>> ret = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs,
    ...                    niter=10, accept_test=mybounds)

    """
    x0 = np.array(x0)

    # set up the np.random generator
    rng = check_random_state(seed)

    # set up minimizer
    if minimizer_kwargs is None:
        minimizer_kwargs = dict()
    wrapped_minimizer = MinimizerWrapper(scipy.optimize.minimize, func,
                                         **minimizer_kwargs)

    # set up step-taking algorithm
    if take_step is not None:
        if not callable(take_step):
            raise TypeError("take_step must be callable")
        # if take_step.stepsize exists then use AdaptiveStepsize to control
        # take_step.stepsize
        if hasattr(take_step, "stepsize"):
            take_step_wrapped = AdaptiveStepsize(take_step, interval=interval,
                                                 verbose=disp)
        else:
            take_step_wrapped = take_step
    else:
        # use default
        displace = RandomDisplacement(stepsize=stepsize, random_gen=rng)
        take_step_wrapped = AdaptiveStepsize(displace, interval=interval,
                                             verbose=disp)

    # set up accept tests
    accept_tests = []
    if accept_test is not None:
        if not callable(accept_test):
            raise TypeError("accept_test must be callable")
        accept_tests = [accept_test]

    # use default
    metropolis = Metropolis(T, random_gen=rng)
    accept_tests.append(metropolis)

    if niter_success is None:
        niter_success = niter + 2

    bh = BasinHoppingRunner(x0, wrapped_minimizer, take_step_wrapped,
                            accept_tests, disp=disp)

    # The wrapped minimizer is called once during construction of
    # BasinHoppingRunner, so run the callback
    if callable(callback):
        callback(bh.storage.minres.x, bh.storage.minres.fun, True)

    # start main iteration loop
    count, i = 0, 0
    message = ["requested number of basinhopping iterations completed"
               " successfully"]
    for i in range(niter):
        new_global_min = bh.one_cycle()

        if callable(callback):
            # should we pass a copy of x?
            val = callback(bh.xtrial, bh.energy_trial, bh.accept)
            if val is not None:
                if val:
                    message = ["callback function requested stop early by"
                               "returning True"]
                    break

        count += 1
        if new_global_min:
            count = 0
        elif count > niter_success:
            message = ["success condition satisfied"]
            break

    # prepare return object
    res = bh.res
    res.lowest_optimization_result = bh.storage.get_lowest()
    res.x = np.copy(res.lowest_optimization_result.x)
    res.fun = res.lowest_optimization_result.fun
    res.message = message
    res.nit = i + 1
    return res


def _test_func2d_nograd(x):
    f = (cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
         + 1.010876184442655)
    return f


def _test_func2d(x):
    f = (cos(14.5 * x[0] - 0.3) + (x[0] + 0.2) * x[0] + cos(14.5 * x[1] -
         0.3) + (x[1] + 0.2) * x[1] + x[0] * x[1] + 1.963879482144252)
    df = np.zeros(2)
    df[0] = -14.5 * sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2 + x[1]
    df[1] = -14.5 * sin(14.5 * x[1] - 0.3) + 2. * x[1] + 0.2 + x[0]
    return f, df


if __name__ == "__main__":
    print("\n\nminimize a 2-D function without gradient")
    # minimum expected at ~[-0.195, -0.1]
    kwargs = {"method": "L-BFGS-B"}
    x0 = np.array([1.0, 1.])
    scipy.optimize.minimize(_test_func2d_nograd, x0, **kwargs)
    ret = basinhopping(_test_func2d_nograd, x0, minimizer_kwargs=kwargs,
                       niter=200, disp=False)
    print("minimum expected at  func([-0.195, -0.1]) = 0.0")
    print(ret)

    print("\n\ntry a harder 2-D problem")
    kwargs = {"method": "L-BFGS-B", "jac": True}
    x0 = np.array([1.0, 1.0])
    ret = basinhopping(_test_func2d, x0, minimizer_kwargs=kwargs, niter=200,
                       disp=False)
    print("minimum expected at ~, func([-0.19415263, -0.19415263]) = 0")
    print(ret)
