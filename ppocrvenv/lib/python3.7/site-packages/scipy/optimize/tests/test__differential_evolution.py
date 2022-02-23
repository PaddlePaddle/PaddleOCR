"""
Unit tests for the differential global minimization algorithm.
"""
import multiprocessing
import platform

from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
                                                   _ConstraintWrapper)
from scipy.optimize import differential_evolution
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
                                         LinearConstraint)
from scipy.optimize import rosen
from scipy.sparse import csr_matrix
from scipy._lib._pep440 import Version

import numpy as np
from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_equal,
                           assert_string_equal, assert_, suppress_warnings)
from pytest import raises as assert_raises, warns
import pytest


class TestDifferentialEvolutionSolver:

    def setup_method(self):
        self.old_seterr = np.seterr(invalid='raise')
        self.limits = np.array([[0., 0.],
                                [2., 2.]])
        self.bounds = [(0., 2.), (0., 2.)]

        self.dummy_solver = DifferentialEvolutionSolver(self.quadratic,
                                                        [(0, 100)])

        # dummy_solver2 will be used to test mutation strategies
        self.dummy_solver2 = DifferentialEvolutionSolver(self.quadratic,
                                                         [(0, 1)],
                                                         popsize=7,
                                                         mutation=0.5)
        # create a population that's only 7 members long
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        population = np.atleast_2d(np.arange(0.1, 0.8, 0.1)).T
        self.dummy_solver2.population = population

    def teardown_method(self):
        np.seterr(**self.old_seterr)

    def quadratic(self, x):
        return x[0]**2

    def test__strategy_resolves(self):
        # test that the correct mutation function is resolved by
        # different requested strategy arguments
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best1exp')
        assert_equal(solver.strategy, 'best1exp')
        assert_equal(solver.mutation_func.__name__, '_best1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best1bin')
        assert_equal(solver.strategy, 'best1bin')
        assert_equal(solver.mutation_func.__name__, '_best1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand1bin')
        assert_equal(solver.strategy, 'rand1bin')
        assert_equal(solver.mutation_func.__name__, '_rand1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand1exp')
        assert_equal(solver.strategy, 'rand1exp')
        assert_equal(solver.mutation_func.__name__, '_rand1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand2exp')
        assert_equal(solver.strategy, 'rand2exp')
        assert_equal(solver.mutation_func.__name__, '_rand2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best2bin')
        assert_equal(solver.strategy, 'best2bin')
        assert_equal(solver.mutation_func.__name__, '_best2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand2bin')
        assert_equal(solver.strategy, 'rand2bin')
        assert_equal(solver.mutation_func.__name__, '_rand2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='rand2exp')
        assert_equal(solver.strategy, 'rand2exp')
        assert_equal(solver.mutation_func.__name__, '_rand2')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='randtobest1bin')
        assert_equal(solver.strategy, 'randtobest1bin')
        assert_equal(solver.mutation_func.__name__, '_randtobest1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='randtobest1exp')
        assert_equal(solver.strategy, 'randtobest1exp')
        assert_equal(solver.mutation_func.__name__, '_randtobest1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='currenttobest1bin')
        assert_equal(solver.strategy, 'currenttobest1bin')
        assert_equal(solver.mutation_func.__name__, '_currenttobest1')

        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='currenttobest1exp')
        assert_equal(solver.strategy, 'currenttobest1exp')
        assert_equal(solver.mutation_func.__name__, '_currenttobest1')

    def test__mutate1(self):
        # strategies */1/*, i.e. rand/1/bin, best/1/exp, etc.
        result = np.array([0.05])
        trial = self.dummy_solver2._best1((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

        result = np.array([0.25])
        trial = self.dummy_solver2._rand1((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

    def test__mutate2(self):
        # strategies */2/*, i.e. rand/2/bin, best/2/exp, etc.
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        result = np.array([-0.1])
        trial = self.dummy_solver2._best2((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

        result = np.array([0.1])
        trial = self.dummy_solver2._rand2((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

    def test__randtobest1(self):
        # strategies randtobest/1/*
        result = np.array([0.15])
        trial = self.dummy_solver2._randtobest1((2, 3, 4, 5, 6))
        assert_allclose(trial, result)

    def test__currenttobest1(self):
        # strategies currenttobest/1/*
        result = np.array([0.1])
        trial = self.dummy_solver2._currenttobest1(1, (2, 3, 4, 5, 6))
        assert_allclose(trial, result)

    def test_can_init_with_dithering(self):
        mutation = (0.5, 1)
        solver = DifferentialEvolutionSolver(self.quadratic,
                                             self.bounds,
                                             mutation=mutation)

        assert_equal(solver.dither, list(mutation))

    def test_invalid_mutation_values_arent_accepted(self):
        func = rosen
        mutation = (0.5, 3)
        assert_raises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        mutation = (-1, 1)
        assert_raises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        mutation = (0.1, np.nan)
        assert_raises(ValueError,
                          DifferentialEvolutionSolver,
                          func,
                          self.bounds,
                          mutation=mutation)

        mutation = 0.5
        solver = DifferentialEvolutionSolver(func,
                                             self.bounds,
                                             mutation=mutation)
        assert_equal(0.5, solver.scale)
        assert_equal(None, solver.dither)

    def test__scale_parameters(self):
        trial = np.array([0.3])
        assert_equal(30, self.dummy_solver._scale_parameters(trial))

        # it should also work with the limits reversed
        self.dummy_solver.limits = np.array([[100], [0.]])
        assert_equal(30, self.dummy_solver._scale_parameters(trial))

    def test__unscale_parameters(self):
        trial = np.array([30])
        assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))

        # it should also work with the limits reversed
        self.dummy_solver.limits = np.array([[100], [0.]])
        assert_equal(0.3, self.dummy_solver._unscale_parameters(trial))

    def test__ensure_constraint(self):
        trial = np.array([1.1, -100, 0.9, 2., 300., -0.00001])
        self.dummy_solver._ensure_constraint(trial)

        assert_equal(trial[2], 0.9)
        assert_(np.logical_and(trial >= 0, trial <= 1).all())

    def test_differential_evolution(self):
        # test that the Jmin of DifferentialEvolutionSolver
        # is the same as the function evaluation
        solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)])
        result = solver.solve()
        assert_almost_equal(result.fun, self.quadratic(result.x))

    def test_best_solution_retrieval(self):
        # test that the getter property method for the best solution works.
        solver = DifferentialEvolutionSolver(self.quadratic, [(-2, 2)])
        result = solver.solve()
        assert_almost_equal(result.x, solver.x)

    def test_callback_terminates(self):
        # test that if the callback returns true, then the minimization halts
        bounds = [(0, 2), (0, 2)]
        expected_msg = 'callback function requested stop early by returning True'

        def callback_python_true(param, convergence=0.):
            return True

        result = differential_evolution(rosen, bounds, callback=callback_python_true)
        assert_string_equal(result.message, expected_msg)

        def callback_evaluates_true(param, convergence=0.):
            # DE should stop if bool(self.callback) is True
            return [10]

        result = differential_evolution(rosen, bounds, callback=callback_evaluates_true)
        assert_string_equal(result.message, expected_msg)

        def callback_evaluates_false(param, convergence=0.):
            return []

        result = differential_evolution(rosen, bounds, callback=callback_evaluates_false)
        assert result.success

    def test_args_tuple_is_passed(self):
        # test that the args tuple is passed to the cost function properly.
        bounds = [(-10, 10)]
        args = (1., 2., 3.)

        def quadratic(x, *args):
            if type(args) != tuple:
                raise ValueError('args should be a tuple')
            return args[0] + args[1] * x + args[2] * x**2.

        result = differential_evolution(quadratic,
                                        bounds,
                                        args=args,
                                        polish=True)
        assert_almost_equal(result.fun, 2 / 3.)

    def test_init_with_invalid_strategy(self):
        # test that passing an invalid strategy raises ValueError
        func = rosen
        bounds = [(-3, 3)]
        assert_raises(ValueError,
                          differential_evolution,
                          func,
                          bounds,
                          strategy='abc')

    def test_bounds_checking(self):
        # test that the bounds checking works
        func = rosen
        bounds = [(-3)]
        assert_raises(ValueError,
                          differential_evolution,
                          func,
                          bounds)
        bounds = [(-3, 3), (3, 4, 5)]
        assert_raises(ValueError,
                          differential_evolution,
                          func,
                          bounds)

        # test that we can use a new-type Bounds object
        result = differential_evolution(rosen, Bounds([0, 0], [2, 2]))
        assert_almost_equal(result.x, (1., 1.))

    def test_select_samples(self):
        # select_samples should return 5 separate random numbers.
        limits = np.arange(12., dtype='float64').reshape(2, 6)
        bounds = list(zip(limits[0, :], limits[1, :]))
        solver = DifferentialEvolutionSolver(None, bounds, popsize=1)
        candidate = 0
        r1, r2, r3, r4, r5 = solver._select_samples(candidate, 5)
        assert_equal(
            len(np.unique(np.array([candidate, r1, r2, r3, r4, r5]))), 6)

    def test_maxiter_stops_solve(self):
        # test that if the maximum number of iterations is exceeded
        # the solver stops.
        solver = DifferentialEvolutionSolver(rosen, self.bounds, maxiter=1)
        result = solver.solve()
        assert_equal(result.success, False)
        assert_equal(result.message,
                        'Maximum number of iterations has been exceeded.')

    def test_maxfun_stops_solve(self):
        # test that if the maximum number of function evaluations is exceeded
        # during initialisation the solver stops
        solver = DifferentialEvolutionSolver(rosen, self.bounds, maxfun=1,
                                             polish=False)
        result = solver.solve()

        assert_equal(result.nfev, 2)
        assert_equal(result.success, False)
        assert_equal(result.message,
                     'Maximum number of function evaluations has '
                     'been exceeded.')

        # test that if the maximum number of function evaluations is exceeded
        # during the actual minimisation, then the solver stops.
        # Have to turn polishing off, as this will still occur even if maxfun
        # is reached. For popsize=5 and len(bounds)=2, then there are only 10
        # function evaluations during initialisation.
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             popsize=5,
                                             polish=False,
                                             maxfun=40)
        result = solver.solve()

        assert_equal(result.nfev, 41)
        assert_equal(result.success, False)
        assert_equal(result.message,
                         'Maximum number of function evaluations has '
                              'been exceeded.')

        # now repeat for updating='deferred version
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             popsize=5,
                                             polish=False,
                                             maxfun=40,
                                             updating='deferred')
        result = solver.solve()

        assert_equal(result.nfev, 40)
        assert_equal(result.success, False)
        assert_equal(result.message,
                         'Maximum number of function evaluations has '
                              'been reached.')

    def test_quadratic(self):
        # test the quadratic function from object
        solver = DifferentialEvolutionSolver(self.quadratic,
                                             [(-100, 100)],
                                             tol=0.02)
        solver.solve()
        assert_equal(np.argmin(solver.population_energies), 0)

    def test_quadratic_from_diff_ev(self):
        # test the quadratic function from differential_evolution function
        differential_evolution(self.quadratic,
                               [(-100, 100)],
                               tol=0.02)

    def test_seed_gives_repeatability(self):
        result = differential_evolution(self.quadratic,
                                        [(-100, 100)],
                                        polish=False,
                                        seed=1,
                                        tol=0.5)
        result2 = differential_evolution(self.quadratic,
                                        [(-100, 100)],
                                        polish=False,
                                        seed=1,
                                        tol=0.5)
        assert_equal(result.x, result2.x)
        assert_equal(result.nfev, result2.nfev)

    @pytest.mark.skipif(Version(np.__version__) < Version('1.17'),
                        reason='Generator not available for numpy, < 1.17')
    def test_random_generator(self):
        # check that np.random.Generator can be used (numpy >= 1.17)
        # obtain a np.random.Generator object
        rng = np.random.default_rng()

        inits = ['random', 'latinhypercube', 'sobol', 'halton']
        for init in inits:
            differential_evolution(self.quadratic,
                                   [(-100, 100)],
                                   polish=False,
                                   seed=rng,
                                   tol=0.5,
                                   init=init)

    def test_exp_runs(self):
        # test whether exponential mutation loop runs
        solver = DifferentialEvolutionSolver(rosen,
                                             self.bounds,
                                             strategy='best1exp',
                                             maxiter=1)

        solver.solve()

    def test_gh_4511_regression(self):
        # This modification of the differential evolution docstring example
        # uses a custom popsize that had triggered an off-by-one error.
        # Because we do not care about solving the optimization problem in
        # this test, we use maxiter=1 to reduce the testing time.
        bounds = [(-5, 5), (-5, 5)]
        # result = differential_evolution(rosen, bounds, popsize=1815,
        #                                 maxiter=1)

        # the original issue arose because of rounding error in arange, with
        # linspace being a much better solution. 1815 is quite a large popsize
        # to use and results in a long test time (~13s). I used the original
        # issue to figure out the lowest number of samples that would cause
        # this rounding error to occur, 49.
        differential_evolution(rosen, bounds, popsize=49, maxiter=1)

    def test_calculate_population_energies(self):
        # if popsize is 3, then the overall generation has size (6,)
        solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=3)
        solver._calculate_population_energies(solver.population)
        solver._promote_lowest_energy()
        assert_equal(np.argmin(solver.population_energies), 0)

        # initial calculation of the energies should require 6 nfev.
        assert_equal(solver._nfev, 6)

    def test_iteration(self):
        # test that DifferentialEvolutionSolver is iterable
        # if popsize is 3, then the overall generation has size (6,)
        solver = DifferentialEvolutionSolver(rosen, self.bounds, popsize=3,
                                             maxfun=12)
        x, fun = next(solver)
        assert_equal(np.size(x, 0), 2)

        # 6 nfev are required for initial calculation of energies, 6 nfev are
        # required for the evolution of the 6 population members.
        assert_equal(solver._nfev, 12)

        # the next generation should halt because it exceeds maxfun
        assert_raises(StopIteration, next, solver)

        # check a proper minimisation can be done by an iterable solver
        solver = DifferentialEvolutionSolver(rosen, self.bounds)
        _, fun_prev = next(solver)
        for i, soln in enumerate(solver):
            x_current, fun_current = soln
            assert(fun_prev >= fun_current)
            _, fun_prev = x_current, fun_current
            # need to have this otherwise the solver would never stop.
            if i == 50:
                break

    def test_convergence(self):
        solver = DifferentialEvolutionSolver(rosen, self.bounds, tol=0.2,
                                             polish=False)
        solver.solve()
        assert_(solver.convergence < 0.2)

    def test_maxiter_none_GH5731(self):
        # Pre 0.17 the previous default for maxiter and maxfun was None.
        # the numerical defaults are now 1000 and np.inf. However, some scripts
        # will still supply None for both of those, this will raise a TypeError
        # in the solve method.
        solver = DifferentialEvolutionSolver(rosen, self.bounds, maxiter=None,
                                             maxfun=None)
        solver.solve()

    def test_population_initiation(self):
        # test the different modes of population initiation

        # init must be either 'latinhypercube' or 'random'
        # raising ValueError is something else is passed in
        assert_raises(ValueError,
                      DifferentialEvolutionSolver,
                      *(rosen, self.bounds),
                      **{'init': 'rubbish'})

        solver = DifferentialEvolutionSolver(rosen, self.bounds)

        # check that population initiation:
        # 1) resets _nfev to 0
        # 2) all population energies are np.inf
        solver.init_population_random()
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        solver.init_population_lhs()
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        solver.init_population_qmc(qmc_engine='halton')
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        solver = DifferentialEvolutionSolver(rosen, self.bounds, init='sobol')
        solver.init_population_qmc(qmc_engine='sobol')
        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))

        # we should be able to initialize with our own array
        population = np.linspace(-1, 3, 10).reshape(5, 2)
        solver = DifferentialEvolutionSolver(rosen, self.bounds,
                                             init=population,
                                             strategy='best2bin',
                                             atol=0.01, seed=1, popsize=5)

        assert_equal(solver._nfev, 0)
        assert_(np.all(np.isinf(solver.population_energies)))
        assert_(solver.num_population_members == 5)
        assert_(solver.population_shape == (5, 2))

        # check that the population was initialized correctly
        unscaled_population = np.clip(solver._unscale_parameters(population),
                                      0, 1)
        assert_almost_equal(solver.population[:5], unscaled_population)

        # population values need to be clipped to bounds
        assert_almost_equal(np.min(solver.population[:5]), 0)
        assert_almost_equal(np.max(solver.population[:5]), 1)

        # shouldn't be able to initialize with an array if it's the wrong shape
        # this would have too many parameters
        population = np.linspace(-1, 3, 15).reshape(5, 3)
        assert_raises(ValueError,
                      DifferentialEvolutionSolver,
                      *(rosen, self.bounds),
                      **{'init': population})

        # provide an initial solution
        # bounds are [(0, 2), (0, 2)]
        x0 = np.random.uniform(low=0.0, high=2.0, size=2)
        solver = DifferentialEvolutionSolver(
            rosen, self.bounds, x0=x0
        )
        # parameters are scaled to unit interval
        assert_allclose(solver.population[0], x0 / 2.0)

    def test_x0(self):
        # smoke test that checks that x0 is usable.
        res = differential_evolution(rosen, self.bounds, x0=[0.2, 0.8])
        assert res.success

        # check what happens if some of the x0 lay outside the bounds
        with assert_raises(ValueError):
            differential_evolution(rosen, self.bounds, x0=[0.2, 2.1])

    def test_infinite_objective_function(self):
        # Test that there are no problems if the objective function
        # returns inf on some runs
        def sometimes_inf(x):
            if x[0] < .5:
                return np.inf
            return x[1]
        bounds = [(0, 1), (0, 1)]
        differential_evolution(sometimes_inf, bounds=bounds, disp=False)

    def test_deferred_updating(self):
        # check setting of deferred updating, with default workers
        bounds = [(0., 2.), (0., 2.)]
        solver = DifferentialEvolutionSolver(rosen, bounds, updating='deferred')
        assert_(solver._updating == 'deferred')
        assert_(solver._mapwrapper._mapfunc is map)
        solver.solve()

    def test_immediate_updating(self):
        # check setting of immediate updating, with default workers
        bounds = [(0., 2.), (0., 2.)]
        solver = DifferentialEvolutionSolver(rosen, bounds)
        assert_(solver._updating == 'immediate')

        # should raise a UserWarning because the updating='immediate'
        # is being overridden by the workers keyword
        with warns(UserWarning):
            with DifferentialEvolutionSolver(rosen, bounds, workers=2) as solver:
                pass
        assert_(solver._updating == 'deferred')

    def test_parallel(self):
        # smoke test for parallelization with deferred updating
        bounds = [(0., 2.), (0., 2.)]
        with multiprocessing.Pool(2) as p, DifferentialEvolutionSolver(
                rosen, bounds, updating='deferred', workers=p.map) as solver:
            assert_(solver._mapwrapper.pool is not None)
            assert_(solver._updating == 'deferred')
            solver.solve()

        with DifferentialEvolutionSolver(rosen, bounds, updating='deferred',
                                         workers=2) as solver:
            assert_(solver._mapwrapper.pool is not None)
            assert_(solver._updating == 'deferred')
            solver.solve()

    def test_converged(self):
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)])
        solver.solve()
        assert_(solver.converged())

    def test_constraint_violation_fn(self):
        def constr_f(x):
            return [x[0] + x[1]]

        def constr_f2(x):
            return [x[0]**2 + x[1], x[0] - x[1]]

        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)

        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc))

        cv = solver._constraint_violation_fn([1.0, 1.0])
        assert_almost_equal(cv, 0.1)

        nlc2 = NonlinearConstraint(constr_f2, -np.inf, 1.8)
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc, nlc2))

        # for multiple constraints the constraint violations should
        # be concatenated.
        cv = solver._constraint_violation_fn([1.2, 1.])
        assert_almost_equal(cv, [0.3, 0.64, 0])

        cv = solver._constraint_violation_fn([2., 2.])
        assert_almost_equal(cv, [2.1, 4.2, 0])

        # should accept valid values
        cv = solver._constraint_violation_fn([0.5, 0.5])
        assert_almost_equal(cv, [0., 0., 0.])

    def test_constraint_population_feasibilities(self):
        def constr_f(x):
            return [x[0] + x[1]]

        def constr_f2(x):
            return [x[0]**2 + x[1], x[0] - x[1]]

        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)

        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc))

        # are population feasibilities correct
        # [0.5, 0.5] corresponds to scaled values of [1., 1.]
        feas, cv = solver._calculate_population_feasibilities(
            np.array([[0.5, 0.5], [1., 1.]]))
        assert_equal(feas, [False, False])
        assert_almost_equal(cv, np.array([[0.1], [2.1]]))
        assert cv.shape == (2, 1)

        nlc2 = NonlinearConstraint(constr_f2, -np.inf, 1.8)
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc, nlc2))

        feas, cv = solver._calculate_population_feasibilities(
            np.array([[0.5, 0.5], [0.6, 0.5]]))
        assert_equal(feas, [False, False])
        assert_almost_equal(cv, np.array([[0.1, 0.2, 0], [0.3, 0.64, 0]]))

        feas, cv = solver._calculate_population_feasibilities(
            np.array([[0.5, 0.5], [1., 1.]]))
        assert_equal(feas, [False, False])
        assert_almost_equal(cv, np.array([[0.1, 0.2, 0], [2.1, 4.2, 0]]))
        assert cv.shape == (2, 3)

        feas, cv = solver._calculate_population_feasibilities(
            np.array([[0.25, 0.25], [1., 1.]]))
        assert_equal(feas, [True, False])
        assert_almost_equal(cv, np.array([[0.0, 0.0, 0.], [2.1, 4.2, 0]]))
        assert cv.shape == (2, 3)

    def test_constraint_solve(self):
        def constr_f(x):
            return np.array([x[0] + x[1]])

        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)

        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc))

        # trust-constr warns if the constraint function is linear
        with warns(UserWarning):
            res = solver.solve()

        assert constr_f(res.x) <= 1.9
        assert res.success

    def test_impossible_constraint(self):
        def constr_f(x):
            return np.array([x[0] + x[1]])

        nlc = NonlinearConstraint(constr_f, -np.inf, -1)

        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc), popsize=3,
                                             seed=1)

        # a UserWarning is issued because the 'trust-constr' polishing is
        # attempted on the least infeasible solution found.
        with warns(UserWarning):
            res = solver.solve()

        assert res.maxcv > 0
        assert not res.success

        # test _promote_lowest_energy works when none of the population is
        # feasible. In this case, the solution with the lowest constraint
        # violation should be promoted.
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc), polish=False)
        next(solver)
        assert not solver.feasible.all()
        assert not np.isfinite(solver.population_energies).all()

        # now swap two of the entries in the population
        l = 20
        cv = solver.constraint_violation[0]

        solver.population_energies[[0, l]] = solver.population_energies[[l, 0]]
        solver.population[[0, l], :] = solver.population[[l, 0], :]
        solver.constraint_violation[[0, l], :] = (
            solver.constraint_violation[[l, 0], :])

        solver._promote_lowest_energy()
        assert_equal(solver.constraint_violation[0], cv)

    def test_accept_trial(self):
        # _accept_trial(self, energy_trial, feasible_trial, cv_trial,
        #               energy_orig, feasible_orig, cv_orig)
        def constr_f(x):
            return [x[0] + x[1]]
        nlc = NonlinearConstraint(constr_f, -np.inf, 1.9)
        solver = DifferentialEvolutionSolver(rosen, [(0, 2), (0, 2)],
                                             constraints=(nlc))
        fn = solver._accept_trial
        # both solutions are feasible, select lower energy
        assert fn(0.1, True, np.array([0.]), 1.0, True, np.array([0.]))
        assert (fn(1.0, True, np.array([0.]), 0.1, True, np.array([0.]))
               == False)
        assert fn(0.1, True, np.array([0.]), 0.1, True, np.array([0.]))

        # trial is feasible, original is not
        assert fn(9.9, True, np.array([0.]), 1.0, False, np.array([1.]))

        # trial and original are infeasible
        # cv_trial have to be <= cv_original to be better
        assert (fn(0.1, False, np.array([0.5, 0.5]),
                  1.0, False, np.array([1., 1.0])))
        assert (fn(0.1, False, np.array([0.5, 0.5]),
                  1.0, False, np.array([1., 0.50])))
        assert (fn(1.0, False, np.array([0.5, 0.5]),
                  1.0, False, np.array([1., 0.4])) == False)

    def test_constraint_wrapper(self):
        lb = np.array([0, 20, 30])
        ub = np.array([0.5, np.inf, 70])
        x0 = np.array([1, 2, 3])
        pc = _ConstraintWrapper(Bounds(lb, ub), x0)
        assert (pc.violation(x0) > 0).any()
        assert (pc.violation([0.25, 21, 31]) == 0).all()

        x0 = np.array([1, 2, 3, 4])
        A = np.array([[1, 2, 3, 4], [5, 0, 0, 6], [7, 0, 8, 0]])
        pc = _ConstraintWrapper(LinearConstraint(A, -np.inf, 0), x0)
        assert (pc.violation(x0) > 0).any()
        assert (pc.violation([-10, 2, -10, 4]) == 0).all()

        pc = _ConstraintWrapper(LinearConstraint(csr_matrix(A), -np.inf, 0),
                                x0)
        assert (pc.violation(x0) > 0).any()
        assert (pc.violation([-10, 2, -10, 4]) == 0).all()

        def fun(x):
            return A.dot(x)

        nonlinear = NonlinearConstraint(fun, -np.inf, 0)
        pc = _ConstraintWrapper(nonlinear, [-10, 2, -10, 4])
        assert (pc.violation(x0) > 0).any()
        assert (pc.violation([-10, 2, -10, 4]) == 0).all()

    def test_constraint_wrapper_violation(self):
        def cons_f(x):
            return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])

        nlc = NonlinearConstraint(cons_f, [-1, -0.8500], [2, 2])
        pc = _ConstraintWrapper(nlc, [0.5, 1])
        assert np.size(pc.bounds[0]) == 2

        assert_array_equal(pc.violation([0.5, 1]), [0., 0.])
        assert_almost_equal(pc.violation([0.5, 1.2]), [0., 0.1])
        assert_almost_equal(pc.violation([1.2, 1.2]), [0.64, 0])
        assert_almost_equal(pc.violation([0.1, -1.2]), [0.19, 0])
        assert_almost_equal(pc.violation([0.1, 2]), [0.01, 1.14])

    def test_L1(self):
        # Lampinen ([5]) test problem 1

        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            fun = np.sum(5*x[1:5]) - 5*x[1:5]@x[1:5] - np.sum(x[5:])
            return fun

        A = np.zeros((10, 14))  # 1-indexed to match reference
        A[1, [1, 2, 10, 11]] = 2, 2, 1, 1
        A[2, [1, 10]] = -8, 1
        A[3, [4, 5, 10]] = -2, -1, 1
        A[4, [1, 3, 10, 11]] = 2, 2, 1, 1
        A[5, [2, 11]] = -8, 1
        A[6, [6, 7, 11]] = -2, -1, 1
        A[7, [2, 3, 11, 12]] = 2, 2, 1, 1
        A[8, [3, 12]] = -8, 1
        A[9, [8, 9, 12]] = -2, -1, 1
        A = A[1:, 1:]

        b = np.array([10, 0, 0, 10, 0, 0, 10, 0, 0])

        L = LinearConstraint(A, -np.inf, b)

        bounds = [(0, 1)]*9 + [(0, 100)]*3 + [(0, 1)]

        # using a lower popsize to speed the test up
        res = differential_evolution(f, bounds, strategy='best1bin', seed=1234,
                                     constraints=(L), popsize=2)

        x_opt = (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1)
        f_opt = -15

        assert_allclose(f(x_opt), f_opt)
        assert res.success
        assert_allclose(res.x, x_opt, atol=5e-4)
        assert_allclose(res.fun, f_opt, atol=5e-3)
        assert_(np.all(A@res.x <= b))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

        # now repeat the same solve, using the same overall constraints,
        # but using a sparse matrix for the LinearConstraint instead of an
        # array

        L = LinearConstraint(csr_matrix(A), -np.inf, b)

        # using a lower popsize to speed the test up
        res = differential_evolution(f, bounds, strategy='best1bin', seed=1234,
                                     constraints=(L), popsize=2)

        assert_allclose(f(x_opt), f_opt)
        assert res.success
        assert_allclose(res.x, x_opt, atol=5e-4)
        assert_allclose(res.fun, f_opt, atol=5e-3)
        assert_(np.all(A@res.x <= b))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

        # now repeat the same solve, using the same overall constraints,
        # but specify half the constraints in terms of LinearConstraint,
        # and the other half by NonlinearConstraint
        def c1(x):
            x = np.hstack(([0], x))
            return [2*x[2] + 2*x[3] + x[11] + x[12],
                    -8*x[3] + x[12]]

        def c2(x):
            x = np.hstack(([0], x))
            return -2*x[8] - x[9] + x[12]

        L = LinearConstraint(A[:5, :], -np.inf, b[:5])
        L2 = LinearConstraint(A[5:6, :], -np.inf, b[5:6])
        N = NonlinearConstraint(c1, -np.inf, b[6:8])
        N2 = NonlinearConstraint(c2, -np.inf, b[8:9])
        constraints = (L, N, L2, N2)

        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            res = differential_evolution(f, bounds, strategy='rand1bin',
                                         seed=1234, constraints=constraints,
                                         popsize=2)

        assert_allclose(res.x, x_opt, atol=5e-4)
        assert_allclose(res.fun, f_opt, atol=5e-3)
        assert_(np.all(A@res.x <= b))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def test_L2(self):
        # Lampinen ([5]) test problem 2

        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            fun = ((x[1]-10)**2 + 5*(x[2]-12)**2 + x[3]**4 + 3*(x[4]-11)**2 +
                   10*x[5]**6 + 7*x[6]**2 + x[7]**4 - 4*x[6]*x[7] - 10*x[6] -
                   8*x[7])
            return fun

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [127 - 2*x[1]**2 - 3*x[2]**4 - x[3] - 4*x[4]**2 - 5*x[5],
                    196 - 23*x[1] - x[2]**2 - 6*x[6]**2 + 8*x[7],
                    282 - 7*x[1] - 3*x[2] - 10*x[3]**2 - x[4] + x[5],
                    -4*x[1]**2 - x[2]**2 + 3*x[1]*x[2] - 2*x[3]**2 -
                    5*x[6] + 11*x[7]]

        N = NonlinearConstraint(c1, 0, np.inf)
        bounds = [(-10, 10)]*7
        constraints = (N)

        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            res = differential_evolution(f, bounds, strategy='rand1bin',
                                         seed=1234, constraints=constraints)

        f_opt = 680.6300599487869
        x_opt = (2.330499, 1.951372, -0.4775414, 4.365726,
                 -0.6244870, 1.038131, 1.594227)

        assert_allclose(f(x_opt), f_opt)
        assert_allclose(res.fun, f_opt)
        assert_allclose(res.x, x_opt, atol=1e-5)
        assert res.success
        assert_(np.all(np.array(c1(res.x)) >= 0))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def test_L3(self):
        # Lampinen ([5]) test problem 3

        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            fun = (x[1]**2 + x[2]**2 + x[1]*x[2] - 14*x[1] - 16*x[2] +
                   (x[3]-10)**2 + 4*(x[4]-5)**2 + (x[5]-3)**2 + 2*(x[6]-1)**2 +
                   5*x[7]**2 + 7*(x[8]-11)**2 + 2*(x[9]-10)**2 +
                   (x[10] - 7)**2 + 45
                   )
            return fun  # maximize

        A = np.zeros((4, 11))
        A[1, [1, 2, 7, 8]] = -4, -5, 3, -9
        A[2, [1, 2, 7, 8]] = -10, 8, 17, -2
        A[3, [1, 2, 9, 10]] = 8, -2, -5, 2
        A = A[1:, 1:]
        b = np.array([-105, 0, -12])

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [3*x[1] - 6*x[2] - 12*(x[9]-8)**2 + 7*x[10],
                    -3*(x[1]-2)**2 - 4*(x[2]-3)**2 - 2*x[3]**2 + 7*x[4] + 120,
                    -x[1]**2 - 2*(x[2]-2)**2 + 2*x[1]*x[2] - 14*x[5] + 6*x[6],
                    -5*x[1]**2 - 8*x[2] - (x[3]-6)**2 + 2*x[4] + 40,
                    -0.5*(x[1]-8)**2 - 2*(x[2]-4)**2 - 3*x[5]**2 + x[6] + 30]

        L = LinearConstraint(A, b, np.inf)
        N = NonlinearConstraint(c1, 0, np.inf)
        bounds = [(-10, 10)]*10
        constraints = (L, N)

        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            res = differential_evolution(f, bounds, seed=1234,
                                         constraints=constraints, popsize=3)

        x_opt = (2.171996, 2.363683, 8.773926, 5.095984, 0.9906548,
                 1.430574, 1.321644, 9.828726, 8.280092, 8.375927)
        f_opt = 24.3062091

        assert_allclose(f(x_opt), f_opt, atol=1e-5)
        assert_allclose(res.x, x_opt, atol=1e-6)
        assert_allclose(res.fun, f_opt, atol=1e-5)
        assert res.success
        assert_(np.all(A @ res.x >= b))
        assert_(np.all(np.array(c1(res.x)) >= 0))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def test_L4(self):
        # Lampinen ([5]) test problem 4
        def f(x):
            return np.sum(x[:3])

        A = np.zeros((4, 9))
        A[1, [4, 6]] = 0.0025, 0.0025
        A[2, [5, 7, 4]] = 0.0025, 0.0025, -0.0025
        A[3, [8, 5]] = 0.01, -0.01
        A = A[1:, 1:]
        b = np.array([1, 1, 1])

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [x[1]*x[6] - 833.33252*x[4] - 100*x[1] + 83333.333,
                    x[2]*x[7] - 1250*x[5] - x[2]*x[4] + 1250*x[4],
                    x[3]*x[8] - 1250000 - x[3]*x[5] + 2500*x[5]]

        L = LinearConstraint(A, -np.inf, 1)
        N = NonlinearConstraint(c1, 0, np.inf)

        bounds = [(100, 10000)] + [(1000, 10000)]*2 + [(10, 1000)]*5
        constraints = (L, N)

        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            res = differential_evolution(f, bounds, strategy='rand1bin',
                                     seed=1234, constraints=constraints,
                                     popsize=3)

        f_opt = 7049.248

        x_opt = [579.306692, 1359.97063, 5109.9707, 182.0177, 295.601172,
                217.9823, 286.416528, 395.601172]

        assert_allclose(f(x_opt), f_opt, atol=0.001)
        assert_allclose(res.fun, f_opt, atol=0.001)

        # use higher tol here for 32-bit Windows, see gh-11693
        if (platform.system() == 'Windows' and np.dtype(np.intp).itemsize < 8):
            assert_allclose(res.x, x_opt, rtol=2.4e-6, atol=0.0035)
        else:
            # tolerance determined from macOS + MKL failure, see gh-12701
            assert_allclose(res.x, x_opt, rtol=5e-6, atol=0.0024)

        assert res.success
        assert_(np.all(A @ res.x <= b))
        assert_(np.all(np.array(c1(res.x)) >= 0))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def test_L5(self):
        # Lampinen ([5]) test problem 5

        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            fun = (np.sin(2*np.pi*x[1])**3*np.sin(2*np.pi*x[2]) /
                   (x[1]**3*(x[1]+x[2])))
            return -fun  # maximize

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [x[1]**2 - x[2] + 1,
                    1 - x[1] + (x[2]-4)**2]

        N = NonlinearConstraint(c1, -np.inf, 0)
        bounds = [(0, 10)]*2
        constraints = (N)

        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints)

        x_opt = (1.22797135, 4.24537337)
        f_opt = -0.095825
        assert_allclose(f(x_opt), f_opt, atol=2e-5)
        assert_allclose(res.fun, f_opt, atol=1e-4)
        assert res.success
        assert_(np.all(np.array(c1(res.x)) <= 0))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def test_L6(self):
        # Lampinen ([5]) test problem 6
        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            fun = (x[1]-10)**3 + (x[2] - 20)**3
            return fun

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [(x[1]-5)**2 + (x[2] - 5)**2 - 100,
                    -(x[1]-6)**2 - (x[2] - 5)**2 + 82.81]

        N = NonlinearConstraint(c1, 0, np.inf)
        bounds = [(13, 100), (0, 100)]
        constraints = (N)
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints, tol=1e-7)
        x_opt = (14.095, 0.84296)
        f_opt = -6961.814744

        assert_allclose(f(x_opt), f_opt, atol=1e-6)
        assert_allclose(res.fun, f_opt, atol=0.001)
        assert_allclose(res.x, x_opt, atol=1e-4)
        assert res.success
        assert_(np.all(np.array(c1(res.x)) >= 0))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def test_L7(self):
        # Lampinen ([5]) test problem 7
        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            fun = (5.3578547*x[3]**2 + 0.8356891*x[1]*x[5] +
                   37.293239*x[1] - 40792.141)
            return fun

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [
                    85.334407 + 0.0056858*x[2]*x[5] + 0.0006262*x[1]*x[4] -
                    0.0022053*x[3]*x[5],

                    80.51249 + 0.0071317*x[2]*x[5] + 0.0029955*x[1]*x[2] +
                    0.0021813*x[3]**2,

                    9.300961 + 0.0047026*x[3]*x[5] + 0.0012547*x[1]*x[3] +
                    0.0019085*x[3]*x[4]
                    ]

        N = NonlinearConstraint(c1, [0, 90, 20], [92, 110, 25])

        bounds = [(78, 102), (33, 45)] + [(27, 45)]*3
        constraints = (N)

        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints)

        # using our best solution, rather than Lampinen/Koziel. Koziel solution
        # doesn't satisfy constraints, Lampinen f_opt just plain wrong.
        x_opt = [78.00000686, 33.00000362, 29.99526064, 44.99999971,
                 36.77579979]

        f_opt = -30665.537578

        assert_allclose(f(x_opt), f_opt)
        assert_allclose(res.x, x_opt, atol=1e-3)
        assert_allclose(res.fun, f_opt, atol=1e-3)

        assert res.success
        assert_(np.all(np.array(c1(res.x)) >= np.array([0, 90, 20])))
        assert_(np.all(np.array(c1(res.x)) <= np.array([92, 110, 25])))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    @pytest.mark.slow
    @pytest.mark.xfail(platform.machine() == 'ppc64le',
                       reason="fails on ppc64le")
    def test_L8(self):
        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            fun = 3*x[1] + 0.000001*x[1]**3 + 2*x[2] + 0.000002/3*x[2]**3
            return fun

        A = np.zeros((3, 5))
        A[1, [4, 3]] = 1, -1
        A[2, [3, 4]] = 1, -1
        A = A[1:, 1:]
        b = np.array([-.55, -.55])

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [
                    1000*np.sin(-x[3]-0.25) + 1000*np.sin(-x[4]-0.25) +
                    894.8 - x[1],
                    1000*np.sin(x[3]-0.25) + 1000*np.sin(x[3]-x[4]-0.25) +
                    894.8 - x[2],
                    1000*np.sin(x[4]-0.25) + 1000*np.sin(x[4]-x[3]-0.25) +
                    1294.8
                    ]
        L = LinearConstraint(A, b, np.inf)
        N = NonlinearConstraint(c1, np.full(3, -0.001), np.full(3, 0.001))

        bounds = [(0, 1200)]*2+[(-.55, .55)]*2
        constraints = (L, N)

        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            # original Lampinen test was with rand1bin, but that takes a
            # huge amount of CPU time. Changing strategy to best1bin speeds
            # things up a lot
            res = differential_evolution(f, bounds, strategy='best1bin',
                                         seed=1234, constraints=constraints,
                                         maxiter=5000)

        x_opt = (679.9453, 1026.067, 0.1188764, -0.3962336)
        f_opt = 5126.4981

        assert_allclose(f(x_opt), f_opt, atol=1e-3)
        assert_allclose(res.x[:2], x_opt[:2], atol=2e-3)
        assert_allclose(res.x[2:], x_opt[2:], atol=2e-3)
        assert_allclose(res.fun, f_opt, atol=2e-2)
        assert res.success
        assert_(np.all(A@res.x >= b))
        assert_(np.all(np.array(c1(res.x)) >= -0.001))
        assert_(np.all(np.array(c1(res.x)) <= 0.001))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))

    def test_L9(self):
        # Lampinen ([5]) test problem 9

        def f(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return x[1]**2 + (x[2]-1)**2

        def c1(x):
            x = np.hstack(([0], x))  # 1-indexed to match reference
            return [x[2] - x[1]**2]

        N = NonlinearConstraint(c1, [-.001], [0.001])

        bounds = [(-1, 1)]*2
        constraints = (N)
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234,
                                     constraints=constraints)

        x_opt = [np.sqrt(2)/2, 0.5]
        f_opt = 0.75

        assert_allclose(f(x_opt), f_opt)
        assert_allclose(np.abs(res.x), x_opt, atol=1e-3)
        assert_allclose(res.fun, f_opt, atol=1e-3)
        assert res.success
        assert_(np.all(np.array(c1(res.x)) >= -0.001))
        assert_(np.all(np.array(c1(res.x)) <= 0.001))
        assert_(np.all(res.x >= np.array(bounds)[:, 0]))
        assert_(np.all(res.x <= np.array(bounds)[:, 1]))
