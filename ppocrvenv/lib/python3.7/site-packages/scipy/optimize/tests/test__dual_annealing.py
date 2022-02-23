# Dual annealing unit tests implementation.
# Copyright (c) 2018 Sylvain Gubian <sylvain.gubian@pmi.com>,
# Yang Xiang <yang.xiang@pmi.com>
# Author: Sylvain Gubian, PMP S.A.
"""
Unit tests for the dual annealing global optimizer
"""
from scipy.optimize import dual_annealing
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
from scipy._lib._util import check_random_state
from scipy._lib._pep440 import Version


class TestDualAnnealing:

    def setup_method(self):
        # A function that returns always infinity for initialization tests
        self.weirdfunc = lambda x: np.inf
        # 2-D bounds for testing function
        self.ld_bounds = [(-5.12, 5.12)] * 2
        # 4-D bounds for testing function
        self.hd_bounds = self.ld_bounds * 4
        # Number of values to be generated for testing visit function
        self.nbtestvalues = 5000
        self.high_temperature = 5230
        self.low_temperature = 0.1
        self.qv = 2.62
        self.seed = 1234
        self.rs = check_random_state(self.seed)
        self.nb_fun_call = 0
        self.ngev = 0

    def callback(self, x, f, context):
        # For testing callback mechanism. Should stop for e <= 1 as
        # the callback function returns True
        if f <= 1.0:
            return True

    def func(self, x, args=()):
        # Using Rastrigin function for performing tests
        if args:
            shift = args
        else:
            shift = 0
        y = np.sum((x - shift) ** 2 - 10 * np.cos(2 * np.pi * (
            x - shift))) + 10 * np.size(x) + shift
        self.nb_fun_call += 1
        return y

    def rosen_der_wrapper(self, x, args=()):
        self.ngev += 1
        return rosen_der(x, *args)

    # FIXME: there are some discontinuities in behaviour as a function of `qv`,
    #        this needs investigating - see gh-12384
    @pytest.mark.parametrize('qv', [1.1, 1.41, 2, 2.62, 2.9])
    def test_visiting_stepping(self, qv):
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        dim = lower.size
        vd = VisitingDistribution(lower, upper, qv, self.rs)
        values = np.zeros(dim)
        x_step_low = vd.visiting(values, 0, self.high_temperature)
        # Make sure that only the first component is changed
        assert_equal(np.not_equal(x_step_low, 0), True)
        values = np.zeros(dim)
        x_step_high = vd.visiting(values, dim, self.high_temperature)
        # Make sure that component other than at dim has changed
        assert_equal(np.not_equal(x_step_high[0], 0), True)

    @pytest.mark.parametrize('qv', [2.25, 2.62, 2.9])
    def test_visiting_dist_high_temperature(self, qv):
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        vd = VisitingDistribution(lower, upper, qv, self.rs)
        # values = np.zeros(self.nbtestvalues)
        # for i in np.arange(self.nbtestvalues):
        #     values[i] = vd.visit_fn(self.high_temperature)
        values = vd.visit_fn(self.high_temperature, self.nbtestvalues)

        # Visiting distribution is a distorted version of Cauchy-Lorentz
        # distribution, and as no 1st and higher moments (no mean defined,
        # no variance defined).
        # Check that big tails values are generated
        assert_array_less(np.min(values), 1e-10)
        assert_array_less(1e+10, np.max(values))

    def test_reset(self):
        owf = ObjectiveFunWrapper(self.weirdfunc)
        lu = list(zip(*self.ld_bounds))
        lower = np.array(lu[0])
        upper = np.array(lu[1])
        es = EnergyState(lower, upper)
        assert_raises(ValueError, es.reset, owf, check_random_state(None))

    def test_low_dim(self):
        ret = dual_annealing(
            self.func, self.ld_bounds, seed=self.seed)
        assert_allclose(ret.fun, 0., atol=1e-12)
        assert ret.success

    def test_high_dim(self):
        ret = dual_annealing(self.func, self.hd_bounds, seed=self.seed)
        assert_allclose(ret.fun, 0., atol=1e-12)
        assert ret.success

    def test_low_dim_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             no_local_search=True, seed=self.seed)
        assert_allclose(ret.fun, 0., atol=1e-4)

    def test_high_dim_no_ls(self):
        ret = dual_annealing(self.func, self.hd_bounds,
                             no_local_search=True, seed=self.seed)
        assert_allclose(ret.fun, 0., atol=1e-4)

    def test_nb_fun_call(self):
        ret = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        assert_equal(self.nb_fun_call, ret.nfev)

    def test_nb_fun_call_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             no_local_search=True, seed=self.seed)
        assert_equal(self.nb_fun_call, ret.nfev)

    def test_max_reinit(self):
        assert_raises(ValueError, dual_annealing, self.weirdfunc,
                      self.ld_bounds)

    def test_reproduce(self):
        res1 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        res2 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        res3 = dual_annealing(self.func, self.ld_bounds, seed=self.seed)
        # If we have reproducible results, x components found has to
        # be exactly the same, which is not the case with no seeding
        assert_equal(res1.x, res2.x)
        assert_equal(res1.x, res3.x)

    @pytest.mark.skipif(Version(np.__version__) < Version('1.17'),
                        reason='Generator not available for numpy, < 1.17')
    def test_rand_gen(self):
        # check that np.random.Generator can be used (numpy >= 1.17)
        # obtain a np.random.Generator object
        rng = np.random.default_rng(1)

        res1 = dual_annealing(self.func, self.ld_bounds, seed=rng)
        # seed again
        rng = np.random.default_rng(1)
        res2 = dual_annealing(self.func, self.ld_bounds, seed=rng)
        # If we have reproducible results, x components found has to
        # be exactly the same, which is not the case with no seeding
        assert_equal(res1.x, res2.x)

    def test_bounds_integrity(self):
        wrong_bounds = [(-5.12, 5.12), (1, 0), (5.12, 5.12)]
        assert_raises(ValueError, dual_annealing, self.func,
                      wrong_bounds)

    def test_bound_validity(self):
        invalid_bounds = [(-5, 5), (-np.inf, 0), (-5, 5)]
        assert_raises(ValueError, dual_annealing, self.func,
                      invalid_bounds)
        invalid_bounds = [(-5, 5), (0, np.inf), (-5, 5)]
        assert_raises(ValueError, dual_annealing, self.func,
                      invalid_bounds)
        invalid_bounds = [(-5, 5), (0, np.nan), (-5, 5)]
        assert_raises(ValueError, dual_annealing, self.func,
                      invalid_bounds)

    def test_local_search_option_bounds(self):
        func = lambda x: np.sum((x-5) * (x-1))
        bounds = list(zip([-6, -5], [6, 5]))
        # Test bounds can be passed (see gh-10831)

        with np.testing.suppress_warnings() as sup:
            sup.record(RuntimeWarning, "Values in x were outside bounds ")

            dual_annealing(
                func,
                bounds=bounds,
                local_search_options={"method": "SLSQP", "bounds": bounds})

        with np.testing.suppress_warnings() as sup:
            sup.record(RuntimeWarning, "Method CG cannot handle ")

            dual_annealing(
                func,
                bounds=bounds,
                local_search_options={"method": "CG", "bounds": bounds})

            # Verify warning happened for Method cannot handle bounds.
            assert sup.log

    def test_max_fun_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds, maxfun=100,
                             seed=self.seed)

        ls_max_iter = min(max(
            len(self.ld_bounds) * LocalSearchWrapper.LS_MAXITER_RATIO,
            LocalSearchWrapper.LS_MAXITER_MIN),
            LocalSearchWrapper.LS_MAXITER_MAX)
        assert ret.nfev <= 100 + ls_max_iter
        assert not ret.success

    def test_max_fun_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             no_local_search=True, maxfun=500, seed=self.seed)
        assert ret.nfev <= 500
        assert not ret.success

    def test_maxiter(self):
        ret = dual_annealing(self.func, self.ld_bounds, maxiter=700,
                             seed=self.seed)
        assert ret.nit <= 700

    # Testing that args are passed correctly for dual_annealing
    def test_fun_args_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             args=((3.14159,)), seed=self.seed)
        assert_allclose(ret.fun, 3.14159, atol=1e-6)

    # Testing that args are passed correctly for pure simulated annealing
    def test_fun_args_no_ls(self):
        ret = dual_annealing(self.func, self.ld_bounds,
                             args=((3.14159, )), no_local_search=True,
                             seed=self.seed)
        assert_allclose(ret.fun, 3.14159, atol=1e-4)

    def test_callback_stop(self):
        # Testing that callback make the algorithm stop for
        # fun value <= 1.0 (see callback method)
        ret = dual_annealing(self.func, self.ld_bounds,
                             callback=self.callback, seed=self.seed)
        assert ret.fun <= 1.0
        assert 'stop early' in ret.message[0]
        assert not ret.success

    @pytest.mark.parametrize('method, atol', [
        ('Nelder-Mead', 2e-5),
        ('COBYLA', 1e-5),
        ('Powell', 1e-8),
        ('CG', 1e-8),
        ('BFGS', 1e-8),
        ('TNC', 1e-8),
        ('SLSQP', 2e-7),
    ])
    def test_multi_ls_minimizer(self, method, atol):
        ret = dual_annealing(self.func, self.ld_bounds,
                             local_search_options=dict(method=method),
                             seed=self.seed)
        assert_allclose(ret.fun, 0., atol=atol)

    def test_wrong_restart_temp(self):
        assert_raises(ValueError, dual_annealing, self.func,
                      self.ld_bounds, restart_temp_ratio=1)
        assert_raises(ValueError, dual_annealing, self.func,
                      self.ld_bounds, restart_temp_ratio=0)

    def test_gradient_gnev(self):
        minimizer_opts = {
            'jac': self.rosen_der_wrapper,
        }
        ret = dual_annealing(rosen, self.ld_bounds,
                             local_search_options=minimizer_opts,
                             seed=self.seed)
        assert ret.njev == self.ngev

    def test_from_docstring(self):
        func = lambda x: np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
        lw = [-5.12] * 10
        up = [5.12] * 10
        ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
        assert_allclose(ret.x,
                        [-4.26437714e-09, -3.91699361e-09, -1.86149218e-09,
                         -3.97165720e-09, -6.29151648e-09, -6.53145322e-09,
                         -3.93616815e-09, -6.55623025e-09, -6.05775280e-09,
                         -5.00668935e-09], atol=4e-8)
        assert_allclose(ret.fun, 0.000000, atol=5e-13)

    @pytest.mark.parametrize('new_e, temp_step, accepted, accept_rate', [
        (0, 100, 1000, 1.0097587941791923),
        (0, 2, 1000, 1.2599210498948732),
        (10, 100, 878, 0.8786035869128718),
        (10, 60, 695, 0.6812920690579612),
        (2, 100, 990, 0.9897404249173424),
    ])
    def test_accept_reject_probabilistic(
            self, new_e, temp_step, accepted, accept_rate):
        # Test accepts unconditionally with e < current_energy and
        # probabilistically with e > current_energy

        rs = check_random_state(123)

        count_accepted = 0
        iterations = 1000

        accept_param = -5
        current_energy = 1
        for _ in range(iterations):
            energy_state = EnergyState(lower=None, upper=None)
            # Set energy state with current_energy, any location.
            energy_state.update_current(current_energy, [0])

            chain = StrategyChain(
                accept_param, None, None, None, rs, energy_state)
            # Normally this is set in run()
            chain.temperature_step = temp_step

            # Check if update is accepted.
            chain.accept_reject(j=1, e=new_e, x_visit=[2])
            if energy_state.current_energy == new_e:
                count_accepted += 1

        assert count_accepted == accepted

        # Check accept rate
        pqv = 1 - (1 - accept_param) * (new_e - current_energy) / temp_step
        rate = 0 if pqv <= 0 else np.exp(np.log(pqv) / (1 - accept_param))

        assert_allclose(rate, accept_rate)
