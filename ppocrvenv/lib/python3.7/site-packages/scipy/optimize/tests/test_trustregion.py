"""
Unit tests for trust-region optimization routines.

To run it in its simplest form::
  nosetests test_optimize.py

"""
import itertools
from copy import deepcopy
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
from scipy.optimize import (minimize, rosen, rosen_der, rosen_hess,
                            rosen_hess_prod, BFGS)
from scipy.optimize._differentiable_functions import FD_METHODS
import pytest


class Accumulator:
    """ This is for testing callbacks."""
    def __init__(self):
        self.count = 0
        self.accum = None

    def __call__(self, x):
        self.count += 1
        if self.accum is None:
            self.accum = np.array(x)
        else:
            self.accum += x


class TestTrustRegionSolvers:

    def setup_method(self):
        self.x_opt = [1.0, 1.0]
        self.easy_guess = [2.0, 2.0]
        self.hard_guess = [-1.2, 1.0]

    def test_dogleg_accuracy(self):
        # test the accuracy and the return_all option
        x0 = self.hard_guess
        r = minimize(rosen, x0, jac=rosen_der, hess=rosen_hess, tol=1e-8,
                     method='dogleg', options={'return_all': True},)
        assert_allclose(x0, r['allvecs'][0])
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(r['x'], self.x_opt)

    def test_dogleg_callback(self):
        # test the callback mechanism and the maxiter and return_all options
        accumulator = Accumulator()
        maxiter = 5
        r = minimize(rosen, self.hard_guess, jac=rosen_der, hess=rosen_hess,
                     callback=accumulator, method='dogleg',
                     options={'return_all': True, 'maxiter': maxiter},)
        assert_equal(accumulator.count, maxiter)
        assert_equal(len(r['allvecs']), maxiter+1)
        assert_allclose(r['x'], r['allvecs'][-1])
        assert_allclose(sum(r['allvecs'][1:]), accumulator.accum)

    def test_solver_concordance(self):
        # Assert that dogleg uses fewer iterations than ncg on the Rosenbrock
        # test function, although this does not necessarily mean
        # that dogleg is faster or better than ncg even for this function
        # and especially not for other test functions.
        f = rosen
        g = rosen_der
        h = rosen_hess
        for x0 in (self.easy_guess, self.hard_guess):
            r_dogleg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                method='dogleg', options={'return_all': True})
            r_trust_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                   method='trust-ncg',
                                   options={'return_all': True})
            r_trust_krylov = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                   method='trust-krylov',
                                   options={'return_all': True})
            r_ncg = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                             method='newton-cg', options={'return_all': True})
            r_iterative = minimize(f, x0, jac=g, hess=h, tol=1e-8,
                                   method='trust-exact',
                                   options={'return_all': True})
            assert_allclose(self.x_opt, r_dogleg['x'])
            assert_allclose(self.x_opt, r_trust_ncg['x'])
            assert_allclose(self.x_opt, r_trust_krylov['x'])
            assert_allclose(self.x_opt, r_ncg['x'])
            assert_allclose(self.x_opt, r_iterative['x'])
            assert_(len(r_dogleg['allvecs']) < len(r_ncg['allvecs']))

    def test_trust_ncg_hessp(self):
        for x0 in (self.easy_guess, self.hard_guess, self.x_opt):
            r = minimize(rosen, x0, jac=rosen_der, hessp=rosen_hess_prod,
                         tol=1e-8, method='trust-ncg')
            assert_allclose(self.x_opt, r['x'])

    def test_trust_ncg_start_in_optimum(self):
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess,
                     tol=1e-8, method='trust-ncg')
        assert_allclose(self.x_opt, r['x'])

    def test_trust_krylov_start_in_optimum(self):
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess,
                     tol=1e-8, method='trust-krylov')
        assert_allclose(self.x_opt, r['x'])

    def test_trust_exact_start_in_optimum(self):
        r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=rosen_hess,
                     tol=1e-8, method='trust-exact')
        assert_allclose(self.x_opt, r['x'])

    def test_finite_differences(self):
        # if the Hessian is estimated by finite differences or
        # a HessianUpdateStrategy (and no hessp is provided) then creation
        # of a hessp is possible.
        # GH13754
        methods = ["trust-ncg", "trust-krylov", "dogleg"]
        product = itertools.product(
            FD_METHODS + (BFGS,),
            methods
        )
        # a hessian needs to be specified for trustregion
        for method in methods:
            with pytest.raises(ValueError):
                minimize(rosen, x0=self.x_opt, jac=rosen_der, method=method)

        # estimate hessian by finite differences. In _trustregion.py this
        # creates a hessp from the LinearOperator/HessianUpdateStrategy
        # that's returned from ScalarFunction.
        for fd, method in product:
            hess = fd
            if fd == BFGS:
                # don't want to use the same object over and over
                hess = deepcopy(fd())

            r = minimize(rosen, x0=self.x_opt, jac=rosen_der, hess=hess,
                         tol=1e-8, method=method)
            assert_allclose(self.x_opt, r['x'])
