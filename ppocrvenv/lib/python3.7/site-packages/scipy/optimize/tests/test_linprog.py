"""
Unit test for Linear Programming
"""
import sys

import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
                           assert_array_less, assert_warns, suppress_warnings)
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
import scipy.sparse
import pytest

has_umfpack = True
try:
    from scikits.umfpack import UmfpackWarning
except ImportError:
    has_umfpack = False

has_cholmod = True
try:
    import sksparse
    from sksparse.cholmod import cholesky as cholmod
except ImportError:
    has_cholmod = False


def _assert_iteration_limit_reached(res, maxiter):
    assert_(not res.success, "Incorrectly reported success")
    assert_(res.success < maxiter, "Incorrectly reported number of iterations")
    assert_equal(res.status, 1, "Failed to report iteration limit reached")


def _assert_infeasible(res):
    # res: linprog result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 2, "failed to report infeasible status")


def _assert_unbounded(res):
    # res: linprog result object
    assert_(not res.success, "incorrectly reported success")
    assert_equal(res.status, 3, "failed to report unbounded status")


def _assert_unable_to_find_basic_feasible_sol(res):
    # res: linprog result object

    # The status may be either 2 or 4 depending on why the feasible solution
    # could not be found. If the undelying problem is expected to not have a
    # feasible solution, _assert_infeasible should be used.
    assert_(not res.success, "incorrectly reported success")
    assert_(res.status in (2, 4), "failed to report optimization failure")


def _assert_success(res, desired_fun=None, desired_x=None,
                    rtol=1e-8, atol=1e-8):
    # res: linprog result object
    # desired_fun: desired objective function value or None
    # desired_x: desired solution or None
    if not res.success:
        msg = "linprog status {0}, message: {1}".format(res.status,
                                                        res.message)
        raise AssertionError(msg)

    assert_equal(res.status, 0)
    if desired_fun is not None:
        assert_allclose(res.fun, desired_fun,
                        err_msg="converged to an unexpected objective value",
                        rtol=rtol, atol=atol)
    if desired_x is not None:
        assert_allclose(res.x, desired_x,
                        err_msg="converged to an unexpected solution",
                        rtol=rtol, atol=atol)


def magic_square(n):
    """
    Generates a linear program for which integer solutions represent an
    n x n magic square; binary decision variables represent the presence
    (or absence) of an integer 1 to n^2 in each position of the square.
    """

    np.random.seed(0)
    M = n * (n**2 + 1) / 2

    numbers = np.arange(n**4) // n**2 + 1

    numbers = numbers.reshape(n**2, n, n)

    zeros = np.zeros((n**2, n, n))

    A_list = []
    b_list = []

    # Rule 1: use every number exactly once
    for i in range(n**2):
        A_row = zeros.copy()
        A_row[i, :, :] = 1
        A_list.append(A_row.flatten())
        b_list.append(1)

    # Rule 2: Only one number per square
    for i in range(n):
        for j in range(n):
            A_row = zeros.copy()
            A_row[:, i, j] = 1
            A_list.append(A_row.flatten())
            b_list.append(1)

    # Rule 3: sum of rows is M
    for i in range(n):
        A_row = zeros.copy()
        A_row[:, i, :] = numbers[:, i, :]
        A_list.append(A_row.flatten())
        b_list.append(M)

    # Rule 4: sum of columns is M
    for i in range(n):
        A_row = zeros.copy()
        A_row[:, :, i] = numbers[:, :, i]
        A_list.append(A_row.flatten())
        b_list.append(M)

    # Rule 5: sum of diagonals is M
    A_row = zeros.copy()
    A_row[:, range(n), range(n)] = numbers[:, range(n), range(n)]
    A_list.append(A_row.flatten())
    b_list.append(M)
    A_row = zeros.copy()
    A_row[:, range(n), range(-1, -n - 1, -1)] = \
        numbers[:, range(n), range(-1, -n - 1, -1)]
    A_list.append(A_row.flatten())
    b_list.append(M)

    A = np.array(np.vstack(A_list), dtype=float)
    b = np.array(b_list, dtype=float)
    c = np.random.rand(A.shape[1])

    return A, b, c, numbers


def lpgen_2d(m, n):
    """ -> A b c LP test: m*n vars, m+n constraints
        row sums == n/m, col sums == 1
        https://gist.github.com/denis-bz/8647461
    """
    np.random.seed(0)
    c = - np.random.exponential(size=(m, n))
    Arow = np.zeros((m, m * n))
    brow = np.zeros(m)
    for j in range(m):
        j1 = j + 1
        Arow[j, j * n:j1 * n] = 1
        brow[j] = n / m

    Acol = np.zeros((n, m * n))
    bcol = np.zeros(n)
    for j in range(n):
        j1 = j + 1
        Acol[j, j::n] = 1
        bcol[j] = 1

    A = np.vstack((Arow, Acol))
    b = np.hstack((brow, bcol))

    return A, b, c.ravel()


def very_random_gen(seed=0):
    np.random.seed(seed)
    m_eq, m_ub, n = 10, 20, 50
    c = np.random.rand(n)-0.5
    A_ub = np.random.rand(m_ub, n)-0.5
    b_ub = np.random.rand(m_ub)-0.5
    A_eq = np.random.rand(m_eq, n)-0.5
    b_eq = np.random.rand(m_eq)-0.5
    lb = -np.random.rand(n)
    ub = np.random.rand(n)
    lb[lb < -np.random.rand()] = -np.inf
    ub[ub > np.random.rand()] = np.inf
    bounds = np.vstack((lb, ub)).T
    return c, A_ub, b_ub, A_eq, b_eq, bounds


def nontrivial_problem():
    c = [-1, 8, 4, -6]
    A_ub = [[-7, -7, 6, 9],
            [1, -1, -3, 0],
            [10, -10, -7, 7],
            [6, -1, 3, 4]]
    b_ub = [-3, 6, -6, 6]
    A_eq = [[-10, 1, 1, -8]]
    b_eq = [-4]
    x_star = [101 / 1391, 1462 / 1391, 0, 752 / 1391]
    f_star = 7083 / 1391
    return c, A_ub, b_ub, A_eq, b_eq, x_star, f_star


def l1_regression_prob(seed=0, m=8, d=9, n=100):
    '''
    Training data is {(x0, y0), (x1, y2), ..., (xn-1, yn-1)}
        x in R^d
        y in R
    n: number of training samples
    d: dimension of x, i.e. x in R^d
    phi: feature map R^d -> R^m
    m: dimension of feature space
    '''
    np.random.seed(seed)
    phi = np.random.normal(0, 1, size=(m, d))  # random feature mapping
    w_true = np.random.randn(m)
    x = np.random.normal(0, 1, size=(d, n))  # features
    y = w_true @ (phi @ x) + np.random.normal(0, 1e-5, size=n)  # measurements

    # construct the problem
    c = np.ones(m+n)
    c[:m] = 0
    A_ub = scipy.sparse.lil_matrix((2*n, n+m))
    idx = 0
    for ii in range(n):
        A_ub[idx, :m] = phi @ x[:, ii]
        A_ub[idx, m+ii] = -1
        A_ub[idx+1, :m] = -1*phi @ x[:, ii]
        A_ub[idx+1, m+ii] = -1
        idx += 2
    A_ub = A_ub.tocsc()
    b_ub = np.zeros(2*n)
    b_ub[0::2] = y
    b_ub[1::2] = -y
    bnds = [(None, None)]*m + [(0, None)]*n
    return c, A_ub, b_ub, bnds


def generic_callback_test(self):
    # Check that callback is as advertised
    last_cb = {}

    def cb(res):
        message = res.pop('message')
        complete = res.pop('complete')

        assert_(res.pop('phase') in (1, 2))
        assert_(res.pop('status') in range(4))
        assert_(isinstance(res.pop('nit'), int))
        assert_(isinstance(complete, bool))
        assert_(isinstance(message, str))

        last_cb['x'] = res['x']
        last_cb['fun'] = res['fun']
        last_cb['slack'] = res['slack']
        last_cb['con'] = res['con']

    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, callback=cb, method=self.method)

    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])
    assert_allclose(last_cb['fun'], res['fun'])
    assert_allclose(last_cb['x'], res['x'])
    assert_allclose(last_cb['con'], res['con'])
    assert_allclose(last_cb['slack'], res['slack'])


def test_unknown_solvers_and_options():
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]

    assert_raises(ValueError, linprog,
                  c, A_ub=A_ub, b_ub=b_ub, method='ekki-ekki-ekki')
    assert_raises(ValueError, linprog,
                  c, A_ub=A_ub, b_ub=b_ub, method='highs-ekki')
    assert_raises(ValueError, linprog, c, A_ub=A_ub, b_ub=b_ub,
                  options={"rr_method": 'ekki-ekki-ekki'})


def test_choose_solver():
    # 'highs' chooses 'dual'
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]

    res = linprog(c, A_ub, b_ub, method='highs')
    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])


A_ub = None
b_ub = None
A_eq = None
b_eq = None
bounds = None

################
# Common Tests #
################


class LinprogCommonTests:
    """
    Base class for `linprog` tests. Generally, each test will be performed
    once for every derived class of LinprogCommonTests, each of which will
    typically change self.options and/or self.method. Effectively, these tests
    are run for many combination of method (simplex, revised simplex, and
    interior point) and options (such as pivoting rule or sparse treatment).
    """

    ##################
    # Targeted Tests #
    ##################

    def test_callback(self):
        generic_callback_test(self)

    def test_disp(self):
        # test that display option does not break anything.
        A, b, c = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                      options={"disp": True})
        _assert_success(res, desired_fun=-64.049494229)

    def test_docstring_example(self):
        # Example from linprog docstring.
        c = [-1, 4]
        A = [[-3, 1], [1, 2]]
        b = [6, 4]
        x0_bounds = (None, None)
        x1_bounds = (-3, None)
        res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),
                      options=self.options, method=self.method)
        _assert_success(res, desired_fun=-22)

    def test_type_error(self):
        # (presumably) checks that linprog recognizes type errors
        # This is tested more carefully in test__linprog_clean_inputs.py
        c = [1]
        A_eq = [[1]]
        b_eq = "hello"
        assert_raises(TypeError, linprog,
                      c, A_eq=A_eq, b_eq=b_eq,
                      method=self.method, options=self.options)

    def test_aliasing_b_ub(self):
        # (presumably) checks that linprog does not modify b_ub
        # This is tested more carefully in test__linprog_clean_inputs.py
        c = np.array([1.0])
        A_ub = np.array([[1.0]])
        b_ub_orig = np.array([3.0])
        b_ub = b_ub_orig.copy()
        bounds = (-4.0, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-4, desired_x=[-4])
        assert_allclose(b_ub_orig, b_ub)

    def test_aliasing_b_eq(self):
        # (presumably) checks that linprog does not modify b_eq
        # This is tested more carefully in test__linprog_clean_inputs.py
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq_orig = np.array([3.0])
        b_eq = b_eq_orig.copy()
        bounds = (-4.0, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])
        assert_allclose(b_eq_orig, b_eq)

    def test_non_ndarray_args(self):
        # (presumably) checks that linprog accepts list in place of arrays
        # This is tested more carefully in test__linprog_clean_inputs.py
        c = [1.0]
        A_ub = [[1.0]]
        b_ub = [3.0]
        A_eq = [[1.0]]
        b_eq = [2.0]
        bounds = (-1.0, 10.0)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=2, desired_x=[2])

    def test_unknown_options(self):
        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]

        def f(c, A_ub=None, b_ub=None, A_eq=None,
              b_eq=None, bounds=None, options={}):
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    method=self.method, options=options)

        o = {key: self.options[key] for key in self.options}
        o['spam'] = 42

        assert_warns(OptimizeWarning, f,
                     c, A_ub=A_ub, b_ub=b_ub, options=o)

    def test_invalid_inputs(self):

        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    method=self.method, options=self.options)

        # Test ill-formatted bounds
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4)])
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4), (3, 4, 5)])
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, -2), (1, 2)])

        # Test other invalid inputs
        assert_raises(ValueError, f, [1, 2], A_ub=[[1, 2]], b_ub=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_ub=[[1]], b_ub=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1, 2]], b_eq=[1, 2])
        assert_raises(ValueError, f, [1, 2], A_eq=[[1]], b_eq=[1])
        assert_raises(ValueError, f, [1, 2], A_eq=[1], b_eq=1)

        # this last check doesn't make sense for sparse presolve
        if ("_sparse_presolve" in self.options and
                self.options["_sparse_presolve"]):
            return
            # there aren't 3-D sparse matrices

        assert_raises(ValueError, f, [1, 2], A_ub=np.zeros((1, 1, 3)), b_eq=1)

    def test_sparse_constraints(self):
        # gh-13559: improve error message for sparse inputs when unsupported
        def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
            linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                    method=self.method, options=self.options)

        np.random.seed(0)
        m = 100
        n = 150
        A_eq = scipy.sparse.rand(m, n, 0.5)
        x_valid = np.random.randn((n))
        c = np.random.randn((n))
        ub = x_valid + np.random.rand((n))
        lb = x_valid - np.random.rand((n))
        bounds = np.column_stack((lb, ub))
        b_eq = A_eq * x_valid

        if self.method in {'simplex', 'revised simplex'}:
            # simplex and revised simplex should raise error
            with assert_raises(ValueError, match=f"Method '{self.method}' "
                               "does not support sparse constraint matrices."):
                linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                        method=self.method, options=self.options)
        else:
            # other methods should succeed
            options = {**self.options}
            if self.method in {'interior-point'}:
                options['sparse'] = True

            res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                          method=self.method, options=options)
            assert res.success

    def test_maxiter(self):
        # test iteration limit w/ Enzo example
        c = [4, 8, 3, 0, 0, 0]
        A = [
            [2, 5, 3, -1, 0, 0],
            [3, 2.5, 8, 0, -1, 0],
            [8, 10, 4, 0, 0, -1]]
        b = [185, 155, 600]
        np.random.seed(0)
        maxiter = 3
        res = linprog(c, A_eq=A, b_eq=b, method=self.method,
                      options={"maxiter": maxiter})
        _assert_iteration_limit_reached(res, maxiter)
        assert_equal(res.nit, maxiter)

    def test_bounds_fixed(self):

        # Test fixed bounds (upper equal to lower)
        # If presolve option True, test if solution found in presolve (i.e.
        # number of iterations is 0).
        do_presolve = self.options.get('presolve', True)

        res = linprog([1], bounds=(1, 1),
                      method=self.method, options=self.options)
        _assert_success(res, 1, 1)
        if do_presolve:
            assert_equal(res.nit, 0)

        res = linprog([1, 2, 3], bounds=[(5, 5), (-1, -1), (3, 3)],
                      method=self.method, options=self.options)
        _assert_success(res, 12, [5, -1, 3])
        if do_presolve:
            assert_equal(res.nit, 0)

        res = linprog([1, 1], bounds=[(1, 1), (1, 3)],
                      method=self.method, options=self.options)
        _assert_success(res, 2, [1, 1])
        if do_presolve:
            assert_equal(res.nit, 0)

        res = linprog([1, 1, 2], A_eq=[[1, 0, 0], [0, 1, 0]], b_eq=[1, 7],
                      bounds=[(-5, 5), (0, 10), (3.5, 3.5)],
                      method=self.method, options=self.options)
        _assert_success(res, 15, [1, 7, 3.5])
        if do_presolve:
            assert_equal(res.nit, 0)

    def test_bounds_infeasible(self):

        # Test ill-valued bounds (upper less than lower)
        # If presolve option True, test if solution found in presolve (i.e.
        # number of iterations is 0).
        do_presolve = self.options.get('presolve', True)

        res = linprog([1], bounds=(1, -2), method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)

        res = linprog([1], bounds=[(1, -2)], method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)

        res = linprog([1, 2, 3], bounds=[(5, 0), (1, 2), (3, 4)], method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)

    def test_bounds_infeasible_2(self):

        # Test ill-valued bounds (lower inf, upper -inf)
        # If presolve option True, test if solution found in presolve (i.e.
        # number of iterations is 0).
        # For the simplex method, the cases do not result in an
        # infeasible status, but in a RuntimeWarning. This is a
        # consequence of having _presolve() take care of feasibility
        # checks. See issue gh-11618.
        do_presolve = self.options.get('presolve', True)
        simplex_without_presolve = not do_presolve and self.method == 'simplex'

        c = [1, 2, 3]
        bounds_1 = [(1, 2), (np.inf, np.inf), (3, 4)]
        bounds_2 = [(1, 2), (-np.inf, -np.inf), (3, 4)]

        if simplex_without_presolve:
            def g(c, bounds):
                res = linprog(c, bounds=bounds, method=self.method, options=self.options)
                return res

            with pytest.warns(RuntimeWarning):
                with pytest.raises(IndexError):
                    g(c, bounds=bounds_1)

            with pytest.warns(RuntimeWarning):
                with pytest.raises(IndexError):
                    g(c, bounds=bounds_2)
        else:
            res = linprog(c=c, bounds=bounds_1, method=self.method, options=self.options)
            _assert_infeasible(res)
            if do_presolve:
                assert_equal(res.nit, 0)
            res = linprog(c=c, bounds=bounds_2, method=self.method, options=self.options)
            _assert_infeasible(res)
            if do_presolve:
                assert_equal(res.nit, 0)

    def test_empty_constraint_1(self):
        c = [-1, -2]
        res = linprog(c, method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_empty_constraint_2(self):
        c = [-1, 1, -1, 1]
        bounds = [(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)]
        res = linprog(c, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_unbounded(res)
        # Unboundedness detected in presolve requires no iterations
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_empty_constraint_3(self):
        c = [1, -1, 1, -1]
        bounds = [(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)]
        res = linprog(c, bounds=bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 0, -1, 1], desired_fun=-2)

    def test_inequality_constraints(self):
        # Minimize linear function subject to linear inequality constraints.
        #  http://www.dam.brown.edu/people/huiwang/classes/am121/Archive/simplex_121_c.pdf
        c = np.array([3, 2]) * -1  # maximize
        A_ub = [[2, 1],
                [1, 1],
                [1, 0]]
        b_ub = [10, 8, 4]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-18, desired_x=[2, 6])

    def test_inequality_constraints2(self):
        # Minimize linear function subject to linear inequality constraints.
        # http://www.statslab.cam.ac.uk/~ff271/teaching/opt/notes/notes8.pdf
        # (dead link)
        c = [6, 3]
        A_ub = [[0, 3],
                [-1, -1],
                [-2, 1]]
        b_ub = [2, -1, -1]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=5, desired_x=[2 / 3, 1 / 3])

    def test_bounds_simple(self):
        c = [1, 2]
        bounds = (1, 2)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[1, 1])

        bounds = [(1, 2), (1, 2)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[1, 1])

    def test_bounded_below_only_1(self):
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])
        bounds = (1.0, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_bounded_below_only_2(self):
        c = np.ones(3)
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])
        bounds = (0.5, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounded_above_only_1(self):
        c = np.array([1.0])
        A_eq = np.array([[1.0]])
        b_eq = np.array([3.0])
        bounds = (None, 10.0)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3, desired_x=[3])

    def test_bounded_above_only_2(self):
        c = np.ones(3)
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])
        bounds = (-np.inf, 4)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounds_infinity(self):
        c = np.ones(3)
        A_eq = np.eye(3)
        b_eq = np.array([1, 2, 3])
        bounds = (-np.inf, np.inf)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))

    def test_bounds_mixed(self):
        # Problem has one unbounded variable and
        # another with a negative lower bound.
        c = np.array([-1, 4]) * -1  # maximize
        A_ub = np.array([[-3, 1],
                         [1, 2]], dtype=np.float64)
        b_ub = [6, 4]
        x0_bounds = (-np.inf, np.inf)
        x1_bounds = (-3, np.inf)
        bounds = (x0_bounds, x1_bounds)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-80 / 7, desired_x=[-8 / 7, 18 / 7])

    def test_bounds_equal_but_infeasible(self):
        c = [-4, 1]
        A_ub = [[7, -2], [0, 1], [2, -2]]
        b_ub = [14, 0, 3]
        bounds = [(2, 2), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bounds_equal_but_infeasible2(self):
        c = [-4, 1]
        A_eq = [[7, -2], [0, 1], [2, -2]]
        b_eq = [14, 0, 3]
        bounds = [(2, 2), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bounds_equal_no_presolve(self):
        # There was a bug when a lower and upper bound were equal but
        # presolve was not on to eliminate the variable. The bound
        # was being converted to an equality constraint, but the bound
        # was not eliminated, leading to issues in postprocessing.
        c = [1, 2]
        A_ub = [[1, 2], [1.1, 2.2]]
        b_ub = [4, 8]
        bounds = [(1, 2), (2, 2)]

        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)
        _assert_infeasible(res)

    def test_zero_column_1(self):
        m, n = 3, 4
        np.random.seed(0)
        c = np.random.rand(n)
        c[1] = 1
        A_eq = np.random.rand(m, n)
        A_eq[:, 1] = 0
        b_eq = np.random.rand(m)
        A_ub = [[1, 0, 1, 1]]
        b_ub = 3
        bounds = [(-10, 10), (-10, 10), (-10, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-9.7087836730413404)

    def test_zero_column_2(self):
        np.random.seed(0)
        m, n = 2, 4
        c = np.random.rand(n)
        c[1] = -1
        A_eq = np.random.rand(m, n)
        A_eq[:, 1] = 0
        b_eq = np.random.rand(m)

        A_ub = np.random.rand(m, n)
        A_ub[:, 1] = 0
        b_ub = np.random.rand(m)
        bounds = (None, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_unbounded(res)
        # Unboundedness detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_zero_row_1(self):
        c = [1, 2, 3]
        A_eq = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        b_eq = [0, 3, 0]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=3)

    def test_zero_row_2(self):
        A_ub = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        b_ub = [0, 3, 0]
        c = [1, 2, 3]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0)

    def test_zero_row_3(self):
        m, n = 2, 4
        c = np.random.rand(n)
        A_eq = np.random.rand(m, n)
        A_eq[0, :] = 0
        b_eq = np.random.rand(m)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

        # Infeasibility detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_zero_row_4(self):
        m, n = 2, 4
        c = np.random.rand(n)
        A_ub = np.random.rand(m, n)
        A_ub[0, :] = 0
        b_ub = -np.random.rand(m)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

        # Infeasibility detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_singleton_row_eq_1(self):
        c = [1, 1, 1, 2]
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
        b_eq = [1, 2, 2, 4]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

        # Infeasibility detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_singleton_row_eq_2(self):
        c = [1, 1, 1, 2]
        A_eq = [[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]]
        b_eq = [1, 2, 1, 4]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=4)

    def test_singleton_row_ub_1(self):
        c = [1, 1, 1, 2]
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]
        b_ub = [1, 2, -2, 4]
        bounds = [(None, None), (0, None), (0, None), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

        # Infeasibility detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_singleton_row_ub_2(self):
        c = [1, 1, 1, 2]
        A_ub = [[1, 0, 0, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [1, 1, 1, 1]]
        b_ub = [1, 2, -0.5, 4]
        bounds = [(None, None), (0, None), (0, None), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0.5)

    def test_infeasible(self):
        # Test linprog response to an infeasible problem
        c = [-1, -1]
        A_ub = [[1, 0],
                [0, 1],
                [-1, -1]]
        b_ub = [2, 2, -5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_infeasible_inequality_bounds(self):
        c = [1]
        A_ub = [[2]]
        b_ub = 4
        bounds = (5, 6)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_infeasible(res)

        # Infeasibility detected in presolve
        if self.options.get('presolve', True):
            assert_equal(res.nit, 0)

    def test_unbounded(self):
        # Test linprog response to an unbounded problem
        c = np.array([1, 1]) * -1  # maximize
        A_ub = [[-1, 1],
                [-1, -1]]
        b_ub = [-1, -2]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_unbounded_below_no_presolve_corrected(self):
        c = [1]
        bounds = [(None, 1)]

        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False

        res = linprog(c=c, bounds=bounds,
                      method=self.method,
                      options=o)
        if self.method == "revised simplex":
            # Revised simplex has a special pathway for no constraints.
            assert_equal(res.status, 5)
        else:
            _assert_unbounded(res)

    def test_unbounded_no_nontrivial_constraints_1(self):
        """
        Test whether presolve pathway for detecting unboundedness after
        constraint elimination is working.
        """
        c = np.array([0, 0, 0, 1, -1, -1])
        A_ub = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, -1]])
        b_ub = np.array([2, -2, 0])
        bounds = [(None, None), (None, None), (None, None),
                  (-1, 1), (-1, 1), (0, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_unbounded(res)
        if not self.method.lower().startswith("highs"):
            assert_equal(res.x[-1], np.inf)
            assert_equal(res.message[:36],
                         "The problem is (trivially) unbounded")

    def test_unbounded_no_nontrivial_constraints_2(self):
        """
        Test whether presolve pathway for detecting unboundedness after
        constraint elimination is working.
        """
        c = np.array([0, 0, 0, 1, -1, 1])
        A_ub = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1]])
        b_ub = np.array([2, -2, 0])
        bounds = [(None, None), (None, None), (None, None),
                  (-1, 1), (-1, 1), (None, 0)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_unbounded(res)
        if not self.method.lower().startswith("highs"):
            assert_equal(res.x[-1], -np.inf)
            assert_equal(res.message[:36],
                         "The problem is (trivially) unbounded")

    def test_cyclic_recovery(self):
        # Test linprogs recovery from cycling using the Klee-Minty problem
        # Klee-Minty  https://www.math.ubc.ca/~israel/m340/kleemin3.pdf
        c = np.array([100, 10, 1]) * -1  # maximize
        A_ub = [[1, 0, 0],
                [20, 1, 0],
                [200, 20, 1]]
        b_ub = [1, 100, 10000]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 0, 10000], atol=5e-6, rtol=1e-7)

    def test_cyclic_bland(self):
        # Test the effect of Bland's rule on a cycling problem
        c = np.array([-10, 57, 9, 24.])
        A_ub = np.array([[0.5, -5.5, -2.5, 9],
                         [0.5, -1.5, -0.5, 1],
                         [1, 0, 0, 0]])
        b_ub = [0, 0, 1]

        # copy the existing options dictionary but change maxiter
        maxiter = 100
        o = {key: val for key, val in self.options.items()}
        o['maxiter'] = maxiter

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)

        if self.method == 'simplex' and not self.options.get('bland'):
            # simplex cycles without Bland's rule
            _assert_iteration_limit_reached(res, o['maxiter'])
        else:
            # other methods, including simplex with Bland's rule, succeed
            _assert_success(res, desired_x=[1, 0, 1, 0])
        # note that revised simplex skips this test because it may or may not
        # cycle depending on the initial basis

    def test_remove_redundancy_infeasibility(self):
        # mostly a test of redundancy removal, which is carefully tested in
        # test__remove_redundancy.py
        m, n = 10, 10
        c = np.random.rand(n)
        A_eq = np.random.rand(m, n)
        b_eq = np.random.rand(m)
        A_eq[-1, :] = 2 * A_eq[-2, :]
        b_eq[-1] *= -1
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_infeasible(res)

    #################
    # General Tests #
    #################

    def test_nontrivial_problem(self):
        # Problem involves all constraint types,
        # negative resource limits, and rounding issues.
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)

    def test_lpgen_problem(self):
        # Test linprog  with a rather large problem (400 variables,
        # 40 constraints) generated by https://gist.github.com/denis-bz/8647461
        A_ub, b_ub, c = lpgen_2d(20, 20)

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_fun=-64.049494229)

    def test_network_flow(self):
        # A network flow problem with supply and demand at nodes
        # and with costs along directed edges.
        # https://www.princeton.edu/~rvdb/542/lectures/lec10.pdf
        c = [2, 4, 9, 11, 4, 3, 8, 7, 0, 15, 16, 18]
        n, p = -1, 1
        A_eq = [
            [n, n, p, 0, p, 0, 0, 0, 0, p, 0, 0],
            [p, 0, 0, p, 0, p, 0, 0, 0, 0, 0, 0],
            [0, 0, n, n, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, p, p, 0, 0, p, 0],
            [0, 0, 0, 0, n, n, n, 0, p, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, n, n, 0, 0, p],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, n, n, n]]
        b_eq = [0, 19, -16, 33, 0, 0, -36]
        with suppress_warnings() as sup:
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_fun=755, atol=1e-6, rtol=1e-7)

    def test_network_flow_limited_capacity(self):
        # A network flow problem with supply and demand at nodes
        # and with costs and capacities along directed edges.
        # http://blog.sommer-forst.de/2013/04/10/
        c = [2, 2, 1, 3, 1]
        bounds = [
            [0, 4],
            [0, 2],
            [0, 2],
            [0, 3],
            [0, 5]]
        n, p = -1, 1
        A_eq = [
            [n, n, 0, 0, 0],
            [p, 0, n, n, 0],
            [0, p, p, 0, n],
            [0, 0, 0, p, p]]
        b_eq = [-4, 0, 0, 4]

        with suppress_warnings() as sup:
            # this is an UmfpackWarning but I had trouble importing it
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(OptimizeWarning, "Solving system with option...")
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_fun=14)

    def test_simplex_algorithm_wikipedia_example(self):
        # https://en.wikipedia.org/wiki/Simplex_algorithm#Example
        c = [-2, -3, -4]
        A_ub = [
            [3, 2, 1],
            [2, 5, 3]]
        b_ub = [10, 15]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=-20)

    def test_enzo_example(self):
        # https://github.com/scipy/scipy/issues/1779 lp2.py
        #
        # Translated from Octave code at:
        # http://www.ecs.shimane-u.ac.jp/~kyoshida/lpeng.htm
        # and placed under MIT licence by Enzo Michelangeli
        # with permission explicitly granted by the original author,
        # Prof. Kazunobu Yoshida
        c = [4, 8, 3, 0, 0, 0]
        A_eq = [
            [2, 5, 3, -1, 0, 0],
            [3, 2.5, 8, 0, -1, 0],
            [8, 10, 4, 0, 0, -1]]
        b_eq = [185, 155, 600]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=317.5,
                        desired_x=[66.25, 0, 17.5, 0, 183.75, 0],
                        atol=6e-6, rtol=1e-7)

    def test_enzo_example_b(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        c = [2.8, 6.3, 10.8, -2.8, -6.3, -10.8]
        A_eq = [[-1, -1, -1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1]]
        b_eq = [-0.5, 0.4, 0.3, 0.3, 0.3]

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_fun=-1.77,
                        desired_x=[0.3, 0.2, 0.0, 0.0, 0.1, 0.3])

    def test_enzo_example_c_with_degeneracy(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        m = 20
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(1, m + 1) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [0, 0]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0, desired_x=np.zeros(m))

    def test_enzo_example_c_with_unboundedness(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [0, 0]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_unbounded(res)

    def test_enzo_example_c_with_infeasibility(self):
        # rescued from https://github.com/scipy/scipy/pull/218
        m = 50
        c = -np.ones(m)
        tmp = 2 * np.pi * np.arange(m) / (m + 1)
        A_eq = np.vstack((np.cos(tmp) - 1, np.sin(tmp)))
        b_eq = [1, 1]

        o = {key: self.options[key] for key in self.options}
        o["presolve"] = False

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)
        _assert_infeasible(res)

    def test_basic_artificial_vars(self):
        # Problem is chosen to test two phase simplex methods when at the end
        # of phase 1 some artificial variables remain in the basis.
        # Also, for `method='simplex'`, the row in the tableau corresponding
        # with the artificial variables is not all zero.
        c = np.array([-0.1, -0.07, 0.004, 0.004, 0.004, 0.004])
        A_ub = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0],
                         [0, -1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0],
                         [1.0, 1.0, 0, 0, 0, 0]])
        b_ub = np.array([3.0, 3.0, 3.0, 3.0, 20.0])
        A_eq = np.array([[1.0, 0, -1, 1, -1, 1], [0, -1.0, -1, 1, -1, 1]])
        b_eq = np.array([0, 0])
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=0, desired_x=np.zeros_like(c),
                        atol=2e-6)

    def test_optimize_result(self):
        # check all fields in OptimizeResult
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(0)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)
        assert_(res.success)
        assert_(res.nit)
        assert_(not res.status)
        assert_(res.message == "Optimization terminated successfully.")
        assert_allclose(c @ res.x, res.fun)
        assert_allclose(b_eq - A_eq @ res.x, res.con, atol=1e-11)
        assert_allclose(b_ub - A_ub @ res.x, res.slack, atol=1e-11)

    #################
    # Bug Fix Tests #
    #################

    def test_bug_5400(self):
        # https://github.com/scipy/scipy/issues/5400
        bounds = [
            (0, None),
            (0, 100), (0, 100), (0, 100), (0, 100), (0, 100), (0, 100),
            (0, 900), (0, 900), (0, 900), (0, 900), (0, 900), (0, 900),
            (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

        f = 1 / 9
        g = -1e4
        h = -3.1
        A_ub = np.array([
            [1, -2.99, 0, 0, -3, 0, 0, 0, -1, -1, 0, -1, -1, 1, 1, 0, 0, 0, 0],
            [1, 0, -2.9, h, 0, -3, 0, -1, 0, 0, -1, 0, -1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, h, 0, 0, -3, -1, -1, 0, -1, -1, 0, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
            [0, 1.99, -1, -1, 0, 0, 0, -1, f, f, 0, 0, 0, g, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, -1, -1, 0, 0, 0, -1, f, f, 0, g, 0, 0, 0, 0],
            [0, -1, 1.9, 2.1, 0, 0, 0, f, -1, -1, 0, 0, 0, 0, 0, g, 0, 0, 0],
            [0, 0, 0, 0, -1, 2, -1, 0, 0, 0, f, -1, f, 0, 0, 0, g, 0, 0],
            [0, -1, -1, 2.1, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, 0, 0, g, 0],
            [0, 0, 0, 0, -1, -1, 2, 0, 0, 0, f, f, -1, 0, 0, 0, 0, 0, g]])

        b_ub = np.array([
            0.0, 0, 0, 100, 100, 100, 100, 100, 100, 900, 900, 900, 900, 900,
            900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        c = np.array([-1.0, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning,
                       "Solving system with option 'sym_pos'")
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_fun=-106.63507541835018)

    def test_bug_6139(self):
        # linprog(method='simplex') fails to find a basic feasible solution
        # if phase 1 pseudo-objective function is outside the provided tol.
        # https://github.com/scipy/scipy/issues/6139

        # Note: This is not strictly a bug as the default tolerance determines
        # if a result is "close enough" to zero and should not be expected
        # to work for all cases.

        c = np.array([1, 1, 1])
        A_eq = np.array([[1., 0., 0.], [-1000., 0., - 1000.]])
        b_eq = np.array([5.00000000e+00, -1.00000000e+04])
        A_ub = -np.array([[0., 1000000., 1010000.]])
        b_ub = -np.array([10000000.])
        bounds = (None, None)

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)

        _assert_success(res, desired_fun=14.95,
                        desired_x=np.array([5, 4.95, 5]))

    def test_bug_6690(self):
        # linprog simplex used to violate bound constraint despite reporting
        # success.
        # https://github.com/scipy/scipy/issues/6690

        A_eq = np.array([[0, 0, 0, 0.93, 0, 0.65, 0, 0, 0.83, 0]])
        b_eq = np.array([0.9626])
        A_ub = np.array([
            [0, 0, 0, 1.18, 0, 0, 0, -0.2, 0, -0.22],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.43, 0, 0, 0, 0, 0, 0],
            [0, -1.22, -0.25, 0, 0, 0, -2.06, 0, 0, 1.37],
            [0, 0, 0, 0, 0, 0, 0, -0.25, 0, 0]
        ])
        b_ub = np.array([0.615, 0, 0.172, -0.869, -0.022])
        bounds = np.array([
            [-0.84, -0.97, 0.34, 0.4, -0.33, -0.74, 0.47, 0.09, -1.45, -0.73],
            [0.37, 0.02, 2.86, 0.86, 1.18, 0.5, 1.76, 0.17, 0.32, -0.15]
        ]).T
        c = np.array([
            -1.64, 0.7, 1.8, -1.06, -1.16, 0.26, 2.13, 1.53, 0.66, 0.28
            ])

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(OptimizeWarning,
                       "Solving system with option 'cholesky'")
            sup.filter(OptimizeWarning, "Solving system with option 'sym_pos'")
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)

        desired_fun = -1.19099999999
        desired_x = np.array([0.3700, -0.9700, 0.3400, 0.4000, 1.1800,
                              0.5000, 0.4700, 0.0900, 0.3200, -0.7300])
        _assert_success(res, desired_fun=desired_fun, desired_x=desired_x)

        # Add small tol value to ensure arrays are less than or equal.
        atol = 1e-6
        assert_array_less(bounds[:, 0] - atol, res.x)
        assert_array_less(res.x, bounds[:, 1] + atol)

    def test_bug_7044(self):
        # linprog simplex failed to "identify correct constraints" (?)
        # leading to a non-optimal solution if A is rank-deficient.
        # https://github.com/scipy/scipy/issues/7044

        A_eq, b_eq, c, N = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)

        desired_fun = 1.730550597
        _assert_success(res, desired_fun=desired_fun)
        assert_allclose(A_eq.dot(res.x), b_eq)
        assert_array_less(np.zeros(res.x.size) - 1e-5, res.x)

    def test_bug_7237(self):
        # https://github.com/scipy/scipy/issues/7237
        # linprog simplex "explodes" when the pivot value is very
        # close to zero.

        c = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0])
        A_ub = np.array([
            [1., -724., 911., -551., -555., -896., 478., -80., -293.],
            [1., 566., 42., 937., 233., 883., 392., -909., 57.],
            [1., -208., -894., 539., 321., 532., -924., 942., 55.],
            [1., 857., -859., 83., 462., -265., -971., 826., 482.],
            [1., 314., -424., 245., -424., 194., -443., -104., -429.],
            [1., 540., 679., 361., 149., -827., 876., 633., 302.],
            [0., -1., -0., -0., -0., -0., -0., -0., -0.],
            [0., -0., -1., -0., -0., -0., -0., -0., -0.],
            [0., -0., -0., -1., -0., -0., -0., -0., -0.],
            [0., -0., -0., -0., -1., -0., -0., -0., -0.],
            [0., -0., -0., -0., -0., -1., -0., -0., -0.],
            [0., -0., -0., -0., -0., -0., -1., -0., -0.],
            [0., -0., -0., -0., -0., -0., -0., -1., -0.],
            [0., -0., -0., -0., -0., -0., -0., -0., -1.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1.]
            ])
        b_ub = np.array([
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.])
        A_eq = np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1.]])
        b_eq = np.array([[1.]])
        bounds = [(None, None)] * 9

        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_fun=108.568535, atol=1e-6)

    def test_bug_8174(self):
        # https://github.com/scipy/scipy/issues/8174
        # The simplex method sometimes "explodes" if the pivot value is very
        # close to zero.
        A_ub = np.array([
            [22714, 1008, 13380, -2713.5, -1116],
            [-4986, -1092, -31220, 17386.5, 684],
            [-4986, 0, 0, -2713.5, 0],
            [22714, 0, 0, 17386.5, 0]])
        b_ub = np.zeros(A_ub.shape[0])
        c = -np.ones(A_ub.shape[1])
        bounds = [(0, 1)] * A_ub.shape[1]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)

        if self.options.get('tol', 1e-9) < 1e-10 and self.method == 'simplex':
            _assert_unable_to_find_basic_feasible_sol(res)
        else:
            _assert_success(res, desired_fun=-2.0080717488789235, atol=1e-6)

    def test_bug_8174_2(self):
        # Test supplementary example from issue 8174.
        # https://github.com/scipy/scipy/issues/8174
        # https://stackoverflow.com/questions/47717012/linprog-in-scipy-optimize-checking-solution
        c = np.array([1, 0, 0, 0, 0, 0, 0])
        A_ub = -np.identity(7)
        b_ub = np.array([[-2], [-2], [-2], [-2], [-2], [-2], [-2]])
        A_eq = np.array([
            [1, 1, 1, 1, 1, 1, 0],
            [0.3, 1.3, 0.9, 0, 0, 0, -1],
            [0.3, 0, 0, 0, 0, 0, -2/3],
            [0, 0.65, 0, 0, 0, 0, -1/15],
            [0, 0, 0.3, 0, 0, 0, -1/15]
        ])
        b_eq = np.array([[100], [0], [0], [0], [0]])

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_fun=43.3333333331385)

    def test_bug_8561(self):
        # Test that pivot row is chosen correctly when using Bland's rule
        # This was originally written for the simplex method with
        # Bland's rule only, but it doesn't hurt to test all methods/options
        # https://github.com/scipy/scipy/issues/8561
        c = np.array([7, 0, -4, 1.5, 1.5])
        A_ub = np.array([
            [4, 5.5, 1.5, 1.0, -3.5],
            [1, -2.5, -2, 2.5, 0.5],
            [3, -0.5, 4, -12.5, -7],
            [-1, 4.5, 2, -3.5, -2],
            [5.5, 2, -4.5, -1, 9.5]])
        b_ub = np.array([0, 0, 0, 0, 1])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=self.options,
                      method=self.method)
        _assert_success(res, desired_x=[0, 0, 19, 16/3, 29/3])

    def test_bug_8662(self):
        # linprog simplex used to report incorrect optimal results
        # https://github.com/scipy/scipy/issues/8662
        c = [-10, 10, 6, 3]
        A_ub = [[8, -8, -4, 6],
                [-8, 8, 4, -6],
                [-4, 4, 8, -4],
                [3, -3, -3, -10]]
        b_ub = [9, -9, -9, -4]
        bounds = [(0, None), (0, None), (0, None), (0, None)]
        desired_fun = 36.0000000000

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res1 = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method, options=self.options)

        # Set boundary condition as a constraint
        A_ub.append([0, 0, -1, 0])
        b_ub.append(0)
        bounds[2] = (None, None)

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res2 = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method, options=self.options)
        rtol = 1e-5
        _assert_success(res1, desired_fun=desired_fun, rtol=rtol)
        _assert_success(res2, desired_fun=desired_fun, rtol=rtol)

    def test_bug_8663(self):
        # exposed a bug in presolve
        # https://github.com/scipy/scipy/issues/8663
        c = [1, 5]
        A_eq = [[0, -7]]
        b_eq = [-6]
        bounds = [(0, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[0, 6./7], desired_fun=5*6./7)

    def test_bug_8664(self):
        # interior-point has trouble with this when presolve is off
        # tested for interior-point with presolve off in TestLinprogIPSpecific
        # https://github.com/scipy/scipy/issues/8664
        c = [4]
        A_ub = [[2], [5]]
        b_ub = [4, 4]
        A_eq = [[0], [-8], [9]]
        b_eq = [3, 2, 10]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sup.filter(OptimizeWarning, "Solving system with option...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_infeasible(res)

    def test_bug_8973(self):
        """
        Test whether bug described at:
        https://github.com/scipy/scipy/issues/8973
        was fixed.
        """
        c = np.array([0, 0, 0, 1, -1])
        A_ub = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        b_ub = np.array([2, -2])
        bounds = [(None, None), (None, None), (None, None), (-1, 1), (-1, 1)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        # solution vector x is not unique
        _assert_success(res, desired_fun=-2)
        # HiGHS IPM had an issue where the following wasn't true!
        assert_equal(c @ res.x, res.fun)

    def test_bug_8973_2(self):
        """
        Additional test for:
        https://github.com/scipy/scipy/issues/8973
        suggested in
        https://github.com/scipy/scipy/pull/8985
        review by @antonior92
        """
        c = np.zeros(1)
        A_ub = np.array([[1]])
        b_ub = np.array([-2])
        bounds = (None, None)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options)
        _assert_success(res, desired_x=[-2], desired_fun=0)

    def test_bug_10124(self):
        """
        Test for linprog docstring problem
        'disp'=True caused revised simplex failure
        """
        c = np.zeros(1)
        A_ub = np.array([[1]])
        b_ub = np.array([-2])
        bounds = (None, None)
        c = [-1, 4]
        A_ub = [[-3, 1], [1, 2]]
        b_ub = [6, 4]
        bounds = [(None, None), (-3, None)]
        o = {"disp": True}
        o.update(self.options)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=o)
        _assert_success(res, desired_x=[10, -3], desired_fun=-22)

    def test_bug_10349(self):
        """
        Test for redundancy removal tolerance issue
        https://github.com/scipy/scipy/issues/10349
        """
        A_eq = np.array([[1, 1, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0],
                         [0, 1, 0, 0, 0, 1]])
        b_eq = np.array([221, 210, 10, 141, 198, 102])
        c = np.concatenate((0, 1, np.zeros(4)), axis=None)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=self.options)
        _assert_success(res, desired_x=[129, 92, 12, 198, 0, 10], desired_fun=92)

    @pytest.mark.skip(sys.platform == 'darwin',
                      reason="Failing on some local macOS builds, see gh-13846")
    def test_bug_10466(self):
        """
        Test that autoscale fixes poorly-scaled problem
        """
        c = [-8., -0., -8., -0., -8., -0., -0., -0., -0., -0., -0., -0., -0.]
        A_eq = [[1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 1., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0.],
                [1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.]]

        b_eq = [3.14572800e+08, 4.19430400e+08, 5.24288000e+08,
                1.00663296e+09, 1.07374182e+09, 1.07374182e+09,
                1.07374182e+09, 1.07374182e+09, 1.07374182e+09,
                1.07374182e+09]

        o = {}
        # HiGHS methods don't use autoscale option
        if not self.method.startswith("highs"):
            o = {"autoscale": True}
        o.update(self.options)

        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "Solving system with option...")
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            sup.filter(RuntimeWarning, "divide by zero encountered...")
            sup.filter(RuntimeWarning, "overflow encountered...")
            sup.filter(RuntimeWarning, "invalid value encountered...")
            sup.filter(LinAlgWarning, "Ill-conditioned matrix...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=o)
        assert_allclose(res.fun, -8589934560)

#########################
# Method-specific Tests #
#########################


class LinprogSimplexTests(LinprogCommonTests):
    method = "simplex"


class LinprogIPTests(LinprogCommonTests):
    method = "interior-point"


class LinprogRSTests(LinprogCommonTests):
    method = "revised simplex"

    # Revised simplex does not reliably solve these problems.
    # Failure is intermittent due to the random choice of elements to complete
    # the basis after phase 1 terminates. In any case, linprog exists
    # gracefully, reporting numerical difficulties. I do not think this should
    # prevent revised simplex from being merged, as it solves the problems
    # most of the time and solves a broader range of problems than the existing
    # simplex implementation.
    # I believe that the root cause is the same for all three and that this
    # same issue prevents revised simplex from solving many other problems
    # reliably. Somehow the pivoting rule allows the algorithm to pivot into
    # a singular basis. I haven't been able to find a reference that
    # acknowledges this possibility, suggesting that there is a bug. On the
    # other hand, the pivoting rule is quite simple, and I can't find a
    # mistake, which suggests that this is a possibility with the pivoting
    # rule. Hopefully, a better pivoting rule will fix the issue.

    def test_bug_5400(self):
        pytest.skip("Intermittent failure acceptable.")

    def test_bug_8662(self):
        pytest.skip("Intermittent failure acceptable.")

    def test_network_flow(self):
        pytest.skip("Intermittent failure acceptable.")


class LinprogHiGHSTests(LinprogCommonTests):
    def test_callback(self):
        # this is the problem from test_callback
        cb = lambda res: None
        c = np.array([-3, -2])
        A_ub = [[2, 1], [1, 1], [1, 0]]
        b_ub = [10, 8, 4]
        assert_raises(NotImplementedError, linprog, c, A_ub=A_ub, b_ub=b_ub,
                      callback=cb, method=self.method)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method=self.method)
        _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])

    @pytest.mark.parametrize("options",
                             [{"maxiter": -1},
                              {"disp": -1},
                              {"presolve": -1},
                              {"time_limit": -1},
                              {"dual_feasibility_tolerance": -1},
                              {"primal_feasibility_tolerance": -1},
                              {"ipm_optimality_tolerance": -1},
                              {"simplex_dual_edge_weight_strategy": "ekki"},
                              ])
    def test_invalid_option_values(self, options):
        def f(options):
            linprog(1, method=self.method, options=options)
        options.update(self.options)
        assert_warns(OptimizeWarning, f, options=options)

    def test_crossover(self):
        c = np.array([1, 1]) * -1  # maximize
        A_ub = np.array([[1, 1]])
        b_ub = [1]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)
        # there should be nonzero crossover iterations for IPM (only)
        assert_equal(res.crossover_nit == 0, self.method != "highs-ipm")

    def test_marginals(self):
        # Ensure lagrange multipliers are correct by comparing the derivative
        # w.r.t. b_ub/b_eq/ub/lb to the reported duals.
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=0)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)
        lb, ub = bounds.T

        # sensitivity w.r.t. b_ub
        def f_bub(x):
            return linprog(c, A_ub, x, A_eq, b_eq, bounds,
                           method=self.method).fun

        dfdbub = approx_derivative(f_bub, b_ub, method='3-point', f0=res.fun)
        assert_allclose(res.ineqlin.marginals, dfdbub)

        # sensitivity w.r.t. b_eq
        def f_beq(x):
            return linprog(c, A_ub, b_ub, A_eq, x, bounds,
                           method=self.method).fun

        dfdbeq = approx_derivative(f_beq, b_eq, method='3-point', f0=res.fun)
        assert_allclose(res.eqlin.marginals, dfdbeq)

        # sensitivity w.r.t. lb
        def f_lb(x):
            bounds = np.array([x, ub]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method).fun

        with np.errstate(invalid='ignore'):
            # approx_derivative has trouble where lb is infinite
            dfdlb = approx_derivative(f_lb, lb, method='3-point', f0=res.fun)
            dfdlb[~np.isfinite(lb)] = 0

        assert_allclose(res.lower.marginals, dfdlb)

        # sensitivity w.r.t. ub
        def f_ub(x):
            bounds = np.array([lb, x]).T
            return linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                           method=self.method).fun

        with np.errstate(invalid='ignore'):
            dfdub = approx_derivative(f_ub, ub, method='3-point', f0=res.fun)
            dfdub[~np.isfinite(ub)] = 0

        assert_allclose(res.upper.marginals, dfdub)

    def test_dual_feasibility(self):
        # Ensure solution is dual feasible using marginals
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=42)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)

        # KKT dual feasibility equation from Theorem 1 from
        # http://www.personal.psu.edu/cxg286/LPKKT.pdf
        resid = (-c + A_ub.T @ res.ineqlin.marginals +
                 A_eq.T @ res.eqlin.marginals +
                 res.upper.marginals +
                 res.lower.marginals)
        assert_allclose(resid, 0, atol=1e-12)

    def test_complementary_slackness(self):
        # Ensure that the complementary slackness condition is satisfied.
        c, A_ub, b_ub, A_eq, b_eq, bounds = very_random_gen(seed=42)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method=self.method, options=self.options)

        # KKT complementary slackness equation from Theorem 1 from
        # http://www.personal.psu.edu/cxg286/LPKKT.pdf modified for
        # non-zero RHS
        assert np.allclose(res.ineqlin.marginals @ (b_ub - A_ub @ res.x), 0)


################################
# Simplex Option-Specific Tests#
################################


class TestLinprogSimplexDefault(LinprogSimplexTests):

    def setup_method(self):
        self.options = {}

    def test_bug_5400(self):
        pytest.skip("Simplex fails on this problem.")

    def test_bug_7237_low_tol(self):
        # Fails if the tolerance is too strict. Here, we test that
        # even if the solution is wrong, the appropriate error is raised.
        pytest.skip("Simplex fails on this problem.")

    def test_bug_8174_low_tol(self):
        # Fails if the tolerance is too strict. Here, we test that
        # even if the solution is wrong, the appropriate warning is issued.
        self.options.update({'tol': 1e-12})
        with pytest.warns(OptimizeWarning):
            super().test_bug_8174()


class TestLinprogSimplexBland(LinprogSimplexTests):

    def setup_method(self):
        self.options = {'bland': True}

    def test_bug_5400(self):
        pytest.skip("Simplex fails on this problem.")

    def test_bug_8174_low_tol(self):
        # Fails if the tolerance is too strict. Here, we test that
        # even if the solution is wrong, the appropriate error is raised.
        self.options.update({'tol': 1e-12})
        with pytest.raises(AssertionError):
            with pytest.warns(OptimizeWarning):
                super().test_bug_8174()


class TestLinprogSimplexNoPresolve(LinprogSimplexTests):

    def setup_method(self):
        self.options = {'presolve': False}

    is_32_bit = np.intp(0).itemsize < 8
    is_linux = sys.platform.startswith('linux')

    @pytest.mark.xfail(
        condition=is_32_bit and is_linux,
        reason='Fails with warning on 32-bit linux')
    def test_bug_5400(self):
        super().test_bug_5400()

    def test_bug_6139_low_tol(self):
        # Linprog(method='simplex') fails to find a basic feasible solution
        # if phase 1 pseudo-objective function is outside the provided tol.
        # https://github.com/scipy/scipy/issues/6139
        # Without ``presolve`` eliminating such rows the result is incorrect.
        self.options.update({'tol': 1e-12})
        with pytest.raises(AssertionError, match='linprog status 4'):
            return super().test_bug_6139()

    def test_bug_7237_low_tol(self):
        pytest.skip("Simplex fails on this problem.")

    def test_bug_8174_low_tol(self):
        # Fails if the tolerance is too strict. Here, we test that
        # even if the solution is wrong, the appropriate warning is issued.
        self.options.update({'tol': 1e-12})
        with pytest.warns(OptimizeWarning):
            super().test_bug_8174()

    def test_unbounded_no_nontrivial_constraints_1(self):
        pytest.skip("Tests behavior specific to presolve")

    def test_unbounded_no_nontrivial_constraints_2(self):
        pytest.skip("Tests behavior specific to presolve")


#######################################
# Interior-Point Option-Specific Tests#
#######################################


class TestLinprogIPDense(LinprogIPTests):
    options = {"sparse": False}


if has_cholmod:
    class TestLinprogIPSparseCholmod(LinprogIPTests):
        options = {"sparse": True, "cholesky": True}


if has_umfpack:
    class TestLinprogIPSparseUmfpack(LinprogIPTests):
        options = {"sparse": True, "cholesky": False}

        def test_bug_10466(self):
            pytest.skip("Autoscale doesn't fix everything, and that's OK.")

        def test_network_flow_limited_capacity(self):
            pytest.skip("Failing due to numerical issues on some platforms.")


class TestLinprogIPSparse(LinprogIPTests):
    options = {"sparse": True, "cholesky": False, "sym_pos": False}

    @pytest.mark.xfail_on_32bit("This test is sensitive to machine epsilon level "
                                "perturbations in linear system solution in "
                                "_linprog_ip._sym_solve.")
    def test_bug_6139(self):
        super().test_bug_6139()

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        # Test defined in base class, but can't mark as xfail there
        super().test_bug_6690()

    def test_magic_square_sparse_no_presolve(self):
        # test linprog with a problem with a rank-deficient A_eq matrix
        A_eq, b_eq, c, N = magic_square(3)
        bounds = (0, 1)

        with suppress_warnings() as sup:
            if has_umfpack:
                sup.filter(UmfpackWarning)
            sup.filter(MatrixRankWarning, "Matrix is exactly singular")
            sup.filter(OptimizeWarning, "Solving system with option...")

            o = {key: self.options[key] for key in self.options}
            o["presolve"] = False

            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options=o)
        _assert_success(res, desired_fun=1.730550597)

    def test_sparse_solve_options(self):
        # checking that problem is solved with all column permutation options
        A_eq, b_eq, c, N = magic_square(3)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(OptimizeWarning, "Invalid permc_spec option")
            o = {key: self.options[key] for key in self.options}
            permc_specs = ('NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A',
                           'COLAMD', 'ekki-ekki-ekki')
            # 'ekki-ekki-ekki' raises warning about invalid permc_spec option
            # and uses default
            for permc_spec in permc_specs:
                o["permc_spec"] = permc_spec
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                              method=self.method, options=o)
                _assert_success(res, desired_fun=1.730550597)


class TestLinprogIPSparsePresolve(LinprogIPTests):
    options = {"sparse": True, "_sparse_presolve": True}

    @pytest.mark.xfail_on_32bit("This test is sensitive to machine epsilon level "
                                "perturbations in linear system solution in "
                                "_linprog_ip._sym_solve.")
    def test_bug_6139(self):
        super().test_bug_6139()

    def test_enzo_example_c_with_infeasibility(self):
        pytest.skip('_sparse_presolve=True incompatible with presolve=False')

    @pytest.mark.xfail(reason='Fails with ATLAS, see gh-7877')
    def test_bug_6690(self):
        # Test defined in base class, but can't mark as xfail there
        super().test_bug_6690()


class TestLinprogIPSpecific:
    method = "interior-point"
    # the following tests don't need to be performed separately for
    # sparse presolve, sparse after presolve, and dense

    def test_solver_select(self):
        # check that default solver is selected as expected
        if has_cholmod:
            options = {'sparse': True, 'cholesky': True}
        elif has_umfpack:
            options = {'sparse': True, 'cholesky': False}
        else:
            options = {'sparse': True, 'cholesky': False, 'sym_pos': False}
        A, b, c = lpgen_2d(20, 20)
        res1 = linprog(c, A_ub=A, b_ub=b, method=self.method, options=options)
        res2 = linprog(c, A_ub=A, b_ub=b, method=self.method)  # default solver
        assert_allclose(res1.fun, res2.fun,
                        err_msg="linprog default solver unexpected result",
                        rtol=1e-15, atol=1e-15)

    def test_unbounded_below_no_presolve_original(self):
        # formerly caused segfault in TravisCI w/ "cholesky":True
        c = [-1]
        bounds = [(None, 1)]
        res = linprog(c=c, bounds=bounds,
                      method=self.method,
                      options={"presolve": False, "cholesky": True})
        _assert_success(res, desired_fun=-1)

    def test_cholesky(self):
        # use cholesky factorization and triangular solves
        A, b, c = lpgen_2d(20, 20)
        res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                      options={"cholesky": True})  # only for dense
        _assert_success(res, desired_fun=-64.049494229)

    def test_alternate_initial_point(self):
        # use "improved" initial point
        A, b, c = lpgen_2d(20, 20)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll...")
            sup.filter(OptimizeWarning, "Solving system with option...")
            sup.filter(LinAlgWarning, "Ill-conditioned matrix...")
            res = linprog(c, A_ub=A, b_ub=b, method=self.method,
                          options={"ip": True, "disp": True})
            # ip code is independent of sparse/dense
        _assert_success(res, desired_fun=-64.049494229)

    def test_bug_8664(self):
        # interior-point has trouble with this when presolve is off
        c = [4]
        A_ub = [[2], [5]]
        b_ub = [4, 4]
        A_eq = [[0], [-8], [9]]
        b_eq = [3, 2, 10]
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sup.filter(OptimizeWarning, "Solving system with option...")
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                          method=self.method, options={"presolve": False})
        assert_(not res.success, "Incorrectly reported success")


########################################
# Revised Simplex Option-Specific Tests#
########################################


class TestLinprogRSCommon(LinprogRSTests):
    options = {}

    def test_cyclic_bland(self):
        pytest.skip("Intermittent failure acceptable.")

    def test_nontrivial_problem_with_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_unbounded_variables(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bounds = [(None, None), (None, None), (0, None), (None, None)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bounded_variables(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bounds = [(None, 1), (1, None), (0, None), (.4, .6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_negative_unbounded_variable(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        b_eq = [4]
        x_star = np.array([-219/385, 582/385, 0, 4/10])
        f_star = 3951/385
        bounds = [(None, None), (1, None), (0, None), (.4, .6)]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bad_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bad_guess = [1, 2, 3, .5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=bad_guess)
        assert_equal(res.status, 6)

    def test_redundant_constraints_with_guess(self):
        A, b, c, N = magic_square(3)
        p = np.random.rand(*c.shape)
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear...")
            sup.filter(RuntimeWarning, "invalid value encountered")
            sup.filter(LinAlgWarning)
            res = linprog(c, A_eq=A, b_eq=b, method=self.method)
            res2 = linprog(c, A_eq=A, b_eq=b, method=self.method, x0=res.x)
            res3 = linprog(c + p, A_eq=A, b_eq=b, method=self.method, x0=res.x)
        _assert_success(res2, desired_fun=1.730550597)
        assert_equal(res2.nit, 0)
        _assert_success(res3)
        assert_(res3.nit < res.nit)  # hot start reduces iterations


class TestLinprogRSBland(LinprogRSTests):
    options = {"pivot": "bland"}


############################################
# HiGHS-Simplex-Dual Option-Specific Tests #
############################################


class TestLinprogHiGHSSimplexDual(LinprogHiGHSTests):
    method = "highs-ds"
    options = {}

    def test_lad_regression(self):
        '''The scaled model should be optimal but unscaled model infeasible.'''
        c, A_ub, b_ub, bnds = l1_regression_prob()
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bnds,
                      method=self.method, options=self.options)
        assert_equal(res.status, 4)
        assert_('An optimal solution to the scaled '
                'model was found but' in res.message)
        assert_(res.x is not None)
        assert_(np.all(res.slack > -1e-6))
        assert_(np.all(res.x <= [np.inf if u is None else u for l, u in bnds]))
        assert_(np.all(res.x >= [-np.inf if l is None else l for l, u in bnds]))


###################################
# HiGHS-IPM Option-Specific Tests #
###################################


class TestLinprogHiGHSIPM(LinprogHiGHSTests):
    method = "highs-ipm"
    options = {}


###########################
# Autoscale-Specific Tests#
###########################


class AutoscaleTests:
    options = {"autoscale": True}

    test_bug_6139 = LinprogCommonTests.test_bug_6139
    test_bug_6690 = LinprogCommonTests.test_bug_6690
    test_bug_7237 = LinprogCommonTests.test_bug_7237


class TestAutoscaleIP(AutoscaleTests):
    method = "interior-point"

    def test_bug_6139(self):
        self.options['tol'] = 1e-10
        return AutoscaleTests.test_bug_6139(self)


class TestAutoscaleSimplex(AutoscaleTests):
    method = "simplex"


class TestAutoscaleRS(AutoscaleTests):
    method = "revised simplex"

    def test_nontrivial_problem_with_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=x_star)
        _assert_success(res, desired_fun=f_star, desired_x=x_star)
        assert_equal(res.nit, 0)

    def test_nontrivial_problem_with_bad_guess(self):
        c, A_ub, b_ub, A_eq, b_eq, x_star, f_star = nontrivial_problem()
        bad_guess = [1, 2, 3, .5]
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds,
                      method=self.method, options=self.options, x0=bad_guess)
        assert_equal(res.status, 6)


###########################
# Redundancy Removal Tests#
###########################


class RRTests:
    method = "interior-point"
    LCT = LinprogCommonTests
    # these are a few of the existing tests that have redundancy
    test_RR_infeasibility = LCT.test_remove_redundancy_infeasibility
    test_bug_10349 = LCT.test_bug_10349
    test_bug_7044 = LCT.test_bug_7044
    test_NFLC = LCT.test_network_flow_limited_capacity
    test_enzo_example_b = LCT.test_enzo_example_b


class TestRRSVD(RRTests):
    options = {"rr_method": "SVD"}


class TestRRPivot(RRTests):
    options = {"rr_method": "pivot"}


class TestRRID(RRTests):
    options = {"rr_method": "ID"}
