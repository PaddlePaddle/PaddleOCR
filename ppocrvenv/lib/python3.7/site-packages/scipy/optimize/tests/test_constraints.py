import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
    Bounds, LinearConstraint, NonlinearConstraint, PreparedConstraint,
    new_bounds_to_old, old_bound_to_new, strict_bounds)


class TestStrictBounds(TestCase):
    def test_scalarvalue_unique_enforce_feasibility(self):
        m = 3
        lb = 2
        ub = 4
        enforce_feasibility = False
        strict_lb, strict_ub = strict_bounds(lb, ub,
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [-np.inf, -np.inf, -np.inf])
        assert_array_equal(strict_ub, [np.inf, np.inf, np.inf])

        enforce_feasibility = True
        strict_lb, strict_ub = strict_bounds(lb, ub,
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [2, 2, 2])
        assert_array_equal(strict_ub, [4, 4, 4])

    def test_vectorvalue_unique_enforce_feasibility(self):
        m = 3
        lb = [1, 2, 3]
        ub = [4, 5, 6]
        enforce_feasibility = False
        strict_lb, strict_ub = strict_bounds(lb, ub,
                                              enforce_feasibility,
                                              m)
        assert_array_equal(strict_lb, [-np.inf, -np.inf, -np.inf])
        assert_array_equal(strict_ub, [np.inf, np.inf, np.inf])

        enforce_feasibility = True
        strict_lb, strict_ub = strict_bounds(lb, ub,
                                              enforce_feasibility,
                                              m)
        assert_array_equal(strict_lb, [1, 2, 3])
        assert_array_equal(strict_ub, [4, 5, 6])

    def test_scalarvalue_vector_enforce_feasibility(self):
        m = 3
        lb = 2
        ub = 4
        enforce_feasibility = [False, True, False]
        strict_lb, strict_ub = strict_bounds(lb, ub,
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [-np.inf, 2, -np.inf])
        assert_array_equal(strict_ub, [np.inf, 4, np.inf])

    def test_vectorvalue_vector_enforce_feasibility(self):
        m = 3
        lb = [1, 2, 3]
        ub = [4, 6, np.inf]
        enforce_feasibility = [True, False, True]
        strict_lb, strict_ub = strict_bounds(lb, ub,
                                             enforce_feasibility,
                                             m)
        assert_array_equal(strict_lb, [1, -np.inf, 3])
        assert_array_equal(strict_ub, [4, np.inf, np.inf])


def test_prepare_constraint_infeasible_x0():
    lb = np.array([0, 20, 30])
    ub = np.array([0.5, np.inf, 70])
    x0 = np.array([1, 2, 3])
    enforce_feasibility = np.array([False, True, True], dtype=bool)
    bounds = Bounds(lb, ub, enforce_feasibility)
    pytest.raises(ValueError, PreparedConstraint, bounds, x0)

    pc = PreparedConstraint(Bounds(lb, ub), [1, 2, 3])
    assert (pc.violation([1, 2, 3]) > 0).any()
    assert (pc.violation([0.25, 21, 31]) == 0).all()

    x0 = np.array([1, 2, 3, 4])
    A = np.array([[1, 2, 3, 4], [5, 0, 0, 6], [7, 0, 8, 0]])
    enforce_feasibility = np.array([True, True, True], dtype=bool)
    linear = LinearConstraint(A, -np.inf, 0, enforce_feasibility)
    pytest.raises(ValueError, PreparedConstraint, linear, x0)

    pc = PreparedConstraint(LinearConstraint(A, -np.inf, 0),
                            [1, 2, 3, 4])
    assert (pc.violation([1, 2, 3, 4]) > 0).any()
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()

    def fun(x):
        return A.dot(x)

    def jac(x):
        return A

    def hess(x, v):
        return sps.csr_matrix((4, 4))

    nonlinear = NonlinearConstraint(fun, -np.inf, 0, jac, hess,
                                    enforce_feasibility)
    pytest.raises(ValueError, PreparedConstraint, nonlinear, x0)

    pc = PreparedConstraint(nonlinear, [-10, 2, -10, 4])
    assert (pc.violation([1, 2, 3, 4]) > 0).any()
    assert (pc.violation([-10, 2, -10, 4]) == 0).all()


def test_violation():
    def cons_f(x):
        return np.array([x[0] ** 2 + x[1], x[0] ** 2 - x[1]])

    nlc = NonlinearConstraint(cons_f, [-1, -0.8500], [2, 2])
    pc = PreparedConstraint(nlc, [0.5, 1])

    assert_array_equal(pc.violation([0.5, 1]), [0., 0.])

    np.testing.assert_almost_equal(pc.violation([0.5, 1.2]), [0., 0.1])

    np.testing.assert_almost_equal(pc.violation([1.2, 1.2]), [0.64, 0])

    np.testing.assert_almost_equal(pc.violation([0.1, -1.2]), [0.19, 0])

    np.testing.assert_almost_equal(pc.violation([0.1, 2]), [0.01, 1.14])


def test_new_bounds_to_old():
    lb = np.array([-np.inf, 2, 3])
    ub = np.array([3, np.inf, 10])

    bounds = [(None, 3), (2, None), (3, 10)]
    assert_array_equal(new_bounds_to_old(lb, ub, 3), bounds)

    bounds_single_lb = [(-1, 3), (-1, None), (-1, 10)]
    assert_array_equal(new_bounds_to_old(-1, ub, 3), bounds_single_lb)

    bounds_no_lb = [(None, 3), (None, None), (None, 10)]
    assert_array_equal(new_bounds_to_old(-np.inf, ub, 3), bounds_no_lb)

    bounds_single_ub = [(None, 20), (2, 20), (3, 20)]
    assert_array_equal(new_bounds_to_old(lb, 20, 3), bounds_single_ub)

    bounds_no_ub = [(None, None), (2, None), (3, None)]
    assert_array_equal(new_bounds_to_old(lb, np.inf, 3), bounds_no_ub)

    bounds_single_both = [(1, 2), (1, 2), (1, 2)]
    assert_array_equal(new_bounds_to_old(1, 2, 3), bounds_single_both)

    bounds_no_both = [(None, None), (None, None), (None, None)]
    assert_array_equal(new_bounds_to_old(-np.inf, np.inf, 3), bounds_no_both)


def test_old_bounds_to_new():
    bounds = ([1, 2], (None, 3), (-1, None))
    lb_true = np.array([1, -np.inf, -1])
    ub_true = np.array([2, 3, np.inf])

    lb, ub = old_bound_to_new(bounds)
    assert_array_equal(lb, lb_true)
    assert_array_equal(ub, ub_true)

    bounds = [(-np.inf, np.inf), (np.array([1]), np.array([1]))]
    lb, ub = old_bound_to_new(bounds)

    assert_array_equal(lb, [-np.inf, 1])
    assert_array_equal(ub, [np.inf, 1])


def test_bounds_repr():
    from numpy import array, inf  # so that eval works
    for args in (
        (-1.0, 5.0),
        (-1.0, np.inf, True),
        (np.array([1.0, -np.inf]), np.array([2.0, np.inf])),
        (np.array([1.0, -np.inf]), np.array([2.0, np.inf]), np.array([True, False])),
    ):
        bounds = Bounds(*args)
        bounds2 = eval(repr(Bounds(*args)))
        assert_array_equal(bounds.lb, bounds2.lb)
        assert_array_equal(bounds.ub, bounds2.ub)
        assert_array_equal(bounds.keep_feasible, bounds2.keep_feasible)


def test_Bounds_array():
    # gh13501
    b = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])
    assert isinstance(b.lb, np.ndarray)
    assert isinstance(b.ub, np.ndarray)
