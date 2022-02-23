import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.optimize._constraints import (NonlinearConstraint, Bounds,
                                         PreparedConstraint)
from scipy.optimize._trustregion_constr.canonical_constraint \
    import CanonicalConstraint, initial_constraints_as_canonical


def create_quadratic_function(n, m, rng):
    a = rng.rand(m)
    A = rng.rand(m, n)
    H = rng.rand(m, n, n)
    HT = np.transpose(H, (1, 2, 0))

    def fun(x):
        return a + A.dot(x) + 0.5 * H.dot(x).dot(x)

    def jac(x):
        return A + H.dot(x)

    def hess(x, v):
        return HT.dot(v)

    return fun, jac, hess


def test_bounds_cases():
    # Test 1: no constraints.
    user_constraint = Bounds(-np.inf, np.inf)
    x0 = np.array([-1, 2])
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)

    assert_equal(c.n_eq, 0)
    assert_equal(c.n_ineq, 0)

    c_eq, c_ineq = c.fun(x0)
    assert_array_equal(c_eq, [])
    assert_array_equal(c_ineq, [])

    J_eq, J_ineq = c.jac(x0)
    assert_array_equal(J_eq, np.empty((0, 2)))
    assert_array_equal(J_ineq, np.empty((0, 2)))

    assert_array_equal(c.keep_feasible, [])

    # Test 2: infinite lower bound.
    user_constraint = Bounds(-np.inf, [0, np.inf, 1], [False, True, True])
    x0 = np.array([-1, -2, -3], dtype=float)
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)

    assert_equal(c.n_eq, 0)
    assert_equal(c.n_ineq, 2)

    c_eq, c_ineq = c.fun(x0)
    assert_array_equal(c_eq, [])
    assert_array_equal(c_ineq, [-1, -4])

    J_eq, J_ineq = c.jac(x0)
    assert_array_equal(J_eq, np.empty((0, 3)))
    assert_array_equal(J_ineq, np.array([[1, 0, 0], [0, 0, 1]]))

    assert_array_equal(c.keep_feasible, [False, True])

    # Test 3: infinite upper bound.
    user_constraint = Bounds([0, 1, -np.inf], np.inf, [True, False, True])
    x0 = np.array([1, 2, 3], dtype=float)
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)

    assert_equal(c.n_eq, 0)
    assert_equal(c.n_ineq, 2)

    c_eq, c_ineq = c.fun(x0)
    assert_array_equal(c_eq, [])
    assert_array_equal(c_ineq, [-1, -1])

    J_eq, J_ineq = c.jac(x0)
    assert_array_equal(J_eq, np.empty((0, 3)))
    assert_array_equal(J_ineq, np.array([[-1, 0, 0], [0, -1, 0]]))

    assert_array_equal(c.keep_feasible, [True, False])

    # Test 4: interval constraint.
    user_constraint = Bounds([-1, -np.inf, 2, 3], [1, np.inf, 10, 3],
                             [False, True, True, True])
    x0 = np.array([0, 10, 8, 5])
    prepared_constraint = PreparedConstraint(user_constraint, x0, False)
    c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)

    assert_equal(c.n_eq, 1)
    assert_equal(c.n_ineq, 4)

    c_eq, c_ineq = c.fun(x0)
    assert_array_equal(c_eq, [2])
    assert_array_equal(c_ineq, [-1, -2, -1, -6])

    J_eq, J_ineq = c.jac(x0)
    assert_array_equal(J_eq, [[0, 0, 0, 1]])
    assert_array_equal(J_ineq, [[1, 0, 0, 0],
                                [0, 0, 1, 0],
                                [-1, 0, 0, 0],
                                [0, 0, -1, 0]])

    assert_array_equal(c.keep_feasible, [False, True, False, True])


def test_nonlinear_constraint():
    n = 3
    m = 5
    rng = np.random.RandomState(0)
    x0 = rng.rand(n)

    fun, jac, hess = create_quadratic_function(n, m, rng)
    f = fun(x0)
    J = jac(x0)

    lb = [-10, 3, -np.inf, -np.inf, -5]
    ub = [10, 3, np.inf, 3, np.inf]
    user_constraint = NonlinearConstraint(
        fun, lb, ub, jac, hess, [True, False, False, True, False])

    for sparse_jacobian in [False, True]:
        prepared_constraint = PreparedConstraint(user_constraint, x0,
                                                 sparse_jacobian)
        c = CanonicalConstraint.from_PreparedConstraint(prepared_constraint)

        assert_array_equal(c.n_eq, 1)
        assert_array_equal(c.n_ineq, 4)

        c_eq, c_ineq = c.fun(x0)
        assert_array_equal(c_eq, [f[1] - lb[1]])
        assert_array_equal(c_ineq, [f[3] - ub[3], lb[4] - f[4],
                                    f[0] - ub[0], lb[0] - f[0]])

        J_eq, J_ineq = c.jac(x0)
        if sparse_jacobian:
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()

        assert_array_equal(J_eq, J[1, None])
        assert_array_equal(J_ineq, np.vstack((J[3], -J[4], J[0], -J[0])))

        v_eq = rng.rand(c.n_eq)
        v_ineq = rng.rand(c.n_ineq)
        v = np.zeros(m)
        v[1] = v_eq[0]
        v[3] = v_ineq[0]
        v[4] = -v_ineq[1]
        v[0] = v_ineq[2] - v_ineq[3]
        assert_array_equal(c.hess(x0, v_eq, v_ineq), hess(x0, v))

        assert_array_equal(c.keep_feasible, [True, False, True, True])


def test_concatenation():
    rng = np.random.RandomState(0)
    n = 4
    x0 = rng.rand(n)

    f1 = x0
    J1 = np.eye(n)
    lb1 = [-1, -np.inf, -2, 3]
    ub1 = [1, np.inf, np.inf, 3]
    bounds = Bounds(lb1, ub1, [False, False, True, False])

    fun, jac, hess = create_quadratic_function(n, 5, rng)
    f2 = fun(x0)
    J2 = jac(x0)
    lb2 = [-10, 3, -np.inf, -np.inf, -5]
    ub2 = [10, 3, np.inf, 5, np.inf]
    nonlinear = NonlinearConstraint(
        fun, lb2, ub2, jac, hess, [True, False, False, True, False])

    for sparse_jacobian in [False, True]:
        bounds_prepared = PreparedConstraint(bounds, x0, sparse_jacobian)
        nonlinear_prepared = PreparedConstraint(nonlinear, x0, sparse_jacobian)

        c1 = CanonicalConstraint.from_PreparedConstraint(bounds_prepared)
        c2 = CanonicalConstraint.from_PreparedConstraint(nonlinear_prepared)
        c = CanonicalConstraint.concatenate([c1, c2], sparse_jacobian)

        assert_equal(c.n_eq, 2)
        assert_equal(c.n_ineq, 7)

        c_eq, c_ineq = c.fun(x0)
        assert_array_equal(c_eq, [f1[3] - lb1[3], f2[1] - lb2[1]])
        assert_array_equal(c_ineq, [lb1[2] - f1[2], f1[0] - ub1[0],
                                    lb1[0] - f1[0], f2[3] - ub2[3],
                                    lb2[4] - f2[4], f2[0] - ub2[0],
                                    lb2[0] - f2[0]])

        J_eq, J_ineq = c.jac(x0)
        if sparse_jacobian:
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()

        assert_array_equal(J_eq, np.vstack((J1[3], J2[1])))
        assert_array_equal(J_ineq, np.vstack((-J1[2], J1[0], -J1[0], J2[3],
                                              -J2[4], J2[0], -J2[0])))

        v_eq = rng.rand(c.n_eq)
        v_ineq = rng.rand(c.n_ineq)
        v = np.zeros(5)
        v[1] = v_eq[1]
        v[3] = v_ineq[3]
        v[4] = -v_ineq[4]
        v[0] = v_ineq[5] - v_ineq[6]
        H = c.hess(x0, v_eq, v_ineq).dot(np.eye(n))
        assert_array_equal(H, hess(x0, v))

        assert_array_equal(c.keep_feasible,
                           [True, False, False, True, False, True, True])


def test_empty():
    x = np.array([1, 2, 3])
    c = CanonicalConstraint.empty(3)
    assert_equal(c.n_eq, 0)
    assert_equal(c.n_ineq, 0)

    c_eq, c_ineq = c.fun(x)
    assert_array_equal(c_eq, [])
    assert_array_equal(c_ineq, [])

    J_eq, J_ineq = c.jac(x)
    assert_array_equal(J_eq, np.empty((0, 3)))
    assert_array_equal(J_ineq, np.empty((0, 3)))

    H = c.hess(x, None, None).toarray()
    assert_array_equal(H, np.zeros((3, 3)))


def test_initial_constraints_as_canonical():
    # rng is only used to generate the coefficients of the quadratic
    # function that is used by the nonlinear constraint.
    rng = np.random.RandomState(0)

    x0 = np.array([0.5, 0.4, 0.3, 0.2])
    n = len(x0)

    lb1 = [-1, -np.inf, -2, 3]
    ub1 = [1, np.inf, np.inf, 3]
    bounds = Bounds(lb1, ub1, [False, False, True, False])

    fun, jac, hess = create_quadratic_function(n, 5, rng)
    lb2 = [-10, 3, -np.inf, -np.inf, -5]
    ub2 = [10, 3, np.inf, 5, np.inf]
    nonlinear = NonlinearConstraint(
        fun, lb2, ub2, jac, hess, [True, False, False, True, False])

    for sparse_jacobian in [False, True]:
        bounds_prepared = PreparedConstraint(bounds, x0, sparse_jacobian)
        nonlinear_prepared = PreparedConstraint(nonlinear, x0, sparse_jacobian)

        f1 = bounds_prepared.fun.f
        J1 = bounds_prepared.fun.J
        f2 = nonlinear_prepared.fun.f
        J2 = nonlinear_prepared.fun.J

        c_eq, c_ineq, J_eq, J_ineq = initial_constraints_as_canonical(
            n, [bounds_prepared, nonlinear_prepared], sparse_jacobian)

        assert_array_equal(c_eq, [f1[3] - lb1[3], f2[1] - lb2[1]])
        assert_array_equal(c_ineq, [lb1[2] - f1[2], f1[0] - ub1[0],
                                    lb1[0] - f1[0], f2[3] - ub2[3],
                                    lb2[4] - f2[4], f2[0] - ub2[0],
                                    lb2[0] - f2[0]])

        if sparse_jacobian:
            J1 = J1.toarray()
            J2 = J2.toarray()
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()

        assert_array_equal(J_eq, np.vstack((J1[3], J2[1])))
        assert_array_equal(J_ineq, np.vstack((-J1[2], J1[0], -J1[0], J2[3],
                                              -J2[4], J2[0], -J2[0])))


def test_initial_constraints_as_canonical_empty():
    n = 3
    for sparse_jacobian in [False, True]:
        c_eq, c_ineq, J_eq, J_ineq = initial_constraints_as_canonical(
            n, [], sparse_jacobian)

        assert_array_equal(c_eq, [])
        assert_array_equal(c_ineq, [])

        if sparse_jacobian:
            J_eq = J_eq.toarray()
            J_ineq = J_ineq.toarray()

        assert_array_equal(J_eq, np.empty((0, n)))
        assert_array_equal(J_ineq, np.empty((0, n)))
