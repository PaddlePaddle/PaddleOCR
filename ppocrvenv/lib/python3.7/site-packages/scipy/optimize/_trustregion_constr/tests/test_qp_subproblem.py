import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
    import (eqp_kktfact,
            projected_cg,
            box_intersections,
            sphere_intersections,
            box_sphere_intersections,
            modified_dogleg)
from scipy.optimize._trustregion_constr.projections \
    import projections
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest


class TestEQPDirectFactorization(TestCase):

    # From Example 16.2 Nocedal/Wright "Numerical
    # Optimization" p.452.
    def test_nocedal_example(self):
        H = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = -np.array([3, 0])
        x, lagrange_multipliers = eqp_kktfact(H, c, A, b)
        assert_array_almost_equal(x, [2, -1, 1])
        assert_array_almost_equal(lagrange_multipliers, [3, -2])


class TestSphericalBoundariesIntersections(TestCase):

    def test_2d_sphere_constraints(self):
        # Interior inicial point
        ta, tb, intersect = sphere_intersections([0, 0],
                                                 [1, 0], 0.5)
        assert_array_almost_equal([ta, tb], [0, 0.5])
        assert_equal(intersect, True)

        # No intersection between line and circle
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [0, 1], 1)
        assert_equal(intersect, False)

        # Outside initial point pointing toward outside the circle
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [1, 0], 1)
        assert_equal(intersect, False)

        # Outside initial point pointing toward inside the circle
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [-1, 0], 1.5)
        assert_array_almost_equal([ta, tb], [0.5, 1])
        assert_equal(intersect, True)

        # Initial point on the boundary
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [1, 0], 2)
        assert_array_almost_equal([ta, tb], [0, 0])
        assert_equal(intersect, True)

    def test_2d_sphere_constraints_line_intersections(self):
        # Interior initial point
        ta, tb, intersect = sphere_intersections([0, 0],
                                                 [1, 0], 0.5,
                                                 entire_line=True)
        assert_array_almost_equal([ta, tb], [-0.5, 0.5])
        assert_equal(intersect, True)

        # No intersection between line and circle
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [0, 1], 1,
                                                 entire_line=True)
        assert_equal(intersect, False)

        # Outside initial point pointing toward outside the circle
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [1, 0], 1,
                                                 entire_line=True)
        assert_array_almost_equal([ta, tb], [-3, -1])
        assert_equal(intersect, True)

        # Outside initial point pointing toward inside the circle
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [-1, 0], 1.5,
                                                 entire_line=True)
        assert_array_almost_equal([ta, tb], [0.5, 3.5])
        assert_equal(intersect, True)

        # Initial point on the boundary
        ta, tb, intersect = sphere_intersections([2, 0],
                                                 [1, 0], 2,
                                                 entire_line=True)
        assert_array_almost_equal([ta, tb], [-4, 0])
        assert_equal(intersect, True)


class TestBoxBoundariesIntersections(TestCase):

    def test_2d_box_constraints(self):
        # Box constraint in the direction of vector d
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [1, 1], [3, 3])
        assert_array_almost_equal([ta, tb], [0.5, 1])
        assert_equal(intersect, True)

        # Negative direction
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [1, -3], [3, -1])
        assert_equal(intersect, False)

        # Some constraints are absent (set to +/- inf)
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-np.inf, 1],
                                              [np.inf, np.inf])
        assert_array_almost_equal([ta, tb], [0.5, 1])
        assert_equal(intersect, True)

        # Intersect on the face of the box
        ta, tb, intersect = box_intersections([1, 0], [0, 1],
                                              [1, 1], [3, 3])
        assert_array_almost_equal([ta, tb], [1, 1])
        assert_equal(intersect, True)

        # Interior initial point
        ta, tb, intersect = box_intersections([0, 0], [4, 4],
                                              [-2, -3], [3, 2])
        assert_array_almost_equal([ta, tb], [0, 0.5])
        assert_equal(intersect, True)

        # No intersection between line and box constraints
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-3, -3], [-1, -1])
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-3, 3], [-1, 1])
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-3, -np.inf],
                                              [-1, np.inf])
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([0, 0], [1, 100],
                                              [1, 1], [3, 3])
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([0.99, 0], [0, 2],
                                                         [1, 1], [3, 3])
        assert_equal(intersect, False)

        # Initial point on the boundary
        ta, tb, intersect = box_intersections([2, 2], [0, 1],
                                              [-2, -2], [2, 2])
        assert_array_almost_equal([ta, tb], [0, 0])
        assert_equal(intersect, True)

    def test_2d_box_constraints_entire_line(self):
        # Box constraint in the direction of vector d
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [1, 1], [3, 3],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [0.5, 1.5])
        assert_equal(intersect, True)

        # Negative direction
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [1, -3], [3, -1],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [-1.5, -0.5])
        assert_equal(intersect, True)

        # Some constraints are absent (set to +/- inf)
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-np.inf, 1],
                                              [np.inf, np.inf],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [0.5, np.inf])
        assert_equal(intersect, True)

        # Intersect on the face of the box
        ta, tb, intersect = box_intersections([1, 0], [0, 1],
                                              [1, 1], [3, 3],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [1, 3])
        assert_equal(intersect, True)

        # Interior initial pointoint
        ta, tb, intersect = box_intersections([0, 0], [4, 4],
                                              [-2, -3], [3, 2],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [-0.5, 0.5])
        assert_equal(intersect, True)

        # No intersection between line and box constraints
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-3, -3], [-1, -1],
                                              entire_line=True)
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-3, 3], [-1, 1],
                                              entire_line=True)
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([2, 0], [0, 2],
                                              [-3, -np.inf],
                                              [-1, np.inf],
                                              entire_line=True)
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([0, 0], [1, 100],
                                              [1, 1], [3, 3],
                                              entire_line=True)
        assert_equal(intersect, False)
        ta, tb, intersect = box_intersections([0.99, 0], [0, 2],
                                              [1, 1], [3, 3],
                                              entire_line=True)
        assert_equal(intersect, False)

        # Initial point on the boundary
        ta, tb, intersect = box_intersections([2, 2], [0, 1],
                                              [-2, -2], [2, 2],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [-4, 0])
        assert_equal(intersect, True)

    def test_3d_box_constraints(self):
        # Simple case
        ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, 1],
                                              [1, 1, 1], [3, 3, 3])
        assert_array_almost_equal([ta, tb], [1, 1])
        assert_equal(intersect, True)

        # Negative direction
        ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, -1],
                                              [1, 1, 1], [3, 3, 3])
        assert_equal(intersect, False)

        # Interior point
        ta, tb, intersect = box_intersections([2, 2, 2], [0, -1, 1],
                                              [1, 1, 1], [3, 3, 3])
        assert_array_almost_equal([ta, tb], [0, 1])
        assert_equal(intersect, True)

    def test_3d_box_constraints_entire_line(self):
        # Simple case
        ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, 1],
                                              [1, 1, 1], [3, 3, 3],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [1, 3])
        assert_equal(intersect, True)

        # Negative direction
        ta, tb, intersect = box_intersections([1, 1, 0], [0, 0, -1],
                                              [1, 1, 1], [3, 3, 3],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [-3, -1])
        assert_equal(intersect, True)

        # Interior point
        ta, tb, intersect = box_intersections([2, 2, 2], [0, -1, 1],
                                              [1, 1, 1], [3, 3, 3],
                                              entire_line=True)
        assert_array_almost_equal([ta, tb], [-1, 1])
        assert_equal(intersect, True)


class TestBoxSphereBoundariesIntersections(TestCase):

    def test_2d_box_constraints(self):
        # Both constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-2, 2],
                                                     [-1, -2], [1, 2], 2,
                                                     entire_line=False)
        assert_array_almost_equal([ta, tb], [0, 0.5])
        assert_equal(intersect, True)

        # None of the constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-1, 1],
                                                     [-1, -3], [1, 3], 10,
                                                     entire_line=False)
        assert_array_almost_equal([ta, tb], [0, 1])
        assert_equal(intersect, True)

        # Box constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [-1, -3], [1, 3], 10,
                                                     entire_line=False)
        assert_array_almost_equal([ta, tb], [0, 0.5])
        assert_equal(intersect, True)

        # Spherical constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [-1, -3], [1, 3], 2,
                                                     entire_line=False)
        assert_array_almost_equal([ta, tb], [0, 0.25])
        assert_equal(intersect, True)

        # Infeasible problems
        ta, tb, intersect = box_sphere_intersections([2, 2], [-4, 4],
                                                     [-1, -3], [1, 3], 2,
                                                     entire_line=False)
        assert_equal(intersect, False)
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [2, 4], [2, 4], 2,
                                                     entire_line=False)
        assert_equal(intersect, False)

    def test_2d_box_constraints_entire_line(self):
        # Both constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-2, 2],
                                                     [-1, -2], [1, 2], 2,
                                                     entire_line=True)
        assert_array_almost_equal([ta, tb], [0, 0.5])
        assert_equal(intersect, True)

        # None of the constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-1, 1],
                                                     [-1, -3], [1, 3], 10,
                                                     entire_line=True)
        assert_array_almost_equal([ta, tb], [0, 2])
        assert_equal(intersect, True)

        # Box constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [-1, -3], [1, 3], 10,
                                                     entire_line=True)
        assert_array_almost_equal([ta, tb], [0, 0.5])
        assert_equal(intersect, True)

        # Spherical constraints are active
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [-1, -3], [1, 3], 2,
                                                     entire_line=True)
        assert_array_almost_equal([ta, tb], [0, 0.25])
        assert_equal(intersect, True)

        # Infeasible problems
        ta, tb, intersect = box_sphere_intersections([2, 2], [-4, 4],
                                                     [-1, -3], [1, 3], 2,
                                                     entire_line=True)
        assert_equal(intersect, False)
        ta, tb, intersect = box_sphere_intersections([1, 1], [-4, 4],
                                                     [2, 4], [2, 4], 2,
                                                     entire_line=True)
        assert_equal(intersect, False)


class TestModifiedDogleg(TestCase):

    def test_cauchypoint_equalsto_newtonpoint(self):
        A = np.array([[1, 8]])
        b = np.array([-16])
        _, _, Y = projections(A)
        newton_point = np.array([0.24615385, 1.96923077])

        # Newton point inside boundaries
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf], [np.inf, np.inf])
        assert_array_almost_equal(x, newton_point)

        # Spherical constraint active
        x = modified_dogleg(A, Y, b, 1, [-np.inf, -np.inf], [np.inf, np.inf])
        assert_array_almost_equal(x, newton_point/np.linalg.norm(newton_point))

        # Box constraints active
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf], [0.1, np.inf])
        assert_array_almost_equal(x, (newton_point/newton_point[0]) * 0.1)

    def test_3d_example(self):
        A = np.array([[1, 8, 1],
                      [4, 2, 2]])
        b = np.array([-16, 2])
        Z, LS, Y = projections(A)

        newton_point = np.array([-1.37090909, 2.23272727, -0.49090909])
        cauchy_point = np.array([0.11165723, 1.73068711, 0.16748585])
        origin = np.zeros_like(newton_point)

        # newton_point inside boundaries
        x = modified_dogleg(A, Y, b, 3, [-np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        assert_array_almost_equal(x, newton_point)

        # line between cauchy_point and newton_point contains best point
        # (spherical constraint is active).
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        z = cauchy_point
        d = newton_point-cauchy_point
        t = ((x-z)/(d))
        assert_array_almost_equal(t, np.full(3, 0.40807330))
        assert_array_almost_equal(np.linalg.norm(x), 2)

        # line between cauchy_point and newton_point contains best point
        # (box constraint is active).
        x = modified_dogleg(A, Y, b, 5, [-1, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        z = cauchy_point
        d = newton_point-cauchy_point
        t = ((x-z)/(d))
        assert_array_almost_equal(t, np.full(3, 0.7498195))
        assert_array_almost_equal(x[0], -1)

        # line between origin and cauchy_point contains best point
        # (spherical constraint is active).
        x = modified_dogleg(A, Y, b, 1, [-np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
        z = origin
        d = cauchy_point
        t = ((x-z)/(d))
        assert_array_almost_equal(t, np.full(3, 0.573936265))
        assert_array_almost_equal(np.linalg.norm(x), 1)

        # line between origin and newton_point contains best point
        # (box constraint is active).
        x = modified_dogleg(A, Y, b, 2, [-np.inf, -np.inf, -np.inf],
                            [np.inf, 1, np.inf])
        z = origin
        d = newton_point
        t = ((x-z)/(d))
        assert_array_almost_equal(t, np.full(3, 0.4478827364))
        assert_array_almost_equal(x[1], 1)


class TestProjectCG(TestCase):

    # From Example 16.2 Nocedal/Wright "Numerical
    # Optimization" p.452.
    def test_nocedal_example(self):
        H = csc_matrix([[6, 2, 1],
                        [2, 5, 2],
                        [1, 2, 4]])
        A = csc_matrix([[1, 0, 1],
                        [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b)
        assert_equal(info["stop_cond"], 4)
        assert_equal(info["hits_boundary"], False)
        assert_array_almost_equal(x, [2, -1, 1])

    def test_compare_with_direct_fact(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b, tol=0)
        x_kkt, _ = eqp_kktfact(H, c, A, b)
        assert_equal(info["stop_cond"], 1)
        assert_equal(info["hits_boundary"], False)
        assert_array_almost_equal(x, x_kkt)

    def test_trust_region_infeasible(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 1
        Z, _, Y = projections(A)
        with pytest.raises(ValueError):
            projected_cg(H, c, Z, Y, b, trust_radius=trust_radius)

    def test_trust_region_barely_feasible(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 2.32379000772445021283
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               trust_radius=trust_radius)
        assert_equal(info["stop_cond"], 2)
        assert_equal(info["hits_boundary"], True)
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)
        assert_array_almost_equal(x, -Y.dot(b))

    def test_hits_boundary(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 3
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               trust_radius=trust_radius)
        assert_equal(info["stop_cond"], 2)
        assert_equal(info["hits_boundary"], True)
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)

    def test_negative_curvature_unconstrained(self):
        H = csc_matrix([[1, 2, 1, 3],
                        [2, 0, 2, 4],
                        [1, 2, 0, 2],
                        [3, 4, 2, 0]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 0, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        with pytest.raises(ValueError):
            projected_cg(H, c, Z, Y, b, tol=0)

    def test_negative_curvature(self):
        H = csc_matrix([[1, 2, 1, 3],
                        [2, 0, 2, 4],
                        [1, 2, 0, 2],
                        [3, 4, 2, 0]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 0, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        trust_radius = 1000
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               trust_radius=trust_radius)
        assert_equal(info["stop_cond"], 3)
        assert_equal(info["hits_boundary"], True)
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)

    # The box constraints are inactive at the solution but
    # are active during the iterations.
    def test_inactive_box_constraints(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               lb=[0.5, -np.inf,
                                   -np.inf, -np.inf],
                               return_all=True)
        x_kkt, _ = eqp_kktfact(H, c, A, b)
        assert_equal(info["stop_cond"], 1)
        assert_equal(info["hits_boundary"], False)
        assert_array_almost_equal(x, x_kkt)

    # The box constraints active and the termination is
    # by maximum iterations (infeasible iteraction).
    def test_active_box_constraints_maximum_iterations_reached(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               lb=[0.8, -np.inf,
                                   -np.inf, -np.inf],
                               return_all=True)
        assert_equal(info["stop_cond"], 1)
        assert_equal(info["hits_boundary"], True)
        assert_array_almost_equal(A.dot(x), -b)
        assert_array_almost_equal(x[0], 0.8)

    # The box constraints are active and the termination is
    # because it hits boundary (without infeasible iteraction).
    def test_active_box_constraints_hits_boundaries(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 3
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               ub=[np.inf, np.inf, 1.6, np.inf],
                               trust_radius=trust_radius,
                               return_all=True)
        assert_equal(info["stop_cond"], 2)
        assert_equal(info["hits_boundary"], True)
        assert_array_almost_equal(x[2], 1.6)

    # The box constraints are active and the termination is
    # because it hits boundary (infeasible iteraction).
    def test_active_box_constraints_hits_boundaries_infeasible_iter(self):
        H = csc_matrix([[6, 2, 1, 3],
                        [2, 5, 2, 4],
                        [1, 2, 4, 5],
                        [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 4
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               ub=[np.inf, 0.1, np.inf, np.inf],
                               trust_radius=trust_radius,
                               return_all=True)
        assert_equal(info["stop_cond"], 2)
        assert_equal(info["hits_boundary"], True)
        assert_array_almost_equal(x[1], 0.1)

    # The box constraints are active and the termination is
    # because it hits boundary (no infeasible iteraction).
    def test_active_box_constraints_negative_curvature(self):
        H = csc_matrix([[1, 2, 1, 3],
                        [2, 0, 2, 4],
                        [1, 2, 0, 2],
                        [3, 4, 2, 0]])
        A = csc_matrix([[1, 0, 1, 0],
                        [0, 1, 0, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        trust_radius = 1000
        x, info = projected_cg(H, c, Z, Y, b,
                               tol=0,
                               ub=[np.inf, np.inf, 100, np.inf],
                               trust_radius=trust_radius)
        assert_equal(info["stop_cond"], 3)
        assert_equal(info["hits_boundary"], True)
        assert_array_almost_equal(x[2], 100)
