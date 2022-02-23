import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_array_equal, assert_array_less)
from scipy.optimize import (BFGS, SR1)


class Rosenbrock:
    """Rosenbrock function.

    The following optimization problem:
        minimize sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    """

    def __init__(self, n=2, random_state=0):
        rng = np.random.RandomState(random_state)
        self.x0 = rng.uniform(-1, 1, n)
        self.x_opt = np.ones(n)

    def fun(self, x):
        x = np.asarray(x)
        r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                   axis=0)
        return r

    def grad(self, x):
        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = (200 * (xm - xm_m1**2) -
                     400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
        der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        der[-1] = 200 * (x[-1] - x[-2]**2)
        return der

    def hess(self, x):
        x = np.atleast_1d(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        return H


class TestHessianUpdateStrategy(TestCase):

    def test_hessian_initialization(self):
        quasi_newton = (BFGS(), SR1())

        for qn in quasi_newton:
            qn.initialize(5, 'hess')
            B = qn.get_matrix()

            assert_array_equal(B, np.eye(5))

    # For this list of points, it is known
    # that no exception occur during the
    # Hessian update. Hence no update is
    # skiped or damped.
    def test_rosenbrock_with_no_exception(self):
        # Define auxiliar problem
        prob = Rosenbrock(n=5)
        # Define iteration points
        x_list = [[0.0976270, 0.4303787, 0.2055267, 0.0897663, -0.15269040],
                  [0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286],
                  [0.2142498, -0.0188480, 0.0503822, 0.0347033, 0.03323606],
                  [0.2071680, -0.0185071, 0.0341337, -0.0139298, 0.02881750],
                  [0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699],
                  [0.1382378, -0.0276671, 0.0266161, -0.0074060, 0.02801610],
                  [0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184],
                  [0.2354930, 0.0443711, 0.0173959, 0.0041872, 0.00794563],
                  [0.4168118, 0.1433867, 0.0111714, 0.0126265, -0.00658537],
                  [0.4681972, 0.2153273, 0.0225249, 0.0152704, -0.00463809],
                  [0.6023068, 0.3346815, 0.0731108, 0.0186618, -0.00371541],
                  [0.6415743, 0.3985468, 0.1324422, 0.0214160, -0.00062401],
                  [0.7503690, 0.5447616, 0.2804541, 0.0539851, 0.00242230],
                  [0.7452626, 0.5644594, 0.3324679, 0.0865153, 0.00454960],
                  [0.8059782, 0.6586838, 0.4229577, 0.1452990, 0.00976702],
                  [0.8549542, 0.7226562, 0.4991309, 0.2420093, 0.02772661],
                  [0.8571332, 0.7285741, 0.5279076, 0.2824549, 0.06030276],
                  [0.8835633, 0.7727077, 0.5957984, 0.3411303, 0.09652185],
                  [0.9071558, 0.8299587, 0.6771400, 0.4402896, 0.17469338],
                  [0.9190793, 0.8486480, 0.7163332, 0.5083780, 0.26107691],
                  [0.9371223, 0.8762177, 0.7653702, 0.5773109, 0.32181041],
                  [0.9554613, 0.9119893, 0.8282687, 0.6776178, 0.43162744],
                  [0.9545744, 0.9099264, 0.8270244, 0.6822220, 0.45237623],
                  [0.9688112, 0.9351710, 0.8730961, 0.7546601, 0.56622448],
                  [0.9743227, 0.9491953, 0.9005150, 0.8086497, 0.64505437],
                  [0.9807345, 0.9638853, 0.9283012, 0.8631675, 0.73812581],
                  [0.9886746, 0.9777760, 0.9558950, 0.9123417, 0.82726553],
                  [0.9899096, 0.9803828, 0.9615592, 0.9255600, 0.85822149],
                  [0.9969510, 0.9935441, 0.9864657, 0.9726775, 0.94358663],
                  [0.9979533, 0.9960274, 0.9921724, 0.9837415, 0.96626288],
                  [0.9995981, 0.9989171, 0.9974178, 0.9949954, 0.99023356],
                  [1.0002640, 1.0005088, 1.0010594, 1.0021161, 1.00386912],
                  [0.9998903, 0.9998459, 0.9997795, 0.9995484, 0.99916305],
                  [1.0000008, 0.9999905, 0.9999481, 0.9998903, 0.99978047],
                  [1.0000004, 0.9999983, 1.0000001, 1.0000031, 1.00000297],
                  [0.9999995, 1.0000003, 1.0000005, 1.0000001, 1.00000032],
                  [0.9999999, 0.9999997, 0.9999994, 0.9999989, 0.99999786],
                  [0.9999999, 0.9999999, 0.9999999, 0.9999999, 0.99999991]]
        # Get iteration points
        grad_list = [prob.grad(x) for x in x_list]
        delta_x = [np.array(x_list[i+1])-np.array(x_list[i])
                   for i in range(len(x_list)-1)]
        delta_grad = [grad_list[i+1]-grad_list[i]
                      for i in range(len(grad_list)-1)]
        # Check curvature condition
        for i in range(len(delta_x)):
            s = delta_x[i]
            y = delta_grad[i]
            if np.dot(s, y) <= 0:
                raise ArithmeticError()
        # Define QuasiNewton update
        for quasi_newton in (BFGS(init_scale=1, min_curvature=1e-4),
                             SR1(init_scale=1)):
            hess = deepcopy(quasi_newton)
            inv_hess = deepcopy(quasi_newton)
            hess.initialize(len(x_list[0]), 'hess')
            inv_hess.initialize(len(x_list[0]), 'inv_hess')
            # Compare the hessian and its inverse
            for i in range(len(delta_x)):
                s = delta_x[i]
                y = delta_grad[i]
                hess.update(s, y)
                inv_hess.update(s, y)
                B = hess.get_matrix()
                H = inv_hess.get_matrix()
                assert_array_almost_equal(np.linalg.inv(B), H, decimal=10)
            B_true = prob.hess(x_list[i+1])
            assert_array_less(norm(B - B_true)/norm(B_true), 0.1)

    def test_SR1_skip_update(self):
        # Define auxiliary problem
        prob = Rosenbrock(n=5)
        # Define iteration points
        x_list = [[0.0976270, 0.4303787, 0.2055267, 0.0897663, -0.15269040],
                  [0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286],
                  [0.2142498, -0.0188480, 0.0503822, 0.0347033, 0.03323606],
                  [0.2071680, -0.0185071, 0.0341337, -0.0139298, 0.02881750],
                  [0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699],
                  [0.1382378, -0.0276671, 0.0266161, -0.0074060, 0.02801610],
                  [0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184],
                  [0.2354930, 0.0443711, 0.0173959, 0.0041872, 0.00794563],
                  [0.4168118, 0.1433867, 0.0111714, 0.0126265, -0.00658537],
                  [0.4681972, 0.2153273, 0.0225249, 0.0152704, -0.00463809],
                  [0.6023068, 0.3346815, 0.0731108, 0.0186618, -0.00371541],
                  [0.6415743, 0.3985468, 0.1324422, 0.0214160, -0.00062401],
                  [0.7503690, 0.5447616, 0.2804541, 0.0539851, 0.00242230],
                  [0.7452626, 0.5644594, 0.3324679, 0.0865153, 0.00454960],
                  [0.8059782, 0.6586838, 0.4229577, 0.1452990, 0.00976702],
                  [0.8549542, 0.7226562, 0.4991309, 0.2420093, 0.02772661],
                  [0.8571332, 0.7285741, 0.5279076, 0.2824549, 0.06030276],
                  [0.8835633, 0.7727077, 0.5957984, 0.3411303, 0.09652185],
                  [0.9071558, 0.8299587, 0.6771400, 0.4402896, 0.17469338]]
        # Get iteration points
        grad_list = [prob.grad(x) for x in x_list]
        delta_x = [np.array(x_list[i+1])-np.array(x_list[i])
                   for i in range(len(x_list)-1)]
        delta_grad = [grad_list[i+1]-grad_list[i]
                      for i in range(len(grad_list)-1)]
        hess = SR1(init_scale=1, min_denominator=1e-2)
        hess.initialize(len(x_list[0]), 'hess')
        # Compare the Hessian and its inverse
        for i in range(len(delta_x)-1):
            s = delta_x[i]
            y = delta_grad[i]
            hess.update(s, y)
        # Test skip update
        B = np.copy(hess.get_matrix())
        s = delta_x[17]
        y = delta_grad[17]
        hess.update(s, y)
        B_updated = np.copy(hess.get_matrix())
        assert_array_equal(B, B_updated)

    def test_BFGS_skip_update(self):
        # Define auxiliar problem
        prob = Rosenbrock(n=5)
        # Define iteration points
        x_list = [[0.0976270, 0.4303787, 0.2055267, 0.0897663, -0.15269040],
                  [0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286],
                  [0.2142498, -0.0188480, 0.0503822, 0.0347033, 0.03323606],
                  [0.2071680, -0.0185071, 0.0341337, -0.0139298, 0.02881750],
                  [0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699],
                  [0.1382378, -0.0276671, 0.0266161, -0.0074060, 0.02801610],
                  [0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184]]
        # Get iteration points
        grad_list = [prob.grad(x) for x in x_list]
        delta_x = [np.array(x_list[i+1])-np.array(x_list[i])
                   for i in range(len(x_list)-1)]
        delta_grad = [grad_list[i+1]-grad_list[i]
                      for i in range(len(grad_list)-1)]
        hess = BFGS(init_scale=1, min_curvature=10)
        hess.initialize(len(x_list[0]), 'hess')
        # Compare the Hessian and its inverse
        for i in range(len(delta_x)-1):
            s = delta_x[i]
            y = delta_grad[i]
            hess.update(s, y)
        # Test skip update
        B = np.copy(hess.get_matrix())
        s = delta_x[5]
        y = delta_grad[5]
        hess.update(s, y)
        B_updated = np.copy(hess.get_matrix())
        assert_array_equal(B, B_updated)
