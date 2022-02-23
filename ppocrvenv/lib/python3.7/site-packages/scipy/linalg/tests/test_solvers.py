import os
import numpy as np

from numpy.testing import assert_array_almost_equal
import pytest
from pytest import raises as assert_raises

from scipy.linalg import solve_sylvester
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.linalg import block_diag, solve, LinAlgError
from scipy.sparse.sputils import matrix


def _load_data(name):
    """
    Load npz data file under data/
    Returns a copy of the data, rather than keeping the npz file open.
    """
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'data', name)
    with np.load(filename) as f:
        return dict(f.items())


class TestSolveLyapunov:

    cases = [
        (np.array([[1, 2], [3, 4]]),
         np.array([[9, 10], [11, 12]])),
        # a, q all complex.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a real; q complex.
        (np.array([[1.0, 2.0], [3.0, 5.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a complex; q real.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[2.0, 2.0], [-1.0, 2.0]])),
        # An example from Kitagawa, 1977
        (np.array([[3, 9, 5, 1, 4], [1, 2, 3, 8, 4], [4, 6, 6, 6, 3],
                   [1, 5, 2, 0, 7], [5, 3, 3, 1, 5]]),
         np.array([[2, 4, 1, 0, 1], [4, 1, 0, 2, 0], [1, 0, 3, 0, 3],
                   [0, 2, 0, 1, 0], [1, 0, 3, 0, 4]])),
        # Companion matrix example. a complex; q real; a.shape[0] = 11
        (np.array([[0.100+0.j, 0.091+0.j, 0.082+0.j, 0.073+0.j, 0.064+0.j,
                    0.055+0.j, 0.046+0.j, 0.037+0.j, 0.028+0.j, 0.019+0.j,
                    0.010+0.j],
                   [1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j,
                    0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j,
                    0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j, 0.000+0.j,
                    0.000+0.j],
                   [0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j,
                    0.000+0.j, 0.000+0.j, 0.000+0.j, 0.000+0.j, 1.000+0.j,
                    0.000+0.j]]),
         np.eye(11)),
        # https://github.com/scipy/scipy/issues/4176
        (matrix([[0, 1], [-1/2, -1]]),
         (matrix([0, 3]).T @ matrix([0, 3]).T.T)),
        # https://github.com/scipy/scipy/issues/4176
        (matrix([[0, 1], [-1/2, -1]]),
         (np.array(matrix([0, 3]).T @ matrix([0, 3]).T.T))),
        ]

    def test_continuous_squareness_and_shape(self):
        nsq = np.ones((3, 2))
        sq = np.eye(3)
        assert_raises(ValueError, solve_continuous_lyapunov, nsq, sq)
        assert_raises(ValueError, solve_continuous_lyapunov, sq, nsq)
        assert_raises(ValueError, solve_continuous_lyapunov, sq, np.eye(2))

    def check_continuous_case(self, a, q):
        x = solve_continuous_lyapunov(a, q)
        assert_array_almost_equal(
                          np.dot(a, x) + np.dot(x, a.conj().transpose()), q)

    def check_discrete_case(self, a, q, method=None):
        x = solve_discrete_lyapunov(a, q, method=method)
        assert_array_almost_equal(
                      np.dot(np.dot(a, x), a.conj().transpose()) - x, -1.0*q)

    def test_cases(self):
        for case in self.cases:
            self.check_continuous_case(case[0], case[1])
            self.check_discrete_case(case[0], case[1])
            self.check_discrete_case(case[0], case[1], method='direct')
            self.check_discrete_case(case[0], case[1], method='bilinear')


def test_solve_continuous_are():
    mat6 = _load_data('carex_6_data.npz')
    mat15 = _load_data('carex_15_data.npz')
    mat18 = _load_data('carex_18_data.npz')
    mat19 = _load_data('carex_19_data.npz')
    mat20 = _load_data('carex_20_data.npz')
    cases = [
        # Carex examples taken from (with default parameters):
        # [1] P.BENNER, A.J. LAUB, V. MEHRMANN: 'A Collection of Benchmark
        #     Examples for the Numerical Solution of Algebraic Riccati
        #     Equations II: Continuous-Time Case', Tech. Report SPC 95_23,
        #     Fak. f. Mathematik, TU Chemnitz-Zwickau (Germany), 1995.
        #
        # The format of the data is (a, b, q, r, knownfailure), where
        # knownfailure is None if the test passes or a string
        # indicating the reason for failure.
        #
        # Test Case 0: carex #1
        (np.diag([1.], 1),
         np.array([[0], [1]]),
         block_diag(1., 2.),
         1,
         None),
        # Test Case 1: carex #2
        (np.array([[4, 3], [-4.5, -3.5]]),
         np.array([[1], [-1]]),
         np.array([[9, 6], [6, 4.]]),
         1,
         None),
        # Test Case 2: carex #3
        (np.array([[0, 1, 0, 0],
                   [0, -1.89, 0.39, -5.53],
                   [0, -0.034, -2.98, 2.43],
                   [0.034, -0.0011, -0.99, -0.21]]),
         np.array([[0, 0], [0.36, -1.6], [-0.95, -0.032], [0.03, 0]]),
         np.array([[2.313, 2.727, 0.688, 0.023],
                   [2.727, 4.271, 1.148, 0.323],
                   [0.688, 1.148, 0.313, 0.102],
                   [0.023, 0.323, 0.102, 0.083]]),
         np.eye(2),
         None),
        # Test Case 3: carex #4
        (np.array([[-0.991, 0.529, 0, 0, 0, 0, 0, 0],
                   [0.522, -1.051, 0.596, 0, 0, 0, 0, 0],
                   [0, 0.522, -1.118, 0.596, 0, 0, 0, 0],
                   [0, 0, 0.522, -1.548, 0.718, 0, 0, 0],
                   [0, 0, 0, 0.922, -1.64, 0.799, 0, 0],
                   [0, 0, 0, 0, 0.922, -1.721, 0.901, 0],
                   [0, 0, 0, 0, 0, 0.922, -1.823, 1.021],
                   [0, 0, 0, 0, 0, 0, 0.922, -1.943]]),
         np.array([[3.84, 4.00, 37.60, 3.08, 2.36, 2.88, 3.08, 3.00],
                   [-2.88, -3.04, -2.80, -2.32, -3.32, -3.82, -4.12, -3.96]]
                  ).T * 0.001,
         np.array([[1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.1],
                   [0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.5, 0.0, 0.0, 0.1, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                   [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]),
         np.eye(2),
         None),
        # Test Case 4: carex #5
        (np.array(
          [[-4.019, 5.120, 0., 0., -2.082, 0., 0., 0., 0.870],
           [-0.346, 0.986, 0., 0., -2.340, 0., 0., 0., 0.970],
           [-7.909, 15.407, -4.069, 0., -6.450, 0., 0., 0., 2.680],
           [-21.816, 35.606, -0.339, -3.870, -17.800, 0., 0., 0., 7.390],
           [-60.196, 98.188, -7.907, 0.340, -53.008, 0., 0., 0., 20.400],
           [0, 0, 0, 0, 94.000, -147.200, 0., 53.200, 0.],
           [0, 0, 0, 0, 0, 94.000, -147.200, 0, 0],
           [0, 0, 0, 0, 0, 12.800, 0.000, -31.600, 0],
           [0, 0, 0, 0, 12.800, 0.000, 0.000, 18.800, -31.600]]),
         np.array([[0.010, -0.011, -0.151],
                   [0.003, -0.021, 0.000],
                   [0.009, -0.059, 0.000],
                   [0.024, -0.162, 0.000],
                   [0.068, -0.445, 0.000],
                   [0.000, 0.000, 0.000],
                   [0.000, 0.000, 0.000],
                   [0.000, 0.000, 0.000],
                   [0.000, 0.000, 0.000]]),
         np.eye(9),
         np.eye(3),
         None),
        # Test Case 5: carex #6
        (mat6['A'], mat6['B'], mat6['Q'], mat6['R'], None),
        # Test Case 6: carex #7
        (np.array([[1, 0], [0, -2.]]),
         np.array([[1e-6], [0]]),
         np.ones((2, 2)),
         1.,
         'Bad residual accuracy'),
        # Test Case 7: carex #8
        (block_diag(-0.1, -0.02),
         np.array([[0.100, 0.000], [0.001, 0.010]]),
         np.array([[100, 1000], [1000, 10000]]),
         np.ones((2, 2)) + block_diag(1e-6, 0),
         None),
        # Test Case 8: carex #9
        (np.array([[0, 1e6], [0, 0]]),
         np.array([[0], [1.]]),
         np.eye(2),
         1.,
         None),
        # Test Case 9: carex #10
        (np.array([[1.0000001, 1], [1., 1.0000001]]),
         np.eye(2),
         np.eye(2),
         np.eye(2),
         None),
        # Test Case 10: carex #11
        (np.array([[3, 1.], [4, 2]]),
         np.array([[1], [1]]),
         np.array([[-11, -5], [-5, -2.]]),
         1.,
         None),
        # Test Case 11: carex #12
        (np.array([[7000000., 2000000., -0.],
                   [2000000., 6000000., -2000000.],
                   [0., -2000000., 5000000.]]) / 3,
         np.eye(3),
         np.array([[1., -2., -2.], [-2., 1., -2.], [-2., -2., 1.]]).dot(
                np.diag([1e-6, 1, 1e6])).dot(
            np.array([[1., -2., -2.], [-2., 1., -2.], [-2., -2., 1.]])) / 9,
         np.eye(3) * 1e6,
         'Bad Residual Accuracy'),
        # Test Case 12: carex #13
        (np.array([[0, 0.4, 0, 0],
                   [0, 0, 0.345, 0],
                   [0, -0.524e6, -0.465e6, 0.262e6],
                   [0, 0, 0, -1e6]]),
         np.array([[0, 0, 0, 1e6]]).T,
         np.diag([1, 0, 1, 0]),
         1.,
         None),
        # Test Case 13: carex #14
        (np.array([[-1e-6, 1, 0, 0],
                   [-1, -1e-6, 0, 0],
                   [0, 0, 1e-6, 1],
                   [0, 0, -1, 1e-6]]),
         np.ones((4, 1)),
         np.ones((4, 4)),
         1.,
         None),
        # Test Case 14: carex #15
        (mat15['A'], mat15['B'], mat15['Q'], mat15['R'], None),
        # Test Case 15: carex #16
        (np.eye(64, 64, k=-1) + np.eye(64, 64)*(-2.) + np.rot90(
                 block_diag(1, np.zeros((62, 62)), 1)) + np.eye(64, 64, k=1),
         np.eye(64),
         np.eye(64),
         np.eye(64),
         None),
        # Test Case 16: carex #17
        (np.diag(np.ones((20, )), 1),
         np.flipud(np.eye(21, 1)),
         np.eye(21, 1) * np.eye(21, 1).T,
         1,
         'Bad Residual Accuracy'),
        # Test Case 17: carex #18
        (mat18['A'], mat18['B'], mat18['Q'], mat18['R'], None),
        # Test Case 18: carex #19
        (mat19['A'], mat19['B'], mat19['Q'], mat19['R'],
         'Bad Residual Accuracy'),
        # Test Case 19: carex #20
        (mat20['A'], mat20['B'], mat20['Q'], mat20['R'],
         'Bad Residual Accuracy')
        ]
    # Makes the minimum precision requirements customized to the test.
    # Here numbers represent the number of decimals that agrees with zero
    # matrix when the solution x is plugged in to the equation.
    #
    # res = array([[8e-3,1e-16],[1e-16,1e-20]]) --> min_decimal[k] = 2
    #
    # If the test is failing use "None" for that entry.
    #
    min_decimal = (14, 12, 13, 14, 11, 6, None, 5, 7, 14, 14,
                   None, 9, 14, 13, 14, None, 12, None, None)

    def _test_factory(case, dec):
        """Checks if 0 = XA + A'X - XB(R)^{-1} B'X + Q is true"""
        a, b, q, r, knownfailure = case
        if knownfailure:
            pytest.xfail(reason=knownfailure)

        x = solve_continuous_are(a, b, q, r)
        res = x.dot(a) + a.conj().T.dot(x) + q
        out_fact = x.dot(b)
        res -= out_fact.dot(solve(np.atleast_2d(r), out_fact.conj().T))
        assert_array_almost_equal(res, np.zeros_like(res), decimal=dec)

    for ind, case in enumerate(cases):
        _test_factory(case, min_decimal[ind])


def test_solve_discrete_are():

    cases = [
        # Darex examples taken from (with default parameters):
        # [1] P.BENNER, A.J. LAUB, V. MEHRMANN: 'A Collection of Benchmark
        #     Examples for the Numerical Solution of Algebraic Riccati
        #     Equations II: Discrete-Time Case', Tech. Report SPC 95_23,
        #     Fak. f. Mathematik, TU Chemnitz-Zwickau (Germany), 1995.
        # [2] T. GUDMUNDSSON, C. KENNEY, A.J. LAUB: 'Scaling of the
        #     Discrete-Time Algebraic Riccati Equation to Enhance Stability
        #     of the Schur Solution Method', IEEE Trans.Aut.Cont., vol.37(4)
        #
        # The format of the data is (a, b, q, r, knownfailure), where
        # knownfailure is None if the test passes or a string
        # indicating the reason for failure.
        #
        # TEST CASE 0 : Complex a; real b, q, r
        (np.array([[2, 1-2j], [0, -3j]]),
         np.array([[0], [1]]),
         np.array([[1, 0], [0, 2]]),
         np.array([[1]]),
         None),
        # TEST CASE 1 :Real a, q, r; complex b
        (np.array([[2, 1], [0, -1]]),
         np.array([[-2j], [1j]]),
         np.array([[1, 0], [0, 2]]),
         np.array([[1]]),
         None),
        # TEST CASE 2 : Real a, b; complex q, r
        (np.array([[3, 1], [0, -1]]),
         np.array([[1, 2], [1, 3]]),
         np.array([[1, 1+1j], [1-1j, 2]]),
         np.array([[2, -2j], [2j, 3]]),
         None),
        # TEST CASE 3 : User-reported gh-2251 (Trac #1732)
        (np.array([[0.63399379, 0.54906824, 0.76253406],
                   [0.5404729, 0.53745766, 0.08731853],
                   [0.27524045, 0.84922129, 0.4681622]]),
         np.array([[0.96861695], [0.05532739], [0.78934047]]),
         np.eye(3),
         np.eye(1),
         None),
        # TEST CASE 4 : darex #1
        (np.array([[4, 3], [-4.5, -3.5]]),
         np.array([[1], [-1]]),
         np.array([[9, 6], [6, 4]]),
         np.array([[1]]),
         None),
        # TEST CASE 5 : darex #2
        (np.array([[0.9512, 0], [0, 0.9048]]),
         np.array([[4.877, 4.877], [-1.1895, 3.569]]),
         np.array([[0.005, 0], [0, 0.02]]),
         np.array([[1/3, 0], [0, 3]]),
         None),
        # TEST CASE 6 : darex #3
        (np.array([[2, -1], [1, 0]]),
         np.array([[1], [0]]),
         np.array([[0, 0], [0, 1]]),
         np.array([[0]]),
         None),
        # TEST CASE 7 : darex #4 (skipped the gen. Ric. term S)
        (np.array([[0, 1], [0, -1]]),
         np.array([[1, 0], [2, 1]]),
         np.array([[-4, -4], [-4, 7]]) * (1/11),
         np.array([[9, 3], [3, 1]]),
         None),
        # TEST CASE 8 : darex #5
        (np.array([[0, 1], [0, 0]]),
         np.array([[0], [1]]),
         np.array([[1, 2], [2, 4]]),
         np.array([[1]]),
         None),
        # TEST CASE 9 : darex #6
        (np.array([[0.998, 0.067, 0, 0],
                   [-.067, 0.998, 0, 0],
                   [0, 0, 0.998, 0.153],
                   [0, 0, -.153, 0.998]]),
         np.array([[0.0033, 0.0200],
                   [0.1000, -.0007],
                   [0.0400, 0.0073],
                   [-.0028, 0.1000]]),
         np.array([[1.87, 0, 0, -0.244],
                   [0, 0.744, 0.205, 0],
                   [0, 0.205, 0.589, 0],
                   [-0.244, 0, 0, 1.048]]),
         np.eye(2),
         None),
        # TEST CASE 10 : darex #7
        (np.array([[0.984750, -.079903, 0.0009054, -.0010765],
                   [0.041588, 0.998990, -.0358550, 0.0126840],
                   [-.546620, 0.044916, -.3299100, 0.1931800],
                   [2.662400, -.100450, -.9245500, -.2632500]]),
         np.array([[0.0037112, 0.0007361],
                   [-.0870510, 9.3411e-6],
                   [-1.198440, -4.1378e-4],
                   [-3.192700, 9.2535e-4]]),
         np.eye(4)*1e-2,
         np.eye(2),
         None),
        # TEST CASE 11 : darex #8
        (np.array([[-0.6000000, -2.2000000, -3.6000000, -5.4000180],
                   [1.0000000, 0.6000000, 0.8000000, 3.3999820],
                   [0.0000000, 1.0000000, 1.8000000, 3.7999820],
                   [0.0000000, 0.0000000, 0.0000000, -0.9999820]]),
         np.array([[1.0, -1.0, -1.0, -1.0],
                   [0.0, 1.0, -1.0, -1.0],
                   [0.0, 0.0, 1.0, -1.0],
                   [0.0, 0.0, 0.0, 1.0]]),
         np.array([[2, 1, 3, 6],
                   [1, 2, 2, 5],
                   [3, 2, 6, 11],
                   [6, 5, 11, 22]]),
         np.eye(4),
         None),
        # TEST CASE 12 : darex #9
        (np.array([[95.4070, 1.9643, 0.3597, 0.0673, 0.0190],
                   [40.8490, 41.3170, 16.0840, 4.4679, 1.1971],
                   [12.2170, 26.3260, 36.1490, 15.9300, 12.3830],
                   [4.1118, 12.8580, 27.2090, 21.4420, 40.9760],
                   [0.1305, 0.5808, 1.8750, 3.6162, 94.2800]]) * 0.01,
         np.array([[0.0434, -0.0122],
                   [2.6606, -1.0453],
                   [3.7530, -5.5100],
                   [3.6076, -6.6000],
                   [0.4617, -0.9148]]) * 0.01,
         np.eye(5),
         np.eye(2),
         None),
        # TEST CASE 13 : darex #10
        (np.kron(np.eye(2), np.diag([1, 1], k=1)),
         np.kron(np.eye(2), np.array([[0], [0], [1]])),
         np.array([[1, 1, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, -1, 0],
                   [0, 0, 0, -1, 1, 0],
                   [0, 0, 0, 0, 0, 0]]),
         np.array([[3, 0], [0, 1]]),
         None),
        # TEST CASE 14 : darex #11
        (0.001 * np.array(
         [[870.1, 135.0, 11.59, .5014, -37.22, .3484, 0, 4.242, 7.249],
          [76.55, 897.4, 12.72, 0.5504, -40.16, .3743, 0, 4.53, 7.499],
          [-127.2, 357.5, 817, 1.455, -102.8, .987, 0, 11.85, 18.72],
          [-363.5, 633.9, 74.91, 796.6, -273.5, 2.653, 0, 31.72, 48.82],
          [-960, 1645.9, -128.9, -5.597, 71.42, 7.108, 0, 84.52, 125.9],
          [-664.4, 112.96, -88.89, -3.854, 84.47, 13.6, 0, 144.3, 101.6],
          [-410.2, 693, -54.71, -2.371, 66.49, 12.49, .1063, 99.97, 69.67],
          [-179.9, 301.7, -23.93, -1.035, 60.59, 22.16, 0, 213.9, 35.54],
          [-345.1, 580.4, -45.96, -1.989, 105.6, 19.86, 0, 219.1, 215.2]]),
         np.array([[4.7600, -0.5701, -83.6800],
                   [0.8790, -4.7730, -2.7300],
                   [1.4820, -13.1200, 8.8760],
                   [3.8920, -35.1300, 24.8000],
                   [10.3400, -92.7500, 66.8000],
                   [7.2030, -61.5900, 38.3400],
                   [4.4540, -36.8300, 20.2900],
                   [1.9710, -15.5400, 6.9370],
                   [3.7730, -30.2800, 14.6900]]) * 0.001,
         np.diag([50, 0, 0, 0, 50, 0, 0, 0, 0]),
         np.eye(3),
         None),
        # TEST CASE 15 : darex #12 - numerically least accurate example
        (np.array([[0, 1e6], [0, 0]]),
         np.array([[0], [1]]),
         np.eye(2),
         np.array([[1]]),
         None),
        # TEST CASE 16 : darex #13
        (np.array([[16, 10, -2],
                  [10, 13, -8],
                  [-2, -8, 7]]) * (1/9),
         np.eye(3),
         1e6 * np.eye(3),
         1e6 * np.eye(3),
         None),
        # TEST CASE 17 : darex #14
        (np.array([[1 - 1/1e8, 0, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]]),
         np.array([[1e-08], [0], [0], [0]]),
         np.diag([0, 0, 0, 1]),
         np.array([[0.25]]),
         None),
        # TEST CASE 18 : darex #15
        (np.eye(100, k=1),
         np.flipud(np.eye(100, 1)),
         np.eye(100),
         np.array([[1]]),
         None)
        ]

    # Makes the minimum precision requirements customized to the test.
    # Here numbers represent the number of decimals that agrees with zero
    # matrix when the solution x is plugged in to the equation.
    #
    # res = array([[8e-3,1e-16],[1e-16,1e-20]]) --> min_decimal[k] = 2
    #
    # If the test is failing use "None" for that entry.
    #
    min_decimal = (12, 14, 13, 14, 13, 16, 18, 14, 14, 13,
                   14, 13, 13, 14, 12, 2, 5, 6, 10)

    def _test_factory(case, dec):
        """Checks if X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q) is true"""
        a, b, q, r, knownfailure = case
        if knownfailure:
            pytest.xfail(reason=knownfailure)

        x = solve_discrete_are(a, b, q, r)
        res = a.conj().T.dot(x.dot(a)) - x + q
        res -= a.conj().T.dot(x.dot(b)).dot(
                    solve(r+b.conj().T.dot(x.dot(b)), b.conj().T).dot(x.dot(a))
                    )
        assert_array_almost_equal(res, np.zeros_like(res), decimal=dec)

    for ind, case in enumerate(cases):
        _test_factory(case, min_decimal[ind])

    # An infeasible example taken from https://arxiv.org/abs/1505.04861v1
    A = np.triu(np.ones((3, 3)))
    A[0, 1] = -1
    B = np.array([[1, 1, 0], [0, 0, 1]]).T
    Q = np.full_like(A, -2) + np.diag([8, -1, -1.9])
    R = np.diag([-10, 0.1])
    assert_raises(LinAlgError, solve_continuous_are, A, B, Q, R)


def test_solve_generalized_continuous_are():
    cases = [
        # Two random examples differ by s term
        # in the absence of any literature for demanding examples.
        (np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                   [4.617139e-02, 6.948286e-01, 3.444608e-02],
                   [9.713178e-02, 3.170995e-01, 4.387444e-01]]),
         np.array([[3.815585e-01, 1.868726e-01],
                   [7.655168e-01, 4.897644e-01],
                   [7.951999e-01, 4.455862e-01]]),
         np.eye(3),
         np.eye(2),
         np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],
                   [7.093648e-01, 6.797027e-01, 1.189977e-01],
                   [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
         np.zeros((3, 2)),
         None),
        (np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                   [4.617139e-02, 6.948286e-01, 3.444608e-02],
                   [9.713178e-02, 3.170995e-01, 4.387444e-01]]),
         np.array([[3.815585e-01, 1.868726e-01],
                   [7.655168e-01, 4.897644e-01],
                   [7.951999e-01, 4.455862e-01]]),
         np.eye(3),
         np.eye(2),
         np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],
                   [7.093648e-01, 6.797027e-01, 1.189977e-01],
                   [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
         np.ones((3, 2)),
         None)
        ]

    min_decimal = (10, 10)

    def _test_factory(case, dec):
        """Checks if X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q) is true"""
        a, b, q, r, e, s, knownfailure = case
        if knownfailure:
            pytest.xfail(reason=knownfailure)

        x = solve_continuous_are(a, b, q, r, e, s)
        res = a.conj().T.dot(x.dot(e)) + e.conj().T.dot(x.dot(a)) + q
        out_fact = e.conj().T.dot(x).dot(b) + s
        res -= out_fact.dot(solve(np.atleast_2d(r), out_fact.conj().T))
        assert_array_almost_equal(res, np.zeros_like(res), decimal=dec)

    for ind, case in enumerate(cases):
        _test_factory(case, min_decimal[ind])


def test_solve_generalized_discrete_are():
    mat20170120 = _load_data('gendare_20170120_data.npz')

    cases = [
        # Two random examples differ by s term
        # in the absence of any literature for demanding examples.
        (np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                   [4.617139e-02, 6.948286e-01, 3.444608e-02],
                   [9.713178e-02, 3.170995e-01, 4.387444e-01]]),
         np.array([[3.815585e-01, 1.868726e-01],
                   [7.655168e-01, 4.897644e-01],
                   [7.951999e-01, 4.455862e-01]]),
         np.eye(3),
         np.eye(2),
         np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],
                   [7.093648e-01, 6.797027e-01, 1.189977e-01],
                   [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
         np.zeros((3, 2)),
         None),
        (np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                   [4.617139e-02, 6.948286e-01, 3.444608e-02],
                   [9.713178e-02, 3.170995e-01, 4.387444e-01]]),
         np.array([[3.815585e-01, 1.868726e-01],
                   [7.655168e-01, 4.897644e-01],
                   [7.951999e-01, 4.455862e-01]]),
         np.eye(3),
         np.eye(2),
         np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],
                   [7.093648e-01, 6.797027e-01, 1.189977e-01],
                   [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
         np.ones((3, 2)),
         None),
        # user-reported (under PR-6616) 20-Jan-2017
        # tests against the case where E is None but S is provided
        (mat20170120['A'],
         mat20170120['B'],
         mat20170120['Q'],
         mat20170120['R'],
         None,
         mat20170120['S'],
         None),
        ]

    min_decimal = (11, 11, 16)

    def _test_factory(case, dec):
        """Checks if X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q) is true"""
        a, b, q, r, e, s, knownfailure = case
        if knownfailure:
            pytest.xfail(reason=knownfailure)

        x = solve_discrete_are(a, b, q, r, e, s)
        if e is None:
            e = np.eye(a.shape[0])
        if s is None:
            s = np.zeros_like(b)
        res = a.conj().T.dot(x.dot(a)) - e.conj().T.dot(x.dot(e)) + q
        res -= (a.conj().T.dot(x.dot(b)) + s).dot(
                    solve(r+b.conj().T.dot(x.dot(b)),
                          (b.conj().T.dot(x.dot(a)) + s.conj().T)
                          )
                )
        assert_array_almost_equal(res, np.zeros_like(res), decimal=dec)

    for ind, case in enumerate(cases):
        _test_factory(case, min_decimal[ind])


def test_are_validate_args():

    def test_square_shape():
        nsq = np.ones((3, 2))
        sq = np.eye(3)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, nsq, 1, 1, 1)
            assert_raises(ValueError, x, sq, sq, nsq, 1)
            assert_raises(ValueError, x, sq, sq, sq, nsq)
            assert_raises(ValueError, x, sq, sq, sq, sq, nsq)

    def test_compatible_sizes():
        nsq = np.ones((3, 2))
        sq = np.eye(4)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sq, nsq, 1, 1)
            assert_raises(ValueError, x, sq, sq, sq, sq, sq, nsq)
            assert_raises(ValueError, x, sq, sq, np.eye(3), sq)
            assert_raises(ValueError, x, sq, sq, sq, np.eye(3))
            assert_raises(ValueError, x, sq, sq, sq, sq, np.eye(3))

    def test_symmetry():
        nsym = np.arange(9).reshape(3, 3)
        sym = np.eye(3)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sym, sym, nsym, sym)
            assert_raises(ValueError, x, sym, sym, sym, nsym)

    def test_singularity():
        sing = np.full((3, 3), 1e12)
        sing[2, 2] -= 1
        sq = np.eye(3)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sq, sq, sq, sq, sing)

        assert_raises(ValueError, solve_continuous_are, sq, sq, sq, sing)

    def test_finiteness():
        nm = np.full((2, 2), np.nan)
        sq = np.eye(2)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, nm, sq, sq, sq)
            assert_raises(ValueError, x, sq, nm, sq, sq)
            assert_raises(ValueError, x, sq, sq, nm, sq)
            assert_raises(ValueError, x, sq, sq, sq, nm)
            assert_raises(ValueError, x, sq, sq, sq, sq, nm)
            assert_raises(ValueError, x, sq, sq, sq, sq, sq, nm)


class TestSolveSylvester:

    cases = [
        # a, b, c all real.
        (np.array([[1, 2], [0, 4]]),
         np.array([[5, 6], [0, 8]]),
         np.array([[9, 10], [11, 12]])),
        # a, b, c all real, 4x4. a and b have non-trival 2x2 blocks in their
        # quasi-triangular form.
        (np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 2.0, 0.0],
                   [0, 0, 3.0, -4],
                   [0, 0, 2, 5]]),
         np.array([[2.0, 0, 0, 1.0],
                   [0, 1.0, 0.0, 0.0],
                   [0, 0, 1.0, -1],
                   [0, 0, 1, 1]]),
         np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 1.0]])),
        # a, b, c all complex.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[-1.0, 2j], [3.0, 4.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a and b real; c complex.
        (np.array([[1.0, 2.0], [3.0, 5.0]]),
         np.array([[-1.0, 0], [3.0, 4.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a and c complex; b real.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[-1.0, 0], [3.0, 4.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a complex; b and c real.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[-1.0, 0], [3.0, 4.0]]),
         np.array([[2.0, 2.0], [-1.0, 2.0]])),
        # not square matrices, real
        (np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]]),
         np.array([[2, 3], [4, 5]]),
         np.array([[1, 2], [3, 4], [5, 6]])),
        # not square matrices, complex
        (np.array([[8, 1j, 6+2j], [3, 5, 7], [4, 9, 2]]),
         np.array([[2, 3], [4, 5-1j]]),
         np.array([[1, 2j], [3, 4j], [5j, 6+7j]])),
    ]

    def check_case(self, a, b, c):
        x = solve_sylvester(a, b, c)
        assert_array_almost_equal(np.dot(a, x) + np.dot(x, b), c)

    def test_cases(self):
        for case in self.cases:
            self.check_case(case[0], case[1], case[2])

    def test_trivial(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0]])
        c = np.array([2.0, 2.0]).reshape(-1, 1)
        x = solve_sylvester(a, b, c)
        assert_array_almost_equal(x, np.array([1.0, 1.0]).reshape(-1, 1))
