import warnings

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           assert_, suppress_warnings)
from pytest import raises as assert_raises

from scipy.signal import (ss2tf, tf2ss, lsim2, impulse2, step2, lti,
                          dlti, bode, freqresp, lsim, impulse, step,
                          abcd_normalize, place_poles,
                          TransferFunction, StateSpace, ZerosPolesGain)
from scipy.signal.filter_design import BadCoefficients
import scipy.linalg as linalg
from scipy.sparse.sputils import matrix


def _assert_poles_close(P1,P2, rtol=1e-8, atol=1e-8):
    """
    Check each pole in P1 is close to a pole in P2 with a 1e-8
    relative tolerance or 1e-8 absolute tolerance (useful for zero poles).
    These tolerances are very strict but the systems tested are known to
    accept these poles so we should not be far from what is requested.
    """
    P2 = P2.copy()
    for p1 in P1:
        found = False
        for p2_idx in range(P2.shape[0]):
            if np.allclose([np.real(p1), np.imag(p1)],
                           [np.real(P2[p2_idx]), np.imag(P2[p2_idx])],
                           rtol, atol):
                found = True
                np.delete(P2, p2_idx)
                break
        if not found:
            raise ValueError("Can't find pole " + str(p1) + " in " + str(P2))


class TestPlacePoles:

    def _check(self, A, B, P, **kwargs):
        """
        Perform the most common tests on the poles computed by place_poles
        and return the Bunch object for further specific tests
        """
        fsf = place_poles(A, B, P, **kwargs)
        expected, _ = np.linalg.eig(A - np.dot(B, fsf.gain_matrix))
        _assert_poles_close(expected,fsf.requested_poles)
        _assert_poles_close(expected,fsf.computed_poles)
        _assert_poles_close(P,fsf.requested_poles)
        return fsf

    def test_real(self):
        # Test real pole placement using KNV and YT0 algorithm and example 1 in
        # section 4 of the reference publication (see place_poles docstring)
        A = np.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146,0]).reshape(4, 2)
        P = np.array([-0.2, -0.5, -5.0566, -8.6659])

        # Check that both KNV and YT compute correct K matrix
        self._check(A, B, P, method='KNV0')
        self._check(A, B, P, method='YT')

        # Try to reach the specific case in _YT_real where two singular
        # values are almost equal. This is to improve code coverage but I
        # have no way to be sure this code is really reached

        # on some architectures this can lead to a RuntimeWarning invalid
        # value in divide (see gh-7590), so suppress it for now
        with np.errstate(invalid='ignore'):
            self._check(A, B, (2,2,3,3))

    def test_complex(self):
        # Test complex pole placement on a linearized car model, taken from L.
        # Jaulin, Automatique pour la robotique, Cours et Exercices, iSTE
        # editions p 184/185
        A = np.array([0,7,0,0,0,0,0,7/3.,0,0,0,0,0,0,0,0]).reshape(4,4)
        B = np.array([0,0,0,0,1,0,0,1]).reshape(4,2)
        # Test complex poles on YT
        P = np.array([-3, -1, -2-1j, -2+1j])
        self._check(A, B, P)

        # Try to reach the specific case in _YT_complex where two singular
        # values are almost equal. This is to improve code coverage but I
        # have no way to be sure this code is really reached

        P = [0-1e-6j,0+1e-6j,-10,10]
        self._check(A, B, P, maxiter=1000)

        # Try to reach the specific case in _YT_complex where the rank two
        # update yields two null vectors. This test was found via Monte Carlo.

        A = np.array(
                    [-2148,-2902, -2267, -598, -1722, -1829, -165, -283, -2546,
                   -167, -754, -2285, -543, -1700, -584, -2978, -925, -1300,
                   -1583, -984, -386, -2650, -764, -897, -517, -1598, 2, -1709,
                   -291, -338, -153, -1804, -1106, -1168, -867, -2297]
                   ).reshape(6,6)

        B = np.array(
                    [-108, -374, -524, -1285, -1232, -161, -1204, -672, -637,
                     -15, -483, -23, -931, -780, -1245, -1129, -1290, -1502,
                     -952, -1374, -62, -964, -930, -939, -792, -756, -1437,
                     -491, -1543, -686]
                     ).reshape(6,5)
        P = [-25.-29.j, -25.+29.j, 31.-42.j, 31.+42.j, 33.-41.j, 33.+41.j]
        self._check(A, B, P)

        # Use a lot of poles to go through all cases for update_order
        # in _YT_loop

        big_A = np.ones((11,11))-np.eye(11)
        big_B = np.ones((11,10))-np.diag([1]*10,1)[:,1:]
        big_A[:6,:6] = A
        big_B[:6,:5] = B

        P = [-10,-20,-30,40,50,60,70,-20-5j,-20+5j,5+3j,5-3j]
        self._check(big_A, big_B, P)

        #check with only complex poles and only real poles
        P = [-10,-20,-30,-40,-50,-60,-70,-80,-90,-100]
        self._check(big_A[:-1,:-1], big_B[:-1,:-1], P)
        P = [-10+10j,-20+20j,-30+30j,-40+40j,-50+50j,
             -10-10j,-20-20j,-30-30j,-40-40j,-50-50j]
        self._check(big_A[:-1,:-1], big_B[:-1,:-1], P)

        # need a 5x5 array to ensure YT handles properly when there
        # is only one real pole and several complex
        A = np.array([0,7,0,0,0,0,0,7/3.,0,0,0,0,0,0,0,0,
                      0,0,0,5,0,0,0,0,9]).reshape(5,5)
        B = np.array([0,0,0,0,1,0,0,1,2,3]).reshape(5,2)
        P = np.array([-2, -3+1j, -3-1j, -1+1j, -1-1j])
        place_poles(A, B, P)

        # same test with an odd number of real poles > 1
        # this is another specific case of YT
        P = np.array([-2, -3, -4, -1+1j, -1-1j])
        self._check(A, B, P)

    def test_tricky_B(self):
        # check we handle as we should the 1 column B matrices and
        # n column B matrices (with n such as shape(A)=(n, n))
        A = np.array([1.380, -0.2077, 6.715, -5.676, -0.5814, -4.290, 0,
                      0.6750, 1.067, 4.273, -6.654, 5.893, 0.0480, 4.273,
                      1.343, -2.104]).reshape(4, 4)
        B = np.array([0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0, 1, 2, 3, 4,
                      5, 6, 7, 8]).reshape(4, 4)

        # KNV or YT are not called here, it's a specific case with only
        # one unique solution
        P = np.array([-0.2, -0.5, -5.0566, -8.6659])
        fsf = self._check(A, B, P)
        # rtol and nb_iter should be set to np.nan as the identity can be
        # used as transfer matrix
        assert_equal(fsf.rtol, np.nan)
        assert_equal(fsf.nb_iter, np.nan)

        # check with complex poles too as they trigger a specific case in
        # the specific case :-)
        P = np.array((-2+1j,-2-1j,-3,-2))
        fsf = self._check(A, B, P)
        assert_equal(fsf.rtol, np.nan)
        assert_equal(fsf.nb_iter, np.nan)

        #now test with a B matrix with only one column (no optimisation)
        B = B[:,0].reshape(4,1)
        P = np.array((-2+1j,-2-1j,-3,-2))
        fsf = self._check(A, B, P)

        #  we can't optimize anything, check they are set to 0 as expected
        assert_equal(fsf.rtol, 0)
        assert_equal(fsf.nb_iter, 0)

    def test_errors(self):
        # Test input mistakes from user
        A = np.array([0,7,0,0,0,0,0,7/3.,0,0,0,0,0,0,0,0]).reshape(4,4)
        B = np.array([0,0,0,0,1,0,0,1]).reshape(4,2)

        #should fail as the method keyword is invalid
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
                      method="foo")

        #should fail as poles are not 1D array
        assert_raises(ValueError, place_poles, A, B,
                      np.array((-2.1,-2.2,-2.3,-2.4)).reshape(4,1))

        #should fail as A is not a 2D array
        assert_raises(ValueError, place_poles, A[:,:,np.newaxis], B,
                      (-2.1,-2.2,-2.3,-2.4))

        #should fail as B is not a 2D array
        assert_raises(ValueError, place_poles, A, B[:,:,np.newaxis],
                      (-2.1,-2.2,-2.3,-2.4))

        #should fail as there are too many poles
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4,-3))

        #should fail as there are not enough poles
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3))

        #should fail as the rtol is greater than 1
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
                      rtol=42)

        #should fail as maxiter is smaller than 1
        assert_raises(ValueError, place_poles, A, B, (-2.1,-2.2,-2.3,-2.4),
                      maxiter=-42)

        # should fail as ndim(B) is two
        assert_raises(ValueError, place_poles, A, B, (-2,-2,-2,-2))

        #unctrollable system
        assert_raises(ValueError, place_poles, np.ones((4,4)),
                      np.ones((4,2)), (1,2,3,4))

        # Should not raise ValueError as the poles can be placed but should
        # raise a warning as the convergence is not reached
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fsf = place_poles(A, B, (-1,-2,-3,-4), rtol=1e-16, maxiter=42)
            assert_(len(w) == 1)
            assert_(issubclass(w[-1].category, UserWarning))
            assert_("Convergence was not reached after maxiter iterations"
                    in str(w[-1].message))
            assert_equal(fsf.nb_iter, 42)

        # should fail as a complex misses its conjugate
        assert_raises(ValueError, place_poles, A, B, (-2+1j,-2-1j,-2+3j,-2))

        # should fail as A is not square
        assert_raises(ValueError, place_poles, A[:,:3], B, (-2,-3,-4,-5))

        # should fail as B has not the same number of lines as A
        assert_raises(ValueError, place_poles, A, B[:3,:], (-2,-3,-4,-5))

        # should fail as KNV0 does not support complex poles
        assert_raises(ValueError, place_poles, A, B,
                      (-2+1j,-2-1j,-2+3j,-2-3j), method="KNV0")


class TestSS2TF:

    def check_matrix_shapes(self, p, q, r):
        ss2tf(np.zeros((p, p)),
              np.zeros((p, q)),
              np.zeros((r, p)),
              np.zeros((r, q)), 0)

    def test_shapes(self):
        # Each tuple holds:
        #   number of states, number of inputs, number of outputs
        for p, q, r in [(3, 3, 3), (1, 3, 3), (1, 1, 1)]:
            self.check_matrix_shapes(p, q, r)

    def test_basic(self):
        # Test a round trip through tf2ss and ss2tf.
        b = np.array([1.0, 3.0, 5.0])
        a = np.array([1.0, 2.0, 3.0])

        A, B, C, D = tf2ss(b, a)
        assert_allclose(A, [[-2, -3], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[1, 2]], rtol=1e-13)
        assert_allclose(D, [[1]], rtol=1e-14)

        bb, aa = ss2tf(A, B, C, D)
        assert_allclose(bb[0], b, rtol=1e-13)
        assert_allclose(aa, a, rtol=1e-13)

    def test_zero_order_round_trip(self):
        # See gh-5760
        tf = (2, 1)
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[0]], rtol=1e-13)
        assert_allclose(B, [[0]], rtol=1e-13)
        assert_allclose(C, [[0]], rtol=1e-13)
        assert_allclose(D, [[2]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[2, 0]], rtol=1e-13)
        assert_allclose(den, [1, 0], rtol=1e-13)

        tf = ([[5], [2]], 1)
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[0]], rtol=1e-13)
        assert_allclose(B, [[0]], rtol=1e-13)
        assert_allclose(C, [[0], [0]], rtol=1e-13)
        assert_allclose(D, [[5], [2]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[5, 0], [2, 0]], rtol=1e-13)
        assert_allclose(den, [1, 0], rtol=1e-13)

    def test_simo_round_trip(self):
        # See gh-5753
        tf = ([[1, 2], [1, 1]], [1, 2])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-2]], rtol=1e-13)
        assert_allclose(B, [[1]], rtol=1e-13)
        assert_allclose(C, [[0], [-1]], rtol=1e-13)
        assert_allclose(D, [[1], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[1, 2], [1, 1]], rtol=1e-13)
        assert_allclose(den, [1, 2], rtol=1e-13)

        tf = ([[1, 0, 1], [1, 1, 1]], [1, 1, 1])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-1, -1], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[-1, 0], [0, 0]], rtol=1e-13)
        assert_allclose(D, [[1], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[1, 0, 1], [1, 1, 1]], rtol=1e-13)
        assert_allclose(den, [1, 1, 1], rtol=1e-13)

        tf = ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-2, -3, -4], [1, 0, 0], [0, 1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0], [0]], rtol=1e-13)
        assert_allclose(C, [[1, 2, 3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(D, [[0], [0]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, 2, 3], [0, 1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 2, 3, 4], rtol=1e-13)

        tf = (np.array([1, [2, 3]], dtype=object), [1, 6])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6]], rtol=1e-31)
        assert_allclose(B, [[1]], rtol=1e-31)
        assert_allclose(C, [[1], [-9]], rtol=1e-31)
        assert_allclose(D, [[0], [2]], rtol=1e-31)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1], [2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6], rtol=1e-13)

        tf = (np.array([[1, -3], [1, 2, 3]], dtype=object), [1, 6, 5])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6, -5], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[1, -3], [-4, -2]], rtol=1e-13)
        assert_allclose(D, [[0], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, -3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6, 5], rtol=1e-13)

    def test_all_int_arrays(self):
        A = [[0, 1, 0], [0, 0, 1], [-3, -4, -2]]
        B = [[0], [0], [1]]
        C = [[5, 1, 0]]
        D = [[0]]
        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0.0, 0.0, 1.0, 5.0]], rtol=1e-13, atol=1e-14)
        assert_allclose(den, [1.0, 2.0, 4.0, 3.0], rtol=1e-13)

    def test_multioutput(self):
        # Regression test for gh-2669.

        # 4 states
        A = np.array([[-1.0, 0.0, 1.0, 0.0],
                      [-1.0, 0.0, 2.0, 0.0],
                      [-4.0, 0.0, 3.0, 0.0],
                      [-8.0, 8.0, 0.0, 4.0]])

        # 1 input
        B = np.array([[0.3],
                      [0.0],
                      [7.0],
                      [0.0]])

        # 3 outputs
        C = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [8.0, 8.0, 0.0, 0.0]])

        D = np.array([[0.0],
                      [0.0],
                      [1.0]])

        # Get the transfer functions for all the outputs in one call.
        b_all, a = ss2tf(A, B, C, D)

        # Get the transfer functions for each output separately.
        b0, a0 = ss2tf(A, B, C[0], D[0])
        b1, a1 = ss2tf(A, B, C[1], D[1])
        b2, a2 = ss2tf(A, B, C[2], D[2])

        # Check that we got the same results.
        assert_allclose(a0, a, rtol=1e-13)
        assert_allclose(a1, a, rtol=1e-13)
        assert_allclose(a2, a, rtol=1e-13)
        assert_allclose(b_all, np.vstack((b0, b1, b2)), rtol=1e-13, atol=1e-14)


class TestLsim:
    def lti_nowarn(self, *args):
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(*args)
        return system

    def test_first_order(self):
        # y' = -y
        # exact solution is y(t) = exp(-t)
        system = self.lti_nowarn(-1.,1.,1.,0.)
        t = np.linspace(0,5)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[1.0])
        expected_x = np.exp(-tout)
        assert_almost_equal(x, expected_x)
        assert_almost_equal(y, expected_x)

    def test_integrator(self):
        # integrator: y' = u
        system = self.lti_nowarn(0., 1., 1., 0.)
        t = np.linspace(0,5)
        u = t
        tout, y, x = lsim(system, u, t)
        expected_x = 0.5 * tout**2
        assert_almost_equal(x, expected_x)
        assert_almost_equal(y, expected_x)

    def test_double_integrator(self):
        # double integrator: y'' = 2u
        A = matrix([[0., 1.], [0., 0.]])
        B = matrix([[0.], [1.]])
        C = matrix([[2., 0.]])
        system = self.lti_nowarn(A, B, C, 0.)
        t = np.linspace(0,5)
        u = np.ones_like(t)
        tout, y, x = lsim(system, u, t)
        expected_x = np.transpose(np.array([0.5 * tout**2, tout]))
        expected_y = tout**2
        assert_almost_equal(x, expected_x)
        assert_almost_equal(y, expected_y)

    def test_jordan_block(self):
        # Non-diagonalizable A matrix
        #   x1' + x1 = x2
        #   x2' + x2 = u
        #   y = x1
        # Exact solution with u = 0 is y(t) = t exp(-t)
        A = matrix([[-1., 1.], [0., -1.]])
        B = matrix([[0.], [1.]])
        C = matrix([[1., 0.]])
        system = self.lti_nowarn(A, B, C, 0.)
        t = np.linspace(0,5)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[0.0, 1.0])
        expected_y = tout * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_miso(self):
        # A system with two state variables, two inputs, and one output.
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([1.0, 0.0])
        D = np.zeros((1,2))
        system = self.lti_nowarn(A, B, C, D)

        t = np.linspace(0, 5.0, 101)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[1.0, 1.0])
        expected_y = np.exp(-tout)
        expected_x0 = np.exp(-tout)
        expected_x1 = np.exp(-2.0*tout)
        assert_almost_equal(y, expected_y)
        assert_almost_equal(x[:,0], expected_x0)
        assert_almost_equal(x[:,1], expected_x1)

    def test_nonzero_initial_time(self):
        system = self.lti_nowarn(-1.,1.,1.,0.)
        t = np.linspace(1,2)
        u = np.zeros_like(t)
        tout, y, x = lsim(system, u, t, X0=[1.0])
        expected_y = np.exp(-tout)
        assert_almost_equal(y, expected_y)


class Test_lsim2:

    def test_01(self):
        t = np.linspace(0,10,1001)
        u = np.zeros_like(t)
        # First order system: x'(t) + x(t) = u(t), x(0) = 1.
        # Exact solution is x(t) = exp(-t).
        system = ([1.0],[1.0,1.0])
        tout, y, x = lsim2(system, u, t, X0=[1.0])
        expected_x = np.exp(-tout)
        assert_almost_equal(x[:,0], expected_x)

    def test_02(self):
        t = np.array([0.0, 1.0, 1.0, 3.0])
        u = np.array([0.0, 0.0, 1.0, 1.0])
        # Simple integrator: x'(t) = u(t)
        system = ([1.0],[1.0,0.0])
        tout, y, x = lsim2(system, u, t, X0=[1.0])
        expected_x = np.maximum(1.0, tout)
        assert_almost_equal(x[:,0], expected_x)

    def test_03(self):
        t = np.array([0.0, 1.0, 1.0, 1.1, 1.1, 2.0])
        u = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        # Simple integrator:  x'(t) = u(t)
        system = ([1.0],[1.0, 0.0])
        tout, y, x = lsim2(system, u, t, hmax=0.01)
        expected_x = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1])
        assert_almost_equal(x[:,0], expected_x)

    def test_04(self):
        t = np.linspace(0, 10, 1001)
        u = np.zeros_like(t)
        # Second order system with a repeated root: x''(t) + 2*x(t) + x(t) = 0.
        # With initial conditions x(0)=1.0 and x'(t)=0.0, the exact solution
        # is (1-t)*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y, x = lsim2(system, u, t, X0=[1.0, 0.0])
        expected_x = (1.0 - tout) * np.exp(-tout)
        assert_almost_equal(x[:,0], expected_x)

    def test_05(self):
        # The call to lsim2 triggers a "BadCoefficients" warning from
        # scipy.signal.filter_design, but the test passes.  I think the warning
        # is related to the incomplete handling of multi-input systems in
        # scipy.signal.

        # A system with two state variables, two inputs, and one output.
        A = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        C = np.array([1.0, 0.0])
        D = np.zeros((1, 2))

        t = np.linspace(0, 10.0, 101)
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            tout, y, x = lsim2((A,B,C,D), T=t, X0=[1.0, 1.0])
        expected_y = np.exp(-tout)
        expected_x0 = np.exp(-tout)
        expected_x1 = np.exp(-2.0 * tout)
        assert_almost_equal(y, expected_y)
        assert_almost_equal(x[:,0], expected_x0)
        assert_almost_equal(x[:,1], expected_x1)

    def test_06(self):
        # Test use of the default values of the arguments `T` and `U`.
        # Second order system with a repeated root: x''(t) + 2*x(t) + x(t) = 0.
        # With initial conditions x(0)=1.0 and x'(t)=0.0, the exact solution
        # is (1-t)*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y, x = lsim2(system, X0=[1.0, 0.0])
        expected_x = (1.0 - tout) * np.exp(-tout)
        assert_almost_equal(x[:,0], expected_x)


class _TestImpulseFuncs:
    # Common tests for impulse/impulse2 (= self.func)

    def test_01(self):
        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system)
        expected_y = np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_02(self):
        # Specify the desired time values for the output.

        # First order system: x'(t) + x(t) = u(t)
        # Exact impulse response is x(t) = exp(-t).
        system = ([1.0], [1.0,1.0])
        n = 21
        t = np.linspace(0, 2.0, n)
        tout, y = self.func(system, T=t)
        assert_equal(tout.shape, (n,))
        assert_almost_equal(tout, t)
        expected_y = np.exp(-t)
        assert_almost_equal(y, expected_y)

    def test_03(self):
        # Specify an initial condition as a scalar.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=3.0)
        expected_y = 4.0 * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_04(self):
        # Specify an initial condition as a list.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact impulse response is x(t) = 4*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=[3.0])
        expected_y = 4.0 * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_05(self):
        # Simple integrator: x'(t) = u(t)
        system = ([1.0], [1.0,0.0])
        tout, y = self.func(system)
        expected_y = np.ones_like(tout)
        assert_almost_equal(y, expected_y)

    def test_06(self):
        # Second order system with a repeated root:
        #     x''(t) + 2*x(t) + x(t) = u(t)
        # The exact impulse response is t*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = self.func(system)
        expected_y = tout * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_array_like(self):
        # Test that function can accept sequences, scalars.
        system = ([1.0], [1.0, 2.0, 1.0])
        # TODO: add meaningful test where X0 is a list
        tout, y = self.func(system, X0=[3], T=[5, 6])
        tout, y = self.func(system, X0=[3], T=[5])

    def test_array_like2(self):
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = self.func(system, X0=3, T=5)


class TestImpulse2(_TestImpulseFuncs):
    def setup_method(self):
        self.func = impulse2


class TestImpulse(_TestImpulseFuncs):
    def setup_method(self):
        self.func = impulse


class _TestStepFuncs:
    def test_01(self):
        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system)
        expected_y = 1.0 - np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_02(self):
        # Specify the desired time values for the output.

        # First order system: x'(t) + x(t) = u(t)
        # Exact step response is x(t) = 1 - exp(-t).
        system = ([1.0], [1.0,1.0])
        n = 21
        t = np.linspace(0, 2.0, n)
        tout, y = self.func(system, T=t)
        assert_equal(tout.shape, (n,))
        assert_almost_equal(tout, t)
        expected_y = 1 - np.exp(-t)
        assert_almost_equal(y, expected_y)

    def test_03(self):
        # Specify an initial condition as a scalar.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=3.0)
        expected_y = 1 + 2.0*np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_04(self):
        # Specify an initial condition as a list.

        # First order system: x'(t) + x(t) = u(t), x(0)=3.0
        # Exact step response is x(t) = 1 + 2*exp(-t).
        system = ([1.0], [1.0,1.0])
        tout, y = self.func(system, X0=[3.0])
        expected_y = 1 + 2.0*np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_05(self):
        # Simple integrator: x'(t) = u(t)
        # Exact step response is x(t) = t.
        system = ([1.0],[1.0,0.0])
        tout, y = self.func(system)
        expected_y = tout
        assert_almost_equal(y, expected_y)

    def test_06(self):
        # Second order system with a repeated root:
        #     x''(t) + 2*x(t) + x(t) = u(t)
        # The exact step response is 1 - (1 + t)*exp(-t).
        system = ([1.0], [1.0, 2.0, 1.0])
        tout, y = self.func(system)
        expected_y = 1 - (1 + tout) * np.exp(-tout)
        assert_almost_equal(y, expected_y)

    def test_array_like(self):
        # Test that function can accept sequences, scalars.
        system = ([1.0], [1.0, 2.0, 1.0])
        # TODO: add meaningful test where X0 is a list
        tout, y = self.func(system, T=[5, 6])


class TestStep2(_TestStepFuncs):
    def setup_method(self):
        self.func = step2

    def test_05(self):
        # This test is almost the same as the one it overwrites in the base
        # class.  The only difference is the tolerances passed to step2:
        # the default tolerances are not accurate enough for this test

        # Simple integrator: x'(t) = u(t)
        # Exact step response is x(t) = t.
        system = ([1.0], [1.0,0.0])
        tout, y = self.func(system, atol=1e-10, rtol=1e-8)
        expected_y = tout
        assert_almost_equal(y, expected_y)


class TestStep(_TestStepFuncs):
    def setup_method(self):
        self.func = step

    def test_complex_input(self):
        # Test that complex input doesn't raise an error.
        # `step` doesn't seem to have been designed for complex input, but this
        # works and may be used, so add regression test.  See gh-2654.
        step(([], [-1], 1+0j))


class TestLti:
    def test_lti_instantiation(self):
        # Test that lti can be instantiated with sequences, scalars.
        # See PR-225.

        # TransferFunction
        s = lti([1], [-1])
        assert_(isinstance(s, TransferFunction))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)

        # ZerosPolesGain
        s = lti(np.array([]), np.array([-1]), 1)
        assert_(isinstance(s, ZerosPolesGain))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)

        # StateSpace
        s = lti([], [-1], 1)
        s = lti([1], [-1], 1, 3)
        assert_(isinstance(s, StateSpace))
        assert_(isinstance(s, lti))
        assert_(not isinstance(s, dlti))
        assert_(s.dt is None)


class TestStateSpace:
    def test_initialization(self):
        # Check that all initializations work
        StateSpace(1, 1, 1, 1)
        StateSpace([1], [2], [3], [4])
        StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]),
                   np.array([[1, 0]]), np.array([[0]]))

    def test_conversion(self):
        # Check the conversion functions
        s = StateSpace(1, 2, 3, 4)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(StateSpace(s) is not s)
        assert_(s.to_ss() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_tf() and to_zpk()

        # Getters
        s = StateSpace(1, 1, 1, 1)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])
        assert_(s.dt is None)

    def test_operators(self):
        # Test +/-/* operators on systems

        class BadType:
            pass

        s1 = StateSpace(np.array([[-0.5, 0.7], [0.3, -0.8]]),
                        np.array([[1], [0]]),
                        np.array([[1, 0]]),
                        np.array([[0]]),
                        )

        s2 = StateSpace(np.array([[-0.2, -0.1], [0.4, -0.1]]),
                        np.array([[1], [0]]),
                        np.array([[1, 0]]),
                        np.array([[0]])
                        )

        s_discrete = s1.to_discrete(0.1)
        s2_discrete = s2.to_discrete(0.2)
        s3_discrete = s2.to_discrete(0.1)

        # Impulse response
        t = np.linspace(0, 1, 100)
        u = np.zeros_like(t)
        u[0] = 1

        # Test multiplication
        for typ in (int, float, complex, np.float32, np.complex128, np.array):
            assert_allclose(lsim(typ(2) * s1, U=u, T=t)[1],
                            typ(2) * lsim(s1, U=u, T=t)[1])

            assert_allclose(lsim(s1 * typ(2), U=u, T=t)[1],
                            lsim(s1, U=u, T=t)[1] * typ(2))

            assert_allclose(lsim(s1 / typ(2), U=u, T=t)[1],
                            lsim(s1, U=u, T=t)[1] / typ(2))

            with assert_raises(TypeError):
                typ(2) / s1

        assert_allclose(lsim(s1 * 2, U=u, T=t)[1],
                        lsim(s1, U=2 * u, T=t)[1])

        assert_allclose(lsim(s1 * s2, U=u, T=t)[1],
                        lsim(s1, U=lsim(s2, U=u, T=t)[1], T=t)[1],
                        atol=1e-5)

        with assert_raises(TypeError):
            s1 / s1

        with assert_raises(TypeError):
            s1 * s_discrete

        with assert_raises(TypeError):
            # Check different discretization constants
            s_discrete * s2_discrete

        with assert_raises(TypeError):
            s1 * BadType()

        with assert_raises(TypeError):
            BadType() * s1

        with assert_raises(TypeError):
            s1 / BadType()

        with assert_raises(TypeError):
            BadType() / s1

        # Test addition
        assert_allclose(lsim(s1 + 2, U=u, T=t)[1],
                        2 * u + lsim(s1, U=u, T=t)[1])

        # Check for dimension mismatch
        with assert_raises(ValueError):
            s1 + np.array([1, 2])

        with assert_raises(ValueError):
            np.array([1, 2]) + s1

        with assert_raises(TypeError):
            s1 + s_discrete

        with assert_raises(ValueError):
            s1 / np.array([[1, 2], [3, 4]])

        with assert_raises(TypeError):
            # Check different discretization constants
            s_discrete + s2_discrete

        with assert_raises(TypeError):
            s1 + BadType()

        with assert_raises(TypeError):
            BadType() + s1

        assert_allclose(lsim(s1 + s2, U=u, T=t)[1],
                        lsim(s1, U=u, T=t)[1] + lsim(s2, U=u, T=t)[1])

        # Test subtraction
        assert_allclose(lsim(s1 - 2, U=u, T=t)[1],
                        -2 * u + lsim(s1, U=u, T=t)[1])

        assert_allclose(lsim(2 - s1, U=u, T=t)[1],
                        2 * u + lsim(-s1, U=u, T=t)[1])

        assert_allclose(lsim(s1 - s2, U=u, T=t)[1],
                        lsim(s1, U=u, T=t)[1] - lsim(s2, U=u, T=t)[1])

        with assert_raises(TypeError):
            s1 - BadType()

        with assert_raises(TypeError):
            BadType() - s1

        s = s_discrete + s3_discrete
        assert_(s.dt == 0.1)

        s = s_discrete * s3_discrete
        assert_(s.dt == 0.1)

        s = 3 * s_discrete
        assert_(s.dt == 0.1)

        s = -s_discrete
        assert_(s.dt == 0.1)

class TestTransferFunction:
    def test_initialization(self):
        # Check that all initializations work
        TransferFunction(1, 1)
        TransferFunction([1], [2])
        TransferFunction(np.array([1]), np.array([2]))

    def test_conversion(self):
        # Check the conversion functions
        s = TransferFunction([1, 0], [1, -1])
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(TransferFunction(s) is not s)
        assert_(s.to_tf() is not s)

    def test_properties(self):
        # Test setters/getters for cross class properties.
        # This implicitly tests to_ss() and to_zpk()

        # Getters
        s = TransferFunction([1, 0], [1, -1])
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])


class TestZerosPolesGain:
    def test_initialization(self):
        # Check that all initializations work
        ZerosPolesGain(1, 1, 1)
        ZerosPolesGain([1], [2], 1)
        ZerosPolesGain(np.array([1]), np.array([2]), 1)

    def test_conversion(self):
        #Check the conversion functions
        s = ZerosPolesGain(1, 2, 3)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(ZerosPolesGain(s) is not s)
        assert_(s.to_zpk() is not s)


class Test_abcd_normalize:
    def setup_method(self):
        self.A = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.B = np.array([[-1.0], [5.0]])
        self.C = np.array([[4.0, 5.0]])
        self.D = np.array([[2.5]])

    def test_no_matrix_fails(self):
        assert_raises(ValueError, abcd_normalize)

    def test_A_nosquare_fails(self):
        assert_raises(ValueError, abcd_normalize, [1, -1],
                      self.B, self.C, self.D)

    def test_AB_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_AC_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      [[4.0], [5.0]], self.D)

    def test_CD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      self.C, [2.5, 0])

    def test_BD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_normalized_matrices_unchanged(self):
        A, B, C, D = abcd_normalize(self.A, self.B, self.C, self.D)
        assert_equal(A, self.A)
        assert_equal(B, self.B)
        assert_equal(C, self.C)
        assert_equal(D, self.D)

    def test_shapes(self):
        A, B, C, D = abcd_normalize(self.A, self.B, [1, 0], 0)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape[0], C.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(B.shape[1], D.shape[1])

    def test_zero_dimension_is_not_none1(self):
        B_ = np.zeros((2, 0))
        D_ = np.zeros((0, 0))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, D=D_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(D, D_)
        assert_equal(C.shape[0], D_.shape[0])
        assert_equal(C.shape[1], self.A.shape[0])

    def test_zero_dimension_is_not_none2(self):
        B_ = np.zeros((2, 0))
        C_ = np.zeros((0, 2))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, C=C_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(C, C_)
        assert_equal(D.shape[0], C_.shape[0])
        assert_equal(D.shape[1], B_.shape[1])

    def test_missing_A(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))

    def test_missing_B(self):
        A, B, C, D = abcd_normalize(A=self.A, C=self.C, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))

    def test_missing_C(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, D=self.D)
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_D(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, C=self.C)
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_AB(self):
        A, B, C, D = abcd_normalize(C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(A.shape, (self.C.shape[1], self.C.shape[1]))
        assert_equal(B.shape, (self.C.shape[1], self.D.shape[1]))

    def test_missing_AC(self):
        A, B, C, D = abcd_normalize(B=self.B, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(C.shape, (self.D.shape[0], self.B.shape[0]))

    def test_missing_AD(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_BC(self):
        A, B, C, D = abcd_normalize(A=self.A, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_ABC_fails(self):
        assert_raises(ValueError, abcd_normalize, D=self.D)

    def test_missing_BD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, C=self.C)

    def test_missing_CD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, B=self.B)


class Test_bode:

    def test_01(self):
        # Test bode() magnitude calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        # cutoff: 1 rad/s, slope: -20 dB/decade
        #   H(s=0.1) ~= 0 dB
        #   H(s=1) ~= -3 dB
        #   H(s=10) ~= -20 dB
        #   H(s=100) ~= -40 dB
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = bode(system, w=w)
        expected_mag = [0, -3, -20, -40]
        assert_almost_equal(mag, expected_mag, decimal=1)

    def test_02(self):
        # Test bode() phase calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   angle(H(s=0.1)) ~= -5.7 deg
        #   angle(H(s=1)) ~= -45 deg
        #   angle(H(s=10)) ~= -84.3 deg
        system = lti([1], [1, 1])
        w = [0.1, 1, 10]
        w, mag, phase = bode(system, w=w)
        expected_phase = [-5.7, -45, -84.3]
        assert_almost_equal(phase, expected_phase, decimal=1)

    def test_03(self):
        # Test bode() magnitude calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = bode(system, w=w)
        jw = w * 1j
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_mag = 20.0 * np.log10(abs(y))
        assert_almost_equal(mag, expected_mag)

    def test_04(self):
        # Test bode() phase calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, mag, phase = bode(system, w=w)
        jw = w * 1j
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
        expected_phase = np.arctan2(y.imag, y.real) * 180.0 / np.pi
        assert_almost_equal(phase, expected_phase)

    def test_05(self):
        # Test that bode() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        n = 10
        # Expected range is from 0.01 to 10.
        expected_w = np.logspace(-2, 1, n)
        w, mag, phase = bode(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_06(self):
        # Test that bode() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = lti([1], [1, 0])
        w, mag, phase = bode(system, n=2)
        assert_equal(w[0], 0.01)  # a fail would give not-a-number

    def test_07(self):
        # bode() should not fail on a system with pure imaginary poles.
        # The test passes if bode doesn't raise an exception.
        system = lti([1], [1, 0, 100])
        w, mag, phase = bode(system, n=2)

    def test_08(self):
        # Test that bode() return continuous phase, issues/2331.
        system = lti([], [-10, -30, -40, -60, -70], 1)
        w, mag, phase = system.bode(w=np.logspace(-3, 40, 100))
        assert_almost_equal(min(phase), -450, decimal=15)

    def test_from_state_space(self):
        # Ensure that bode works with a system that was created from the
        # state space representation matrices A, B, C, D.  In this case,
        # system.num will be a 2-D array with shape (1, n+1), where (n,n)
        # is the shape of A.
        # A Butterworth lowpass filter is used, so we know the exact
        # frequency response.
        a = np.array([1.0, 2.0, 2.0, 1.0])
        A = linalg.companion(a).T
        B = np.array([[0.0], [0.0], [1.0]])
        C = np.array([[1.0, 0.0, 0.0]])
        D = np.array([[0.0]])
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(A, B, C, D)
            w, mag, phase = bode(system, n=100)

        expected_magnitude = 20 * np.log10(np.sqrt(1.0 / (1.0 + w**6)))
        assert_almost_equal(mag, expected_magnitude)


class Test_freqresp:

    def test_output_manual(self):
        # Test freqresp() output calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 1 / (s + 1),
        #   re(H(s=0.1)) ~= 0.99
        #   re(H(s=1)) ~= 0.5
        #   re(H(s=10)) ~= 0.0099
        system = lti([1], [1, 1])
        w = [0.1, 1, 10]
        w, H = freqresp(system, w=w)
        expected_re = [0.99, 0.5, 0.0099]
        expected_im = [-0.099, -0.5, -0.099]
        assert_almost_equal(H.real, expected_re, decimal=1)
        assert_almost_equal(H.imag, expected_im, decimal=1)

    def test_output(self):
        # Test freqresp() output calculation.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([1], [1, 1])
        w = [0.1, 1, 10, 100]
        w, H = freqresp(system, w=w)
        s = w * 1j
        expected = np.polyval(system.num, s) / np.polyval(system.den, s)
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 1 / (s + 1)
        # Expected range is from 0.01 to 10.
        system = lti([1], [1, 1])
        n = 10
        expected_w = np.logspace(-2, 1, n)
        w, H = freqresp(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_zero(self):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = lti([1], [1, 0])
        w, H = freqresp(system, n=2)
        assert_equal(w[0], 0.01)  # a fail would give not-a-number

    def test_from_state_space(self):
        # Ensure that freqresp works with a system that was created from the
        # state space representation matrices A, B, C, D.  In this case,
        # system.num will be a 2-D array with shape (1, n+1), where (n,n) is
        # the shape of A.
        # A Butterworth lowpass filter is used, so we know the exact
        # frequency response.
        a = np.array([1.0, 2.0, 2.0, 1.0])
        A = linalg.companion(a).T
        B = np.array([[0.0],[0.0],[1.0]])
        C = np.array([[1.0, 0.0, 0.0]])
        D = np.array([[0.0]])
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            system = lti(A, B, C, D)
            w, H = freqresp(system, n=100)
        s = w * 1j
        expected = (1.0 / (1.0 + 2*s + 2*s**2 + s**3))
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)

    def test_from_zpk(self):
        # 4th order low-pass filter: H(s) = 1 / (s + 1)
        system = lti([],[-1]*4,[1])
        w = [0.1, 1, 10, 100]
        w, H = freqresp(system, w=w)
        s = w * 1j
        expected = 1 / (s + 1)**4
        assert_almost_equal(H.real, expected.real)
        assert_almost_equal(H.imag, expected.imag)

