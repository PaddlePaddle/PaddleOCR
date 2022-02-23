import numpy as np
from numpy.testing import \
                          assert_array_almost_equal, assert_almost_equal, \
                          assert_allclose, assert_equal

import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim2, lti
from scipy.signal import tf2ss, impulse2, dimpulse, step2, dstep

# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# March 29, 2011


class TestC2D:
    def test_zoh(self):
        ac = np.eye(2)
        bc = np.full((2, 1), 0.5)
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = np.array([[0.0], [0.0], [-0.33]])

        ad_truth = 1.648721270700128 * np.eye(2)
        bd_truth = np.full((2, 1), 0.324360635350064)
        # c and d in discrete should be equal to their continuous counterparts
        dt_requested = 0.5

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='zoh')

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cc, cd)
        assert_array_almost_equal(dc, dd)
        assert_almost_equal(dt_requested, dt)

    def test_foh(self):
        ac = np.eye(2)
        bc = np.full((2, 1), 0.5)
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = np.array([[0.0], [0.0], [-0.33]])

        # True values are verified with Matlab
        ad_truth = 1.648721270700128 * np.eye(2)
        bd_truth = np.full((2, 1), 0.420839287058789)
        cd_truth = cc
        dd_truth = np.array([[0.260262223725224],
                             [0.297442541400256],
                             [-0.144098411624840]])
        dt_requested = 0.5

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='foh')

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

    def test_impulse(self):
        ac = np.eye(2)
        bc = np.full((2, 1), 0.5)
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = np.array([[0.0], [0.0], [0.0]])

        # True values are verified with Matlab
        ad_truth = 1.648721270700128 * np.eye(2)
        bd_truth = np.full((2, 1), 0.412180317675032)
        cd_truth = cc
        dd_truth = np.array([[0.4375], [0.5], [0.3125]])
        dt_requested = 0.5

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='impulse')

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

    def test_gbt(self):
        ac = np.eye(2)
        bc = np.full((2, 1), 0.5)
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = np.array([[0.0], [0.0], [-0.33]])

        dt_requested = 0.5
        alpha = 1.0 / 3.0

        ad_truth = 1.6 * np.eye(2)
        bd_truth = np.full((2, 1), 0.3)
        cd_truth = np.array([[0.9, 1.2],
                             [1.2, 1.2],
                             [1.2, 0.3]])
        dd_truth = np.array([[0.175],
                             [0.2],
                             [-0.205]])

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='gbt', alpha=alpha)

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)

    def test_euler(self):
        ac = np.eye(2)
        bc = np.full((2, 1), 0.5)
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = np.array([[0.0], [0.0], [-0.33]])

        dt_requested = 0.5

        ad_truth = 1.5 * np.eye(2)
        bd_truth = np.full((2, 1), 0.25)
        cd_truth = np.array([[0.75, 1.0],
                             [1.0, 1.0],
                             [1.0, 0.25]])
        dd_truth = dc

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='euler')

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

    def test_backward_diff(self):
        ac = np.eye(2)
        bc = np.full((2, 1), 0.5)
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = np.array([[0.0], [0.0], [-0.33]])

        dt_requested = 0.5

        ad_truth = 2.0 * np.eye(2)
        bd_truth = np.full((2, 1), 0.5)
        cd_truth = np.array([[1.5, 2.0],
                             [2.0, 2.0],
                             [2.0, 0.5]])
        dd_truth = np.array([[0.875],
                             [1.0],
                             [0.295]])

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='backward_diff')

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)

    def test_bilinear(self):
        ac = np.eye(2)
        bc = np.full((2, 1), 0.5)
        cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
        dc = np.array([[0.0], [0.0], [-0.33]])

        dt_requested = 0.5

        ad_truth = (5.0 / 3.0) * np.eye(2)
        bd_truth = np.full((2, 1), 1.0 / 3.0)
        cd_truth = np.array([[1.0, 4.0 / 3.0],
                             [4.0 / 3.0, 4.0 / 3.0],
                             [4.0 / 3.0, 1.0 / 3.0]])
        dd_truth = np.array([[0.291666666666667],
                             [1.0 / 3.0],
                             [-0.121666666666667]])

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='bilinear')

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

        # Same continuous system again, but change sampling rate

        ad_truth = 1.4 * np.eye(2)
        bd_truth = np.full((2, 1), 0.2)
        cd_truth = np.array([[0.9, 1.2], [1.2, 1.2], [1.2, 0.3]])
        dd_truth = np.array([[0.175], [0.2], [-0.205]])

        dt_requested = 1.0 / 3.0

        ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
                                 method='bilinear')

        assert_array_almost_equal(ad_truth, ad)
        assert_array_almost_equal(bd_truth, bd)
        assert_array_almost_equal(cd_truth, cd)
        assert_array_almost_equal(dd_truth, dd)
        assert_almost_equal(dt_requested, dt)

    def test_transferfunction(self):
        numc = np.array([0.25, 0.25, 0.5])
        denc = np.array([0.75, 0.75, 1.0])

        numd = np.array([[1.0 / 3.0, -0.427419169438754, 0.221654141101125]])
        dend = np.array([1.0, -1.351394049721225, 0.606530659712634])

        dt_requested = 0.5

        num, den, dt = c2d((numc, denc), dt_requested, method='zoh')

        assert_array_almost_equal(numd, num)
        assert_array_almost_equal(dend, den)
        assert_almost_equal(dt_requested, dt)

    def test_zerospolesgain(self):
        zeros_c = np.array([0.5, -0.5])
        poles_c = np.array([1.j / np.sqrt(2), -1.j / np.sqrt(2)])
        k_c = 1.0

        zeros_d = [1.23371727305860, 0.735356894461267]
        polls_d = [0.938148335039729 + 0.346233593780536j,
                   0.938148335039729 - 0.346233593780536j]
        k_d = 1.0

        dt_requested = 0.5

        zeros, poles, k, dt = c2d((zeros_c, poles_c, k_c), dt_requested,
                                  method='zoh')

        assert_array_almost_equal(zeros_d, zeros)
        assert_array_almost_equal(polls_d, poles)
        assert_almost_equal(k_d, k)
        assert_almost_equal(dt_requested, dt)

    def test_gbt_with_sio_tf_and_zpk(self):
        """Test method='gbt' with alpha=0.25 for tf and zpk cases."""
        # State space coefficients for the continuous SIO system.
        A = -1.0
        B = 1.0
        C = 1.0
        D = 0.5

        # The continuous transfer function coefficients.
        cnum, cden = ss2tf(A, B, C, D)

        # Continuous zpk representation
        cz, cp, ck = ss2zpk(A, B, C, D)

        h = 1.0
        alpha = 0.25

        # Explicit formulas, in the scalar case.
        Ad = (1 + (1 - alpha) * h * A) / (1 - alpha * h * A)
        Bd = h * B / (1 - alpha * h * A)
        Cd = C / (1 - alpha * h * A)
        Dd = D + alpha * C * Bd

        # Convert the explicit solution to tf
        dnum, dden = ss2tf(Ad, Bd, Cd, Dd)

        # Compute the discrete tf using cont2discrete.
        c2dnum, c2dden, dt = c2d((cnum, cden), h, method='gbt', alpha=alpha)

        assert_allclose(dnum, c2dnum)
        assert_allclose(dden, c2dden)

        # Convert explicit solution to zpk.
        dz, dp, dk = ss2zpk(Ad, Bd, Cd, Dd)

        # Compute the discrete zpk using cont2discrete.
        c2dz, c2dp, c2dk, dt = c2d((cz, cp, ck), h, method='gbt', alpha=alpha)

        assert_allclose(dz, c2dz)
        assert_allclose(dp, c2dp)
        assert_allclose(dk, c2dk)

    def test_discrete_approx(self):
        """
        Test that the solution to the discrete approximation of a continuous
        system actually approximates the solution to the continuous system.
        This is an indirect test of the correctness of the implementation
        of cont2discrete.
        """

        def u(t):
            return np.sin(2.5 * t)

        a = np.array([[-0.01]])
        b = np.array([[1.0]])
        c = np.array([[1.0]])
        d = np.array([[0.2]])
        x0 = 1.0

        t = np.linspace(0, 10.0, 101)
        dt = t[1] - t[0]
        u1 = u(t)

        # Use lsim2 to compute the solution to the continuous system.
        t, yout, xout = lsim2((a, b, c, d), T=t, U=u1, X0=x0,
                              rtol=1e-9, atol=1e-11)

        # Convert the continuous system to a discrete approximation.
        dsys = c2d((a, b, c, d), dt, method='bilinear')

        # Use dlsim with the pairwise averaged input to compute the output
        # of the discrete system.
        u2 = 0.5 * (u1[:-1] + u1[1:])
        t2 = t[:-1]
        td2, yd2, xd2 = dlsim(dsys, u=u2.reshape(-1, 1), t=t2, x0=x0)

        # ymid is the average of consecutive terms of the "exact" output
        # computed by lsim2.  This is what the discrete approximation
        # actually approximates.
        ymid = 0.5 * (yout[:-1] + yout[1:])

        assert_allclose(yd2.ravel(), ymid, rtol=1e-4)

    def test_simo_tf(self):
        # See gh-5753
        tf = ([[1, 0], [1, 1]], [1, 1])
        num, den, dt = c2d(tf, 0.01)

        assert_equal(dt, 0.01)  # sanity check
        assert_allclose(den, [1, -0.990404983], rtol=1e-3)
        assert_allclose(num, [[1, -1], [1, -0.99004983]], rtol=1e-3)

    def test_multioutput(self):
        ts = 0.01  # time step

        tf = ([[1, -3], [1, 5]], [1, 1])
        num, den, dt = c2d(tf, ts)

        tf1 = (tf[0][0], tf[1])
        num1, den1, dt1 = c2d(tf1, ts)

        tf2 = (tf[0][1], tf[1])
        num2, den2, dt2 = c2d(tf2, ts)

        # Sanity checks
        assert_equal(dt, dt1)
        assert_equal(dt, dt2)

        # Check that we get the same results
        assert_allclose(num, np.vstack((num1, num2)), rtol=1e-13)

        # Single input, so the denominator should
        # not be multidimensional like the numerator
        assert_allclose(den, den1, rtol=1e-13)
        assert_allclose(den, den2, rtol=1e-13)

class TestC2dLti:
    def test_c2d_ss(self):
        # StateSpace
        A = np.array([[-0.3, 0.1], [0.2, -0.7]])
        B = np.array([[0], [1]])
        C = np.array([[1, 0]])
        D = 0

        A_res = np.array([[0.985136404135682, 0.004876671474795],
                          [0.009753342949590, 0.965629718236502]])
        B_res = np.array([[0.000122937599964], [0.049135527547844]])

        sys_ssc = lti(A, B, C, D)
        sys_ssd = sys_ssc.to_discrete(0.05)

        assert_allclose(sys_ssd.A, A_res)
        assert_allclose(sys_ssd.B, B_res)
        assert_allclose(sys_ssd.C, C)
        assert_allclose(sys_ssd.D, D)

    def test_c2d_tf(self):

        sys = lti([0.5, 0.3], [1.0, 0.4])
        sys = sys.to_discrete(0.005)

        # Matlab results
        num_res = np.array([0.5, -0.485149004980066])
        den_res = np.array([1.0, -0.980198673306755])

        # Somehow a lot of numerical errors
        assert_allclose(sys.den, den_res, atol=0.02)
        assert_allclose(sys.num, num_res, atol=0.02)


class TestC2dInvariants:
    # Some test cases for checking the invariances.
    # Array of triplets: (system, sample time, number of samples)
    cases = [
        (tf2ss([1, 1], [1, 1.5, 1]), 0.25, 10),
        (tf2ss([1, 2], [1, 1.5, 3, 1]), 0.5, 10),
        (tf2ss(0.1, [1, 1, 2, 1]), 0.5, 10),
    ]

    # Some options for lsim2 and derived routines
    tolerances = {'rtol': 1e-9, 'atol': 1e-11}

    # Check that systems discretized with the impulse-invariant
    # method really hold the invariant
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_impulse_invariant(self, sys, sample_time, samples_number):
        time = np.arange(samples_number) * sample_time
        _, yout_cont = impulse2(sys, T=time, **self.tolerances)
        _, yout_disc = dimpulse(c2d(sys, sample_time, method='impulse'),
                                n=len(time))
        assert_allclose(sample_time * yout_cont.ravel(), yout_disc[0].ravel())

    # Step invariant should hold for ZOH discretized systems
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_step_invariant(self, sys, sample_time, samples_number):
        time = np.arange(samples_number) * sample_time
        _, yout_cont = step2(sys, T=time, **self.tolerances)
        _, yout_disc = dstep(c2d(sys, sample_time, method='zoh'), n=len(time))
        assert_allclose(yout_cont.ravel(), yout_disc[0].ravel())

    # Linear invariant should hold for FOH discretized systems
    @pytest.mark.parametrize("sys,sample_time,samples_number", cases)
    def test_linear_invariant(self, sys, sample_time, samples_number):
        time = np.arange(samples_number) * sample_time
        _, yout_cont, _ = lsim2(sys, T=time, U=time, **self.tolerances)
        _, yout_disc, _ = dlsim(c2d(sys, sample_time, method='foh'), u=time)
        assert_allclose(yout_cont.ravel(), yout_disc.ravel())
