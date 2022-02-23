# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# April 4, 2011

import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_, assert_almost_equal,
                           suppress_warnings)
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
                          StateSpace, TransferFunction, ZerosPolesGain,
                          dfreqresp, dbode, BadCoefficients)


class TestDLTI:

    def test_dlsim(self):

        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = np.asarray([[0.1, 0.3]])
        d = np.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        # Create an input matrix with inputs down the columns (3 cols) and its
        # respective time input vector
        u = np.hstack((np.linspace(0, 4.0, num=5)[:, np.newaxis],
                       np.full((5, 1), 0.01),
                       np.full((5, 1), -0.002)))
        t_in = np.linspace(0, 2.0, num=5)

        # Define the known result
        yout_truth = np.array([[-0.001,
                                -0.00073,
                                0.039446,
                                0.0915387,
                                0.13195948]]).T
        xout_truth = np.asarray([[0, 0],
                                 [0.0012, 0.0005],
                                 [0.40233, 0.00071],
                                 [1.163368, -0.079327],
                                 [2.2402985, -0.3035679]])

        tout, yout, xout = dlsim((a, b, c, d, dt), u, t_in)

        assert_array_almost_equal(yout_truth, yout)
        assert_array_almost_equal(xout_truth, xout)
        assert_array_almost_equal(t_in, tout)

        # Make sure input with single-dimension doesn't raise error
        dlsim((1, 2, 3), 4)

        # Interpolated control - inputs should have different time steps
        # than the discrete model uses internally
        u_sparse = u[[0, 4], :]
        t_sparse = np.asarray([0.0, 2.0])

        tout, yout, xout = dlsim((a, b, c, d, dt), u_sparse, t_sparse)

        assert_array_almost_equal(yout_truth, yout)
        assert_array_almost_equal(xout_truth, xout)
        assert_equal(len(tout), yout.shape[0])

        # Transfer functions (assume dt = 0.5)
        num = np.asarray([1.0, -0.1])
        den = np.asarray([0.3, 1.0, 0.2])
        yout_truth = np.array([[0.0,
                                0.0,
                                3.33333333333333,
                                -4.77777777777778,
                                23.0370370370370]]).T

        # Assume use of the first column of the control input built earlier
        tout, yout = dlsim((num, den, 0.5), u[:, 0], t_in)

        assert_array_almost_equal(yout, yout_truth)
        assert_array_almost_equal(t_in, tout)

        # Retest the same with a 1-D input vector
        uflat = np.asarray(u[:, 0])
        uflat = uflat.reshape((5,))
        tout, yout = dlsim((num, den, 0.5), uflat, t_in)

        assert_array_almost_equal(yout, yout_truth)
        assert_array_almost_equal(t_in, tout)

        # zeros-poles-gain representation
        zd = np.array([0.5, -0.5])
        pd = np.array([1.j / np.sqrt(2), -1.j / np.sqrt(2)])
        k = 1.0
        yout_truth = np.array([[0.0, 1.0, 2.0, 2.25, 2.5]]).T

        tout, yout = dlsim((zd, pd, k, 0.5), u[:, 0], t_in)

        assert_array_almost_equal(yout, yout_truth)
        assert_array_almost_equal(t_in, tout)

        # Raise an error for continuous-time systems
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dlsim, system, u)

    def test_dstep(self):

        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = np.asarray([[0.1, 0.3]])
        d = np.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        # Because b.shape[1] == 3, dstep should result in a tuple of three
        # result vectors
        yout_step_truth = (np.asarray([0.0, 0.04, 0.052, 0.0404, 0.00956,
                                       -0.036324, -0.093318, -0.15782348,
                                       -0.226628324, -0.2969374948]),
                           np.asarray([-0.1, -0.075, -0.058, -0.04815,
                                       -0.04453, -0.0461895, -0.0521812,
                                       -0.061588875, -0.073549579,
                                       -0.08727047595]),
                           np.asarray([0.0, -0.01, -0.013, -0.0101, -0.00239,
                                       0.009081, 0.0233295, 0.03945587,
                                       0.056657081, 0.0742343737]))

        tout, yout = dstep((a, b, c, d, dt), n=10)

        assert_equal(len(yout), 3)

        for i in range(0, len(yout)):
            assert_equal(yout[i].shape[0], 10)
            assert_array_almost_equal(yout[i].flatten(), yout_step_truth[i])

        # Check that the other two inputs (tf, zpk) will work as well
        tfin = ([1.0], [1.0, 1.0], 0.5)
        yout_tfstep = np.asarray([0.0, 1.0, 0.0])
        tout, yout = dstep(tfin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfstep)

        zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = dstep(zpkin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfstep)

        # Raise an error for continuous-time systems
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dstep, system)

    def test_dimpulse(self):

        a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
        b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
        c = np.asarray([[0.1, 0.3]])
        d = np.asarray([[0.0, -0.1, 0.0]])
        dt = 0.5

        # Because b.shape[1] == 3, dimpulse should result in a tuple of three
        # result vectors
        yout_imp_truth = (np.asarray([0.0, 0.04, 0.012, -0.0116, -0.03084,
                                      -0.045884, -0.056994, -0.06450548,
                                      -0.068804844, -0.0703091708]),
                          np.asarray([-0.1, 0.025, 0.017, 0.00985, 0.00362,
                                      -0.0016595, -0.0059917, -0.009407675,
                                      -0.011960704, -0.01372089695]),
                          np.asarray([0.0, -0.01, -0.003, 0.0029, 0.00771,
                                      0.011471, 0.0142485, 0.01612637,
                                      0.017201211, 0.0175772927]))

        tout, yout = dimpulse((a, b, c, d, dt), n=10)

        assert_equal(len(yout), 3)

        for i in range(0, len(yout)):
            assert_equal(yout[i].shape[0], 10)
            assert_array_almost_equal(yout[i].flatten(), yout_imp_truth[i])

        # Check that the other two inputs (tf, zpk) will work as well
        tfin = ([1.0], [1.0, 1.0], 0.5)
        yout_tfimpulse = np.asarray([0.0, 1.0, -1.0])
        tout, yout = dimpulse(tfin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)

        zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
        tout, yout = dimpulse(zpkin, n=3)
        assert_equal(len(yout), 1)
        assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)

        # Raise an error for continuous-time systems
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dimpulse, system)

    def test_dlsim_trivial(self):
        a = np.array([[0.0]])
        b = np.array([[0.0]])
        c = np.array([[0.0]])
        d = np.array([[0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        tout, yout, xout = dlsim((a, b, c, d, 1), u)
        assert_array_equal(tout, np.arange(float(n)))
        assert_array_equal(yout, np.zeros((n, 1)))
        assert_array_equal(xout, np.zeros((n, 1)))

    def test_dlsim_simple1d(self):
        a = np.array([[0.5]])
        b = np.array([[0.0]])
        c = np.array([[1.0]])
        d = np.array([[0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
        assert_array_equal(tout, np.arange(float(n)))
        expected = (0.5 ** np.arange(float(n))).reshape(-1, 1)
        assert_array_equal(yout, expected)
        assert_array_equal(xout, expected)

    def test_dlsim_simple2d(self):
        lambda1 = 0.5
        lambda2 = 0.25
        a = np.array([[lambda1, 0.0],
                      [0.0, lambda2]])
        b = np.array([[0.0],
                      [0.0]])
        c = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        d = np.array([[0.0],
                      [0.0]])
        n = 5
        u = np.zeros(n).reshape(-1, 1)
        tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
        assert_array_equal(tout, np.arange(float(n)))
        # The analytical solution:
        expected = (np.array([lambda1, lambda2]) **
                                np.arange(float(n)).reshape(-1, 1))
        assert_array_equal(yout, expected)
        assert_array_equal(xout, expected)

    def test_more_step_and_impulse(self):
        lambda1 = 0.5
        lambda2 = 0.75
        a = np.array([[lambda1, 0.0],
                      [0.0, lambda2]])
        b = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
        c = np.array([[1.0, 1.0]])
        d = np.array([[0.0, 0.0]])

        n = 10

        # Check a step response.
        ts, ys = dstep((a, b, c, d, 1), n=n)

        # Create the exact step response.
        stp0 = (1.0 / (1 - lambda1)) * (1.0 - lambda1 ** np.arange(n))
        stp1 = (1.0 / (1 - lambda2)) * (1.0 - lambda2 ** np.arange(n))

        assert_allclose(ys[0][:, 0], stp0)
        assert_allclose(ys[1][:, 0], stp1)

        # Check an impulse response with an initial condition.
        x0 = np.array([1.0, 1.0])
        ti, yi = dimpulse((a, b, c, d, 1), n=n, x0=x0)

        # Create the exact impulse response.
        imp = (np.array([lambda1, lambda2]) **
                            np.arange(-1, n + 1).reshape(-1, 1))
        imp[0, :] = 0.0
        # Analytical solution to impulse response
        y0 = imp[:n, 0] + np.dot(imp[1:n + 1, :], x0)
        y1 = imp[:n, 1] + np.dot(imp[1:n + 1, :], x0)

        assert_allclose(yi[0][:, 0], y0)
        assert_allclose(yi[1][:, 0], y1)

        # Check that dt=0.1, n=3 gives 3 time values.
        system = ([1.0], [1.0, -0.5], 0.1)
        t, (y,) = dstep(system, n=3)
        assert_allclose(t, [0, 0.1, 0.2])
        assert_array_equal(y.T, [[0, 1.0, 1.5]])
        t, (y,) = dimpulse(system, n=3)
        assert_allclose(t, [0, 0.1, 0.2])
        assert_array_equal(y.T, [[0, 1, 0.5]])


class TestDlti:
    def test_dlti_instantiation(self):
        # Test that lti can be instantiated.

        dt = 0.05
        # TransferFunction
        s = dlti([1], [-1], dt=dt)
        assert_(isinstance(s, TransferFunction))
        assert_(isinstance(s, dlti))
        assert_(not isinstance(s, lti))
        assert_equal(s.dt, dt)

        # ZerosPolesGain
        s = dlti(np.array([]), np.array([-1]), 1, dt=dt)
        assert_(isinstance(s, ZerosPolesGain))
        assert_(isinstance(s, dlti))
        assert_(not isinstance(s, lti))
        assert_equal(s.dt, dt)

        # StateSpace
        s = dlti([1], [-1], 1, 3, dt=dt)
        assert_(isinstance(s, StateSpace))
        assert_(isinstance(s, dlti))
        assert_(not isinstance(s, lti))
        assert_equal(s.dt, dt)

        # Number of inputs
        assert_raises(ValueError, dlti, 1)
        assert_raises(ValueError, dlti, 1, 1, 1, 1, 1)


class TestStateSpaceDisc:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        StateSpace(1, 1, 1, 1, dt=dt)
        StateSpace([1], [2], [3], [4], dt=dt)
        StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]),
                   np.array([[1, 0]]), np.array([[0]]), dt=dt)
        StateSpace(1, 1, 1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = StateSpace(1, 2, 3, 4, dt=0.05)
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
        s = StateSpace(1, 1, 1, 1, dt=0.05)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])


class TestTransferFunction:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        TransferFunction(1, 1, dt=dt)
        TransferFunction([1], [2], dt=dt)
        TransferFunction(np.array([1]), np.array([2]), dt=dt)
        TransferFunction(1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = TransferFunction([1, 0], [1, -1], dt=0.05)
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
        s = TransferFunction([1, 0], [1, -1], dt=0.05)
        assert_equal(s.poles, [1])
        assert_equal(s.zeros, [0])


class TestZerosPolesGain:
    def test_initialization(self):
        # Check that all initializations work
        dt = 0.05
        ZerosPolesGain(1, 1, 1, dt=dt)
        ZerosPolesGain([1], [2], 1, dt=dt)
        ZerosPolesGain(np.array([1]), np.array([2]), 1, dt=dt)
        ZerosPolesGain(1, 1, 1, dt=True)

    def test_conversion(self):
        # Check the conversion functions
        s = ZerosPolesGain(1, 2, 3, dt=0.05)
        assert_(isinstance(s.to_ss(), StateSpace))
        assert_(isinstance(s.to_tf(), TransferFunction))
        assert_(isinstance(s.to_zpk(), ZerosPolesGain))

        # Make sure copies work
        assert_(ZerosPolesGain(s) is not s)
        assert_(s.to_zpk() is not s)


class Test_dfreqresp:

    def test_manual(self):
        # Test dfreqresp() real part calculation (manual sanity check).
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        w = [0.1, 1, 10]
        w, H = dfreqresp(system, w=w)

        # test real
        expected_re = [1.2383, 0.4130, -0.7553]
        assert_almost_equal(H.real, expected_re, decimal=4)

        # test imag
        expected_im = [-0.1555, -1.0214, 0.3955]
        assert_almost_equal(H.imag, expected_im, decimal=4)

    def test_auto(self):
        # Test dfreqresp() real part calculation.
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        w = [0.1, 1, 10, 100]
        w, H = dfreqresp(system, w=w)
        jw = np.exp(w * 1j)
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)

        # test real
        expected_re = y.real
        assert_almost_equal(H.real, expected_re)

        # test imag
        expected_im = y.imag
        assert_almost_equal(H.imag, expected_im)

    def test_freq_range(self):
        # Test that freqresp() finds a reasonable frequency range.
        # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
        # Expected range is from 0.01 to 10.
        system = TransferFunction(1, [1, -0.2], dt=0.1)
        n = 10
        expected_w = np.linspace(0, np.pi, 10, endpoint=False)
        w, H = dfreqresp(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_one(self):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = TransferFunction([1], [1, -1], dt=0.1)

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message="divide by zero")
            sup.filter(RuntimeWarning, message="invalid value encountered")
            w, H = dfreqresp(system, n=2)
        assert_equal(w[0], 0.)  # a fail would give not-a-number

    def test_error(self):
        # Raise an error for continuous-time systems
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dfreqresp, system)

    def test_from_state_space(self):
        # H(z) = 2 / z^3 - 0.5 * z^2

        system_TF = dlti([2], [1, -0.5, 0, 0])

        A = np.array([[0.5, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])
        B = np.array([[1, 0, 0]]).T
        C = np.array([[0, 0, 2]])
        D = 0

        system_SS = dlti(A, B, C, D)
        w = 10.0**np.arange(-3,0,.5)
        with suppress_warnings() as sup:
            sup.filter(BadCoefficients)
            w1, H1 = dfreqresp(system_TF, w=w)
            w2, H2 = dfreqresp(system_SS, w=w)

        assert_almost_equal(H1, H2)

    def test_from_zpk(self):
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        system_ZPK = dlti([],[0.2],0.3)
        system_TF = dlti(0.3, [1, -0.2])
        w = [0.1, 1, 10, 100]
        w1, H1 = dfreqresp(system_ZPK, w=w)
        w2, H2 = dfreqresp(system_TF, w=w)
        assert_almost_equal(H1, H2)


class Test_bode:

    def test_manual(self):
        # Test bode() magnitude calculation (manual sanity check).
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        dt = 0.1
        system = TransferFunction(0.3, [1, -0.2], dt=dt)
        w = [0.1, 0.5, 1, np.pi]
        w2, mag, phase = dbode(system, w=w)

        # Test mag
        expected_mag = [-8.5329, -8.8396, -9.6162, -12.0412]
        assert_almost_equal(mag, expected_mag, decimal=4)

        # Test phase
        expected_phase = [-7.1575, -35.2814, -67.9809, -180.0000]
        assert_almost_equal(phase, expected_phase, decimal=4)

        # Test frequency
        assert_equal(np.array(w) / dt, w2)

    def test_auto(self):
        # Test bode() magnitude calculation.
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        system = TransferFunction(0.3, [1, -0.2], dt=0.1)
        w = np.array([0.1, 0.5, 1, np.pi])
        w2, mag, phase = dbode(system, w=w)
        jw = np.exp(w * 1j)
        y = np.polyval(system.num, jw) / np.polyval(system.den, jw)

        # Test mag
        expected_mag = 20.0 * np.log10(abs(y))
        assert_almost_equal(mag, expected_mag)

        # Test phase
        expected_phase = np.rad2deg(np.angle(y))
        assert_almost_equal(phase, expected_phase)

    def test_range(self):
        # Test that bode() finds a reasonable frequency range.
        # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
        dt = 0.1
        system = TransferFunction(0.3, [1, -0.2], dt=0.1)
        n = 10
        # Expected range is from 0.01 to 10.
        expected_w = np.linspace(0, np.pi, n, endpoint=False) / dt
        w, mag, phase = dbode(system, n=n)
        assert_almost_equal(w, expected_w)

    def test_pole_one(self):
        # Test that freqresp() doesn't fail on a system with a pole at 0.
        # integrator, pole at zero: H(s) = 1 / s
        system = TransferFunction([1], [1, -1], dt=0.1)

        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, message="divide by zero")
            sup.filter(RuntimeWarning, message="invalid value encountered")
            w, mag, phase = dbode(system, n=2)
        assert_equal(w[0], 0.)  # a fail would give not-a-number

    def test_imaginary(self):
        # bode() should not fail on a system with pure imaginary poles.
        # The test passes if bode doesn't raise an exception.
        system = TransferFunction([1], [1, 0, 100], dt=0.1)
        dbode(system, n=2)

    def test_error(self):
        # Raise an error for continuous-time systems
        system = lti([1], [1, 1])
        assert_raises(AttributeError, dbode, system)


class TestTransferFunctionZConversion:
    """Test private conversions between 'z' and 'z**-1' polynomials."""

    def test_full(self):
        # Numerator and denominator same order
        num = [2, 3, 4]
        den = [5, 6, 7]
        num2, den2 = TransferFunction._z_to_zinv(num, den)
        assert_equal(num, num2)
        assert_equal(den, den2)

        num2, den2 = TransferFunction._zinv_to_z(num, den)
        assert_equal(num, num2)
        assert_equal(den, den2)

    def test_numerator(self):
        # Numerator lower order than denominator
        num = [2, 3]
        den = [5, 6, 7]
        num2, den2 = TransferFunction._z_to_zinv(num, den)
        assert_equal([0, 2, 3], num2)
        assert_equal(den, den2)

        num2, den2 = TransferFunction._zinv_to_z(num, den)
        assert_equal([2, 3, 0], num2)
        assert_equal(den, den2)

    def test_denominator(self):
        # Numerator higher order than denominator
        num = [2, 3, 4]
        den = [5, 6]
        num2, den2 = TransferFunction._z_to_zinv(num, den)
        assert_equal(num, num2)
        assert_equal([0, 5, 6], den2)

        num2, den2 = TransferFunction._zinv_to_z(num, den)
        assert_equal(num, num2)
        assert_equal([5, 6, 0], den2)

