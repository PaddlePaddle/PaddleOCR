import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_equal, assert_,
                           assert_allclose, assert_warns)
from pytest import raises as assert_raises
import pytest

from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
    firwin, firwin2, freqz, remez, firls, minimum_phase


def test_kaiser_beta():
    b = kaiser_beta(58.7)
    assert_almost_equal(b, 0.1102 * 50.0)
    b = kaiser_beta(22.0)
    assert_almost_equal(b, 0.5842 + 0.07886)
    b = kaiser_beta(21.0)
    assert_equal(b, 0.0)
    b = kaiser_beta(10.0)
    assert_equal(b, 0.0)


def test_kaiser_atten():
    a = kaiser_atten(1, 1.0)
    assert_equal(a, 7.95)
    a = kaiser_atten(2, 1/np.pi)
    assert_equal(a, 2.285 + 7.95)


def test_kaiserord():
    assert_raises(ValueError, kaiserord, 1.0, 1.0)
    numtaps, beta = kaiserord(2.285 + 7.95 - 0.001, 1/np.pi)
    assert_equal((numtaps, beta), (2, 0.0))


class TestFirwin:

    def check_response(self, h, expected_response, tol=.05):
        N = len(h)
        alpha = 0.5 * (N-1)
        m = np.arange(0,N) - alpha   # time indices of taps
        for freq, expected in expected_response:
            actual = abs(np.sum(h*np.exp(-1.j*np.pi*m*freq)))
            mse = abs(actual-expected)**2
            assert_(mse < tol, 'response not as expected, mse=%g > %g'
               % (mse, tol))

    def test_response(self):
        N = 51
        f = .5
        # increase length just to try even/odd
        h = firwin(N, f)  # low-pass from 0 to f
        self.check_response(h, [(.25,1), (.75,0)])

        h = firwin(N+1, f, window='nuttall')  # specific window
        self.check_response(h, [(.25,1), (.75,0)])

        h = firwin(N+2, f, pass_zero=False)  # stop from 0 to f --> high-pass
        self.check_response(h, [(.25,0), (.75,1)])

        f1, f2, f3, f4 = .2, .4, .6, .8
        h = firwin(N+3, [f1, f2], pass_zero=False)  # band-pass filter
        self.check_response(h, [(.1,0), (.3,1), (.5,0)])

        h = firwin(N+4, [f1, f2])  # band-stop filter
        self.check_response(h, [(.1,1), (.3,0), (.5,1)])

        h = firwin(N+5, [f1, f2, f3, f4], pass_zero=False, scale=False)
        self.check_response(h, [(.1,0), (.3,1), (.5,0), (.7,1), (.9,0)])

        h = firwin(N+6, [f1, f2, f3, f4])  # multiband filter
        self.check_response(h, [(.1,1), (.3,0), (.5,1), (.7,0), (.9,1)])

        h = firwin(N+7, 0.1, width=.03)  # low-pass
        self.check_response(h, [(.05,1), (.75,0)])

        h = firwin(N+8, 0.1, pass_zero=False)  # high-pass
        self.check_response(h, [(.05,0), (.75,1)])

    def mse(self, h, bands):
        """Compute mean squared error versus ideal response across frequency
        band.
          h -- coefficients
          bands -- list of (left, right) tuples relative to 1==Nyquist of
            passbands
        """
        w, H = freqz(h, worN=1024)
        f = w/np.pi
        passIndicator = np.zeros(len(w), bool)
        for left, right in bands:
            passIndicator |= (f >= left) & (f < right)
        Hideal = np.where(passIndicator, 1, 0)
        mse = np.mean(abs(abs(H)-Hideal)**2)
        return mse

    def test_scaling(self):
        """
        For one lowpass, bandpass, and highpass example filter, this test
        checks two things:
          - the mean squared error over the frequency domain of the unscaled
            filter is smaller than the scaled filter (true for rectangular
            window)
          - the response of the scaled filter is exactly unity at the center
            of the first passband
        """
        N = 11
        cases = [
            ([.5], True, (0, 1)),
            ([0.2, .6], False, (.4, 1)),
            ([.5], False, (1, 1)),
        ]
        for cutoff, pass_zero, expected_response in cases:
            h = firwin(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
            hs = firwin(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
            if len(cutoff) == 1:
                if pass_zero:
                    cutoff = [0] + cutoff
                else:
                    cutoff = cutoff + [1]
            assert_(self.mse(h, [cutoff]) < self.mse(hs, [cutoff]),
                'least squares violation')
            self.check_response(hs, [expected_response], 1e-12)


class TestFirWinMore:
    """Different author, different style, different tests..."""

    def test_lowpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='lowpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_highpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)

        # Ensure that ntaps is odd.
        ntaps |= 1

        kwargs = dict(cutoff=0.5, window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='highpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_bandpass(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.3, 0.7], window=('kaiser', beta), scale=False)
        taps = firwin(ntaps, pass_zero=False, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.2, 0.3-width/2, 0.3+width/2, 0.5,
                                0.7-width/2, 0.7+width/2, 0.8, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        taps_str = firwin(ntaps, pass_zero='bandpass', **kwargs)
        assert_allclose(taps, taps_str)

    def test_bandstop_multi(self):
        width = 0.04
        ntaps, beta = kaiserord(120, width)
        kwargs = dict(cutoff=[0.2, 0.5, 0.8], window=('kaiser', beta),
                      scale=False)
        taps = firwin(ntaps, **kwargs)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 0.1, 0.2-width/2, 0.2+width/2, 0.35,
                                0.5-width/2, 0.5+width/2, 0.65,
                                0.8-width/2, 0.8+width/2, 0.9, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                decimal=5)

        taps_str = firwin(ntaps, pass_zero='bandstop', **kwargs)
        assert_allclose(taps, taps_str)

    def test_fs_nyq(self):
        """Test the fs and nyq keywords."""
        nyquist = 1000
        width = 40.0
        relative_width = width/nyquist
        ntaps, beta = kaiserord(120, relative_width)
        taps = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta),
                        pass_zero=False, scale=False, fs=2*nyquist)

        # Check the symmetry of taps.
        assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])

        # Check the gain at a few samples where we know it should be approximately 0 or 1.
        freq_samples = np.array([0.0, 200, 300-width/2, 300+width/2, 500,
                                700-width/2, 700+width/2, 800, 1000])
        freqs, response = freqz(taps, worN=np.pi*freq_samples/nyquist)
        assert_array_almost_equal(np.abs(response),
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)

        taps2 = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta),
                        pass_zero=False, scale=False, nyq=nyquist)
        assert_allclose(taps2, taps)

    def test_bad_cutoff(self):
        """Test that invalid cutoff argument raises ValueError."""
        # cutoff values must be greater than 0 and less than 1.
        assert_raises(ValueError, firwin, 99, -0.5)
        assert_raises(ValueError, firwin, 99, 1.5)
        # Don't allow 0 or 1 in cutoff.
        assert_raises(ValueError, firwin, 99, [0, 0.5])
        assert_raises(ValueError, firwin, 99, [0.5, 1])
        # cutoff values must be strictly increasing.
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
        assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
        # Must have at least one cutoff value.
        assert_raises(ValueError, firwin, 99, [])
        # 2D array not allowed.
        assert_raises(ValueError, firwin, 99, [[0.1, 0.2],[0.3, 0.4]])
        # cutoff values must be less than nyq.
        assert_raises(ValueError, firwin, 99, 50.0, nyq=40)
        assert_raises(ValueError, firwin, 99, [10, 20, 30], nyq=25)
        assert_raises(ValueError, firwin, 99, 50.0, fs=80)
        assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)

    def test_even_highpass_raises_value_error(self):
        """Test that attempt to create a highpass filter with an even number
        of taps raises a ValueError exception."""
        assert_raises(ValueError, firwin, 40, 0.5, pass_zero=False)
        assert_raises(ValueError, firwin, 40, [.25, 0.5])

    def test_bad_pass_zero(self):
        """Test degenerate pass_zero cases."""
        with assert_raises(ValueError, match='pass_zero must be'):
            firwin(41, 0.5, pass_zero='foo')
        with assert_raises(TypeError, match='cannot be interpreted'):
            firwin(41, 0.5, pass_zero=1.)
        for pass_zero in ('lowpass', 'highpass'):
            with assert_raises(ValueError, match='cutoff must have one'):
                firwin(41, [0.5, 0.6], pass_zero=pass_zero)
        for pass_zero in ('bandpass', 'bandstop'):
            with assert_raises(ValueError, match='must have at least two'):
                firwin(41, [0.5], pass_zero=pass_zero)


class TestFirwin2:

    def test_invalid_args(self):
        # `freq` and `gain` have different lengths.
        with assert_raises(ValueError, match='must be of same length'):
            firwin2(50, [0, 0.5, 1], [0.0, 1.0])
        # `nfreqs` is less than `ntaps`.
        with assert_raises(ValueError, match='ntaps must be less than nfreqs'):
            firwin2(50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
        # Decreasing value in `freq`
        with assert_raises(ValueError, match='must be nondecreasing'):
            firwin2(50, [0, 0.5, 0.4, 1.0], [0, .25, .5, 1.0])
        # Value in `freq` repeated more than once.
        with assert_raises(ValueError, match='must not occur more than twice'):
            firwin2(50, [0, .1, .1, .1, 1.0], [0.0, 0.5, 0.75, 1.0, 1.0])
        # `freq` does not start at 0.0.
        with assert_raises(ValueError, match='start with 0'):
            firwin2(50, [0.5, 1.0], [0.0, 1.0])
        # `freq` does not end at fs/2.
        with assert_raises(ValueError, match='end with fs/2'):
            firwin2(50, [0.0, 0.5], [0.0, 1.0])
        # Value 0 is repeated in `freq`
        with assert_raises(ValueError, match='0 must not be repeated'):
            firwin2(50, [0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        # Value fs/2 is repeated in `freq`
        with assert_raises(ValueError, match='fs/2 must not be repeated'):
            firwin2(50, [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
        # Value in `freq` that is too close to a repeated number
        with assert_raises(ValueError, match='cannot contain numbers '
                                             'that are too close'):
            firwin2(50, [0.0, 0.5 - np.finfo(float).eps * 0.5, 0.5, 0.5, 1.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0])

        # Type II filter, but the gain at nyquist frequency is not zero.
        with assert_raises(ValueError, match='Type II filter'):
            firwin2(16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])

        # Type III filter, but the gains at nyquist and zero rate are not zero.
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)
        with assert_raises(ValueError, match='Type III filter'):
            firwin2(17, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0], antisymmetric=True)

        # Type IV filter, but the gain at zero rate is not zero.
        with assert_raises(ValueError, match='Type IV filter'):
            firwin2(16, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], antisymmetric=True)

    def test01(self):
        width = 0.04
        beta = 12.0
        ntaps = 400
        # Filter is 1 from w=0 to w=0.5, then decreases linearly from 1 to 0 as w
        # increases from w=0.5 to w=1  (w=1 is the Nyquist frequency).
        freq = [0.0, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0]
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2,
                                                        0.75, 1.0-width/2])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                        [1.0, 1.0, 1.0, 1.0-width, 0.5, width], decimal=5)

    def test02(self):
        width = 0.04
        beta = 12.0
        # ntaps must be odd for positive gain at Nyquist.
        ntaps = 401
        # An ideal highpass filter.
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [0.0, 0.0, 1.0, 1.0]
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.25, 0.5-width, 0.5+width, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test03(self):
        width = 0.02
        ntaps, beta = kaiserord(120, width)
        # ntaps must be odd for positive gain at Nyquist.
        ntaps = int(ntaps) | 1
        freq = [0.0, 0.4, 0.4, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
        freq_samples = np.array([0.0, 0.4-width, 0.4+width, 0.45,
                                    0.5-width, 0.5+width, 0.75, 1.0])
        freqs, response = freqz(taps, worN=np.pi*freq_samples)
        assert_array_almost_equal(np.abs(response),
                    [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)

    def test04(self):
        """Test firwin2 when window=None."""
        ntaps = 5
        # Ideal lowpass: gain is 1 on [0,0.5], and 0 on [0.5, 1.0]
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0]
        taps = firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
        alpha = 0.5 * (ntaps - 1)
        m = np.arange(0, ntaps) - alpha
        h = 0.5 * sinc(0.5 * m)
        assert_array_almost_equal(h, taps)

    def test05(self):
        """Test firwin2 for calculating Type IV filters"""
        ntaps = 1500

        freq = [0.0, 1.0]
        gain = [0.0, 1.0]
        taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2:][::-1])

        freqs, response = freqz(taps, worN=2048)
        assert_array_almost_equal(abs(response), freqs / np.pi, decimal=4)

    def test06(self):
        """Test firwin2 for calculating Type III filters"""
        ntaps = 1501

        freq = [0.0, 0.5, 0.55, 1.0]
        gain = [0.0, 0.5, 0.0, 0.0]
        taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
        assert_equal(taps[ntaps // 2], 0.0)
        assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2 + 1:][::-1])

        freqs, response1 = freqz(taps, worN=2048)
        response2 = np.interp(freqs / np.pi, freq, gain)
        assert_array_almost_equal(abs(response1), response2, decimal=3)

    def test_fs_nyq(self):
        taps1 = firwin2(80, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
        taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], fs=120.0)
        assert_array_almost_equal(taps1, taps2)
        taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], nyq=60.0)
        assert_array_almost_equal(taps1, taps2)

    def test_tuple(self):
        taps1 = firwin2(150, (0.0, 0.5, 0.5, 1.0), (1.0, 1.0, 0.0, 0.0))
        taps2 = firwin2(150, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
        assert_array_almost_equal(taps1, taps2)

    def test_input_modyfication(self):
        freq1 = np.array([0.0, 0.5, 0.5, 1.0])
        freq2 = np.array(freq1)
        firwin2(80, freq1, [1.0, 1.0, 0.0, 0.0])
        assert_equal(freq1, freq2)


class TestRemez:

    def test_bad_args(self):
        assert_raises(ValueError, remez, 11, [0.1, 0.4], [1], type='pooka')

    def test_hilbert(self):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band

        # design an unity gain hilbert bandpass filter from w to 0.5-w
        h = remez(11, [a, 0.5-a], [1], type='hilbert')

        # make sure the filter has correct # of taps
        assert_(len(h) == N, "Number of Taps")

        # make sure it is type III (anti-symmetric tap coefficients)
        assert_array_almost_equal(h[:(N-1)//2], -h[:-(N-1)//2-1:-1])

        # Since the requested response is symmetric, all even coefficients
        # should be zero (or in this case really small)
        assert_((abs(h[1::2]) < 1e-15).all(), "Even Coefficients Equal Zero")

        # now check the frequency response
        w, H = freqz(h, 1)
        f = w/2/np.pi
        Hmag = abs(H)

        # should have a zero at 0 and pi (in this case close to zero)
        assert_((Hmag[[0, -1]] < 0.02).all(), "Zero at zero and pi")

        # check that the pass band is close to unity
        idx = np.logical_and(f > a, f < 0.5-a)
        assert_((abs(Hmag[idx] - 1) < 0.015).all(), "Pass Band Close To Unity")

    def test_compare(self):
        # test comparison to MATLAB
        k = [0.024590270518440, -0.041314581814658, -0.075943803756711,
             -0.003530911231040, 0.193140296954975, 0.373400753484939,
             0.373400753484939, 0.193140296954975, -0.003530911231040,
             -0.075943803756711, -0.041314581814658, 0.024590270518440]
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], Hz=2.)
        assert_allclose(h, k)
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
        assert_allclose(h, k)

        h = [-0.038976016082299, 0.018704846485491, -0.014644062687875,
             0.002879152556419, 0.016849978528150, -0.043276706138248,
             0.073641298245579, -0.103908158578635, 0.129770906801075,
             -0.147163447297124, 0.153302248456347, -0.147163447297124,
             0.129770906801075, -0.103908158578635, 0.073641298245579,
             -0.043276706138248, 0.016849978528150, 0.002879152556419,
             -0.014644062687875, 0.018704846485491, -0.038976016082299]
        assert_allclose(remez(21, [0, 0.8, 0.9, 1], [0, 1], Hz=2.), h)
        assert_allclose(remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.), h)


class TestFirls:

    def test_bad_args(self):
        # even numtaps
        assert_raises(ValueError, firls, 10, [0.1, 0.2], [0, 0])
        # odd bands
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.4], [0, 0, 0])
        # len(bands) != len(desired)
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.4], [0, 0, 0])
        # non-monotonic bands
        assert_raises(ValueError, firls, 11, [0.2, 0.1], [0, 0])
        assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.3], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.3, 0.4, 0.1, 0.2], [0] * 4)
        assert_raises(ValueError, firls, 11, [0.1, 0.3, 0.2, 0.4], [0] * 4)
        # negative desired
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [-1, 1])
        # len(weight) != len(pairs)
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [1, 2])
        # negative weight
        assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [-1])

    def test_firls(self):
        N = 11  # number of taps in the filter
        a = 0.1  # width of the transition band

        # design a halfband symmetric low-pass filter
        h = firls(11, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)

        # make sure the filter has correct # of taps
        assert_equal(len(h), N)

        # make sure it is symmetric
        midx = (N-1) // 2
        assert_array_almost_equal(h[:midx], h[:-midx-1:-1])

        # make sure the center tap is 0.5
        assert_almost_equal(h[midx], 0.5)

        # For halfband symmetric, odd coefficients (except the center)
        # should be zero (really small)
        hodd = np.hstack((h[1:midx:2], h[-midx+1::2]))
        assert_array_almost_equal(hodd, 0)

        # now check the frequency response
        w, H = freqz(h, 1)
        f = w/2/np.pi
        Hmag = np.abs(H)

        # check that the pass band is close to unity
        idx = np.logical_and(f > 0, f < a)
        assert_array_almost_equal(Hmag[idx], 1, decimal=3)

        # check that the stop band is close to zero
        idx = np.logical_and(f > 0.5-a, f < 0.5)
        assert_array_almost_equal(Hmag[idx], 0, decimal=3)

    def test_compare(self):
        # compare to OCTAVE output
        taps = firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], [1, 2])
        # >> taps = firls(8, [0 0.5 0.55 1], [1 1 0 0], [1, 2]);
        known_taps = [-6.26930101730182e-04, -1.03354450635036e-01,
                      -9.81576747564301e-03, 3.17271686090449e-01,
                      5.11409425599933e-01, 3.17271686090449e-01,
                      -9.81576747564301e-03, -1.03354450635036e-01,
                      -6.26930101730182e-04]
        assert_allclose(taps, known_taps)

        # compare to MATLAB output
        taps = firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], [1, 2])
        # >> taps = firls(10, [0 0.5 0.5 1], [1 1 0 0], [1, 2]);
        known_taps = [
            0.058545300496815, -0.014233383714318, -0.104688258464392,
            0.012403323025279, 0.317930861136062, 0.488047220029700,
            0.317930861136062, 0.012403323025279, -0.104688258464392,
            -0.014233383714318, 0.058545300496815]
        assert_allclose(taps, known_taps)

        # With linear changes:
        taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], fs=20)
        # >> taps = firls(6, [0, 0.1, 0.2, 0.3, 0.4, 0.5], [1, 0, 0, 1, 1, 0])
        known_taps = [
            1.156090832768218, -4.1385894727395849, 7.5288619164321826,
            -8.5530572592947856, 7.5288619164321826, -4.1385894727395849,
            1.156090832768218]
        assert_allclose(taps, known_taps)

        taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], nyq=10)
        assert_allclose(taps, known_taps)

        with pytest.raises(ValueError, match='between 0 and 1'):
            firls(7, [0, 1], [0, 1], nyq=0.5)

    def test_rank_deficient(self):
        # solve() runs but warns (only sometimes, so here we don't use match)
        x = firls(21, [0, 0.1, 0.9, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.)
        assert_allclose(np.abs(h[:2]), 1., atol=1e-5)
        assert_allclose(np.abs(h[-2:]), 0., atol=1e-6)
        # switch to pinvh (tolerances could be higher with longer
        # filters, but using shorter ones is faster computationally and
        # the idea is the same)
        x = firls(101, [0, 0.01, 0.99, 1], [1, 1, 0, 0])
        w, h = freqz(x, fs=2.)
        mask = w < 0.01
        assert mask.sum() > 3
        assert_allclose(np.abs(h[mask]), 1., atol=1e-4)
        mask = w > 0.99
        assert mask.sum() > 3
        assert_allclose(np.abs(h[mask]), 0., atol=1e-4)


class TestMinimumPhase:

    def test_bad_args(self):
        # not enough taps
        assert_raises(ValueError, minimum_phase, [1.])
        assert_raises(ValueError, minimum_phase, [1., 1.])
        assert_raises(ValueError, minimum_phase, np.full(10, 1j))
        assert_raises(ValueError, minimum_phase, 'foo')
        assert_raises(ValueError, minimum_phase, np.ones(10), n_fft=8)
        assert_raises(ValueError, minimum_phase, np.ones(10), method='foo')
        assert_warns(RuntimeWarning, minimum_phase, np.arange(3))

    def test_homomorphic(self):
        # check that it can recover frequency responses of arbitrary
        # linear-phase filters

        # for some cases we can get the actual filter back
        h = [1, -1]
        h_new = minimum_phase(np.convolve(h, h[::-1]))
        assert_allclose(h_new, h, rtol=0.05)

        # but in general we only guarantee we get the magnitude back
        rng = np.random.RandomState(0)
        for n in (2, 3, 10, 11, 15, 16, 17, 20, 21, 100, 101):
            h = rng.randn(n)
            h_new = minimum_phase(np.convolve(h, h[::-1]))
            assert_allclose(np.abs(fft(h_new)),
                            np.abs(fft(h)), rtol=1e-4)

    def test_hilbert(self):
        # compare to MATLAB output of reference implementation

        # f=[0 0.3 0.5 1];
        # a=[1 1 0 0];
        # h=remez(11,f,a);
        h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
        k = [0.349585548646686, 0.373552164395447, 0.326082685363438,
             0.077152207480935, -0.129943946349364, -0.059355880509749]
        m = minimum_phase(h, 'hilbert')
        assert_allclose(m, k, rtol=5e-3)

        # f=[0 0.8 0.9 1];
        # a=[0 0 1 1];
        # h=remez(20,f,a);
        h = remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.)
        k = [0.232486803906329, -0.133551833687071, 0.151871456867244,
             -0.157957283165866, 0.151739294892963, -0.129293146705090,
             0.100787844523204, -0.065832656741252, 0.035361328741024,
             -0.014977068692269, -0.158416139047557]
        m = minimum_phase(h, 'hilbert', n_fft=2**19)
        assert_allclose(m, k, rtol=2e-3)
