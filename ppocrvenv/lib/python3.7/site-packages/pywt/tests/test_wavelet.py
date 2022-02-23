#!/usr/bin/env python
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose, assert_

import pywt


def test_wavelet_properties():
    w = pywt.Wavelet('db3')

    # Name
    assert_(w.name == 'db3')
    assert_(w.short_family_name == 'db')
    assert_(w.family_name, 'Daubechies')

    # String representation
    fields = ('Family name', 'Short name', 'Filters length', 'Orthogonal',
              'Biorthogonal', 'Symmetry')
    for field in fields:
        assert_(field in str(w))

    # Filter coefficients
    dec_lo = [0.03522629188210, -0.08544127388224, -0.13501102001039,
              0.45987750211933, 0.80689150931334, 0.33267055295096]
    dec_hi = [-0.33267055295096, 0.80689150931334, -0.45987750211933,
              -0.13501102001039, 0.08544127388224, 0.03522629188210]
    rec_lo = [0.33267055295096, 0.80689150931334, 0.45987750211933,
              -0.13501102001039, -0.08544127388224, 0.03522629188210]
    rec_hi = [0.03522629188210, 0.08544127388224, -0.13501102001039,
              -0.45987750211933, 0.80689150931334, -0.33267055295096]
    assert_allclose(w.dec_lo, dec_lo)
    assert_allclose(w.dec_hi, dec_hi)
    assert_allclose(w.rec_lo, rec_lo)
    assert_allclose(w.rec_hi, rec_hi)

    assert_(len(w.filter_bank) == 4)

    # Orthogonality
    assert_(w.orthogonal)
    assert_(w.biorthogonal)

    # Symmetry
    assert_(w.symmetry)

    # Vanishing moments
    assert_(w.vanishing_moments_phi == 0)
    assert_(w.vanishing_moments_psi == 3)


def test_wavelet_coefficients():
    families = ('db', 'sym', 'coif', 'bior', 'rbio')
    wavelets = sum([pywt.wavelist(name) for name in families], [])
    for wavelet in wavelets:
        if (pywt.Wavelet(wavelet).orthogonal):
            check_coefficients_orthogonal(wavelet)
        elif(pywt.Wavelet(wavelet).biorthogonal):
            check_coefficients_biorthogonal(wavelet)
        else:
            check_coefficients(wavelet)


def check_coefficients_orthogonal(wavelet):

    epsilon = 5e-11
    level = 5
    w = pywt.Wavelet(wavelet)
    phi, psi, x = w.wavefun(level=level)

    # Lowpass filter coefficients sum to sqrt2
    res = np.sum(w.dec_lo)-np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # sum even coef = sum odd coef = 1 / sqrt(2)
    res = np.sum(w.dec_lo[::2])-1./np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)

    res = np.sum(w.dec_lo[1::2])-1./np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # Highpass filter coefficients sum to zero
    res = np.sum(w.dec_hi)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # Scaling function integrates to unity

    res = np.sum(phi) - 2**level
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # Wavelet function is orthogonal to the scaling function at the same scale
    res = np.sum(phi*psi)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # The lowpass and highpass filter coefficients are orthogonal
    res = np.sum(np.array(w.dec_lo)*np.array(w.dec_hi))
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)


def check_coefficients_biorthogonal(wavelet):

    epsilon = 5e-11
    level = 5
    w = pywt.Wavelet(wavelet)
    phi_d, psi_d, phi_r, psi_r, x = w.wavefun(level=level)

    # Lowpass filter coefficients sum to sqrt2
    res = np.sum(w.dec_lo)-np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # sum even coef = sum odd coef = 1 / sqrt(2)
    res = np.sum(w.dec_lo[::2])-1./np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    res = np.sum(w.dec_lo[1::2])-1./np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # Highpass filter coefficients sum to zero
    res = np.sum(w.dec_hi)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # Scaling function integrates to unity
    res = np.sum(phi_d) - 2**level
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    res = np.sum(phi_r) - 2**level
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)


def check_coefficients(wavelet):
    epsilon = 5e-11
    level = 10
    w = pywt.Wavelet(wavelet)
    # Lowpass filter coefficients sum to sqrt2
    res = np.sum(w.dec_lo)-np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # sum even coef = sum odd coef = 1 / sqrt(2)
    res = np.sum(w.dec_lo[::2])-1./np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)

    res = np.sum(w.dec_lo[1::2])-1./np.sqrt(2)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)
    # Highpass filter coefficients sum to zero
    res = np.sum(w.dec_hi)
    msg = ('[RMS_REC > EPSILON] for Wavelet: %s, rms=%.3g' % (wavelet, res))
    assert_(res < epsilon, msg=msg)


class _CustomHaarFilterBank(object):
    @property
    def filter_bank(self):
        val = np.sqrt(2) / 2
        return ([val]*2, [-val, val], [val]*2, [val, -val])


def test_custom_wavelet():
    haar_custom1 = pywt.Wavelet('Custom Haar Wavelet',
                                filter_bank=_CustomHaarFilterBank())
    haar_custom1.orthogonal = True
    haar_custom1.biorthogonal = True

    val = np.sqrt(2) / 2
    filter_bank = ([val]*2, [-val, val], [val]*2, [val, -val])
    haar_custom2 = pywt.Wavelet('Custom Haar Wavelet',
                                filter_bank=filter_bank)

    # check expected default wavelet properties
    assert_(~haar_custom2.orthogonal)
    assert_(~haar_custom2.biorthogonal)
    assert_(haar_custom2.symmetry == 'unknown')
    assert_(haar_custom2.family_name == '')
    assert_(haar_custom2.short_family_name == '')
    assert_(haar_custom2.vanishing_moments_phi == 0)
    assert_(haar_custom2.vanishing_moments_psi == 0)

    # Some properties can be set by the user
    haar_custom2.orthogonal = True
    haar_custom2.biorthogonal = True


def test_wavefun_sym3():
    w = pywt.Wavelet('sym3')
    # sym3 is an orthogonal wavelet, so 3 outputs from wavefun
    phi, psi, x = w.wavefun(level=3)
    assert_(phi.size == 41)
    assert_(psi.size == 41)
    assert_(x.size == 41)

    assert_allclose(x, np.linspace(0, 5, num=x.size))
    phi_expect = np.array([0.00000000e+00, 1.04132926e-01, 2.52574126e-01,
                           3.96525521e-01, 5.70356539e-01, 7.18934305e-01,
                           8.70293448e-01, 1.05363620e+00, 1.24921722e+00,
                           1.15296888e+00, 9.41669683e-01, 7.55875887e-01,
                           4.96118565e-01, 3.28293151e-01, 1.67624969e-01,
                           -7.33690312e-02, -3.35452855e-01, -3.31221131e-01,
                           -2.32061503e-01, -1.66854239e-01, -4.34091324e-02,
                           -2.86152390e-02, -3.63563035e-02, 2.06034491e-02,
                           8.30280254e-02, 7.17779073e-02, 3.85914311e-02,
                           1.47527100e-02, -2.31896077e-02, -1.86122172e-02,
                           -1.56211329e-03, -8.70615088e-04, 3.20760857e-03,
                           2.34142153e-03, -7.73737194e-04, -2.99879354e-04,
                           1.23636238e-04, 0.00000000e+00, 0.00000000e+00,
                           0.00000000e+00, 0.00000000e+00])

    psi_expect = np.array([0.00000000e+00, 1.10265752e-02, 2.67449277e-02,
                           4.19878574e-02, 6.03947231e-02, 7.61275365e-02,
                           9.21548684e-02, 1.11568926e-01, 1.32278887e-01,
                           6.45829680e-02, -3.97635130e-02, -1.38929884e-01,
                           -2.62428322e-01, -3.62246804e-01, -4.62843343e-01,
                           -5.89607507e-01, -7.25363076e-01, -3.36865858e-01,
                           2.67715108e-01, 8.40176767e-01, 1.55574430e+00,
                           1.18688954e+00, 4.20276324e-01, -1.51697311e-01,
                           -9.42076108e-01, -7.93172332e-01, -3.26343710e-01,
                           -1.24552779e-01, 2.12909254e-01, 1.75770320e-01,
                           1.47523075e-02, 8.22192707e-03, -3.02920592e-02,
                           -2.21119497e-02, 7.30703025e-03, 2.83200488e-03,
                           -1.16759765e-03, 0.00000000e+00, 0.00000000e+00,
                           0.00000000e+00, 0.00000000e+00])

    assert_allclose(phi, phi_expect)
    assert_allclose(psi, psi_expect)


def test_wavefun_bior13():
    w = pywt.Wavelet('bior1.3')
    # bior1.3 is not an orthogonal wavelet, so 5 outputs from wavefun
    phi_d, psi_d, phi_r, psi_r, x = w.wavefun(level=3)
    for arr in [phi_d, psi_d, phi_r, psi_r]:
        assert_(arr.size == 40)

    phi_d_expect = np.array([0., -0.00195313, 0.00195313, 0.01757813,
                             0.01367188, 0.00390625, -0.03515625, -0.12890625,
                             -0.15234375, -0.125, -0.09375, -0.0625, 0.03125,
                             0.15234375, 0.37890625, 0.78515625, 0.99609375,
                             1.08203125, 1.13671875, 1.13671875, 1.08203125,
                             0.99609375, 0.78515625, 0.37890625, 0.15234375,
                             0.03125, -0.0625, -0.09375, -0.125, -0.15234375,
                             -0.12890625, -0.03515625, 0.00390625, 0.01367188,
                             0.01757813, 0.00195313, -0.00195313, 0., 0., 0.])
    phi_r_expect = np.zeros(x.size, dtype=np.float64)
    phi_r_expect[15:23] = 1

    psi_d_expect = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0.015625, -0.015625, -0.140625, -0.109375,
                             -0.03125, 0.28125, 1.03125, 1.21875, 1.125, 0.625,
                             -0.625, -1.125, -1.21875, -1.03125, -0.28125,
                             0.03125, 0.109375, 0.140625, 0.015625, -0.015625,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    psi_r_expect = np.zeros(x.size, dtype=np.float64)
    psi_r_expect[7:15] = -0.125
    psi_r_expect[15:19] = 1
    psi_r_expect[19:23] = -1
    psi_r_expect[23:31] = 0.125

    assert_allclose(x, np.linspace(0, 5, x.size, endpoint=False))
    assert_allclose(phi_d, phi_d_expect, rtol=1e-5, atol=1e-9)
    assert_allclose(phi_r, phi_r_expect, rtol=1e-10, atol=1e-12)
    assert_allclose(psi_d, psi_d_expect, rtol=1e-10, atol=1e-12)
    assert_allclose(psi_r, psi_r_expect, rtol=1e-10, atol=1e-12)


def test_wavelet_pickle(tmpdir):
    wavelet = pywt.Wavelet('sym4')
    filename = os.path.join(tmpdir, 'wav.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(wavelet, f)
    with open(filename, 'rb') as f:
        wavelet2 = pickle.load(f)
    assert isinstance(wavelet2, pywt.Wavelet)
    assert wavelet2.name == wavelet.name
