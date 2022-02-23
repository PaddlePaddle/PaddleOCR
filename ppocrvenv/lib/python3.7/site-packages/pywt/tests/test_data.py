import os
import numpy as np
from numpy.testing import assert_allclose, assert_raises, assert_

import pywt.data

data_dir = os.path.join(os.path.dirname(__file__), 'data')
wavelab_data_file = os.path.join(data_dir, 'wavelab_test_signals.npz')
wavelab_result_dict = np.load(wavelab_data_file)


def test_data_aero():
    aero = pywt.data.aero()

    ref = np.array([[178, 178, 179],
                    [170, 173, 171],
                    [185, 174, 171]])

    assert_allclose(aero[:3, :3], ref)


def test_data_ascent():
    ascent = pywt.data.ascent()

    ref = np.array([[83, 83, 83],
                    [82, 82, 83],
                    [80, 81, 83]])

    assert_allclose(ascent[:3, :3], ref)


def test_data_camera():
    camera = pywt.data.camera()

    ref = np.array([[200, 200, 200],
                    [200, 199, 199],
                    [199, 199, 199]])

    assert_allclose(camera[:3, :3], ref)


def test_data_ecg():
    ecg = pywt.data.ecg()

    ref = np.array([-86, -87, -87])

    assert_allclose(ecg[:3], ref)


def test_wavelab_signals():
    """Comparison with results generated using WaveLab"""
    rtol = atol = 1e-12

    # get a list of the available signals
    available_signals = pywt.data.demo_signal('list')
    assert_('Doppler' in available_signals)

    for signal in available_signals:
        # reference dictionary has lowercase names for the keys
        key = signal.replace('-', '_').lower()
        val = wavelab_result_dict[key]
        if key in ['gabor', 'sineoneoverx']:
            # these functions do not allow a size to be provided
            assert_allclose(val, pywt.data.demo_signal(signal),
                            rtol=rtol, atol=atol)
            assert_raises(ValueError, pywt.data.demo_signal, key, val.size)
        else:
            assert_allclose(val, pywt.data.demo_signal(signal, val.size),
                            rtol=rtol, atol=atol)
            # these functions require a size to be provided
            assert_raises(ValueError, pywt.data.demo_signal, key)

    # ValueError on unrecognized signal type
    assert_raises(ValueError, pywt.data.demo_signal, 'unknown_signal', 512)

    # ValueError on invalid length
    assert_raises(ValueError, pywt.data.demo_signal, 'Doppler', 0)
