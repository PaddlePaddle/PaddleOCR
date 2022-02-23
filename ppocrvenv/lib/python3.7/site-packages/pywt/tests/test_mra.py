#!/usr/bin/env python

import numpy as np
import pytest
from numpy.testing import assert_allclose

import pywt
from pywt import data

# tolerances used in accuracy comparisons
tol_single = 1e-6
tol_double = 1e-13
atol = 1e-7


####
# 1d mra tests
####

@pytest.mark.parametrize('wavelet', ['db2', 'sym4', 'coif5'])
@pytest.mark.parametrize('transform', ['dwt', 'swt'])
@pytest.mark.parametrize('mode', pywt.Modes.modes)
@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'complex64', 'complex128']
)
def test_mra_roundtrip(wavelet, transform, mode, dtype):
    x = data.ecg()[:64].astype(dtype)
    if x.dtype.kind == 'c':
        # fill some data for the imaginary channel
        x.imag = x[::-1].real

    if transform == 'swt':
        # swt mode only supports periodization
        if mode != 'periodization':
            with pytest.raises(ValueError):
                pywt.mra(x, wavelet, transform=transform, mode=mode)
            return

    coeffs = pywt.mra(x, wavelet, transform=transform, mode=mode)
    assert isinstance(coeffs, list)
    assert isinstance(coeffs[0], np.ndarray)
    # assert all(isinstance(coeffs[i], dict) for i in range(1, len(coeffs)))

    y = pywt.imra(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


@pytest.mark.parametrize('wavelet', ['rbio1.3', 'bior2.4'])
@pytest.mark.parametrize('transform', ['dwt', 'swt'])
def test_mra_warns_on_non_orthogonal(wavelet, transform):
    dtype = np.float64
    x = data.ecg()[:64].astype(dtype)

    assert not pywt.Wavelet(wavelet).orthogonal

    if transform == 'swt':
        # bi-orthogonal wavelets raise a warning for SWT case
        msg = 'norm=True, but the wavelet is not orthogonal'
        with pytest.warns(UserWarning, match=msg):
            coeffs = pywt.mra(x, wavelet, transform=transform)
    else:
        coeffs = pywt.mra(x, wavelet, transform=transform)

    y = pywt.imra(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


@pytest.mark.parametrize('axis', [0, -1, 1, 2, -3])
@pytest.mark.parametrize('ndim', [1, 2, 3])
@pytest.mark.parametrize('transform', ['dwt', 'swt'])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_mra_axis(transform, ndim, axis, dtype):
    # Test transforms over a specific axis of 1D, 2D or 3D data
    if ndim == 1:
        x = data.ecg()[:64]
    elif ndim == 2:
        x = data.camera()[:64, :32]
    elif ndim == 3:
        x = data.camera()[:48, :8]
        x = np.stack((x,) * 8, axis=-1)
    x = x.astype(dtype, copy=False)

    # out of range axis
    if axis < -x.ndim or axis >= x.ndim:
        with pytest.raises(np.AxisError):
            pywt.mra(x, 'db1', transform=transform, axis=axis)
        return

    coeffs = pywt.mra(x, 'db1', transform=transform, axis=axis)
    y = pywt.imra(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


####
# 2d mra tests
####

@pytest.mark.parametrize('wavelet', ['db2', 'sym4', 'coif5'])
@pytest.mark.parametrize('transform', ['dwt2', 'swt2'])
@pytest.mark.parametrize('mode', pywt.Modes.modes)
@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'complex64', 'complex128']
)
def test_mra2_roundtrip(wavelet, transform, mode, dtype):
    x = data.camera()[:32, :16].astype(dtype, copy=False)
    if x.dtype.kind == 'c':
        # fill some data for the imaginary channel
        x.imag = x[::-1, :].real

    if transform == 'swt2':
        # swt mode only supports periodization
        if mode != 'periodization':
            with pytest.raises(ValueError):
                pywt.mra2(x, wavelet, transform=transform, mode=mode)
            return

    coeffs = pywt.mra2(x, wavelet, transform=transform, mode=mode)
    assert isinstance(coeffs, list)
    assert isinstance(coeffs[0], np.ndarray)
    # assert all(isinstance(coeffs[i], dict) for i in range(1, len(coeffs)))

    y = pywt.imra2(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


@pytest.mark.parametrize('wavelet', ['rbio1.3', 'bior2.4'])
@pytest.mark.parametrize('transform', ['dwt2', 'swt2'])
def test_mra2_warns_on_non_orthogonal(wavelet, transform):
    dtype = np.float64
    x = data.camera()[:32, :8].astype(dtype, copy=False)

    assert not pywt.Wavelet(wavelet).orthogonal

    if transform == 'swt2':
        # bi-orthogonal wavelets raise a warning for SWT case
        msg = 'norm=True, but the wavelets used are not orthogonal'
        with pytest.warns(UserWarning, match=msg):
            coeffs = pywt.mra2(x, wavelet, transform=transform)
    else:
        coeffs = pywt.mra2(x, wavelet, transform=transform)

    y = pywt.imra2(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


@pytest.mark.parametrize('transform', ['dwt2', 'swt2'])
@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('axes', [(0, 1), (-2, -1), (0, 2), (-3, 1), (0, 4)])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
def test_mra2_axes(transform, axes, ndim, dtype):
    # Test transforms over various axes of 2D or 3D data.
    x = data.camera()[:32, :16].astype(dtype, copy=False)
    if ndim == 3:
        x = np.stack((x,) * 8, axis=-1)

    # out of range axis
    if any([axis < -x.ndim or axis >= x.ndim for axis in axes]):
        with pytest.raises(np.AxisError):
            pywt.mra2(x, 'db1', transform=transform, axes=axes)
        return

    coeffs = pywt.mra2(x, 'db1', transform=transform, axes=axes)
    y = pywt.imra2(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


####
# nd mra tests
####

@pytest.mark.parametrize('wavelet', ['sym2', ])
@pytest.mark.parametrize('transform', ['dwtn', 'swtn'])
@pytest.mark.parametrize('mode', pywt.Modes.modes)
@pytest.mark.parametrize(
    'dtype', ['float32', 'float64', 'complex64', 'complex128']
)
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_mran_roundtrip(wavelet, transform, mode, dtype, ndim):
    if ndim == 1:
        x = data.ecg()[:48].astype(dtype, copy=False)
    elif ndim == 2:
        x = data.camera()[:16, :8].astype(dtype, copy=False)
    elif ndim == 3:
        x = data.camera()[:16, :8].astype(dtype, copy=False)
        x = np.stack((x,) * 8, axis=-1)

    if x.dtype.kind == 'c':
        # fill some data for the imaginary channel
        x.imag = x[::-1, ...].real

    if transform == 'swtn':
        # swt mode only supports periodization
        if mode != 'periodization':
            with pytest.raises(ValueError):
                pywt.mran(x, wavelet, transform=transform, mode=mode)
            return

    coeffs = pywt.mran(x, wavelet, transform=transform, mode=mode)
    assert isinstance(coeffs, list)
    assert isinstance(coeffs[0], np.ndarray)
    # assert all(isinstance(coeffs[i], dict) for i in range(1, len(coeffs)))

    y = pywt.imran(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


@pytest.mark.parametrize('wavelet', ['rbio1.3', 'bior2.4'])
@pytest.mark.parametrize('transform', ['dwtn', 'swtn'])
def test_mran_warns_on_non_orthogonal(wavelet, transform):
    dtype = np.float64
    x = data.camera()[:32, :8].astype(dtype, copy=False)

    assert not pywt.Wavelet(wavelet).orthogonal

    if transform == 'swtn':
        # bi-orthogonal wavelets raise a warning for SWT case
        msg = 'norm=True, but the wavelets used are not orthogonal'
        with pytest.warns(UserWarning, match=msg):
            coeffs = pywt.mran(x, wavelet, transform=transform)
    else:
        coeffs = pywt.mran(x, wavelet, transform=transform)

    y = pywt.imran(coeffs)
    rtol = tol_single if x.real.dtype.kind == 'f' else tol_double
    assert_allclose(x, y, rtol=rtol, atol=rtol)


@pytest.mark.parametrize(
    'axes', [(0, 1), (-2, -1), (0, 2), (-3, 1), (0, 4), (-3, -2, -1),
             (0, 2, 1), (0, 5, 1), (0,), (1,), (2,), (-2,),  (-3,), (-4,)])
@pytest.mark.parametrize('transform', ['dwtn', 'swtn'])
def test_mran_axes(axes, transform):
    # Test with transforms over 1, 2 or 3 axes of 3d data.
    # Cases with out of range axes are also tested
    dtype = np.float64
    x = data.camera()[:32, :16].astype(dtype, copy=False)
    x3d = np.stack((x,) * 8, axis=-1)

    # out of range axis
    if any([axis < -x.ndim or axis >= x.ndim for axis in axes]):
        with pytest.raises(np.AxisError):
            pywt.mran(x, 'db1', transform='dwtn', axes=axes)
        return

    coeffs = pywt.mran(x3d, 'db1', transform='dwtn', axes=axes)
    y = pywt.imran(coeffs)
    rtol = tol_single if x3d.real.dtype.kind == 'f' else tol_double
    assert_allclose(x3d, y, rtol=rtol, atol=rtol)
