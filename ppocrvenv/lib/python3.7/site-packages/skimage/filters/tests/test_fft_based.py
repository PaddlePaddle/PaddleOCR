import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule

from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth


def _fft_centered(x):
    return fftmodule.fftshift(fftmodule.fftn(fftmodule.fftshift(x)))


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64,
                                   np.uint8, np.int32])
def test_butterworth_2D_zeros_dtypes(dtype):
    im = np.zeros((4, 4), dtype=dtype)
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert filtered.dtype == _supported_float_type(dtype)
    assert_array_equal(im, filtered)


@pytest.mark.parametrize("high_pass", [True, False])
def test_butterworth_2D(high_pass):
    # rough check of high-pass vs. low-pass behavior via relative energy
    im = np.random.randn(64, 128)
    filtered = butterworth(
        im,
        cutoff_frequency_ratio=0.20,
        high_pass=high_pass,
    )

    # Compute the energy at the outer edges of the Fourier domain
    # before and after filtering.
    im_fft = _fft_centered(im)
    im_fft = np.real(im_fft * np.conj(im_fft))
    filtered_fft = _fft_centered(filtered)
    filtered_fft = np.real(filtered_fft * np.conj(filtered_fft))
    outer_mask = np.ones(im.shape, dtype=bool)
    outer_mask[4:-4, 4:-4] = 0
    abs_filt_outer = filtered_fft[outer_mask].mean()
    abs_im_outer = im_fft[outer_mask].mean()

    # Compute energy near the center of the Fourier domain
    inner_sl = tuple(slice(s // 2 - 4, s // 2 + 4) for s in im.shape)
    abs_filt_inner = filtered_fft[inner_sl].mean()
    abs_im_inner = im_fft[inner_sl].mean()

    if high_pass:
        assert abs_filt_outer > 0.9 * abs_im_outer
        assert abs_filt_inner < 0.1 * abs_im_inner
    else:
        assert abs_filt_outer < 0.1 * abs_im_outer
        assert abs_filt_inner > 0.9 * abs_im_inner


@pytest.mark.parametrize("high_pass", [True, False])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_butterworth_2D_realfft(high_pass, dtype):
    """Filtering a real-valued array is equivalent to filtering a
       complex-valued array where the imaginary part is zero.
    """
    im = np.random.randn(32, 64).astype(dtype)
    kwargs = dict(
        cutoff_frequency_ratio=0.20,
        high_pass=high_pass
    )

    expected_dtype = _supported_float_type(im.dtype)
    filtered_real = butterworth(im, **kwargs)
    assert filtered_real.dtype == expected_dtype

    cplx_dtype = np.promote_types(im.dtype, np.complex64)
    filtered_cplx = butterworth(im.astype(cplx_dtype), **kwargs)
    assert filtered_cplx.real.dtype == expected_dtype

    if expected_dtype == np.float64:
        rtol = atol = 1e-13
    else:
        rtol = atol = 1e-5
    assert_allclose(filtered_real, filtered_cplx.real, rtol=rtol, atol=atol)


def test_butterworth_3D_zeros():
    im = np.zeros((3, 4, 5))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert_array_equal(im, filtered)


def test_butterworth_4D_zeros():
    im = np.zeros((3, 4, 5, 6))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert_array_equal(im, filtered)


@pytest.mark.parametrize(
    "chan, dtype",
    [(0, np.float64), (1, np.complex128), (2, np.uint8), (3, np.int64)],
)
def test_butterworth_4D_channel(chan, dtype):
    im = np.zeros((3, 4, 5, 6), dtype=dtype)
    filtered = butterworth(im, channel_axis=chan)
    assert filtered.shape == im.shape


def test_butterworth_correctness_bw():
    small = coins()[180:190, 260:270]
    filtered = butterworth(small,
                           cutoff_frequency_ratio=0.2)
    correct = np.array(
        [
            [ 28.63019362, -17.69023786,  26.95346957,  20.57423019, -15.1933463 , -28.05828136, -35.25135674, -25.70376951, -43.37121955, -16.87688457],
            [  4.62077869,  36.5726672 ,  28.41926375, -22.86436829, -25.32375274, -19.94182623,  -2.9666164 ,   6.62250413,   3.55910886, -33.15358921],
            [ 25.00377084,  34.2948942 , -15.13862785, -15.34354183, -12.68722526,  12.82729905,   5.21622357,  11.41087761,  16.33690526, -50.39790969],
            [ 72.62331496, -14.7924709 , -22.14868895,  -7.47854864,   9.66784721,  24.37625693,  12.5479457 ,  -1.38194367,   2.40079497, -26.61141413],
            [ 21.85962078, -56.73932031, -14.82425429,   4.10524297, -19.16561768, -48.19021687,   5.0258744 ,  28.82432166,   0.66992097,   9.8378842 ],
            [-54.93417679,  -5.12037233,  19.2956981 ,  38.56431593,  27.95408908,  -3.53103389,  23.75329532,  -6.92615359,  -8.50823024,   7.05743093],
            [-30.51016624,  -9.99691211,  -7.1080672 ,  23.67643315,   1.61919726,  12.94103905, -29.08141699, -11.56937511,  22.70988847,  32.04063285],
            [ -7.51780937, -30.27899181,  -2.57782655,  -1.58947887,  -2.13564576, -11.34039302,   1.59165041,  14.39173421, -14.15148821,  -2.21664717],
            [ 14.81167298,  -3.75274782,  18.41459894,  15.80334075, -19.7477109 ,  -3.68619619,  -2.9513036 , -10.17325791,  18.32438702,  18.68003971],
            [-50.53430811,  12.14152989,  17.69341877,   9.1858496 ,  12.1470914 ,   1.45865179,  61.08961357,  29.76775029, -11.04603619,  24.18621404],
        ]
    )
    assert_allclose(filtered, correct)


def test_butterworth_correctness_rgb():
    small = astronaut()[135:145, 205:215]
    filtered = butterworth(small,
                           cutoff_frequency_ratio=0.3,
                           high_pass=True,
                           channel_axis=-1)
    correct = np.array([
        [
            [-5.30292781e-01,  2.17985072e+00,  2.86622486e+00],
            [ 6.39360740e+00,  9.30643715e+00,  8.61226660e+00],
            [ 5.47978436e+00,  1.03641402e+01,  1.02940329e+01],
            [ 8.88312002e+00,  7.29681652e+00,  8.16021235e+00],
            [ 4.67693778e+00,  6.33135145e-01, -2.51296407e+00],
            [ 3.21039522e+00, -9.14893931e-01, -2.11543661e+00],
            [ 4.61985125e-02, -6.10682453e+00, -1.72837650e+00],
            [-4.59492989e+00, -7.35887525e+00, -1.03532871e+01],
            [-3.67859542e+00, -4.36371621e+00, -5.67371459e+00],
            [-4.38264080e+00, -6.08362280e+00, -9.20394882e+00],
        ],

        [
            [-2.46191390e+00, -2.10996960e+00, -1.41287606e+00],
            [ 4.87042304e-01,  4.70686760e-01,  2.90817746e+00],
            [ 9.33095004e-01, -2.11867564e-01,  3.10917925e+00],
            [-2.35660768e+00, -1.35043153e+00, -2.67062162e+00],
            [-1.22363424e+00,  1.11155488e-01,  1.25392954e+00],
            [-1.05667680e+00,  1.58195605e-01,  6.11873557e-01],
            [-4.12128910e+00, -3.55994486e+00, -8.75303054e+00],
            [ 2.47171790e+00,  2.70762582e+00,  5.69543552e+00],
            [ 6.97042504e-01, -2.24173305e+00,  3.26477871e-01],
            [ 5.00195333e-01, -2.66024743e+00, -1.87479563e+00],
        ],

        [
            [-4.40136260e+00, -4.02254309e+00, -4.89246563e+00],
            [-4.64563864e+00, -6.21442755e+00, -9.31399553e+00],
            [-2.11532959e+00, -2.58844609e+00, -4.20629743e+00],
            [-3.40862389e+00, -3.29511853e+00, -4.78220207e+00],
            [-8.06768327e-01, -4.01651211e+00, -2.84783939e+00],
            [ 1.72379068e+00,  1.00930709e+00,  2.57575911e+00],
            [-2.13771052e+00, -1.75564076e+00, -2.88676819e+00],
            [-2.72015191e-01, -1.61610409e-01,  2.15906305e+00],
            [-3.80356741e+00, -7.30201675e-01, -3.79800352e+00],
            [ 1.43534281e-01, -2.95121861e+00, -2.67070135e+00],
        ],

        [
            [ 1.03480271e+00,  6.34545011e+00,  3.53610283e+00],
            [ 7.44740677e+00,  9.97350707e+00,  1.25152734e+01],
            [ 3.10493189e+00,  5.15553793e+00,  6.48354940e+00],
            [-7.89260096e-01,  3.04573015e-01, -1.43829810e+00],
            [-1.46298411e+00,  1.23095495e+00, -1.33983509e+00],
            [ 2.82986807e+00,  2.80546223e+00,  6.39492794e+00],
            [-9.15293187e-01,  2.88688464e+00, -9.69417480e-01],
            [ 4.50217964e+00,  2.90410068e+00,  5.39107589e+00],
            [-5.71608069e-01,  1.78198962e+00, -3.72062011e-01],
            [ 7.43870617e+00,  8.78780364e+00,  9.91142612e+00]],

        [
            [-1.01243616e+01, -1.46725955e+01, -1.63691866e+01],
            [ 1.06512445e+01,  7.84699418e+00,  1.05428678e+01],
            [ 4.75343829e+00,  6.11329861e+00,  2.81633365e+00],
            [ 7.78936796e+00,  9.63684277e+00,  1.21495065e+01],
            [ 5.19238043e+00,  5.38165743e+00,  8.03025884e+00],
            [ 1.67424214e+00,  2.25530135e+00,  2.44161390e-01],
            [-3.18012002e-01,  1.99405335e+00, -4.33960644e-01],
            [-1.21700957e+00, -2.65973900e+00, -6.31515766e-01],
            [-4.87805104e+00, -5.55289609e+00, -8.50052504e+00],
            [ 1.43493808e+01,  1.77252074e+01,  1.92810954e+01],
        ],

        [
            [-4.21712178e+01, -4.47409535e+01, -4.16375264e+01],
            [ 1.18585381e+00,  1.33732681e+00, -1.45927283e+00],
            [-4.83275742e+00, -7.14344851e+00, -7.59812923e+00],
            [-7.13716513e+00, -1.10025632e+01, -1.16111397e+01],
            [-5.00275373e+00, -4.20410732e+00, -4.93030043e+00],
            [ 1.98421851e+00,  2.68393141e+00,  3.14898078e+00],
            [ 1.97471502e+00, -2.11937555e+00,  2.04674150e+00],
            [ 3.42916035e+00,  4.98808524e+00,  6.74436447e+00],
            [-3.29035900e-01, -9.88239773e-01,  2.38909382e-02],
            [ 2.58646940e+01,  2.40294324e+01,  2.64535438e+01],
        ],

        [
            [-2.72408341e+01, -2.51637965e+01, -2.95566971e+01],
            [ 4.83667855e+00,  5.63749968e+00,  6.97366639e+00],
            [ 6.18182884e+00,  2.89519333e+00,  6.15697112e+00],
            [-1.03326540e+00, -4.04702873e+00, -5.50872246e+00],
            [-6.92401355e+00, -8.99374166e+00, -9.43201766e+00],
            [-7.24967366e+00, -1.16225741e+01, -1.26385982e+01],
            [-7.79470220e+00, -9.36143025e+00, -7.13686778e+00],
            [-1.22393834e+01, -1.24094588e+01, -1.70498337e+01],
            [-4.65034860e+00, -3.93071471e+00, -4.65788605e+00],
            [ 2.47461715e+01,  2.46389560e+01,  3.11843915e+01],
        ],
        [
            [ 1.22510987e+00, -3.30423292e+00, -1.02785428e+01],
            [-2.41285934e+01, -1.60883211e+01, -1.97064909e+01],
            [ 9.77242420e+00,  1.80847757e+01,  2.01088714e+01],
            [ 1.21335913e+01,  1.16812821e+01,  1.20531919e+01],
            [-1.64052514e+00, -4.45404672e+00, -3.83103216e+00],
            [ 9.97519545e-01, -6.32678881e+00, -4.89843180e-01],
            [ 1.07464325e+01,  2.62880708e+01,  1.74691665e+01],
            [ 2.31869805e+00,  1.91935135e+00, -9.65612833e-01],
            [-9.23026361e+00, -1.75138706e+01, -1.39243019e+01],
            [ 4.60784836e+00,  5.60845273e+00,  5.28255564e+00],
        ],

        [
            [ 9.68795318e+00,  2.78851276e+00,  5.45620353e+00],
            [-1.02076906e+01, -1.47926224e+01, -1.43257049e+01],
            [-2.17025353e+00,  6.23446752e+00,  5.21771748e+00],
            [ 1.57074742e+01,  2.17163634e+01,  2.55600809e+01],
            [ 7.03884578e+00,  1.29273058e+01,  7.50960315e+00],
            [-6.69896692e+00, -1.83433042e+01, -1.60702492e+01],
            [ 7.44877725e+00,  1.28971365e+01,  1.10234666e+01],
            [ 5.25663607e+00,  9.80648891e+00,  1.22955858e+01],
            [-7.44903684e+00, -1.92670342e+01, -1.68232131e+01],
            [ 1.25609220e+01,  2.09808909e+01,  2.21425299e+01],
        ],
        [
            [ 2.87825597e+00,  3.37945227e+00,  3.05777360e+00],
            [ 1.18858884e+00, -5.27430874e+00, -6.96009863e+00],
            [-7.55910235e+00, -2.12068126e+01, -2.06925790e+01],
            [-1.47217788e+01, -1.45626702e+01, -1.56493571e+01],
            [-5.60886203e+00,  8.81908697e-01,  5.47367282e+00],
            [-1.00478644e+01, -8.01471176e+00, -7.45670458e+00],
            [ 3.61521638e+00,  8.99194959e+00,  4.93826323e+00],
            [ 7.87025438e+00,  1.34804191e+01,  1.96899695e+01],
            [-5.50012037e+00, -6.40490471e+00, -1.17265188e+01],
            [ 6.17010624e+00,  1.56199152e+01,  1.79889524e+01],
        ],
    ])
    assert_allclose(filtered, correct)
