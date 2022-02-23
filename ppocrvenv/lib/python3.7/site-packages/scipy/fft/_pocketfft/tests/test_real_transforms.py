from os.path import join, dirname
from typing import Callable, Dict, Tuple, Union, Type

import numpy as np
from numpy.testing import (
    assert_array_almost_equal, assert_equal, assert_allclose)
import pytest
from pytest import raises as assert_raises

from scipy.fft._pocketfft.realtransforms import (
    dct, idct, dst, idst, dctn, idctn, dstn, idstn)

fftpack_test_dir = join(dirname(__file__), '..', '..', '..', 'fftpack', 'tests')

MDATA_COUNT = 8
FFTWDATA_COUNT = 14

def is_longdouble_binary_compatible():
    try:
        one = np.frombuffer(
            b'\x00\x00\x00\x00\x00\x00\x00\x80\xff\x3f\x00\x00\x00\x00\x00\x00',
            dtype='<f16')
        return one == np.longfloat(1.)
    except TypeError:
        return False


def get_reference_data():
    ref = getattr(globals(), '__reference_data', None)
    if ref is not None:
        return ref

    # Matlab reference data
    MDATA = np.load(join(fftpack_test_dir, 'test.npz'))
    X = [MDATA['x%d' % i] for i in range(MDATA_COUNT)]
    Y = [MDATA['y%d' % i] for i in range(MDATA_COUNT)]

    # FFTW reference data: the data are organized as follows:
    #    * SIZES is an array containing all available sizes
    #    * for every type (1, 2, 3, 4) and every size, the array dct_type_size
    #    contains the output of the DCT applied to the input np.linspace(0, size-1,
    #    size)
    FFTWDATA_DOUBLE = np.load(join(fftpack_test_dir, 'fftw_double_ref.npz'))
    FFTWDATA_SINGLE = np.load(join(fftpack_test_dir, 'fftw_single_ref.npz'))
    FFTWDATA_SIZES = FFTWDATA_DOUBLE['sizes']
    assert len(FFTWDATA_SIZES) == FFTWDATA_COUNT

    if is_longdouble_binary_compatible():
        FFTWDATA_LONGDOUBLE = np.load(
            join(fftpack_test_dir, 'fftw_longdouble_ref.npz'))
    else:
        FFTWDATA_LONGDOUBLE = {k: v.astype(np.longfloat)
                               for k,v in FFTWDATA_DOUBLE.items()}

    ref = {
        'FFTWDATA_LONGDOUBLE': FFTWDATA_LONGDOUBLE,
        'FFTWDATA_DOUBLE': FFTWDATA_DOUBLE,
        'FFTWDATA_SINGLE': FFTWDATA_SINGLE,
        'FFTWDATA_SIZES': FFTWDATA_SIZES,
        'X': X,
        'Y': Y
    }

    globals()['__reference_data'] = ref
    return ref


@pytest.fixture(params=range(FFTWDATA_COUNT))
def fftwdata_size(request):
    return get_reference_data()['FFTWDATA_SIZES'][request.param]

@pytest.fixture(params=range(MDATA_COUNT))
def mdata_x(request):
    return get_reference_data()['X'][request.param]


@pytest.fixture(params=range(MDATA_COUNT))
def mdata_xy(request):
    ref = get_reference_data()
    y = ref['Y'][request.param]
    x = ref['X'][request.param]
    return x, y


def fftw_dct_ref(type, size, dt):
    x = np.linspace(0, size-1, size).astype(dt)
    dt = np.result_type(np.float32, dt)
    if dt == np.double:
        data = get_reference_data()['FFTWDATA_DOUBLE']
    elif dt == np.float32:
        data = get_reference_data()['FFTWDATA_SINGLE']
    elif dt == np.longfloat:
        data = get_reference_data()['FFTWDATA_LONGDOUBLE']
    else:
        raise ValueError()
    y = (data['dct_%d_%d' % (type, size)]).astype(dt)
    return x, y, dt


def fftw_dst_ref(type, size, dt):
    x = np.linspace(0, size-1, size).astype(dt)
    dt = np.result_type(np.float32, dt)
    if dt == np.double:
        data = get_reference_data()['FFTWDATA_DOUBLE']
    elif dt == np.float32:
        data = get_reference_data()['FFTWDATA_SINGLE']
    elif dt == np.longfloat:
        data = get_reference_data()['FFTWDATA_LONGDOUBLE']
    else:
        raise ValueError()
    y = (data['dst_%d_%d' % (type, size)]).astype(dt)
    return x, y, dt


def ref_2d(func, x, **kwargs):
    """Calculate 2-D reference data from a 1d transform"""
    x = np.array(x, copy=True)
    for row in range(x.shape[0]):
        x[row, :] = func(x[row, :], **kwargs)
    for col in range(x.shape[1]):
        x[:, col] = func(x[:, col], **kwargs)
    return x


def naive_dct1(x, norm=None):
    """Calculate textbook definition version of DCT-I."""
    x = np.array(x, copy=True)
    N = len(x)
    M = N-1
    y = np.zeros(N)
    m0, m = 1, 2
    if norm == 'ortho':
        m0 = np.sqrt(1.0/M)
        m = np.sqrt(2.0/M)
    for k in range(N):
        for n in range(1, N-1):
            y[k] += m*x[n]*np.cos(np.pi*n*k/M)
        y[k] += m0 * x[0]
        y[k] += m0 * x[N-1] * (1 if k % 2 == 0 else -1)
    if norm == 'ortho':
        y[0] *= 1/np.sqrt(2)
        y[N-1] *= 1/np.sqrt(2)
    return y


def naive_dst1(x, norm=None):
    """Calculate textbook definition version of DST-I."""
    x = np.array(x, copy=True)
    N = len(x)
    M = N+1
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += 2*x[n]*np.sin(np.pi*(n+1.0)*(k+1.0)/M)
    if norm == 'ortho':
        y *= np.sqrt(0.5/M)
    return y


def naive_dct4(x, norm=None):
    """Calculate textbook definition version of DCT-IV."""
    x = np.array(x, copy=True)
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n]*np.cos(np.pi*(n+0.5)*(k+0.5)/(N))
    if norm == 'ortho':
        y *= np.sqrt(2.0/N)
    else:
        y *= 2
    return y


def naive_dst4(x, norm=None):
    """Calculate textbook definition version of DST-IV."""
    x = np.array(x, copy=True)
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        for n in range(N):
            y[k] += x[n]*np.sin(np.pi*(n+0.5)*(k+0.5)/(N))
    if norm == 'ortho':
        y *= np.sqrt(2.0/N)
    else:
        y *= 2
    return y


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128, np.longcomplex])
@pytest.mark.parametrize('transform', [dct, dst, idct, idst])
def test_complex(transform, dtype):
    y = transform(1j*np.arange(5, dtype=dtype))
    x = 1j*transform(np.arange(5))
    assert_array_almost_equal(x, y)


DecMapType = Dict[
    Tuple[Callable[..., np.ndarray], Union[Type[np.floating], Type[int]], int],
    int,
]

# map (tranform, dtype, type) -> decimal
dec_map: DecMapType = {
    # DCT
    (dct, np.double, 1): 13,
    (dct, np.float32, 1): 6,

    (dct, np.double, 2): 14,
    (dct, np.float32, 2): 5,

    (dct, np.double, 3): 14,
    (dct, np.float32, 3): 5,

    (dct, np.double, 4): 13,
    (dct, np.float32, 4): 6,

    # IDCT
    (idct, np.double, 1): 14,
    (idct, np.float32, 1): 6,

    (idct, np.double, 2): 14,
    (idct, np.float32, 2): 5,

    (idct, np.double, 3): 14,
    (idct, np.float32, 3): 5,

    (idct, np.double, 4): 14,
    (idct, np.float32, 4): 6,

    # DST
    (dst, np.double, 1): 13,
    (dst, np.float32, 1): 6,

    (dst, np.double, 2): 14,
    (dst, np.float32, 2): 6,

    (dst, np.double, 3): 14,
    (dst, np.float32, 3): 7,

    (dst, np.double, 4): 13,
    (dst, np.float32, 4): 6,

    # IDST
    (idst, np.double, 1): 14,
    (idst, np.float32, 1): 6,

    (idst, np.double, 2): 14,
    (idst, np.float32, 2): 6,

    (idst, np.double, 3): 14,
    (idst, np.float32, 3): 6,

    (idst, np.double, 4): 14,
    (idst, np.float32, 4): 6,
}

for k,v in dec_map.copy().items():
    if k[1] == np.double:
        dec_map[(k[0], np.longdouble, k[2])] = v
    elif k[1] == np.float32:
        dec_map[(k[0], int, k[2])] = v


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
class TestDCT:
    def test_definition(self, rdt, type, fftwdata_size):
        x, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt)
        y = dct(x, type=type)
        assert_equal(y.dtype, dt)
        dec = dec_map[(dct, rdt, type)]
        assert_allclose(y, yr, rtol=0., atol=np.max(yr)*10**(-dec))

    @pytest.mark.parametrize('size', [7, 8, 9, 16, 32, 64])
    def test_axis(self, rdt, type, size):
        nt = 2
        dec = dec_map[(dct, rdt, type)]
        x = np.random.randn(nt, size)
        y = dct(x, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[j], dct(x[j], type=type),
                                      decimal=dec)

        x = x.T
        y = dct(x, axis=0, type=type)
        for j in range(nt):
            assert_array_almost_equal(y[:,j], dct(x[:,j], type=type),
                                      decimal=dec)


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
def test_dct1_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    dec = dec_map[(dct, rdt, 1)]
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dct(x, norm='ortho', type=1)
    y2 = naive_dct1(x, norm='ortho')
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
def test_dct2_definition_matlab(mdata_xy, rdt):
    # Test correspondence with matlab (orthornomal mode).
    dt = np.result_type(np.float32, rdt)
    x = np.array(mdata_xy[0], dtype=dt)

    yr = mdata_xy[1]
    y = dct(x, norm="ortho", type=2)
    dec = dec_map[(dct, rdt, 2)]
    assert_equal(y.dtype, dt)
    assert_array_almost_equal(y, yr, decimal=dec)


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
def test_dct3_definition_ortho(mdata_x, rdt):
    # Test orthornomal mode.
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dct(x, norm='ortho', type=2)
    xi = dct(y, norm="ortho", type=3)
    dec = dec_map[(dct, rdt, 3)]
    assert_equal(xi.dtype, dt)
    assert_array_almost_equal(xi, x, decimal=dec)


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
def test_dct4_definition_ortho(mdata_x, rdt):
    # Test orthornomal mode.
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dct(x, norm='ortho', type=4)
    y2 = naive_dct4(x, norm='ortho')
    dec = dec_map[(dct, rdt, 4)]
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_idct_definition(fftwdata_size, rdt, type):
    xr, yr, dt = fftw_dct_ref(type, fftwdata_size, rdt)
    x = idct(yr, type=type)
    dec = dec_map[(idct, rdt, type)]
    assert_equal(x.dtype, dt)
    assert_allclose(x, xr, rtol=0., atol=np.max(xr)*10**(-dec))


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_definition(fftwdata_size, rdt, type):
    xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt)
    y = dst(xr, type=type)
    dec = dec_map[(dst, rdt, type)]
    assert_equal(y.dtype, dt)
    assert_allclose(y, yr, rtol=0., atol=np.max(yr)*10**(-dec))


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
def test_dst1_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    dec = dec_map[(dst, rdt, 1)]
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dst(x, norm='ortho', type=1)
    y2 = naive_dst1(x, norm='ortho')
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0., atol=np.max(y2)*10**(-dec))


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
def test_dst4_definition_ortho(rdt, mdata_x):
    # Test orthornomal mode.
    dec = dec_map[(dst, rdt, 4)]
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dst(x, norm='ortho', type=4)
    y2 = naive_dst4(x, norm='ortho')
    assert_equal(y.dtype, dt)
    assert_array_almost_equal(y, y2, decimal=dec)


@pytest.mark.parametrize('rdt', [np.longfloat, np.double, np.float32, int])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
def test_idst_definition(fftwdata_size, rdt, type):
    xr, yr, dt = fftw_dst_ref(type, fftwdata_size, rdt)
    x = idst(yr, type=type)
    dec = dec_map[(idst, rdt, type)]
    assert_equal(x.dtype, dt)
    assert_allclose(x, xr, rtol=0., atol=np.max(xr)*10**(-dec))


@pytest.mark.parametrize('routine', [dct, dst, idct, idst])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.longfloat])
@pytest.mark.parametrize('shape, axis', [
    ((16,), -1), ((16, 2), 0), ((2, 16), 1)
])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('overwrite_x', [True, False])
@pytest.mark.parametrize('norm', [None, 'ortho'])
def test_overwrite(routine, dtype, shape, axis, type, norm, overwrite_x):
    # Check input overwrite behavior
    np.random.seed(1234)
    if np.issubdtype(dtype, np.complexfloating):
        x = np.random.randn(*shape) + 1j*np.random.randn(*shape)
    else:
        x = np.random.randn(*shape)
    x = x.astype(dtype)
    x2 = x.copy()
    routine(x2, type, None, axis, norm, overwrite_x=overwrite_x)

    sig = "%s(%s%r, %r, axis=%r, overwrite_x=%r)" % (
        routine.__name__, x.dtype, x.shape, None, axis, overwrite_x)
    if not overwrite_x:
        assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)


class Test_DCTN_IDCTN:
    dec = 14
    dct_type = [1, 2, 3, 4]
    norms = [None, 'backward', 'ortho', 'forward']
    rstate = np.random.RandomState(1234)
    shape = (32, 16)
    data = rstate.randn(*shape)

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    @pytest.mark.parametrize('axes', [None,
                                      1, (1,), [1],
                                      0, (0,), [0],
                                      (0, 1), [0, 1],
                                      (-2, -1), [-2, -1]])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', ['ortho'])
    def test_axes_round_trip(self, fforward, finverse, axes, dct_type, norm):
        tmp = fforward(self.data, type=dct_type, axes=axes, norm=norm)
        tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)
        assert_array_almost_equal(self.data, tmp, decimal=12)

    @pytest.mark.parametrize('funcn,func', [(dctn, dct), (dstn, dst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_dctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        y1 = funcn(self.data, type=dct_type, axes=None, norm=norm)
        y2 = ref_2d(func, self.data, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize('funcn,func', [(idctn, idct), (idstn, idst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_idctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        fdata = dctn(self.data, type=dct_type, norm=norm)
        y1 = funcn(fdata, type=dct_type, norm=norm)
        y2 = ref_2d(func, fdata, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    def test_axes_and_shape(self, fforward, finverse):
        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            fforward(self.data, s=self.data.shape[0], axes=(0, 1))

        with assert_raises(ValueError,
                           match="when given, axes and shape arguments"
                           " have to be of the same length"):
            fforward(self.data, s=self.data.shape, axes=0)

    @pytest.mark.parametrize('fforward', [dctn, dstn])
    def test_shape(self, fforward):
        tmp = fforward(self.data, s=(128, 128), axes=None)
        assert_equal(tmp.shape, (128, 128))

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn),
                                                   (dstn, idstn)])
    @pytest.mark.parametrize('axes', [1, (1,), [1],
                                      0, (0,), [0]])
    def test_shape_is_none_with_axes(self, fforward, finverse, axes):
        tmp = fforward(self.data, s=None, axes=axes, norm='ortho')
        tmp = finverse(tmp, s=None, axes=axes, norm='ortho')
        assert_array_almost_equal(self.data, tmp, decimal=self.dec)


@pytest.mark.parametrize('func', [dct, dctn, idct, idctn,
                                  dst, dstn, idst, idstn])
def test_swapped_byte_order(func):
    rng = np.random.RandomState(1234)
    x = rng.rand(10)
    swapped_dt = x.dtype.newbyteorder('S')
    assert_allclose(func(x.astype(swapped_dt)), func(x))
