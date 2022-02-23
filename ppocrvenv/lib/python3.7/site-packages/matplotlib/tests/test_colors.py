import copy
import itertools
import unittest.mock

from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64

from numpy.testing import assert_array_equal, assert_array_almost_equal

from matplotlib import cycler
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.colorbar as mcolorbar
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.testing.decorators import image_comparison, check_figures_equal


@pytest.mark.parametrize('N, result', [
    (5, [1, .6, .2, .1, 0]),
    (2, [1, 0]),
    (1, [0]),
])
def test_create_lookup_table(N, result):
    data = [(0.0, 1.0, 1.0), (0.5, 0.2, 0.2), (1.0, 0.0, 0.0)]
    assert_array_almost_equal(mcolors._create_lookup_table(N, data), result)


def test_resample():
    """
    GitHub issue #6025 pointed to incorrect ListedColormap._resample;
    here we test the method for LinearSegmentedColormap as well.
    """
    n = 101
    colorlist = np.empty((n, 4), float)
    colorlist[:, 0] = np.linspace(0, 1, n)
    colorlist[:, 1] = 0.2
    colorlist[:, 2] = np.linspace(1, 0, n)
    colorlist[:, 3] = 0.7
    lsc = mcolors.LinearSegmentedColormap.from_list('lsc', colorlist)
    lc = mcolors.ListedColormap(colorlist)
    # Set some bad values for testing too
    for cmap in [lsc, lc]:
        cmap.set_under('r')
        cmap.set_over('g')
        cmap.set_bad('b')
    lsc3 = lsc._resample(3)
    lc3 = lc._resample(3)
    expected = np.array([[0.0, 0.2, 1.0, 0.7],
                         [0.5, 0.2, 0.5, 0.7],
                         [1.0, 0.2, 0.0, 0.7]], float)
    assert_array_almost_equal(lsc3([0, 0.5, 1]), expected)
    assert_array_almost_equal(lc3([0, 0.5, 1]), expected)
    # Test over/under was copied properly
    assert_array_almost_equal(lsc(np.inf), lsc3(np.inf))
    assert_array_almost_equal(lsc(-np.inf), lsc3(-np.inf))
    assert_array_almost_equal(lsc(np.nan), lsc3(np.nan))
    assert_array_almost_equal(lc(np.inf), lc3(np.inf))
    assert_array_almost_equal(lc(-np.inf), lc3(-np.inf))
    assert_array_almost_equal(lc(np.nan), lc3(np.nan))


def test_register_cmap():
    new_cm = copy.copy(cm.get_cmap("viridis"))
    target = "viridis2"
    cm.register_cmap(target, new_cm)
    assert plt.get_cmap(target) == new_cm

    with pytest.raises(ValueError,
                       match="Arguments must include a name or a Colormap"):
        cm.register_cmap()

    with pytest.warns(UserWarning):
        cm.register_cmap(target, new_cm)

    cm.unregister_cmap(target)
    with pytest.raises(ValueError,
                       match=f'{target!r} is not a valid value for name;'):
        cm.get_cmap(target)
    # test that second time is error free
    cm.unregister_cmap(target)

    with pytest.raises(ValueError, match="You must pass a Colormap instance."):
        cm.register_cmap('nome', cmap='not a cmap')


def test_double_register_builtin_cmap():
    name = "viridis"
    match = f"Trying to re-register the builtin cmap {name!r}."
    with pytest.raises(ValueError, match=match):
        cm.register_cmap(name, cm.get_cmap(name))
    with pytest.warns(UserWarning):
        cm.register_cmap(name, cm.get_cmap(name), override_builtin=True)


def test_unregister_builtin_cmap():
    name = "viridis"
    match = f'cannot unregister {name!r} which is a builtin colormap.'
    with pytest.raises(ValueError, match=match):
        cm.unregister_cmap(name)


def test_colormap_global_set_warn():
    new_cm = plt.get_cmap('viridis')
    # Store the old value so we don't override the state later on.
    orig_cmap = copy.copy(new_cm)
    with pytest.warns(cbook.MatplotlibDeprecationWarning,
                      match="You are modifying the state of a globally"):
        # This should warn now because we've modified the global state
        new_cm.set_under('k')

    # This shouldn't warn because it is a copy
    copy.copy(new_cm).set_under('b')

    # Test that registering and then modifying warns
    plt.register_cmap(name='test_cm', cmap=copy.copy(orig_cmap))
    new_cm = plt.get_cmap('test_cm')
    with pytest.warns(cbook.MatplotlibDeprecationWarning,
                      match="You are modifying the state of a globally"):
        # This should warn now because we've modified the global state
        new_cm.set_under('k')

    # Re-register the original
    with pytest.warns(UserWarning):
        plt.register_cmap(cmap=orig_cmap, override_builtin=True)


def test_colormap_dict_deprecate():
    # Make sure we warn on get and set access into cmap_d
    with pytest.warns(cbook.MatplotlibDeprecationWarning,
                      match="The global colormaps dictionary is no longer"):
        cmap = plt.cm.cmap_d['viridis']

    with pytest.warns(cbook.MatplotlibDeprecationWarning,
                      match="The global colormaps dictionary is no longer"):
        plt.cm.cmap_d['test'] = cmap


def test_colormap_copy():
    cmap = plt.cm.Reds
    copied_cmap = copy.copy(cmap)
    with np.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    cmap2 = copy.copy(copied_cmap)
    cmap2.set_bad('g')
    with np.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    assert_array_equal(ret1, ret2)
    # again with the .copy method:
    cmap = plt.cm.Reds
    copied_cmap = cmap.copy()
    with np.errstate(invalid='ignore'):
        ret1 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    cmap2 = copy.copy(copied_cmap)
    cmap2.set_bad('g')
    with np.errstate(invalid='ignore'):
        ret2 = copied_cmap([-1, 0, .5, 1, np.nan, np.inf])
    assert_array_equal(ret1, ret2)


def test_colormap_equals():
    cmap = plt.get_cmap("plasma")
    cm_copy = cmap.copy()
    # different object id's
    assert cm_copy is not cmap
    # But the same data should be equal
    assert cm_copy == cmap
    # Change the copy
    cm_copy.set_bad('y')
    assert cm_copy != cmap
    # Make sure we can compare different sizes without failure
    cm_copy._lut = cm_copy._lut[:10, :]
    assert cm_copy != cmap
    # Test different names are not equal
    cm_copy = cmap.copy()
    cm_copy.name = "Test"
    assert cm_copy != cmap
    # Test colorbar extends
    cm_copy = cmap.copy()
    cm_copy.colorbar_extend = not cmap.colorbar_extend
    assert cm_copy != cmap


def test_colormap_endian():
    """
    GitHub issue #1005: a bug in putmask caused erroneous
    mapping of 1.0 when input from a non-native-byteorder
    array.
    """
    cmap = cm.get_cmap("jet")
    # Test under, over, and invalid along with values 0 and 1.
    a = [-0.5, 0, 0.5, 1, 1.5, np.nan]
    for dt in ["f2", "f4", "f8"]:
        anative = np.ma.masked_invalid(np.array(a, dtype=dt))
        aforeign = anative.byteswap().newbyteorder()
        assert_array_equal(cmap(anative), cmap(aforeign))


def test_colormap_invalid():
    """
    GitHub issue #9892: Handling of nan's were getting mapped to under
    rather than bad. This tests to make sure all invalid values
    (-inf, nan, inf) are mapped respectively to (under, bad, over).
    """
    cmap = cm.get_cmap("plasma")
    x = np.array([-np.inf, -1, 0, np.nan, .7, 2, np.inf])

    expected = np.array([[0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.,       0.,       0.,       0.],
                         [0.949217, 0.517763, 0.295662, 1.],
                         [0.940015, 0.975158, 0.131326, 1.],
                         [0.940015, 0.975158, 0.131326, 1.]])
    assert_array_equal(cmap(x), expected)

    # Test masked representation (-inf, inf) are now masked
    expected = np.array([[0.,       0.,       0.,       0.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.050383, 0.029803, 0.527975, 1.],
                         [0.,       0.,       0.,       0.],
                         [0.949217, 0.517763, 0.295662, 1.],
                         [0.940015, 0.975158, 0.131326, 1.],
                         [0.,       0.,       0.,       0.]])
    assert_array_equal(cmap(np.ma.masked_invalid(x)), expected)

    # Test scalar representations
    assert_array_equal(cmap(-np.inf), cmap(0))
    assert_array_equal(cmap(np.inf), cmap(1.0))
    assert_array_equal(cmap(np.nan), np.array([0., 0., 0., 0.]))


def test_colormap_return_types():
    """
    Make sure that tuples are returned for scalar input and
    that the proper shapes are returned for ndarrays.
    """
    cmap = cm.get_cmap("plasma")
    # Test return types and shapes
    # scalar input needs to return a tuple of length 4
    assert isinstance(cmap(0.5), tuple)
    assert len(cmap(0.5)) == 4

    # input array returns an ndarray of shape x.shape + (4,)
    x = np.ones(4)
    assert cmap(x).shape == x.shape + (4,)

    # multi-dimensional array input
    x2d = np.zeros((2, 2))
    assert cmap(x2d).shape == x2d.shape + (4,)


def test_BoundaryNorm():
    """
    GitHub issue #1258: interpolation was failing with numpy
    1.7 pre-release.
    """

    boundaries = [0, 1.1, 2.2]
    vals = [-1, 0, 1, 2, 2.2, 4]

    # Without interpolation
    expected = [-1, 0, 0, 1, 2, 2]
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # ncolors != len(boundaries) - 1 triggers interpolation
    expected = [-1, 0, 0, 2, 3, 3]
    ncolors = len(boundaries)
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # with a single region and interpolation
    expected = [-1, 1, 1, 1, 3, 3]
    bn = mcolors.BoundaryNorm([0, 2.2], ncolors)
    assert_array_equal(bn(vals), expected)

    # more boundaries for a third color
    boundaries = [0, 1, 2, 3]
    vals = [-1, 0.1, 1.1, 2.2, 4]
    ncolors = 5
    expected = [-1, 0, 2, 4, 5]
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    assert_array_equal(bn(vals), expected)

    # a scalar as input should not trigger an error and should return a scalar
    boundaries = [0, 1, 2]
    vals = [-1, 0.1, 1.1, 2.2]
    bn = mcolors.BoundaryNorm(boundaries, 2)
    expected = [-1, 0, 1, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # same with interp
    bn = mcolors.BoundaryNorm(boundaries, 3)
    expected = [-1, 0, 2, 3]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # Clipping
    bn = mcolors.BoundaryNorm(boundaries, 3, clip=True)
    expected = [0, 0, 2, 2]
    for v, ex in zip(vals, expected):
        ret = bn(v)
        assert isinstance(ret, int)
        assert_array_equal(ret, ex)
        assert_array_equal(bn([v]), ex)

    # Masked arrays
    boundaries = [0, 1.1, 2.2]
    vals = np.ma.masked_invalid([-1., np.NaN, 0, 1.4, 9])

    # Without interpolation
    ncolors = len(boundaries) - 1
    bn = mcolors.BoundaryNorm(boundaries, ncolors)
    expected = np.ma.masked_array([-1, -99, 0, 1, 2], mask=[0, 1, 0, 0, 0])
    assert_array_equal(bn(vals), expected)

    # With interpolation
    bn = mcolors.BoundaryNorm(boundaries, len(boundaries))
    expected = np.ma.masked_array([-1, -99, 0, 2, 3], mask=[0, 1, 0, 0, 0])
    assert_array_equal(bn(vals), expected)

    # Non-trivial masked arrays
    vals = np.ma.masked_invalid([np.Inf, np.NaN])
    assert np.all(bn(vals).mask)
    vals = np.ma.masked_invalid([np.Inf])
    assert np.all(bn(vals).mask)

    # Incompatible extend and clip
    with pytest.raises(ValueError, match="not compatible"):
        mcolors.BoundaryNorm(np.arange(4), 5, extend='both', clip=True)

    # Too small ncolors argument
    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(np.arange(4), 2)

    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(np.arange(4), 3, extend='min')

    with pytest.raises(ValueError, match="ncolors must equal or exceed"):
        mcolors.BoundaryNorm(np.arange(4), 4, extend='both')

    # Testing extend keyword, with interpolation (large cmap)
    bounds = [1, 2, 3]
    cmap = cm.get_cmap('viridis')
    mynorm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    refnorm = mcolors.BoundaryNorm([0] + bounds + [4], cmap.N)
    x = np.random.randn(100) * 10 + 2
    ref = refnorm(x)
    ref[ref == 0] = -1
    ref[ref == cmap.N - 1] = cmap.N
    assert_array_equal(mynorm(x), ref)

    # Without interpolation
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_over('black')
    cmref.set_under('white')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red', 'black'])

    assert mcolors.same_color(cmref.get_over(), 'black')
    assert mcolors.same_color(cmref.get_under(), 'white')

    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='both')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax

    assert mynorm(bounds[0] - 0.1) == -1  # under
    assert mynorm(bounds[0] + 0.1) == 1   # first bin -> second color
    assert mynorm(bounds[-1] - 0.1) == cmshould.N - 2  # next-to-last color
    assert mynorm(bounds[-1] + 0.1) == cmshould.N  # over

    x = [-1, 1.2, 2.3, 9.6]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2, 3]))
    x = np.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))

    # Just min
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_under('white')
    cmshould = mcolors.ListedColormap(['white', 'blue', 'red'])

    assert mcolors.same_color(cmref.get_under(), 'white')

    assert cmref.N == 2
    assert cmshould.N == 3
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='min')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax
    x = [-1, 1.2, 2.3]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    x = np.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))

    # Just max
    cmref = mcolors.ListedColormap(['blue', 'red'])
    cmref.set_over('black')
    cmshould = mcolors.ListedColormap(['blue', 'red', 'black'])

    assert mcolors.same_color(cmref.get_over(), 'black')

    assert cmref.N == 2
    assert cmshould.N == 3
    refnorm = mcolors.BoundaryNorm(bounds, cmref.N)
    mynorm = mcolors.BoundaryNorm(bounds, cmshould.N, extend='max')
    assert mynorm.vmin == refnorm.vmin
    assert mynorm.vmax == refnorm.vmax
    x = [1.2, 2.3, 4]
    assert_array_equal(cmshould(mynorm(x)), cmshould([0, 1, 2]))
    x = np.random.randn(100) * 10 + 2
    assert_array_equal(cmshould(mynorm(x)), cmref(refnorm(x)))


def test_CenteredNorm():
    np.random.seed(0)

    # Assert equivalence to symmetrical Normalize.
    x = np.random.normal(size=100)
    x_maxabs = np.max(np.abs(x))
    norm_ref = mcolors.Normalize(vmin=-x_maxabs, vmax=x_maxabs)
    norm = mcolors.CenteredNorm()
    assert_array_almost_equal(norm_ref(x), norm(x))

    # Check that vcenter is in the center of vmin and vmax
    # when vcenter is set.
    vcenter = int(np.random.normal(scale=50))
    norm = mcolors.CenteredNorm(vcenter=vcenter)
    norm.autoscale_None([1, 2])
    assert norm.vmax + norm.vmin == 2 * vcenter

    # Check that halfrange can be set without setting vcenter and that it is
    # not reset through autoscale_None.
    norm = mcolors.CenteredNorm(halfrange=1.0)
    norm.autoscale_None([1, 3000])
    assert norm.halfrange == 1.0

    # Check that halfrange input works correctly.
    x = np.random.normal(size=10)
    norm = mcolors.CenteredNorm(vcenter=0.5, halfrange=0.5)
    assert_array_almost_equal(x, norm(x))
    norm = mcolors.CenteredNorm(vcenter=1, halfrange=1)
    assert_array_almost_equal(x, 2 * norm(x))

    # Check that halfrange input works correctly and use setters.
    norm = mcolors.CenteredNorm()
    norm.vcenter = 2
    norm.halfrange = 2
    assert_array_almost_equal(x, 4 * norm(x))

    # Check that prior to adding data, setting halfrange first has same effect.
    norm = mcolors.CenteredNorm()
    norm.halfrange = 2
    norm.vcenter = 2
    assert_array_almost_equal(x, 4 * norm(x))

    # Check that manual change of vcenter adjusts halfrange accordingly.
    norm = mcolors.CenteredNorm()
    assert norm.vcenter == 0
    # add data
    norm(np.linspace(-1.0, 0.0, 10))
    assert norm.vmax == 1.0
    assert norm.halfrange == 1.0
    # set vcenter to 1, which should double halfrange
    norm.vcenter = 1
    assert norm.vmin == -1.0
    assert norm.vmax == 3.0
    assert norm.halfrange == 2.0


@pytest.mark.parametrize("vmin,vmax", [[-1, 2], [3, 1]])
def test_lognorm_invalid(vmin, vmax):
    # Check that invalid limits in LogNorm error
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    with pytest.raises(ValueError):
        norm(1)
    with pytest.raises(ValueError):
        norm.inverse(1)


def test_LogNorm():
    """
    LogNorm ignored clip, now it has the same
    behavior as Normalize, e.g., values > vmax are bigger than 1
    without clip, with clip they are 1.
    """
    ln = mcolors.LogNorm(clip=True, vmax=5)
    assert_array_equal(ln([1, 6]), [0, 1.0])


def test_LogNorm_inverse():
    """
    Test that lists work, and that the inverse works
    """
    norm = mcolors.LogNorm(vmin=0.1, vmax=10)
    assert_array_almost_equal(norm([0.5, 0.4]), [0.349485, 0.30103])
    assert_array_almost_equal([0.5, 0.4], norm.inverse([0.349485, 0.30103]))
    assert_array_almost_equal(norm(0.4), [0.30103])
    assert_array_almost_equal([0.4], norm.inverse([0.30103]))


def test_PowerNorm():
    a = np.array([0, 0.5, 1, 1.5], dtype=float)
    pnorm = mcolors.PowerNorm(1)
    norm = mcolors.Normalize()
    assert_array_almost_equal(norm(a), pnorm(a))

    a = np.array([-0.5, 0, 2, 4, 8], dtype=float)
    expected = [0, 0, 1/16, 1/4, 1]
    pnorm = mcolors.PowerNorm(2, vmin=0, vmax=8)
    assert_array_almost_equal(pnorm(a), expected)
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[2]) == expected[2]
    assert_array_almost_equal(a[1:], pnorm.inverse(pnorm(a))[1:])

    # Clip = True
    a = np.array([-0.5, 0, 1, 8, 16], dtype=float)
    expected = [0, 0, 0, 1, 1]
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=True)
    assert_array_almost_equal(pnorm(a), expected)
    assert pnorm(a[0]) == expected[0]
    assert pnorm(a[-1]) == expected[-1]

    # Clip = True at call time
    a = np.array([-0.5, 0, 1, 8, 16], dtype=float)
    expected = [0, 0, 0, 1, 1]
    pnorm = mcolors.PowerNorm(2, vmin=2, vmax=8, clip=False)
    assert_array_almost_equal(pnorm(a, clip=True), expected)
    assert pnorm(a[0], clip=True) == expected[0]
    assert pnorm(a[-1], clip=True) == expected[-1]


def test_PowerNorm_translation_invariance():
    a = np.array([0, 1/2, 1], dtype=float)
    expected = [0, 1/8, 1]
    pnorm = mcolors.PowerNorm(vmin=0, vmax=1, gamma=3)
    assert_array_almost_equal(pnorm(a), expected)
    pnorm = mcolors.PowerNorm(vmin=-2, vmax=-1, gamma=3)
    assert_array_almost_equal(pnorm(a - 2), expected)


def test_Normalize():
    norm = mcolors.Normalize()
    vals = np.arange(-10, 10, 1, dtype=float)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # Handle integer input correctly (don't overflow when computing max-min,
    # i.e. 127-(-128) here).
    vals = np.array([-128, 127], dtype=np.int8)
    norm = mcolors.Normalize(vals.min(), vals.max())
    assert_array_equal(np.asarray(norm(vals)), [0, 1])

    # Don't lose precision on longdoubles (float128 on Linux):
    # for array inputs...
    vals = np.array([1.2345678901, 9.8765432109], dtype=np.longdouble)
    norm = mcolors.Normalize(vals.min(), vals.max())
    assert_array_equal(np.asarray(norm(vals)), [0, 1])
    # and for scalar ones.
    eps = np.finfo(np.longdouble).resolution
    norm = plt.Normalize(1, 1 + 100 * eps)
    # This returns exactly 0.5 when longdouble is extended precision (80-bit),
    # but only a value close to it when it is quadruple precision (128-bit).
    assert 0 < norm(1 + 50 * eps) < 1


def test_FuncNorm():
    def forward(x):
        return (x**2)
    def inverse(x):
        return np.sqrt(x)

    norm = mcolors.FuncNorm((forward, inverse), vmin=0, vmax=10)
    expected = np.array([0, 0.25, 1])
    input = np.array([0, 5, 10])
    assert_array_almost_equal(norm(input), expected)
    assert_array_almost_equal(norm.inverse(expected), input)

    def forward(x):
        return np.log10(x)
    def inverse(x):
        return 10**x
    norm = mcolors.FuncNorm((forward, inverse), vmin=0.1, vmax=10)
    lognorm = mcolors.LogNorm(vmin=0.1, vmax=10)
    assert_array_almost_equal(norm([0.2, 5, 10]), lognorm([0.2, 5, 10]))
    assert_array_almost_equal(norm.inverse([0.2, 5, 10]),
                              lognorm.inverse([0.2, 5, 10]))


def test_TwoSlopeNorm_autoscale():
    norm = mcolors.TwoSlopeNorm(vcenter=20)
    norm.autoscale([10, 20, 30, 40])
    assert norm.vmin == 10.
    assert norm.vmax == 40.


def test_TwoSlopeNorm_autoscale_None_vmin():
    norm = mcolors.TwoSlopeNorm(2, vmin=0, vmax=None)
    norm.autoscale_None([1, 2, 3, 4, 5])
    assert norm(5) == 1
    assert norm.vmax == 5


def test_TwoSlopeNorm_autoscale_None_vmax():
    norm = mcolors.TwoSlopeNorm(2, vmin=None, vmax=10)
    norm.autoscale_None([1, 2, 3, 4, 5])
    assert norm(1) == 0
    assert norm.vmin == 1


def test_TwoSlopeNorm_scale():
    norm = mcolors.TwoSlopeNorm(2)
    assert norm.scaled() is False
    norm([1, 2, 3, 4])
    assert norm.scaled() is True


def test_TwoSlopeNorm_scaleout_center():
    # test the vmin never goes above vcenter
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    norm([1, 2, 3, 5])
    assert norm.vmin == 0
    assert norm.vmax == 5


def test_TwoSlopeNorm_scaleout_center_max():
    # test the vmax never goes below vcenter
    norm = mcolors.TwoSlopeNorm(vcenter=0)
    norm([-1, -2, -3, -5])
    assert norm.vmax == 0
    assert norm.vmin == -5


def test_TwoSlopeNorm_Even():
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=4)
    vals = np.array([-1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0])
    expected = np.array([0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_Odd():
    norm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=5)
    vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert_array_equal(norm(vals), expected)


def test_TwoSlopeNorm_VminEqualsVcenter():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=-2, vcenter=-2, vmax=2)


def test_TwoSlopeNorm_VmaxEqualsVcenter():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=-2, vcenter=2, vmax=2)


def test_TwoSlopeNorm_VminGTVcenter():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=0, vmax=20)


def test_TwoSlopeNorm_TwoSlopeNorm_VminGTVmax():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=0, vmax=5)


def test_TwoSlopeNorm_VcenterGTVmax():
    with pytest.raises(ValueError):
        mcolors.TwoSlopeNorm(vmin=10, vcenter=25, vmax=20)


def test_TwoSlopeNorm_premature_scaling():
    norm = mcolors.TwoSlopeNorm(vcenter=2)
    with pytest.raises(ValueError):
        norm.inverse(np.array([0.1, 0.5, 0.9]))


def test_SymLogNorm():
    """
    Test SymLogNorm behavior
    """
    norm = mcolors.SymLogNorm(3, vmax=5, linscale=1.2, base=np.e)
    vals = np.array([-30, -1, 2, 6], dtype=float)
    normed_vals = norm(vals)
    expected = [0., 0.53980074, 0.826991, 1.02758204]
    assert_array_almost_equal(normed_vals, expected)
    _inverse_tester(norm, vals)
    _scalar_tester(norm, vals)
    _mask_tester(norm, vals)

    # Ensure that specifying vmin returns the same result as above
    norm = mcolors.SymLogNorm(3, vmin=-30, vmax=5, linscale=1.2, base=np.e)
    normed_vals = norm(vals)
    assert_array_almost_equal(normed_vals, expected)

    # test something more easily checked.
    norm = mcolors.SymLogNorm(1, vmin=-np.e**3, vmax=np.e**3, base=np.e)
    nn = norm([-np.e**3, -np.e**2, -np.e**1, -1,
              0, 1, np.e**1, np.e**2, np.e**3])
    xx = np.array([0., 0.109123, 0.218246, 0.32737, 0.5, 0.67263,
                   0.781754, 0.890877, 1.])
    assert_array_almost_equal(nn, xx)
    norm = mcolors.SymLogNorm(1, vmin=-10**3, vmax=10**3, base=10)
    nn = norm([-10**3, -10**2, -10**1, -1,
              0, 1, 10**1, 10**2, 10**3])
    xx = np.array([0., 0.121622, 0.243243, 0.364865, 0.5, 0.635135,
                   0.756757, 0.878378, 1.])
    assert_array_almost_equal(nn, xx)


def test_SymLogNorm_colorbar():
    """
    Test un-called SymLogNorm in a colorbar.
    """
    norm = mcolors.SymLogNorm(0.1, vmin=-1, vmax=1, linscale=1, base=np.e)
    fig = plt.figure()
    mcolorbar.ColorbarBase(fig.add_subplot(), norm=norm)
    plt.close(fig)


def test_SymLogNorm_single_zero():
    """
    Test SymLogNorm to ensure it is not adding sub-ticks to zero label
    """
    fig = plt.figure()
    norm = mcolors.SymLogNorm(1e-5, vmin=-1, vmax=1, base=np.e)
    cbar = mcolorbar.ColorbarBase(fig.add_subplot(), norm=norm)
    ticks = cbar.get_ticks()
    assert np.count_nonzero(ticks == 0) <= 1
    plt.close(fig)


def _inverse_tester(norm_instance, vals):
    """
    Checks if the inverse of the given normalization is working.
    """
    assert_array_almost_equal(norm_instance.inverse(norm_instance(vals)), vals)


def _scalar_tester(norm_instance, vals):
    """
    Checks if scalars and arrays are handled the same way.
    Tests only for float.
    """
    scalar_result = [norm_instance(float(v)) for v in vals]
    assert_array_almost_equal(scalar_result, norm_instance(vals))


def _mask_tester(norm_instance, vals):
    """
    Checks mask handling
    """
    masked_array = np.ma.array(vals)
    masked_array[0] = np.ma.masked
    assert_array_equal(masked_array.mask, norm_instance(masked_array).mask)


@image_comparison(['levels_and_colors.png'])
def test_cmap_and_norm_from_levels_and_colors():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    data = np.linspace(-2, 4, 49).reshape(7, 7)
    levels = [-1, 2, 2.5, 3]
    colors = ['red', 'green', 'blue', 'yellow', 'black']
    extend = 'both'
    cmap, norm = mcolors.from_levels_and_colors(levels, colors, extend=extend)

    ax = plt.axes()
    m = plt.pcolormesh(data, cmap=cmap, norm=norm)
    plt.colorbar(m)

    # Hide the axes labels (but not the colorbar ones, as they are useful)
    ax.tick_params(labelleft=False, labelbottom=False)


@image_comparison(baseline_images=['boundarynorm_and_colorbar'],
                  extensions=['png'], tol=1.0)
def test_boundarynorm_and_colorbarbase():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure()
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    ax2 = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    ax3 = fig.add_axes([0.05, 0.15, 0.9, 0.15])

    # Set the colormap and bounds
    bounds = [-1, 2, 5, 7, 12, 15]
    cmap = cm.get_cmap('viridis')

    # Default behavior
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    cb1 = mcolorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, extend='both',
                                 orientation='horizontal', spacing='uniform')
    # New behavior
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    cb2 = mcolorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                 orientation='horizontal')

    # User can still force to any extend='' if really needed
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='both')
    cb3 = mcolorbar.ColorbarBase(ax3, cmap=cmap, norm=norm,
                                 extend='neither', orientation='horizontal')


def test_cmap_and_norm_from_levels_and_colors2():
    levels = [-1, 2, 2.5, 3]
    colors = ['red', (0, 1, 0), 'blue', (0.5, 0.5, 0.5), (0.0, 0.0, 0.0, 1.0)]
    clr = mcolors.to_rgba_array(colors)
    bad = (0.1, 0.1, 0.1, 0.1)
    no_color = (0.0, 0.0, 0.0, 0.0)
    masked_value = 'masked_value'

    # Define the test values which are of interest.
    # Note: levels are lev[i] <= v < lev[i+1]
    tests = [('both', None, {-2: clr[0],
                             -1: clr[1],
                             2: clr[2],
                             2.25: clr[2],
                             3: clr[4],
                             3.5: clr[4],
                             masked_value: bad}),

             ('min', -1, {-2: clr[0],
                          -1: clr[1],
                          2: clr[2],
                          2.25: clr[2],
                          3: no_color,
                          3.5: no_color,
                          masked_value: bad}),

             ('max', -1, {-2: no_color,
                          -1: clr[0],
                          2: clr[1],
                          2.25: clr[1],
                          3: clr[3],
                          3.5: clr[3],
                          masked_value: bad}),

             ('neither', -2, {-2: no_color,
                              -1: clr[0],
                              2: clr[1],
                              2.25: clr[1],
                              3: no_color,
                              3.5: no_color,
                              masked_value: bad}),
             ]

    for extend, i1, cases in tests:
        cmap, norm = mcolors.from_levels_and_colors(levels, colors[0:i1],
                                                    extend=extend)
        cmap.set_bad(bad)
        for d_val, expected_color in cases.items():
            if d_val == masked_value:
                d_val = np.ma.array([1], mask=True)
            else:
                d_val = [d_val]
            assert_array_equal(expected_color, cmap(norm(d_val))[0],
                               'Wih extend={0!r} and data '
                               'value={1!r}'.format(extend, d_val))

    with pytest.raises(ValueError):
        mcolors.from_levels_and_colors(levels, colors)


def test_rgb_hsv_round_trip():
    for a_shape in [(500, 500, 3), (500, 3), (1, 3), (3,)]:
        np.random.seed(0)
        tt = np.random.random(a_shape)
        assert_array_almost_equal(
            tt, mcolors.hsv_to_rgb(mcolors.rgb_to_hsv(tt)))
        assert_array_almost_equal(
            tt, mcolors.rgb_to_hsv(mcolors.hsv_to_rgb(tt)))


def test_autoscale_masked():
    # Test for #2336. Previously fully masked data would trigger a ValueError.
    data = np.ma.masked_all((12, 20))
    plt.pcolor(data)
    plt.draw()


@image_comparison(['light_source_shading_topo.png'])
def test_light_source_topo_surface():
    """Shades a DEM using different v.e.'s and blend modes."""
    dem = cbook.get_sample_data('jacksboro_fault_dem.npz', np_load=True)
    elev = dem['elevation']
    dx, dy = dem['dx'], dem['dy']
    # Get the true cellsize in meters for accurate vertical exaggeration
    # Convert from decimal degrees to meters
    dx = 111320.0 * dx * np.cos(dem['ymin'])
    dy = 111320.0 * dy

    ls = mcolors.LightSource(315, 45)
    cmap = cm.gist_earth

    fig, axs = plt.subplots(nrows=3, ncols=3)
    for row, mode in zip(axs, ['hsv', 'overlay', 'soft']):
        for ax, ve in zip(row, [0.1, 1, 10]):
            rgb = ls.shade(elev, cmap, vert_exag=ve, dx=dx, dy=dy,
                           blend_mode=mode)
            ax.imshow(rgb)
            ax.set(xticks=[], yticks=[])


def test_light_source_shading_default():
    """
    Array comparison test for the default "hsv" blend mode. Ensure the
    default result doesn't change without warning.
    """
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z = 10 * np.cos(x**2 + y**2)

    cmap = plt.cm.copper
    ls = mcolors.LightSource(315, 45)
    rgb = ls.shade(z, cmap)

    # Result stored transposed and rounded for more compact display...
    expect = np.array(
        [[[0.00, 0.45, 0.90, 0.90, 0.82, 0.62, 0.28, 0.00],
          [0.45, 0.94, 0.99, 1.00, 1.00, 0.96, 0.65, 0.17],
          [0.90, 0.99, 1.00, 1.00, 1.00, 1.00, 0.94, 0.35],
          [0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.49],
          [0.82, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.41],
          [0.62, 0.96, 1.00, 1.00, 1.00, 1.00, 0.90, 0.07],
          [0.28, 0.65, 0.94, 1.00, 1.00, 0.90, 0.35, 0.01],
          [0.00, 0.17, 0.35, 0.49, 0.41, 0.07, 0.01, 0.00]],

         [[0.00, 0.28, 0.59, 0.72, 0.62, 0.40, 0.18, 0.00],
          [0.28, 0.78, 0.93, 0.92, 0.83, 0.66, 0.39, 0.11],
          [0.59, 0.93, 0.99, 1.00, 0.92, 0.75, 0.50, 0.21],
          [0.72, 0.92, 1.00, 0.99, 0.93, 0.76, 0.51, 0.18],
          [0.62, 0.83, 0.92, 0.93, 0.87, 0.68, 0.42, 0.08],
          [0.40, 0.66, 0.75, 0.76, 0.68, 0.52, 0.23, 0.02],
          [0.18, 0.39, 0.50, 0.51, 0.42, 0.23, 0.00, 0.00],
          [0.00, 0.11, 0.21, 0.18, 0.08, 0.02, 0.00, 0.00]],

         [[0.00, 0.18, 0.38, 0.46, 0.39, 0.26, 0.11, 0.00],
          [0.18, 0.50, 0.70, 0.75, 0.64, 0.44, 0.25, 0.07],
          [0.38, 0.70, 0.91, 0.98, 0.81, 0.51, 0.29, 0.13],
          [0.46, 0.75, 0.98, 0.96, 0.84, 0.48, 0.22, 0.12],
          [0.39, 0.64, 0.81, 0.84, 0.71, 0.31, 0.11, 0.05],
          [0.26, 0.44, 0.51, 0.48, 0.31, 0.10, 0.03, 0.01],
          [0.11, 0.25, 0.29, 0.22, 0.11, 0.03, 0.00, 0.00],
          [0.00, 0.07, 0.13, 0.12, 0.05, 0.01, 0.00, 0.00]],

         [[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]]
         ]).T

    assert_array_almost_equal(rgb, expect, decimal=2)


def test_light_source_shading_empty_mask():
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z0 = 10 * np.cos(x**2 + y**2)
    z1 = np.ma.array(z0)

    cmap = plt.cm.copper
    ls = mcolors.LightSource(315, 45)
    rgb0 = ls.shade(z0, cmap)
    rgb1 = ls.shade(z1, cmap)

    assert_array_almost_equal(rgb0, rgb1)


# Numpy 1.9.1 fixed a bug in masked arrays which resulted in
# additional elements being masked when calculating the gradient thus
# the output is different with earlier numpy versions.
def test_light_source_masked_shading():
    """
    Array comparison test for a surface with a masked portion. Ensures that
    we don't wind up with "fringes" of odd colors around masked regions.
    """
    y, x = np.mgrid[-1.2:1.2:8j, -1.2:1.2:8j]
    z = 10 * np.cos(x**2 + y**2)

    z = np.ma.masked_greater(z, 9.9)

    cmap = plt.cm.copper
    ls = mcolors.LightSource(315, 45)
    rgb = ls.shade(z, cmap)

    # Result stored transposed and rounded for more compact display...
    expect = np.array(
        [[[0.00, 0.46, 0.91, 0.91, 0.84, 0.64, 0.29, 0.00],
          [0.46, 0.96, 1.00, 1.00, 1.00, 0.97, 0.67, 0.18],
          [0.91, 1.00, 1.00, 1.00, 1.00, 1.00, 0.96, 0.36],
          [0.91, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.51],
          [0.84, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 0.44],
          [0.64, 0.97, 1.00, 1.00, 1.00, 1.00, 0.94, 0.09],
          [0.29, 0.67, 0.96, 1.00, 1.00, 0.94, 0.38, 0.01],
          [0.00, 0.18, 0.36, 0.51, 0.44, 0.09, 0.01, 0.00]],

         [[0.00, 0.29, 0.61, 0.75, 0.64, 0.41, 0.18, 0.00],
          [0.29, 0.81, 0.95, 0.93, 0.85, 0.68, 0.40, 0.11],
          [0.61, 0.95, 1.00, 0.78, 0.78, 0.77, 0.52, 0.22],
          [0.75, 0.93, 0.78, 0.00, 0.00, 0.78, 0.54, 0.19],
          [0.64, 0.85, 0.78, 0.00, 0.00, 0.78, 0.45, 0.08],
          [0.41, 0.68, 0.77, 0.78, 0.78, 0.55, 0.25, 0.02],
          [0.18, 0.40, 0.52, 0.54, 0.45, 0.25, 0.00, 0.00],
          [0.00, 0.11, 0.22, 0.19, 0.08, 0.02, 0.00, 0.00]],

         [[0.00, 0.19, 0.39, 0.48, 0.41, 0.26, 0.12, 0.00],
          [0.19, 0.52, 0.73, 0.78, 0.66, 0.46, 0.26, 0.07],
          [0.39, 0.73, 0.95, 0.50, 0.50, 0.53, 0.30, 0.14],
          [0.48, 0.78, 0.50, 0.00, 0.00, 0.50, 0.23, 0.12],
          [0.41, 0.66, 0.50, 0.00, 0.00, 0.50, 0.11, 0.05],
          [0.26, 0.46, 0.53, 0.50, 0.50, 0.11, 0.03, 0.01],
          [0.12, 0.26, 0.30, 0.23, 0.11, 0.03, 0.00, 0.00],
          [0.00, 0.07, 0.14, 0.12, 0.05, 0.01, 0.00, 0.00]],

         [[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
          [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]],
         ]).T

    assert_array_almost_equal(rgb, expect, decimal=2)


def test_light_source_hillshading():
    """
    Compare the current hillshading method against one that should be
    mathematically equivalent. Illuminates a cone from a range of angles.
    """

    def alternative_hillshade(azimuth, elev, z):
        illum = _sph2cart(*_azimuth2math(azimuth, elev))
        illum = np.array(illum)

        dy, dx = np.gradient(-z)
        dy = -dy
        dz = np.ones_like(dy)
        normals = np.dstack([dx, dy, dz])
        normals /= np.linalg.norm(normals, axis=2)[..., None]

        intensity = np.tensordot(normals, illum, axes=(2, 0))
        intensity -= intensity.min()
        intensity /= intensity.ptp()
        return intensity

    y, x = np.mgrid[5:0:-1, :5]
    z = -np.hypot(x - x.mean(), y - y.mean())

    for az, elev in itertools.product(range(0, 390, 30), range(0, 105, 15)):
        ls = mcolors.LightSource(az, elev)
        h1 = ls.hillshade(z)
        h2 = alternative_hillshade(az, elev, z)
        assert_array_almost_equal(h1, h2)


def test_light_source_planar_hillshading():
    """
    Ensure that the illumination intensity is correct for planar surfaces.
    """

    def plane(azimuth, elevation, x, y):
        """
        Create a plane whose normal vector is at the given azimuth and
        elevation.
        """
        theta, phi = _azimuth2math(azimuth, elevation)
        a, b, c = _sph2cart(theta, phi)
        z = -(a*x + b*y) / c
        return z

    def angled_plane(azimuth, elevation, angle, x, y):
        """
        Create a plane whose normal vector is at an angle from the given
        azimuth and elevation.
        """
        elevation = elevation + angle
        if elevation > 90:
            azimuth = (azimuth + 180) % 360
            elevation = (90 - elevation) % 90
        return plane(azimuth, elevation, x, y)

    y, x = np.mgrid[5:0:-1, :5]
    for az, elev in itertools.product(range(0, 390, 30), range(0, 105, 15)):
        ls = mcolors.LightSource(az, elev)

        # Make a plane at a range of angles to the illumination
        for angle in range(0, 105, 15):
            z = angled_plane(az, elev, angle, x, y)
            h = ls.hillshade(z)
            assert_array_almost_equal(h, np.cos(np.radians(angle)))


def test_color_names():
    assert mcolors.to_hex("blue") == "#0000ff"
    assert mcolors.to_hex("xkcd:blue") == "#0343df"
    assert mcolors.to_hex("tab:blue") == "#1f77b4"


def _sph2cart(theta, phi):
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return x, y, z


def _azimuth2math(azimuth, elevation):
    """
    Convert from clockwise-from-north and up-from-horizontal to mathematical
    conventions.
    """
    theta = np.radians((90 - azimuth) % 360)
    phi = np.radians(90 - elevation)
    return theta, phi


def test_pandas_iterable(pd):
    # Using a list or series yields equivalent
    # colormaps, i.e the series isn't seen as
    # a single color
    lst = ['red', 'blue', 'green']
    s = pd.Series(lst)
    cm1 = mcolors.ListedColormap(lst, N=5)
    cm2 = mcolors.ListedColormap(s, N=5)
    assert_array_equal(cm1.colors, cm2.colors)


@pytest.mark.parametrize('name', sorted(plt.colormaps()))
def test_colormap_reversing(name):
    """
    Check the generated _lut data of a colormap and corresponding reversed
    colormap if they are almost the same.
    """
    cmap = plt.get_cmap(name)
    cmap_r = cmap.reversed()
    if not cmap_r._isinit:
        cmap._init()
        cmap_r._init()
    assert_array_almost_equal(cmap._lut[:-3], cmap_r._lut[-4::-1])
    # Test the bad, over, under values too
    assert_array_almost_equal(cmap(-np.inf), cmap_r(np.inf))
    assert_array_almost_equal(cmap(np.inf), cmap_r(-np.inf))
    assert_array_almost_equal(cmap(np.nan), cmap_r(np.nan))


def test_cn():
    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['blue', 'r'])
    assert mcolors.to_hex("C0") == '#0000ff'
    assert mcolors.to_hex("C1") == '#ff0000'

    matplotlib.rcParams['axes.prop_cycle'] = cycler('color',
                                                    ['xkcd:blue', 'r'])
    assert mcolors.to_hex("C0") == '#0343df'
    assert mcolors.to_hex("C1") == '#ff0000'
    assert mcolors.to_hex("C10") == '#0343df'
    assert mcolors.to_hex("C11") == '#ff0000'

    matplotlib.rcParams['axes.prop_cycle'] = cycler('color', ['8e4585', 'r'])

    assert mcolors.to_hex("C0") == '#8e4585'
    # if '8e4585' gets parsed as a float before it gets detected as a hex
    # colour it will be interpreted as a very large number.
    # this mustn't happen.
    assert mcolors.to_rgb("C0")[0] != np.inf


def test_conversions():
    # to_rgba_array("none") returns a (0, 4) array.
    assert_array_equal(mcolors.to_rgba_array("none"), np.zeros((0, 4)))
    assert_array_equal(mcolors.to_rgba_array([]), np.zeros((0, 4)))
    # a list of grayscale levels, not a single color.
    assert_array_equal(
        mcolors.to_rgba_array([".2", ".5", ".8"]),
        np.vstack([mcolors.to_rgba(c) for c in [".2", ".5", ".8"]]))
    # alpha is properly set.
    assert mcolors.to_rgba((1, 1, 1), .5) == (1, 1, 1, .5)
    assert mcolors.to_rgba(".1", .5) == (.1, .1, .1, .5)
    # builtin round differs between py2 and py3.
    assert mcolors.to_hex((.7, .7, .7)) == "#b2b2b2"
    # hex roundtrip.
    hex_color = "#1234abcd"
    assert mcolors.to_hex(mcolors.to_rgba(hex_color), keep_alpha=True) == \
        hex_color


def test_conversions_masked():
    x1 = np.ma.array(['k', 'b'], mask=[True, False])
    x2 = np.ma.array([[0, 0, 0, 1], [0, 0, 1, 1]])
    x2[0] = np.ma.masked
    assert mcolors.to_rgba(x1[0]) == (0, 0, 0, 0)
    assert_array_equal(mcolors.to_rgba_array(x1),
                       [[0, 0, 0, 0], [0, 0, 1, 1]])
    assert_array_equal(mcolors.to_rgba_array(x2), mcolors.to_rgba_array(x1))


def test_to_rgba_array_single_str():
    # single color name is valid
    assert_array_equal(mcolors.to_rgba_array("red"), [(1, 0, 0, 1)])

    # single char color sequence is invalid
    with pytest.raises(ValueError,
                       match="Using a string of single character colors as "
                             "a color sequence is not supported."):
        array = mcolors.to_rgba_array("rgb")


def test_to_rgba_array_alpha_array():
    with pytest.raises(ValueError, match="The number of colors must match"):
        mcolors.to_rgba_array(np.ones((5, 3), float), alpha=np.ones((2,)))
    alpha = [0.5, 0.6]
    c = mcolors.to_rgba_array(np.ones((2, 3), float), alpha=alpha)
    assert_array_equal(c[:, 3], alpha)
    c = mcolors.to_rgba_array(['r', 'g'], alpha=alpha)
    assert_array_equal(c[:, 3], alpha)


def test_failed_conversions():
    with pytest.raises(ValueError):
        mcolors.to_rgba('5')
    with pytest.raises(ValueError):
        mcolors.to_rgba('-1')
    with pytest.raises(ValueError):
        mcolors.to_rgba('nan')
    with pytest.raises(ValueError):
        mcolors.to_rgba('unknown_color')
    with pytest.raises(ValueError):
        # Gray must be a string to distinguish 3-4 grays from RGB or RGBA.
        mcolors.to_rgba(0.4)


def test_grey_gray():
    color_mapping = mcolors._colors_full_map
    for k in color_mapping.keys():
        if 'grey' in k:
            assert color_mapping[k] == color_mapping[k.replace('grey', 'gray')]
        if 'gray' in k:
            assert color_mapping[k] == color_mapping[k.replace('gray', 'grey')]


def test_tableau_order():
    dflt_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

    assert list(mcolors.TABLEAU_COLORS.values()) == dflt_cycle


def test_ndarray_subclass_norm():
    # Emulate an ndarray subclass that handles units
    # which objects when adding or subtracting with other
    # arrays. See #6622 and #8696
    class MyArray(np.ndarray):
        def __isub__(self, other):
            raise RuntimeError

        def __add__(self, other):
            raise RuntimeError

    data = np.arange(-10, 10, 1, dtype=float).reshape((10, 2))
    mydata = data.view(MyArray)

    for norm in [mcolors.Normalize(), mcolors.LogNorm(),
                 mcolors.SymLogNorm(3, vmax=5, linscale=1, base=np.e),
                 mcolors.Normalize(vmin=mydata.min(), vmax=mydata.max()),
                 mcolors.SymLogNorm(3, vmin=mydata.min(), vmax=mydata.max(),
                                    base=np.e),
                 mcolors.PowerNorm(1)]:
        assert_array_equal(norm(mydata), norm(data))
        fig, ax = plt.subplots()
        ax.imshow(mydata, norm=norm)
        fig.canvas.draw()  # Check that no warning is emitted.


def test_same_color():
    assert mcolors.same_color('k', (0, 0, 0))
    assert not mcolors.same_color('w', (1, 1, 0))
    assert mcolors.same_color(['red', 'blue'], ['r', 'b'])
    assert mcolors.same_color('none', 'none')
    assert not mcolors.same_color('none', 'red')
    with pytest.raises(ValueError):
        mcolors.same_color(['r', 'g', 'b'], ['r'])
    with pytest.raises(ValueError):
        mcolors.same_color(['red', 'green'], 'none')


def test_hex_shorthand_notation():
    assert mcolors.same_color("#123", "#112233")
    assert mcolors.same_color("#123a", "#112233aa")


def test_repr_png():
    cmap = plt.get_cmap('viridis')
    png = cmap._repr_png_()
    assert len(png) > 0
    img = Image.open(BytesIO(png))
    assert img.width > 0
    assert img.height > 0
    assert 'Title' in img.text
    assert 'Description' in img.text
    assert 'Author' in img.text
    assert 'Software' in img.text


def test_repr_html():
    cmap = plt.get_cmap('viridis')
    html = cmap._repr_html_()
    assert len(html) > 0
    png = cmap._repr_png_()
    assert base64.b64encode(png).decode('ascii') in html
    assert cmap.name in html
    assert html.startswith('<div')
    assert html.endswith('</div>')


def test_get_under_over_bad():
    cmap = plt.get_cmap('viridis')
    assert_array_equal(cmap.get_under(), cmap(-np.inf))
    assert_array_equal(cmap.get_over(), cmap(np.inf))
    assert_array_equal(cmap.get_bad(), cmap(np.nan))


@pytest.mark.parametrize('kind', ('over', 'under', 'bad'))
def test_non_mutable_get_values(kind):
    cmap = copy.copy(plt.get_cmap('viridis'))
    init_value = getattr(cmap, f'get_{kind}')()
    getattr(cmap, f'set_{kind}')('k')
    black_value = getattr(cmap, f'get_{kind}')()
    assert np.all(black_value == [0, 0, 0, 1])
    assert not np.all(init_value == black_value)


def test_colormap_alpha_array():
    cmap = plt.get_cmap('viridis')
    vals = [-1, 0.5, 2]  # under, valid, over
    with pytest.raises(ValueError, match="alpha is array-like but"):
        cmap(vals, alpha=[1, 1, 1, 1])
    alpha = np.array([0.1, 0.2, 0.3])
    c = cmap(vals, alpha=alpha)
    assert_array_equal(c[:, -1], alpha)
    c = cmap(vals, alpha=alpha, bytes=True)
    assert_array_equal(c[:, -1], (alpha * 255).astype(np.uint8))


def test_colormap_bad_data_with_alpha():
    cmap = plt.get_cmap('viridis')
    c = cmap(np.nan, alpha=0.5)
    assert c == (0, 0, 0, 0)
    c = cmap([0.5, np.nan], alpha=0.5)
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([0.5, np.nan], alpha=[0.1, 0.2])
    assert_array_equal(c[1], (0, 0, 0, 0))
    c = cmap([[np.nan, 0.5], [0, 0]], alpha=0.5)
    assert_array_equal(c[0, 0], (0, 0, 0, 0))
    c = cmap([[np.nan, 0.5], [0, 0]], alpha=np.full((2, 2), 0.5))
    assert_array_equal(c[0, 0], (0, 0, 0, 0))


def test_2d_to_rgba():
    color = np.array([0.1, 0.2, 0.3])
    rgba_1d = mcolors.to_rgba(color.reshape(-1))
    rgba_2d = mcolors.to_rgba(color.reshape((1, -1)))
    assert rgba_1d == rgba_2d


def test_set_dict_to_rgba():
    # downstream libraries do this...
    # note we can't test this because it is not well-ordered
    # so just smoketest:
    colors = set([(0, .5, 1), (1, .2, .5), (.4, 1, .2)])
    res = mcolors.to_rgba_array(colors)
    palette = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1)}
    res = mcolors.to_rgba_array(palette.values())
    exp = np.eye(3)
    np.testing.assert_array_almost_equal(res[:, :-1], exp)


def test_norm_deepcopy():
    norm = mcolors.LogNorm()
    norm.vmin = 0.0002
    norm2 = copy.deepcopy(norm)
    assert norm2.vmin == norm.vmin
    assert isinstance(norm2._scale, mscale.LogScale)
    norm = mcolors.Normalize()
    norm.vmin = 0.0002
    norm2 = copy.deepcopy(norm)
    assert norm2._scale is None
    assert norm2.vmin == norm.vmin


def test_norm_callback():
    increment = unittest.mock.Mock(return_value=None)

    norm = mcolors.Normalize()
    norm.callbacks.connect('changed', increment)
    # Haven't updated anything, so call count should be 0
    assert increment.call_count == 0

    # Now change vmin and vmax to test callbacks
    norm.vmin = 1
    assert increment.call_count == 1
    norm.vmax = 5
    assert increment.call_count == 2
    # callback shouldn't be called if setting to the same value
    norm.vmin = 1
    assert increment.call_count == 2
    norm.vmax = 5
    assert increment.call_count == 2


def test_scalarmappable_norm_update():
    norm = mcolors.Normalize()
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
    # sm doesn't have a stale attribute at first, set it to False
    sm.stale = False
    # The mappable should be stale after updating vmin/vmax
    norm.vmin = 5
    assert sm.stale
    sm.stale = False
    norm.vmax = 5
    assert sm.stale
    sm.stale = False
    norm.clip = True
    assert sm.stale
    # change to the CenteredNorm and TwoSlopeNorm to test those
    # Also make sure that updating the norm directly and with
    # set_norm both update the Norm callback
    norm = mcolors.CenteredNorm()
    sm.norm = norm
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    sm.set_norm(norm)
    sm.stale = False
    norm.vcenter = 1
    assert sm.stale


@check_figures_equal()
def test_norm_update_figs(fig_test, fig_ref):
    ax_ref = fig_ref.add_subplot()
    ax_test = fig_test.add_subplot()

    z = np.arange(100).reshape((10, 10))
    ax_ref.imshow(z, norm=mcolors.Normalize(10, 90))

    # Create the norm beforehand with different limits and then update
    # after adding to the plot
    norm = mcolors.Normalize(0, 1)
    ax_test.imshow(z, norm=norm)
    # Force initial draw to make sure it isn't already stale
    fig_test.canvas.draw()
    norm.vmin, norm.vmax = 10, 90


def test_make_norm_from_scale_name():
    logitnorm = mcolors.make_norm_from_scale(
        mscale.LogitScale, mcolors.Normalize)
    assert logitnorm.__name__ == "LogitScaleNorm"
