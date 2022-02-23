import numpy as np
import pytest

from matplotlib import cm
import matplotlib.colors as mcolors

from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm, LogNorm, PowerNorm, Normalize, NoNorm
)
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator
from matplotlib.testing.decorators import check_figures_equal


def _get_cmap_norms():
    """
    Define a colormap and appropriate norms for each of the four
    possible settings of the extend keyword.

    Helper function for _colorbar_extension_shape and
    colorbar_extension_length.
    """
    # Create a colormap and specify the levels it represents.
    cmap = cm.get_cmap("RdBu", lut=5)
    clevs = [-5., -2.5, -.5, .5, 1.5, 3.5]
    # Define norms for the colormaps.
    norms = dict()
    norms['neither'] = BoundaryNorm(clevs, len(clevs) - 1)
    norms['min'] = BoundaryNorm([-10] + clevs[1:], len(clevs) - 1)
    norms['max'] = BoundaryNorm(clevs[:-1] + [10], len(clevs) - 1)
    norms['both'] = BoundaryNorm([-10] + clevs[1:-1] + [10], len(clevs) - 1)
    return cmap, norms


def _colorbar_extension_shape(spacing):
    """
    Produce 4 colorbars with rectangular extensions for either uniform
    or proportional spacing.

    Helper function for test_colorbar_extension_shape.
    """
    # Get a colormap and appropriate norms for each extension type.
    cmap, norms = _get_cmap_norms()
    # Create a figure and adjust whitespace for subplots.
    fig = plt.figure()
    fig.subplots_adjust(hspace=4)
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # Get the appropriate norm and use it to get colorbar boundaries.
        norm = norms[extension_type]
        boundaries = values = norm.boundaries
        # note that the last value was silently dropped pre 3.3:
        values = values[:-1]
        # Create a subplot.
        cax = fig.add_subplot(4, 1, i + 1)
        # Generate the colorbar.
        Colorbar(cax, cmap=cmap, norm=norm,
                 boundaries=boundaries, values=values,
                 extend=extension_type, extendrect=True,
                 orientation='horizontal', spacing=spacing)
        # Turn off text and ticks.
        cax.tick_params(left=False, labelleft=False,
                        bottom=False, labelbottom=False)
    # Return the figure to the caller.
    return fig


def _colorbar_extension_length(spacing):
    """
    Produce 12 colorbars with variable length extensions for either
    uniform or proportional spacing.

    Helper function for test_colorbar_extension_length.
    """
    # Get a colormap and appropriate norms for each extension type.
    cmap, norms = _get_cmap_norms()
    # Create a figure and adjust whitespace for subplots.
    fig = plt.figure()
    fig.subplots_adjust(hspace=.6)
    for i, extension_type in enumerate(('neither', 'min', 'max', 'both')):
        # Get the appropriate norm and use it to get colorbar boundaries.
        norm = norms[extension_type]
        boundaries = values = norm.boundaries
        values = values[:-1]
        for j, extendfrac in enumerate((None, 'auto', 0.1)):
            # Create a subplot.
            cax = fig.add_subplot(12, 1, i*3 + j + 1)
            # Generate the colorbar.
            Colorbar(cax, cmap=cmap, norm=norm,
                     boundaries=boundaries, values=values,
                     extend=extension_type, extendfrac=extendfrac,
                     orientation='horizontal', spacing=spacing)
            # Turn off text and ticks.
            cax.tick_params(left=False, labelleft=False,
                              bottom=False, labelbottom=False)
    # Return the figure to the caller.
    return fig


@image_comparison(['colorbar_extensions_shape_uniform.png',
                   'colorbar_extensions_shape_proportional.png'])
def test_colorbar_extension_shape():
    """Test rectangular colorbar extensions."""
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    # Create figures for uniform and proportionally spaced colorbars.
    _colorbar_extension_shape('uniform')
    _colorbar_extension_shape('proportional')


@image_comparison(['colorbar_extensions_uniform.png',
                   'colorbar_extensions_proportional.png'],
                  tol=1.0)
def test_colorbar_extension_length():
    """Test variable length colorbar extensions."""
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    # Create figures for uniform and proportionally spaced colorbars.
    _colorbar_extension_length('uniform')
    _colorbar_extension_length('proportional')


@pytest.mark.parametrize('use_gridspec', [True, False])
@image_comparison(['cbar_with_orientation',
                   'cbar_locationing',
                   'double_cbar',
                   'cbar_sharing',
                   ],
                  extensions=['png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_colorbar_positioning(use_gridspec):
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    # -------------------
    plt.figure()
    plt.contourf(data, levels=levels)
    plt.colorbar(orientation='horizontal', use_gridspec=use_gridspec)

    locations = ['left', 'right', 'top', 'bottom']
    plt.figure()
    for i, location in enumerate(locations):
        plt.subplot(2, 2, i + 1)
        plt.contourf(data, levels=levels)
        plt.colorbar(location=location, use_gridspec=use_gridspec)

    # -------------------
    plt.figure()
    # make some other data (random integers)
    data_2nd = np.array([[2, 3, 2, 3], [1.5, 2, 2, 3], [2, 3, 3, 4]])
    # make the random data expand to the shape of the main data
    data_2nd = np.repeat(np.repeat(data_2nd, 10, axis=1), 10, axis=0)

    color_mappable = plt.contourf(data, levels=levels, extend='both')
    # test extend frac here
    hatch_mappable = plt.contourf(data_2nd, levels=[1, 2, 3], colors='none',
                                  hatches=['/', 'o', '+'], extend='max')
    plt.contour(hatch_mappable, colors='black')

    plt.colorbar(color_mappable, location='left', label='variable 1',
                 use_gridspec=use_gridspec)
    plt.colorbar(hatch_mappable, location='right', label='variable 2',
                 use_gridspec=use_gridspec)

    # -------------------
    plt.figure()
    ax1 = plt.subplot(211, anchor='NE', aspect='equal')
    plt.contourf(data, levels=levels)
    ax2 = plt.subplot(223)
    plt.contourf(data, levels=levels)
    ax3 = plt.subplot(224)
    plt.contourf(data, levels=levels)

    plt.colorbar(ax=[ax2, ax3, ax1], location='right', pad=0.0, shrink=0.5,
                 panchor=False, use_gridspec=use_gridspec)
    plt.colorbar(ax=[ax2, ax3, ax1], location='left', shrink=0.5,
                 panchor=False, use_gridspec=use_gridspec)
    plt.colorbar(ax=[ax1], location='bottom', panchor=False,
                 anchor=(0.8, 0.5), shrink=0.6, use_gridspec=use_gridspec)


@image_comparison(['contour_colorbar.png'], remove_text=True)
def test_contour_colorbar():
    fig, ax = plt.subplots(figsize=(4, 2))
    data = np.arange(1200).reshape(30, 40) - 500
    levels = np.array([0, 200, 400, 600, 800, 1000, 1200]) - 500

    CS = ax.contour(data, levels=levels, extend='both')
    fig.colorbar(CS, orientation='horizontal', extend='both')
    fig.colorbar(CS, orientation='vertical')


@image_comparison(['cbar_with_subplots_adjust.png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_gridspec_make_colorbar():
    plt.figure()
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    plt.subplot(121)
    plt.contourf(data, levels=levels)
    plt.colorbar(use_gridspec=True, orientation='vertical')

    plt.subplot(122)
    plt.contourf(data, levels=levels)
    plt.colorbar(use_gridspec=True, orientation='horizontal')

    plt.subplots_adjust(top=0.95, right=0.95, bottom=0.2, hspace=0.25)


@image_comparison(['colorbar_single_scatter.png'], remove_text=True,
                  savefig_kwarg={'dpi': 40})
def test_colorbar_single_scatter():
    # Issue #2642: if a path collection has only one entry,
    # the norm scaling within the colorbar must ensure a
    # finite range, otherwise a zero denominator will occur in _locate.
    plt.figure()
    x = y = [0]
    z = [50]
    cmap = plt.get_cmap('jet', 16)
    cs = plt.scatter(x, y, z, c=z, cmap=cmap)
    plt.colorbar(cs)


@pytest.mark.parametrize('use_gridspec', [False, True],
                         ids=['no gridspec', 'with gridspec'])
def test_remove_from_figure(use_gridspec):
    """
    Test `remove` with the specified ``use_gridspec`` setting
    """
    fig, ax = plt.subplots()
    sc = ax.scatter([1, 2], [3, 4], cmap="spring")
    sc.set_array(np.array([5, 6]))
    pre_position = ax.get_position()
    cb = fig.colorbar(sc, use_gridspec=use_gridspec)
    fig.subplots_adjust()
    cb.remove()
    fig.subplots_adjust()
    post_position = ax.get_position()
    assert (pre_position.get_points() == post_position.get_points()).all()


def test_remove_from_figure_cl():
    """
    Test `remove` with constrained_layout
    """
    fig, ax = plt.subplots(constrained_layout=True)
    sc = ax.scatter([1, 2], [3, 4], cmap="spring")
    sc.set_array(np.array([5, 6]))
    fig.draw_without_rendering()
    pre_position = ax.get_position()
    cb = fig.colorbar(sc)
    cb.remove()
    fig.draw_without_rendering()
    post_position = ax.get_position()
    np.testing.assert_allclose(pre_position.get_points(),
                               post_position.get_points())


def test_colorbarbase():
    # smoke test from #3805
    ax = plt.gca()
    Colorbar(ax, cmap=plt.cm.bone)


@image_comparison(['colorbar_closed_patch.png'], remove_text=True)
def test_colorbar_closed_patch():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_axes([0.05, 0.85, 0.9, 0.1])
    ax2 = fig.add_axes([0.1, 0.65, 0.75, 0.1])
    ax3 = fig.add_axes([0.05, 0.45, 0.9, 0.1])
    ax4 = fig.add_axes([0.05, 0.25, 0.9, 0.1])
    ax5 = fig.add_axes([0.05, 0.05, 0.9, 0.1])

    cmap = cm.get_cmap("RdBu", lut=5)

    im = ax1.pcolormesh(np.linspace(0, 10, 16).reshape((4, 4)), cmap=cmap)

    # The use of a "values" kwarg here is unusual.  It works only
    # because it is matched to the data range in the image and to
    # the number of colors in the LUT.
    values = np.linspace(0, 10, 5)
    cbar_kw = dict(orientation='horizontal', values=values, ticks=[])

    # The wide line is to show that the closed path is being handled
    # correctly.  See PR #4186.
    with rc_context({'axes.linewidth': 16}):
        plt.colorbar(im, cax=ax2, extend='both', extendfrac=0.5, **cbar_kw)
        plt.colorbar(im, cax=ax3, extend='both', **cbar_kw)
        plt.colorbar(im, cax=ax4, extend='both', extendrect=True, **cbar_kw)
        plt.colorbar(im, cax=ax5, extend='neither', **cbar_kw)


def test_colorbar_ticks():
    # test fix for #5673
    fig, ax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    clevs = np.array([-12, -5, 0, 5, 12], dtype=float)
    colors = ['r', 'g', 'b', 'c']
    cs = ax.contourf(X, Y, Z, clevs, colors=colors, extend='neither')
    cbar = fig.colorbar(cs, ax=ax, orientation='horizontal', ticks=clevs)
    assert len(cbar.ax.xaxis.get_ticklocs()) == len(clevs)


def test_colorbar_minorticks_on_off():
    # test for github issue #11510 and PR #11584
    np.random.seed(seed=12345)
    data = np.random.randn(20, 20)
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots()
        # purposefully setting vmin and vmax to odd fractions
        # so as to check for the correct locations of the minor ticks
        im = ax.pcolormesh(data, vmin=-2.3, vmax=3.3)

        cbar = fig.colorbar(im, extend='both')
        # testing after minorticks_on()
        cbar.minorticks_on()
        np.testing.assert_almost_equal(
            cbar.ax.yaxis.get_minorticklocs(),
            [-2.2, -1.8, -1.6, -1.4, -1.2, -0.8, -0.6, -0.4, -0.2,
             0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.2, 2.4, 2.6, 2.8, 3.2])
        # testing after minorticks_off()
        cbar.minorticks_off()
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_minorticklocs(), [])

        im.set_clim(vmin=-1.2, vmax=1.2)
        cbar.minorticks_on()
        np.testing.assert_almost_equal(
            cbar.ax.yaxis.get_minorticklocs(),
            [-1.1, -0.9, -0.8, -0.7, -0.6, -0.4, -0.3, -0.2, -0.1,
             0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3])

    # tests for github issue #13257 and PR #13265
    data = np.random.uniform(low=1, high=10, size=(20, 20))

    fig, ax = plt.subplots()
    im = ax.pcolormesh(data, norm=LogNorm())
    cbar = fig.colorbar(im)
    fig.canvas.draw()
    default_minorticklocks = cbar.ax.yaxis.get_minorticklocs()
    # test that minorticks turn off for LogNorm
    cbar.minorticks_off()
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(), [])

    # test that minorticks turn back on for LogNorm
    cbar.minorticks_on()
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(),
                            default_minorticklocks)

    # test issue #13339: minorticks for LogNorm should stay off
    cbar.minorticks_off()
    cbar.set_ticks([3, 5, 7, 9])
    np.testing.assert_equal(cbar.ax.yaxis.get_minorticklocs(), [])


def test_cbar_minorticks_for_rc_xyminortickvisible():
    """
    issue gh-16468.

    Making sure that minor ticks on the colorbar are turned on
    (internally) using the cbar.minorticks_on() method when
    rcParams['xtick.minor.visible'] = True (for horizontal cbar)
    rcParams['ytick.minor.visible'] = True (for vertical cbar).
    Using cbar.minorticks_on() ensures that the minor ticks
    don't overflow into the extend regions of the colorbar.
    """

    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.visible'] = True

    vmin, vmax = 0.4, 2.6
    fig, ax = plt.subplots()
    im = ax.pcolormesh([[1, 2]], vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(im, extend='both', orientation='vertical')
    assert cbar.ax.yaxis.get_minorticklocs()[0] >= vmin
    assert cbar.ax.yaxis.get_minorticklocs()[-1] <= vmax

    cbar = fig.colorbar(im, extend='both', orientation='horizontal')
    assert cbar.ax.xaxis.get_minorticklocs()[0] >= vmin
    assert cbar.ax.xaxis.get_minorticklocs()[-1] <= vmax


def test_colorbar_autoticks():
    # Test new autotick modes. Needs to be classic because
    # non-classic doesn't go this route.
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(2, 1)
        x = np.arange(-3.0, 4.001)
        y = np.arange(-4.0, 3.001)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        Z = Z[:-1, :-1]
        pcm = ax[0].pcolormesh(X, Y, Z)
        cbar = fig.colorbar(pcm, ax=ax[0], extend='both',
                            orientation='vertical')

        pcm = ax[1].pcolormesh(X, Y, Z)
        cbar2 = fig.colorbar(pcm, ax=ax[1], extend='both',
                             orientation='vertical', shrink=0.4)
        # note only -10 to 10 are visible,
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_ticklocs(),
                                       np.arange(-15, 16, 5))
        # note only -10 to 10 are visible
        np.testing.assert_almost_equal(cbar2.ax.yaxis.get_ticklocs(),
                                       np.arange(-20, 21, 10))


def test_colorbar_autotickslog():
    # Test new autotick modes...
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(2, 1)
        x = np.arange(-3.0, 4.001)
        y = np.arange(-4.0, 3.001)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        Z = Z[:-1, :-1]
        pcm = ax[0].pcolormesh(X, Y, 10**Z, norm=LogNorm())
        cbar = fig.colorbar(pcm, ax=ax[0], extend='both',
                            orientation='vertical')

        pcm = ax[1].pcolormesh(X, Y, 10**Z, norm=LogNorm())
        cbar2 = fig.colorbar(pcm, ax=ax[1], extend='both',
                             orientation='vertical', shrink=0.4)
        # note only -12 to +12 are visible
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_ticklocs(),
                                       10**np.arange(-16., 16.2, 4.))
        # note only -24 to +24 are visible
        np.testing.assert_almost_equal(cbar2.ax.yaxis.get_ticklocs(),
                                       10**np.arange(-24., 25., 12.))


def test_colorbar_get_ticks():
    # test feature for #5792
    plt.figure()
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]

    plt.contourf(data, levels=levels)

    # testing getter for user set ticks
    userTicks = plt.colorbar(ticks=[0, 600, 1200])
    assert userTicks.get_ticks().tolist() == [0, 600, 1200]

    # testing for getter after calling set_ticks
    userTicks.set_ticks([600, 700, 800])
    assert userTicks.get_ticks().tolist() == [600, 700, 800]

    # testing for getter after calling set_ticks with some ticks out of bounds
    # removed #20054: other axes don't trim fixed lists, so colorbars
    # should not either:
    # userTicks.set_ticks([600, 1300, 1400, 1500])
    # assert userTicks.get_ticks().tolist() == [600]

    # testing getter when no ticks are assigned
    defTicks = plt.colorbar(orientation='horizontal')
    np.testing.assert_allclose(defTicks.get_ticks().tolist(), levels)

    # test normal ticks and minor ticks
    fig, ax = plt.subplots()
    x = np.arange(-3.0, 4.001)
    y = np.arange(-4.0, 3.001)
    X, Y = np.meshgrid(x, y)
    Z = X * Y
    Z = Z[:-1, :-1]
    pcm = ax.pcolormesh(X, Y, Z)
    cbar = fig.colorbar(pcm, ax=ax, extend='both',
                        orientation='vertical')
    ticks = cbar.get_ticks()
    np.testing.assert_allclose(ticks, np.arange(-15, 16, 5))
    assert len(cbar.get_ticks(minor=True)) == 0


@pytest.mark.parametrize("extend", ['both', 'min', 'max'])
def test_colorbar_lognorm_extension(extend):
    # Test that colorbar with lognorm is extended correctly
    f, ax = plt.subplots()
    cb = Colorbar(ax, norm=LogNorm(vmin=0.1, vmax=1000.0),
                  orientation='vertical', extend=extend)
    assert cb._values[0] >= 0.0


def test_colorbar_powernorm_extension():
    # Test that colorbar with powernorm is extended correctly
    f, ax = plt.subplots()
    cb = Colorbar(ax, norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0),
                  orientation='vertical', extend='both')
    assert cb._values[0] >= 0.0


def test_colorbar_axes_kw():
    # test fix for #8493: This does only test, that axes-related keywords pass
    # and do not raise an exception.
    plt.figure()
    plt.imshow([[1, 2], [3, 4]])
    plt.colorbar(orientation='horizontal', fraction=0.2, pad=0.2, shrink=0.5,
                 aspect=10, anchor=(0., 0.), panchor=(0., 1.))


def test_colorbar_log_minortick_labels():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots()
        pcm = ax.imshow([[10000, 50000]], norm=LogNorm())
        cb = fig.colorbar(pcm)
        fig.canvas.draw()
        lb = [l.get_text() for l in cb.ax.yaxis.get_ticklabels(which='both')]
        expected = [r'$\mathdefault{10^{4}}$',
                    r'$\mathdefault{2\times10^{4}}$',
                    r'$\mathdefault{3\times10^{4}}$',
                    r'$\mathdefault{4\times10^{4}}$']
        for exp in expected:
            assert exp in lb


def test_colorbar_renorm():
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = 120000*np.exp(-x**2 - y**2)

    fig, ax = plt.subplots()
    im = ax.imshow(z)
    cbar = fig.colorbar(im)
    np.testing.assert_allclose(cbar.ax.yaxis.get_majorticklocs(),
                               np.arange(0, 120000.1, 20000))

    cbar.set_ticks([1, 2, 3])
    assert isinstance(cbar.locator, FixedLocator)

    norm = LogNorm(z.min(), z.max())
    im.set_norm(norm)
    np.testing.assert_allclose(cbar.ax.yaxis.get_majorticklocs(),
                               np.logspace(-10, 7, 18))
    # note that set_norm removes the FixedLocator...
    assert np.isclose(cbar.vmin, z.min())
    cbar.set_ticks([1, 2, 3])
    assert isinstance(cbar.locator, FixedLocator)
    np.testing.assert_allclose(cbar.ax.yaxis.get_majorticklocs(),
                               [1.0, 2.0, 3.0])

    norm = LogNorm(z.min() * 1000, z.max() * 1000)
    im.set_norm(norm)
    assert np.isclose(cbar.vmin, z.min() * 1000)
    assert np.isclose(cbar.vmax, z.max() * 1000)


def test_colorbar_format():
    # make sure that format is passed properly
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = 120000*np.exp(-x**2 - y**2)

    fig, ax = plt.subplots()
    im = ax.imshow(z)
    cbar = fig.colorbar(im, format='%4.2e')
    fig.canvas.draw()
    assert cbar.ax.yaxis.get_ticklabels()[4].get_text() == '8.00e+04'

    # make sure that if we change the clim of the mappable that the
    # formatting is *not* lost:
    im.set_clim([4, 200])
    fig.canvas.draw()
    assert cbar.ax.yaxis.get_ticklabels()[4].get_text() == '2.00e+02'

    # but if we change the norm:
    im.set_norm(LogNorm(vmin=0.1, vmax=10))
    fig.canvas.draw()
    assert (cbar.ax.yaxis.get_ticklabels()[0].get_text() ==
            r'$\mathdefault{10^{-2}}$')


def test_colorbar_scale_reset():
    x, y = np.ogrid[-4:4:31j, -4:4:31j]
    z = 120000*np.exp(-x**2 - y**2)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(z, cmap='RdBu_r', rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.outline.set_edgecolor('red')
    assert cbar.ax.yaxis.get_scale() == 'linear'

    pcm.set_norm(LogNorm(vmin=1, vmax=100))
    assert cbar.ax.yaxis.get_scale() == 'log'
    pcm.set_norm(Normalize(vmin=-20, vmax=20))
    assert cbar.ax.yaxis.get_scale() == 'linear'

    assert cbar.outline.get_edgecolor() == mcolors.to_rgba('red')


def test_colorbar_get_ticks_2():
    plt.rcParams['_internal.classic_mode'] = False
    fig, ax = plt.subplots()
    pc = ax.pcolormesh([[.05, .95]])
    cb = fig.colorbar(pc)
    np.testing.assert_allclose(cb.get_ticks(), [0., 0.2, 0.4, 0.6, 0.8, 1.0])


def test_colorbar_inverted_ticks():
    fig, axs = plt.subplots(2)
    ax = axs[0]
    pc = ax.pcolormesh(10**np.arange(1, 5).reshape(2, 2), norm=LogNorm())
    cbar = fig.colorbar(pc, ax=ax, extend='both')
    ticks = cbar.get_ticks()
    cbar.ax.invert_yaxis()
    np.testing.assert_allclose(ticks, cbar.get_ticks())

    ax = axs[1]
    pc = ax.pcolormesh(np.arange(1, 5).reshape(2, 2))
    cbar = fig.colorbar(pc, ax=ax, extend='both')
    cbar.minorticks_on()
    ticks = cbar.get_ticks()
    minorticks = cbar.get_ticks(minor=True)
    cbar.ax.invert_yaxis()
    np.testing.assert_allclose(ticks, cbar.get_ticks())
    np.testing.assert_allclose(minorticks, cbar.get_ticks(minor=True))


def test_mappable_no_alpha():
    fig, ax = plt.subplots()
    sm = cm.ScalarMappable(norm=mcolors.Normalize(), cmap='viridis')
    fig.colorbar(sm)
    sm.set_cmap('plasma')
    plt.draw()


def test_mappable_2d_alpha():
    fig, ax = plt.subplots()
    x = np.arange(1, 5).reshape(2, 2)/4
    pc = ax.pcolormesh(x, alpha=x)
    cb = fig.colorbar(pc, ax=ax)
    # The colorbar's alpha should be None and the mappable should still have
    # the original alpha array
    assert cb.alpha is None
    assert pc.get_alpha() is x
    fig.draw_without_rendering()


def test_colorbar_label():
    """
    Test the label parameter. It should just be mapped to the xlabel/ylabel of
    the axes, depending on the orientation.
    """
    fig, ax = plt.subplots()
    im = ax.imshow([[1, 2], [3, 4]])
    cbar = fig.colorbar(im, label='cbar')
    assert cbar.ax.get_ylabel() == 'cbar'
    cbar.set_label(None)
    assert cbar.ax.get_ylabel() == ''
    cbar.set_label('cbar 2')
    assert cbar.ax.get_ylabel() == 'cbar 2'

    cbar2 = fig.colorbar(im, label=None)
    assert cbar2.ax.get_ylabel() == ''

    cbar3 = fig.colorbar(im, orientation='horizontal', label='horizontal cbar')
    assert cbar3.ax.get_xlabel() == 'horizontal cbar'


@pytest.mark.parametrize("clim", [(-20000, 20000), (-32768, 0)])
def test_colorbar_int(clim):
    # Check that we cast to float early enough to not
    # overflow ``int16(20000) - int16(-20000)`` or
    # run into ``abs(int16(-32768)) == -32768``.
    fig, ax = plt.subplots()
    im = ax.imshow([[*map(np.int16, clim)]])
    fig.colorbar(im)
    assert (im.norm.vmin, im.norm.vmax) == clim


def test_anchored_cbar_position_using_specgrid():
    data = np.arange(1200).reshape(30, 40)
    levels = [0, 200, 400, 600, 800, 1000, 1200]
    shrink = 0.5
    anchor_y = 0.3
    # right
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='right', anchor=(1, anchor_y), shrink=shrink)

    # the bottom left corner of one ax is (x0, y0)
    # the top right corner of one ax is (x1, y1)
    # p0: the vertical / horizontal position of anchor
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor_y + y0

    np.testing.assert_allclose(
            [cy1, cy0],
            [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])

    # left
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='left', anchor=(1, anchor_y), shrink=shrink)

    # the bottom left corner of one ax is (x0, y0)
    # the top right corner of one ax is (x1, y1)
    # p0: the vertical / horizontal position of anchor
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (y1 - y0) * anchor_y + y0

    np.testing.assert_allclose(
            [cy1, cy0],
            [y1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + y0 * shrink])

    # top
    shrink = 0.5
    anchor_x = 0.3
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='top', anchor=(anchor_x, 1), shrink=shrink)

    # the bottom left corner of one ax is (x0, y0)
    # the top right corner of one ax is (x1, y1)
    # p0: the vertical / horizontal position of anchor
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor_x + x0

    np.testing.assert_allclose(
            [cx1, cx0],
            [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])

    # bottom
    shrink = 0.5
    anchor_x = 0.3
    fig, ax = plt.subplots()
    cs = ax.contourf(data, levels=levels)
    cbar = plt.colorbar(
            cs, ax=ax, use_gridspec=True,
            location='bottom', anchor=(anchor_x, 1), shrink=shrink)

    # the bottom left corner of one ax is (x0, y0)
    # the top right corner of one ax is (x1, y1)
    # p0: the vertical / horizontal position of anchor
    x0, y0, x1, y1 = ax.get_position().extents
    cx0, cy0, cx1, cy1 = cbar.ax.get_position().extents
    p0 = (x1 - x0) * anchor_x + x0

    np.testing.assert_allclose(
            [cx1, cx0],
            [x1 * shrink + (1 - shrink) * p0, p0 * (1 - shrink) + x0 * shrink])


@image_comparison(['colorbar_change_lim_scale.png'], remove_text=True,
                  style='mpl20')
def test_colorbar_change_lim_scale():
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    pc = ax[0].pcolormesh(np.arange(100).reshape(10, 10)+1)
    cb = fig.colorbar(pc, ax=ax[0], extend='both')
    cb.ax.set_yscale('log')

    pc = ax[1].pcolormesh(np.arange(100).reshape(10, 10)+1)
    cb = fig.colorbar(pc, ax=ax[1], extend='both')
    cb.ax.set_ylim([20, 90])


@check_figures_equal(extensions=["png"])
def test_axes_handles_same_functions(fig_ref, fig_test):
    # prove that cax and cb.ax are functionally the same
    for nn, fig in enumerate([fig_ref, fig_test]):
        ax = fig.add_subplot()
        pc = ax.pcolormesh(np.ones(300).reshape(10, 30))
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb = fig.colorbar(pc, cax=cax)
        if nn == 0:
            caxx = cax
        else:
            caxx = cb.ax
        caxx.set_yticks(np.arange(0, 20))
        caxx.set_yscale('log')
        caxx.set_position([0.92, 0.1, 0.02, 0.7])


def test_inset_colorbar_layout():
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3, 6))
    pc = ax.imshow(np.arange(100).reshape(10, 10))
    cax = ax.inset_axes([1.02, 0.1, 0.03, 0.8])
    cb = fig.colorbar(pc, cax=cax)

    fig.draw_without_rendering()
    # make sure this is in the figure. In the colorbar swapping
    # it was being dropped from the list of children...
    np.testing.assert_allclose(cb.ax.get_position().bounds,
                               [0.87, 0.342, 0.0237, 0.315], atol=0.01)
    assert cb.ax in ax.child_axes


@image_comparison(['colorbar_twoslope.png'], remove_text=True,
                  style='mpl20')
def test_twoslope_colorbar():
    # Note that the second tick = 20, and should be in the middle
    # of the colorbar (white)
    # There should be no tick right at the bottom, nor at the top.
    fig, ax = plt.subplots()

    norm = mcolors.TwoSlopeNorm(20, 5, 95)
    pc = ax.pcolormesh(np.arange(1, 11), np.arange(1, 11),
                       np.arange(100).reshape(10, 10),
                       norm=norm, cmap='RdBu_r')
    fig.colorbar(pc)


@check_figures_equal(extensions=["png"])
def test_remove_cb_whose_mappable_has_no_figure(fig_ref, fig_test):
    ax = fig_test.add_subplot()
    cb = fig_test.colorbar(cm.ScalarMappable(), cax=ax)
    cb.remove()


def test_aspects():
    fig, ax = plt.subplots(3, 2, figsize=(8, 8))
    aspects = [20, 20, 10]
    extends = ['neither', 'both', 'both']
    cb = [[None, None, None], [None, None, None]]
    for nn, orient in enumerate(['vertical', 'horizontal']):
        for mm, (aspect, extend) in enumerate(zip(aspects, extends)):
            pc = ax[mm, nn].pcolormesh(np.arange(100).reshape(10, 10))
            cb[nn][mm] = fig.colorbar(pc, ax=ax[mm, nn], orientation=orient,
                                      aspect=aspect, extend=extend)
    fig.draw_without_rendering()
    # check the extends are right ratio:
    np.testing.assert_almost_equal(cb[0][1].ax.get_position().height,
                                   cb[0][0].ax.get_position().height * 0.9,
                                   decimal=2)
    # horizontal
    np.testing.assert_almost_equal(cb[1][1].ax.get_position().width,
                                   cb[1][0].ax.get_position().width * 0.9,
                                   decimal=2)
    # check correct aspect:
    pos = cb[0][0].ax.get_position(original=False)
    np.testing.assert_almost_equal(pos.height, pos.width * 20, decimal=2)
    pos = cb[1][0].ax.get_position(original=False)
    np.testing.assert_almost_equal(pos.height * 20, pos.width, decimal=2)
    # check twice as wide if aspect is 10 instead of 20
    np.testing.assert_almost_equal(
        cb[0][0].ax.get_position(original=False).width * 2,
        cb[0][2].ax.get_position(original=False).width, decimal=2)
    np.testing.assert_almost_equal(
        cb[1][0].ax.get_position(original=False).height * 2,
        cb[1][2].ax.get_position(original=False).height, decimal=2)


@image_comparison(['proportional_colorbars.png'], remove_text=True,
                  style='mpl20')
def test_proportional_colorbars():

    x = y = np.arange(-3.0, 3.01, 0.025)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2) * 2

    levels = [-1.25, -0.5, -0.125, 0.125, 0.5, 1.25]
    cmap = mcolors.ListedColormap(
        ['0.3', '0.5', 'white', 'lightblue', 'steelblue'])
    cmap.set_under('darkred')
    cmap.set_over('crimson')
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    extends = ['neither', 'both']
    spacings = ['uniform', 'proportional']
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            CS3 = axs[i, j].contourf(X, Y, Z, levels, cmap=cmap, norm=norm,
                                     extend=extends[i])
            fig.colorbar(CS3, spacing=spacings[j], ax=axs[i, j])


def test_negative_boundarynorm():
    fig, ax = plt.subplots(figsize=(1, 3))
    cmap = plt.get_cmap("viridis")

    clevs = np.arange(-94, -85)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)

    clevs = np.arange(85, 94)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)

    clevs = np.arange(-3, 3)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)

    clevs = np.arange(-8, 1)
    norm = BoundaryNorm(clevs, cmap.N)
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=ax)
    np.testing.assert_allclose(cb.ax.get_ylim(), [clevs[0], clevs[-1]])
    np.testing.assert_allclose(cb.ax.get_yticks(), clevs)


@image_comparison(['nonorm_colorbars.svg'], remove_text=False,
                  style='mpl20')
def test_nonorm():
    plt.rcParams['svg.fonttype'] = 'none'
    data = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = NoNorm(vmin=min(data), vmax=max(data))
    cmap = cm.get_cmap("viridis", len(data))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, cax=ax, orientation="horizontal")


@image_comparison(['test_boundaries.png'], remove_text=True,
                  style='mpl20')
def test_boundaries():
    np.random.seed(seed=19680808)
    fig, ax = plt.subplots(figsize=(2, 2))
    pc = ax.pcolormesh(np.random.randn(10, 10), cmap='RdBu_r')
    cb = fig.colorbar(pc, ax=ax, boundaries=np.linspace(-3, 3, 7))
