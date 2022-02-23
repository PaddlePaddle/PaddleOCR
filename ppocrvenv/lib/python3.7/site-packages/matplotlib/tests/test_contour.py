import datetime
import platform
import re

import numpy as np
from numpy.testing import assert_array_almost_equal
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
from matplotlib import pyplot as plt, rc_context
from matplotlib.colors import LogNorm
import pytest


def test_contour_shape_1d_valid():

    x = np.arange(10)
    y = np.arange(9)
    z = np.random.random((9, 10))

    fig, ax = plt.subplots()
    ax.contour(x, y, z)


def test_contour_shape_2d_valid():

    x = np.arange(10)
    y = np.arange(9)
    xg, yg = np.meshgrid(x, y)
    z = np.random.random((9, 10))

    fig, ax = plt.subplots()
    ax.contour(xg, yg, z)


@pytest.mark.parametrize("args, message", [
    ((np.arange(9), np.arange(9), np.empty((9, 10))),
     'Length of x (9) must match number of columns in z (10)'),
    ((np.arange(10), np.arange(10), np.empty((9, 10))),
     'Length of y (10) must match number of rows in z (9)'),
    ((np.empty((10, 10)), np.arange(10), np.empty((9, 10))),
     'Number of dimensions of x (2) and y (1) do not match'),
    ((np.arange(10), np.empty((10, 10)), np.empty((9, 10))),
     'Number of dimensions of x (1) and y (2) do not match'),
    ((np.empty((9, 9)), np.empty((9, 10)), np.empty((9, 10))),
     'Shapes of x (9, 9) and z (9, 10) do not match'),
    ((np.empty((9, 10)), np.empty((9, 9)), np.empty((9, 10))),
     'Shapes of y (9, 9) and z (9, 10) do not match'),
    ((np.empty((3, 3, 3)), np.empty((3, 3, 3)), np.empty((9, 10))),
     'Inputs x and y must be 1D or 2D, not 3D'),
    ((np.empty((3, 3, 3)), np.empty((3, 3, 3)), np.empty((3, 3, 3))),
     'Input z must be 2D, not 3D'),
    (([[0]],),  # github issue 8197
     'Input z must be at least a (2, 2) shaped array, but has shape (1, 1)'),
    (([0], [0], [[0]]),
     'Input z must be at least a (2, 2) shaped array, but has shape (1, 1)'),
])
def test_contour_shape_error(args, message):
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match=re.escape(message)):
        ax.contour(*args)


def test_contour_empty_levels():

    x = np.arange(9)
    z = np.random.random((9, 9))

    fig, ax = plt.subplots()
    with pytest.warns(UserWarning) as record:
        ax.contour(x, x, z, levels=[])
    assert len(record) == 1


def test_contour_Nlevels():
    # A scalar levels arg or kwarg should trigger auto level generation.
    # https://github.com/matplotlib/matplotlib/issues/11913
    z = np.arange(12).reshape((3, 4))
    fig, ax = plt.subplots()
    cs1 = ax.contour(z, 5)
    assert len(cs1.levels) > 1
    cs2 = ax.contour(z, levels=5)
    assert (cs1.levels == cs2.levels).all()


def test_contour_badlevel_fmt():
    # Test edge case from https://github.com/matplotlib/matplotlib/issues/9742
    # User supplied fmt for each level as a dictionary, but Matplotlib changed
    # the level to the minimum data value because no contours possible.
    # This was fixed in https://github.com/matplotlib/matplotlib/pull/9743
    x = np.arange(9)
    z = np.zeros((9, 9))

    fig, ax = plt.subplots()
    fmt = {1.: '%1.2f'}
    with pytest.warns(UserWarning) as record:
        cs = ax.contour(x, x, z, levels=[1.])
        ax.clabel(cs, fmt=fmt)
    assert len(record) == 1


def test_contour_uniform_z():

    x = np.arange(9)
    z = np.ones((9, 9))

    fig, ax = plt.subplots()
    with pytest.warns(UserWarning) as record:
        ax.contour(x, x, z)
    assert len(record) == 1


@image_comparison(['contour_manual_labels'], remove_text=True, style='mpl20')
def test_contour_manual_labels():
    x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    z = np.max(np.dstack([abs(x), abs(y)]), 2)

    plt.figure(figsize=(6, 2), dpi=200)
    cs = plt.contour(x, y, z)
    pts = np.array([(1.0, 3.0), (1.0, 4.4), (1.0, 6.0)])
    plt.clabel(cs, manual=pts)
    pts = np.array([(2.0, 3.0), (2.0, 4.4), (2.0, 6.0)])
    plt.clabel(cs, manual=pts, fontsize='small', colors=('r', 'g'))


@image_comparison(['contour_manual_colors_and_levels.png'], remove_text=True)
def test_given_colors_levels_and_extends():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    _, axs = plt.subplots(2, 4)

    data = np.arange(12).reshape(3, 4)

    colors = ['red', 'yellow', 'pink', 'blue', 'black']
    levels = [2, 4, 8, 10]

    for i, ax in enumerate(axs.flat):
        filled = i % 2 == 0.
        extend = ['neither', 'min', 'max', 'both'][i // 2]

        if filled:
            # If filled, we have 3 colors with no extension,
            # 4 colors with one extension, and 5 colors with both extensions
            first_color = 1 if extend in ['max', 'neither'] else None
            last_color = -1 if extend in ['min', 'neither'] else None
            c = ax.contourf(data, colors=colors[first_color:last_color],
                            levels=levels, extend=extend)
        else:
            # If not filled, we have 4 levels and 4 colors
            c = ax.contour(data, colors=colors[:-1],
                           levels=levels, extend=extend)

        plt.colorbar(c, ax=ax)


@image_comparison(['contour_datetime_axis.png'],
                  remove_text=False, style='mpl20')
def test_contour_datetime_axis():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=.15)
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(20)])
    y = np.arange(20)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    plt.subplot(221)
    plt.contour(x, y, z)
    plt.subplot(222)
    plt.contourf(x, y, z)
    x = np.repeat(x[np.newaxis], 20, axis=0)
    y = np.repeat(y[:, np.newaxis], 20, axis=1)
    plt.subplot(223)
    plt.contour(x, y, z)
    plt.subplot(224)
    plt.contourf(x, y, z)
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)


@image_comparison(['contour_test_label_transforms.png'],
                  remove_text=True, style='mpl20',
                  tol=0 if platform.machine() == 'x86_64' else 0.08)
def test_labels():
    # Adapted from pylab_examples example code: contour_demo.py
    # see issues #2475, #2843, and #2818 for explanation
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((X - 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))

    # difference of Gaussians
    Z = 10.0 * (Z2 - Z1)

    fig, ax = plt.subplots(1, 1)
    CS = ax.contour(X, Y, Z)
    disp_units = [(216, 177), (359, 290), (521, 406)]
    data_units = [(-2, .5), (0, -1.5), (2.8, 1)]

    CS.clabel()

    for x, y in data_units:
        CS.add_label_near(x, y, inline=True, transform=None)

    for x, y in disp_units:
        CS.add_label_near(x, y, inline=True, transform=False)


@image_comparison(['contour_corner_mask_False.png',
                   'contour_corner_mask_True.png'],
                  remove_text=True)
def test_corner_mask():
    n = 60
    mask_level = 0.95
    noise_amp = 1.0
    np.random.seed([1])
    x, y = np.meshgrid(np.linspace(0, 2.0, n), np.linspace(0, 2.0, n))
    z = np.cos(7*x)*np.sin(8*y) + noise_amp*np.random.rand(n, n)
    mask = np.random.rand(n, n) >= mask_level
    z = np.ma.array(z, mask=mask)

    for corner_mask in [False, True]:
        plt.figure()
        plt.contourf(z, corner_mask=corner_mask)


def test_contourf_decreasing_levels():
    # github issue 5477.
    z = [[0.1, 0.3], [0.5, 0.7]]
    plt.figure()
    with pytest.raises(ValueError):
        plt.contourf(z, [1.0, 0.0])


def test_contourf_symmetric_locator():
    # github issue 7271
    z = np.arange(12).reshape((3, 4))
    locator = plt.MaxNLocator(nbins=4, symmetric=True)
    cs = plt.contourf(z, locator=locator)
    assert_array_almost_equal(cs.levels, np.linspace(-12, 12, 5))


@pytest.mark.parametrize("args, cls, message", [
    ((), TypeError,
     'function takes exactly 6 arguments (0 given)'),
    ((1, 2, 3, 4, 5, 6), ValueError,
     'Expected 2-dimensional array, got 0'),
    (([[0]], [[0]], [[]], None, True, 0), ValueError,
     'x, y and z must all be 2D arrays with the same dimensions'),
    (([[0]], [[0]], [[0]], None, True, 0), ValueError,
     'x, y and z must all be at least 2x2 arrays'),
    ((*[np.arange(4).reshape((2, 2))] * 3, [[0]], True, 0), ValueError,
     'If mask is set it must be a 2D array with the same dimensions as x.'),
])
def test_internal_cpp_api(args, cls, message):  # Github issue 8197.
    from matplotlib import _contour  # noqa: ensure lazy-loaded module *is* loaded.
    with pytest.raises(cls, match=re.escape(message)):
        mpl._contour.QuadContourGenerator(*args)


def test_internal_cpp_api_2():
    from matplotlib import _contour  # noqa: ensure lazy-loaded module *is* loaded.
    arr = [[0, 1], [2, 3]]
    qcg = mpl._contour.QuadContourGenerator(arr, arr, arr, None, True, 0)
    with pytest.raises(
            ValueError, match=r'filled contour levels must be increasing'):
        qcg.create_filled_contour(1, 0)


def test_circular_contour_warning():
    # Check that almost circular contours don't throw a warning
    x, y = np.meshgrid(np.linspace(-2, 2, 4), np.linspace(-2, 2, 4))
    r = np.hypot(x, y)
    plt.figure()
    cs = plt.contour(x, y, r)
    plt.clabel(cs)


@pytest.mark.parametrize("use_clabeltext, contour_zorder, clabel_zorder",
                         [(True, 123, 1234), (False, 123, 1234),
                          (True, 123, None), (False, 123, None)])
def test_clabel_zorder(use_clabeltext, contour_zorder, clabel_zorder):
    x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    z = np.max(np.dstack([abs(x), abs(y)]), 2)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    cs = ax1.contour(x, y, z, zorder=contour_zorder)
    cs_filled = ax2.contourf(x, y, z, zorder=contour_zorder)
    clabels1 = cs.clabel(zorder=clabel_zorder, use_clabeltext=use_clabeltext)
    clabels2 = cs_filled.clabel(zorder=clabel_zorder,
                                use_clabeltext=use_clabeltext)

    if clabel_zorder is None:
        expected_clabel_zorder = 2+contour_zorder
    else:
        expected_clabel_zorder = clabel_zorder

    for clabel in clabels1:
        assert clabel.get_zorder() == expected_clabel_zorder
    for clabel in clabels2:
        assert clabel.get_zorder() == expected_clabel_zorder


# tol because ticks happen to fall on pixel boundaries so small
# floating point changes in tick location flip which pixel gets
# the tick.
@image_comparison(['contour_log_extension.png'],
                  remove_text=True, style='mpl20',
                  tol=1.444)
def test_contourf_log_extension():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    # Test that contourf with lognorm is extended correctly
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=0.95)

    # make data set with large range e.g. between 1e-8 and 1e10
    data_exp = np.linspace(-7.5, 9.5, 1200)
    data = np.power(10, data_exp).reshape(30, 40)
    # make manual levels e.g. between 1e-4 and 1e-6
    levels_exp = np.arange(-4., 7.)
    levels = np.power(10., levels_exp)

    # original data
    c1 = ax1.contourf(data,
                      norm=LogNorm(vmin=data.min(), vmax=data.max()))
    # just show data in levels
    c2 = ax2.contourf(data, levels=levels,
                      norm=LogNorm(vmin=levels.min(), vmax=levels.max()),
                      extend='neither')
    # extend data from levels
    c3 = ax3.contourf(data, levels=levels,
                      norm=LogNorm(vmin=levels.min(), vmax=levels.max()),
                      extend='both')
    cb = plt.colorbar(c1, ax=ax1)
    assert cb.ax.get_ylim() == (1e-8, 1e10)
    cb = plt.colorbar(c2, ax=ax2)
    assert cb.ax.get_ylim() == (1e-4, 1e6)
    cb = plt.colorbar(c3, ax=ax3)


@image_comparison(['contour_addlines.png'],
                  remove_text=True, style='mpl20', tol=0.03)
# tolerance is because image changed minutely when tick finding on
# colorbars was cleaned up...
def test_contour_addlines():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    fig, ax = plt.subplots()
    np.random.seed(19680812)
    X = np.random.rand(10, 10)*10000
    pcm = ax.pcolormesh(X)
    # add 1000 to make colors visible...
    cont = ax.contour(X+1000)
    cb = fig.colorbar(pcm)
    cb.add_lines(cont)
    assert_array_almost_equal(cb.ax.get_ylim(), [114.3091, 9972.30735], 3)


@image_comparison(baseline_images=['contour_uneven'],
                  extensions=['png'], remove_text=True, style='mpl20')
def test_contour_uneven():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    z = np.arange(24).reshape(4, 6)
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    cs = ax.contourf(z, levels=[2, 4, 6, 10, 20])
    fig.colorbar(cs, ax=ax, spacing='proportional')
    ax = axs[1]
    cs = ax.contourf(z, levels=[2, 4, 6, 10, 20])
    fig.colorbar(cs, ax=ax, spacing='uniform')


@pytest.mark.parametrize(
    "rc_lines_linewidth, rc_contour_linewidth, call_linewidths, expected", [
        (1.23, None, None, 1.23),
        (1.23, 4.24, None, 4.24),
        (1.23, 4.24, 5.02, 5.02)
        ])
def test_contour_linewidth(
        rc_lines_linewidth, rc_contour_linewidth, call_linewidths, expected):

    with rc_context(rc={"lines.linewidth": rc_lines_linewidth,
                        "contour.linewidth": rc_contour_linewidth}):
        fig, ax = plt.subplots()
        X = np.arange(4*3).reshape(4, 3)
        cs = ax.contour(X, linewidths=call_linewidths)
        assert cs.tlinewidths[0][0] == expected


@pytest.mark.backend("pdf")
def test_label_nonagg():
    # This should not crash even if the canvas doesn't have a get_renderer().
    plt.clabel(plt.contour([[1, 2], [3, 4]]))


@image_comparison(baseline_images=['contour_closed_line_loop'],
                  extensions=['png'], remove_text=True)
def test_contour_closed_line_loop():
    # github issue 19568.
    z = [[0, 0, 0], [0, 2, 0], [0, 0, 0], [2, 1, 2]]

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.contour(z, [0.5], linewidths=[20], alpha=0.7)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 3.1)


def test_quadcontourset_reuse():
    # If QuadContourSet returned from one contour(f) call is passed as first
    # argument to another the underlying C++ contour generator will be reused.
    x, y = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    z = x + y
    fig, ax = plt.subplots()
    qcs1 = ax.contourf(x, y, z)
    qcs2 = ax.contour(x, y, z)
    assert qcs2._contour_generator != qcs1._contour_generator
    qcs3 = ax.contour(qcs1, z)
    assert qcs3._contour_generator == qcs1._contour_generator


@image_comparison(baseline_images=['contour_manual'],
                  extensions=['png'], remove_text=True)
def test_contour_manual():
    # Manually specifying contour lines/polygons to plot.
    from matplotlib.contour import ContourSet

    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = 'viridis'

    # Segments only (no 'kind' codes).
    lines0 = [[[2, 0], [1, 2], [1, 3]]]  # Single line.
    lines1 = [[[3, 0], [3, 2]], [[3, 3], [3, 4]]]  # Two lines.
    filled01 = [[[0, 0], [0, 4], [1, 3], [1, 2], [2, 0]]]
    filled12 = [[[2, 0], [3, 0], [3, 2], [1, 3], [1, 2]],  # Two polygons.
                [[1, 4], [3, 4], [3, 3]]]
    ContourSet(ax, [0, 1, 2], [filled01, filled12], filled=True, cmap=cmap)
    ContourSet(ax, [1, 2], [lines0, lines1], linewidths=3, colors=['r', 'k'])

    # Segments and kind codes (1 = MOVETO, 2 = LINETO, 79 = CLOSEPOLY).
    segs = [[[4, 0], [7, 0], [7, 3], [4, 3], [4, 0],
             [5, 1], [5, 2], [6, 2], [6, 1], [5, 1]]]
    kinds = [[1, 2, 2, 2, 79, 1, 2, 2, 2, 79]]  # Polygon containing hole.
    ContourSet(ax, [2, 3], [segs], [kinds], filled=True, cmap=cmap)
    ContourSet(ax, [2], [segs], [kinds], colors='k', linewidths=3)


@image_comparison(baseline_images=['contour_line_start_on_corner_edge'],
                  extensions=['png'], remove_text=True)
def test_contour_line_start_on_corner_edge():
    fig, ax = plt.subplots(figsize=(6, 5))

    x, y = np.meshgrid([0, 1, 2, 3, 4], [0, 1, 2])
    z = 1.2 - (x - 2)**2 + (y - 1)**2
    mask = np.zeros_like(z, dtype=bool)
    mask[1, 1] = mask[1, 3] = True
    z = np.ma.array(z, mask=mask)

    filled = ax.contourf(x, y, z, corner_mask=True)
    cbar = fig.colorbar(filled)
    lines = ax.contour(x, y, z, corner_mask=True, colors='k')
    cbar.add_lines(lines)


@mpl.style.context("default")
def test_contour_autolabel_beyond_powerlimits():
    ax = plt.figure().add_subplot()
    cs = plt.contour(np.geomspace(1e-6, 1e-4, 100).reshape(10, 10),
                     levels=[.25e-5, 1e-5, 4e-5])
    ax.clabel(cs)
    # Currently, the exponent is missing, but that may be fixed in the future.
    assert {text.get_text() for text in ax.texts} == {"0.25", "1.00", "4.00"}
