import functools
import itertools
import platform

import pytest

from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import MouseButton
from matplotlib.cbook import MatplotlibDeprecationWarning
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle

import matplotlib.pyplot as plt
import numpy as np


mpl3d_image_comparison = functools.partial(
    image_comparison, remove_text=True, style='default')


def test_aspect_equal_error():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    with pytest.raises(NotImplementedError):
        ax.set_aspect('equal')


@mpl3d_image_comparison(['bar3d.png'])
def test_bar3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.arange(20)
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', align='edge', color=cs, alpha=0.8)


def test_bar3d_colors():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for c in ['red', 'green', 'blue', 'yellow']:
        xs = np.arange(len(c))
        ys = np.zeros_like(xs)
        zs = np.zeros_like(ys)
        # Color names with same length as xs/ys/zs should not be split into
        # individual letters.
        ax.bar3d(xs, ys, zs, 1, 1, 1, color=c)


@mpl3d_image_comparison(['bar3d_shaded.png'])
def test_bar3d_shaded():
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d + 1  # Avoid triggering bug with zero-depth boxes.

    views = [(-60, 30), (30, 30), (30, -30), (120, -30)]
    fig = plt.figure(figsize=plt.figaspect(1 / len(views)))
    axs = fig.subplots(
        1, len(views),
        subplot_kw=dict(projection='3d')
    )
    for ax, (azim, elev) in zip(axs, views):
        ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)
        ax.view_init(azim=azim, elev=elev)
    fig.canvas.draw()


@mpl3d_image_comparison(['bar3d_notshaded.png'])
def test_bar3d_notshaded():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    z = x2d + y2d
    ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=False)
    fig.canvas.draw()


def test_bar3d_lightsource():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ls = mcolors.LightSource(azdeg=0, altdeg=90)

    length, width = 3, 4
    area = length * width

    x, y = np.meshgrid(np.arange(length), np.arange(width))
    x = x.ravel()
    y = y.ravel()
    dz = x + y

    color = [cm.coolwarm(i/area) for i in range(area)]

    collection = ax.bar3d(x=x, y=y, z=0,
                          dx=1, dy=1, dz=dz,
                          color=color, shade=True, lightsource=ls)

    # Testing that the custom 90° lightsource produces different shading on
    # the top facecolors compared to the default, and that those colors are
    # precisely the colors from the colormap, due to the illumination parallel
    # to the z-axis.
    np.testing.assert_array_equal(color, collection._facecolor3d[1::6])


@mpl3d_image_comparison(['contour3d.png'])
def test_contour3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)


@mpl3d_image_comparison(['contourf3d.png'])
def test_contourf3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)


@mpl3d_image_comparison(['contourf3d_fill.png'])
def test_contourf3d_fill():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    Z = X.clip(0, 0)
    # This produces holes in the z=0 surface that causes rendering errors if
    # the Poly3DCollection is not aware of path code information (issue #4784)
    Z[::5, ::5] = 0.1
    ax.contourf(X, Y, Z, offset=0, levels=[-0.1, 0], cmap=cm.coolwarm)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)


@pytest.mark.parametrize('extend, levels', [['both', [2, 4, 6]],
                                            ['min', [2, 4, 6, 8]],
                                            ['max', [0, 2, 4, 6]]])
@check_figures_equal(extensions=["png"])
def test_contourf3d_extend(fig_test, fig_ref, extend, levels):
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    # Z is in the range [0, 8]
    Z = X**2 + Y**2

    # Manually set the over/under colors to be the end of the colormap
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(255))
    # Set vmin/max to be the min/max values plotted on the reference image
    kwargs = {'vmin': 1, 'vmax': 7, 'cmap': cmap}

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.contourf(X, Y, Z, levels=[0, 2, 4, 6, 8], **kwargs)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.contourf(X, Y, Z, levels, extend=extend, **kwargs)

    for ax in [ax_ref, ax_test]:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-10, 10)


@mpl3d_image_comparison(['tricontour.png'], tol=0.02)
def test_tricontour():
    fig = plt.figure()

    np.random.seed(19680801)
    x = np.random.rand(1000) - 0.5
    y = np.random.rand(1000) - 0.5
    z = -(x**2 + y**2)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.tricontour(x, y, z)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.tricontourf(x, y, z)


def test_contour3d_1d_input():
    # Check that 1D sequences of different length for {x, y} doesn't error
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    nx, ny = 30, 20
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    z = np.random.randint(0, 2, [ny, nx])
    ax.contour(x, y, z, [0.5])


@mpl3d_image_comparison(['lines3d.png'])
def test_lines3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z)


@check_figures_equal(extensions=["png"])
def test_plot_scalar(fig_test, fig_ref):
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.plot([1], [1], "o")
    ax2 = fig_ref.add_subplot(projection='3d')
    ax2.plot(1, 1, "o")


@mpl3d_image_comparison(['mixedsubplot.png'])
def test_mixedsubplots():
    def f(t):
        return np.cos(2*np.pi*t) * np.exp(-t)

    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green')
    ax.grid(True)

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
    R = np.hypot(X, Y)
    Z = np.sin(R)

    ax.plot_surface(X, Y, Z, rcount=40, ccount=40,
                    linewidth=0, antialiased=False)

    ax.set_zlim3d(-1, 1)


@check_figures_equal(extensions=['png'])
def test_tight_layout_text(fig_test, fig_ref):
    # text is currently ignored in tight layout. So the order of text() and
    # tight_layout() calls should not influence the result.
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.text(.5, .5, .5, s='some string')
    fig_test.tight_layout()

    ax2 = fig_ref.add_subplot(projection='3d')
    fig_ref.tight_layout()
    ax2.text(.5, .5, .5, s='some string')


@mpl3d_image_comparison(['scatter3d.png'])
def test_scatter3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               c='r', marker='o')
    x = y = z = np.arange(10, 20)
    ax.scatter(x, y, z, c='b', marker='^')
    z[-1] = 0  # Check that scatter() copies the data.
    # Ensure empty scatters do not break.
    ax.scatter([], [], [], c='r', marker='X')


@mpl3d_image_comparison(['scatter3d_color.png'])
def test_scatter3d_color():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Check that 'none' color works; these two should overlay to produce the
    # same as setting just `color`.
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               facecolor='r', edgecolor='none', marker='o')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               facecolor='none', edgecolor='r', marker='o')

    ax.scatter(np.arange(10, 20), np.arange(10, 20), np.arange(10, 20),
               color='b', marker='s')


@check_figures_equal(extensions=['png'])
def test_scatter3d_modification(fig_ref, fig_test):
    # Changing Path3DCollection properties post-creation should work correctly.
    ax_test = fig_test.add_subplot(projection='3d')
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10),
                        marker='o')
    c.set_facecolor('C1')
    c.set_edgecolor('C2')
    c.set_alpha([0.3, 0.7] * 5)
    assert c.get_depthshade()
    c.set_depthshade(False)
    assert not c.get_depthshade()
    c.set_sizes(np.full(10, 75))
    c.set_linewidths(3)

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o',
                   facecolor='C1', edgecolor='C2', alpha=[0.3, 0.7] * 5,
                   depthshade=False, s=75, linewidths=3)


@pytest.mark.parametrize('depthshade', [True, False])
@check_figures_equal(extensions=['png'])
def test_scatter3d_sorting(fig_ref, fig_test, depthshade):
    """Test that marker properties are correctly sorted."""

    y, x = np.mgrid[:10, :10]
    z = np.arange(x.size).reshape(x.shape)

    sizes = np.full(z.shape, 25)
    sizes[0::2, 0::2] = 100
    sizes[1::2, 1::2] = 100

    facecolors = np.full(z.shape, 'C0')
    facecolors[:5, :5] = 'C1'
    facecolors[6:, :4] = 'C2'
    facecolors[6:, 6:] = 'C3'

    edgecolors = np.full(z.shape, 'C4')
    edgecolors[1:5, 1:5] = 'C5'
    edgecolors[5:9, 1:5] = 'C6'
    edgecolors[5:9, 5:9] = 'C7'

    linewidths = np.full(z.shape, 2)
    linewidths[0::2, 0::2] = 5
    linewidths[1::2, 1::2] = 5

    x, y, z, sizes, facecolors, edgecolors, linewidths = [
        a.flatten()
        for a in [x, y, z, sizes, facecolors, edgecolors, linewidths]
    ]

    ax_ref = fig_ref.add_subplot(projection='3d')
    sets = (np.unique(a) for a in [sizes, facecolors, edgecolors, linewidths])
    for s, fc, ec, lw in itertools.product(*sets):
        subset = (
            (sizes != s) |
            (facecolors != fc) |
            (edgecolors != ec) |
            (linewidths != lw)
        )
        subset = np.ma.masked_array(z, subset, dtype=float)

        # When depth shading is disabled, the colors are passed through as
        # single-item lists; this triggers single path optimization. The
        # following reshaping is a hack to disable that, since the optimization
        # would not occur for the full scatter which has multiple colors.
        fc = np.repeat(fc, sum(~subset.mask))

        ax_ref.scatter(x, y, subset, s=s, fc=fc, ec=ec, lw=lw, alpha=1,
                       depthshade=depthshade)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.scatter(x, y, z, s=sizes, fc=facecolors, ec=edgecolors,
                    lw=linewidths, alpha=1, depthshade=depthshade)


@pytest.mark.parametrize('azim', [-50, 130])  # yellow first, blue first
@check_figures_equal(extensions=['png'])
def test_marker_draw_order_data_reversed(fig_test, fig_ref, azim):
    """
    Test that the draw order does not depend on the data point order.

    For the given viewing angle at azim=-50, the yellow marker should be in
    front. For azim=130, the blue marker should be in front.
    """
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    color = ['b', 'y']
    ax = fig_test.add_subplot(projection='3d')
    ax.scatter(x, y, z, s=3500, c=color)
    ax.view_init(elev=0, azim=azim)
    ax = fig_ref.add_subplot(projection='3d')
    ax.scatter(x[::-1], y[::-1], z[::-1], s=3500, c=color[::-1])
    ax.view_init(elev=0, azim=azim)


@check_figures_equal(extensions=['png'])
def test_marker_draw_order_view_rotated(fig_test, fig_ref):
    """
    Test that the draw order changes with the direction.

    If we rotate *azim* by 180 degrees and exchange the colors, the plot
    plot should look the same again.
    """
    azim = 130
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    color = ['b', 'y']
    ax = fig_test.add_subplot(projection='3d')
    # axis are not exactly invariant under 180 degree rotation -> deactivate
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color)
    ax.view_init(elev=0, azim=azim)
    ax = fig_ref.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.scatter(x, y, z, s=3500, c=color[::-1])  # color reversed
    ax.view_init(elev=0, azim=azim - 180)  # view rotated by 180 degrees


@mpl3d_image_comparison(['plot_3d_from_2d.png'], tol=0.015)
def test_plot_3d_from_2d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = np.arange(0, 5)
    ys = np.arange(5, 10)
    ax.plot(xs, ys, zs=0, zdir='x')
    ax.plot(xs, ys, zs=0, zdir='y')


@mpl3d_image_comparison(['surface3d.png'])
def test_surface3d():
    # Remove this line when this test image is regenerated.
    plt.rcParams['pcolormesh.snap'] = False

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.hypot(X, Y)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rcount=40, ccount=40, cmap=cm.coolwarm,
                           lw=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)


@mpl3d_image_comparison(['surface3d_shaded.png'])
def test_surface3d_shaded():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                    color=[0.25, 1, 0.25], lw=1, antialiased=False)
    ax.set_zlim(-1.01, 1.01)


@mpl3d_image_comparison(['surface3d_masked.png'])
def test_surface3d_masked():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [1, 2, 3, 4, 5, 6, 7, 8]

    x, y = np.meshgrid(x, y)
    matrix = np.array(
        [
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1],
            [-1, -1., 4, 5, 6, 8, 6, 5, 4, 3, -1.],
            [-1, -1., 7, 8, 11, 12, 11, 8, 7, -1., -1.],
            [-1, -1., 8, 9, 10, 16, 10, 9, 10, 7, -1.],
            [-1, -1., -1., 12, 16, 20, 16, 12, 11, -1., -1.],
            [-1, -1., -1., -1., 22, 24, 22, 20, 18, -1., -1.],
            [-1, -1., -1., -1., -1., 28, 26, 25, -1., -1., -1.],
        ]
    )
    z = np.ma.masked_less(matrix, 0)
    norm = mcolors.Normalize(vmax=z.max(), vmin=z.min())
    colors = plt.get_cmap("plasma")(norm(z))
    ax.plot_surface(x, y, z, facecolors=colors)
    ax.view_init(30, -80)


@mpl3d_image_comparison(['surface3d_masked_strides.png'])
def test_surface3d_masked_strides():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x, y = np.mgrid[-6:6.1:1, -6:6.1:1]
    z = np.ma.masked_less(x * y, 2)

    ax.plot_surface(x, y, z, rstride=4, cstride=4)
    ax.view_init(60, -45)


@mpl3d_image_comparison(['text3d.png'], remove_text=False)
def test_text3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)

    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)

    ax.text(1, 1, 1, "red", color='red')
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
    ax.set_xlim3d(0, 10)
    ax.set_ylim3d(0, 10)
    ax.set_zlim3d(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


@check_figures_equal(extensions=['png'])
def test_text3d_modification(fig_ref, fig_test):
    # Modifying the Text position after the fact should work the same as
    # setting it directly.
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.set_xlim3d(0, 10)
    ax_test.set_ylim3d(0, 10)
    ax_test.set_zlim3d(0, 10)
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        t = ax_test.text(0, 0, 0, f'({x}, {y}, {z}), dir={zdir}')
        t.set_position_3d((x, y, z), zdir=zdir)

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.set_xlim3d(0, 10)
    ax_ref.set_ylim3d(0, 10)
    ax_ref.set_zlim3d(0, 10)
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        ax_ref.text(x, y, z, f'({x}, {y}, {z}), dir={zdir}', zdir=zdir)


@mpl3d_image_comparison(['trisurf3d.png'], tol=0.03)
def test_trisurf3d():
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x*y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)


@mpl3d_image_comparison(['trisurf3d_shaded.png'], tol=0.03)
def test_trisurf3d_shaded():
    n_angles = 36
    n_radii = 8
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x*y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, z, color=[1, 0.5, 0], linewidth=0.2)


@mpl3d_image_comparison(['wireframe3d.png'])
def test_wireframe3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=13)


@mpl3d_image_comparison(['wireframe3dzerocstride.png'])
def test_wireframe3dzerocstride():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=0)


@mpl3d_image_comparison(['wireframe3dzerorstride.png'])
def test_wireframe3dzerorstride():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=0, cstride=10)


def test_wireframe3dzerostrideraises():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=0, cstride=0)


def test_mixedsamplesraises():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=10, ccount=50)
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, Z, cstride=50, rcount=10)


@mpl3d_image_comparison(
    ['quiver3d.png', 'quiver3d_pivot_middle.png', 'quiver3d_pivot_tail.png'])
def test_quiver3d():
    x, y, z = np.ogrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2/3)**0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    for pivot in ['tip', 'middle', 'tail']:
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(x, y, z, u, v, w, length=0.1, pivot=pivot, normalize=True)


@check_figures_equal(extensions=["png"])
def test_quiver3d_empty(fig_test, fig_ref):
    fig_ref.add_subplot(projection='3d')
    x = y = z = u = v = w = []
    ax = fig_test.add_subplot(projection='3d')
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)


@mpl3d_image_comparison(['quiver3d_masked.png'])
def test_quiver3d_masked():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Using mgrid here instead of ogrid because masked_where doesn't
    # seem to like broadcasting very much...
    x, y, z = np.mgrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]

    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2/3)**0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)
    u = np.ma.masked_where((-0.4 < x) & (x < 0.1), u, copy=False)
    v = np.ma.masked_where((0.1 < y) & (y < 0.7), v, copy=False)

    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)


def test_patch_modification():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    circle = Circle((0, 0))
    ax.add_patch(circle)
    art3d.patch_2d_to_3d(circle)
    circle.set_facecolor((1.0, 0.0, 0.0, 1))

    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))
    fig.canvas.draw()
    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))


@check_figures_equal(extensions=['png'])
def test_patch_collection_modification(fig_test, fig_ref):
    # Test that modifying Patch3DCollection properties after creation works.
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0., 0.5, 0., 1.], [0.5, 0., 0., 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3)

    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.add_collection3d(c)
    c.set_edgecolor('C2')
    c.set_facecolor(facecolors)
    c.set_alpha(0.7)
    assert c.get_depthshade()
    c.set_depthshade(False)
    assert not c.get_depthshade()

    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0., 0.5, 0., 1.], [0.5, 0., 0., 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3,
                                edgecolor='C2', facecolor=facecolors,
                                alpha=0.7, depthshade=False)

    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.add_collection3d(c)


def test_poly3dcollection_verts_validation():
    poly = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
    with pytest.raises(ValueError, match=r'list of \(N, 3\) array-like'):
        art3d.Poly3DCollection(poly)  # should be Poly3DCollection([poly])

    poly = np.array(poly, dtype=float)
    with pytest.raises(ValueError, match=r'list of \(N, 3\) array-like'):
        art3d.Poly3DCollection(poly)  # should be Poly3DCollection([poly])


@mpl3d_image_comparison(['poly3dcollection_closed.png'])
def test_poly3dcollection_closed():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1, 0.5), closed=True)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, edgecolor='k',
                                facecolor=(1, 0.5, 0.5, 0.5), closed=False)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)


def test_poly_collection_2d_to_3d_empty():
    poly = PolyCollection([])
    art3d.poly_collection_2d_to_3d(poly)
    assert isinstance(poly, art3d.Poly3DCollection)
    assert poly.get_paths() == []

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.add_artist(poly)
    minz = poly.do_3d_projection()
    assert np.isnan(minz)

    # Ensure drawing actually works.
    fig.canvas.draw()


@mpl3d_image_comparison(['poly3dcollection_alpha.png'])
def test_poly3dcollection_alpha():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1), closed=True)
    c1.set_alpha(0.5)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, closed=False)
    # Post-creation modification should work.
    c2.set_facecolor((1, 0.5, 0.5))
    c2.set_edgecolor('k')
    c2.set_alpha(0.5)
    ax.add_collection3d(c1)
    ax.add_collection3d(c2)


@mpl3d_image_comparison(['add_collection3d_zs_array.png'])
def test_add_collection3d_zs_array():
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    norm = plt.Normalize(0, 2*np.pi)
    # 2D LineCollection from x & y values
    lc = LineCollection(segments[:, :, :2], cmap='twilight', norm=norm)
    lc.set_array(np.mod(theta, 2*np.pi))
    # Add 2D collection at z values to ax
    line = ax.add_collection3d(lc, zs=segments[:, :, 2])

    assert line is not None

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(-2, 2)


@mpl3d_image_comparison(['add_collection3d_zs_scalar.png'])
def test_add_collection3d_zs_scalar():
    theta = np.linspace(0, 2 * np.pi, 100)
    z = 1
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    norm = plt.Normalize(0, 2*np.pi)
    lc = LineCollection(segments, cmap='twilight', norm=norm)
    lc.set_array(theta)
    line = ax.add_collection3d(lc, zs=z)

    assert line is not None

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(0, 2)


@mpl3d_image_comparison(['axes3d_labelpad.png'], remove_text=False)
def test_axes3d_labelpad():
    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig, auto_add_to_figure=False))
    # labelpad respects rcParams
    assert ax.xaxis.labelpad == mpl.rcParams['axes.labelpad']
    # labelpad can be set in set_label
    ax.set_xlabel('X LABEL', labelpad=10)
    assert ax.xaxis.labelpad == 10
    ax.set_ylabel('Y LABEL')
    ax.set_zlabel('Z LABEL')
    # or manually
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = -40

    # Tick labels also respect tick.pad (also from rcParams)
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_pad(tick.get_pad() - i * 5)


@mpl3d_image_comparison(['axes3d_cla.png'], remove_text=False)
def test_axes3d_cla():
    # fixed in pull request 4553
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_axis_off()
    ax.cla()  # make sure the axis displayed is 3D (not 2D)


@mpl3d_image_comparison(['axes3d_rotated.png'], remove_text=False)
def test_axes3d_rotated():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(90, 45)  # look down, rotated. Should be square


def test_plotsurface_1d_raises():
    x = np.linspace(0.5, 10, num=100)
    y = np.linspace(0.5, 10, num=100)
    X, Y = np.meshgrid(x, y)
    z = np.random.randn(100)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, z)


def _test_proj_make_M():
    # eye point
    E = np.array([1000, -1000, 2000])
    R = np.array([100, 100, 100])
    V = np.array([0, 0, 1])
    viewM = proj3d.view_transformation(E, R, V)
    perspM = proj3d.persp_transformation(100, -100)
    M = np.dot(perspM, viewM)
    return M


def test_proj_transform():
    M = _test_proj_make_M()

    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    ixs, iys, izs = proj3d.inv_transform(txs, tys, tzs, M)

    np.testing.assert_almost_equal(ixs, xs)
    np.testing.assert_almost_equal(iys, ys)
    np.testing.assert_almost_equal(izs, zs)


def _test_proj_draw_axes(M, s=1, *args, **kwargs):
    xs = [0, s, 0, 0]
    ys = [0, 0, s, 0]
    zs = [0, 0, 0, s]
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    o, ax, ay, az = zip(txs, tys)
    lines = [(o, ax), (o, ay), (o, az)]

    fig, ax = plt.subplots(*args, **kwargs)
    linec = LineCollection(lines)
    ax.add_collection(linec)
    for x, y, t in zip(txs, tys, ['o', 'x', 'y', 'z']):
        ax.text(x, y, t)

    return fig, ax


@mpl3d_image_comparison(['proj3d_axes_cube.png'])
def test_proj_axes_cube():
    M = _test_proj_make_M()

    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    fig, ax = _test_proj_draw_axes(M, s=400)

    ax.scatter(txs, tys, c=tzs)
    ax.plot(txs, tys, c='r')
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)


@mpl3d_image_comparison(['proj3d_axes_cube_ortho.png'])
def test_proj_axes_cube_ortho():
    E = np.array([200, 100, 100])
    R = np.array([0, 0, 0])
    V = np.array([0, 0, 1])
    viewM = proj3d.view_transformation(E, R, V)
    orthoM = proj3d.ortho_transformation(-1, 1)
    M = np.dot(orthoM, viewM)

    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 100
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 100
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 100

    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    fig, ax = _test_proj_draw_axes(M, s=150)

    ax.scatter(txs, tys, s=300-tzs)
    ax.plot(txs, tys, c='r')
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)


def test_rot():
    V = [1, 0, 0, 1]
    rotated_V = proj3d.rot_x(V, np.pi / 6)
    np.testing.assert_allclose(rotated_V, [1, 0, 0, 1])

    V = [0, 1, 0, 1]
    rotated_V = proj3d.rot_x(V, np.pi / 6)
    np.testing.assert_allclose(rotated_V, [0, np.sqrt(3) / 2, 0.5, 1])


def test_world():
    xmin, xmax = 100, 120
    ymin, ymax = -100, 100
    zmin, zmax = 0.1, 0.2
    M = proj3d.world_transformation(xmin, xmax, ymin, ymax, zmin, zmax)
    np.testing.assert_allclose(M,
                               [[5e-2, 0, 0, -5],
                                [0, 5e-3, 0, 5e-1],
                                [0, 0, 1e1, -1],
                                [0, 0, 0, 1]])


@mpl3d_image_comparison(['proj3d_lines_dists.png'])
def test_lines_dists():
    fig, ax = plt.subplots(figsize=(4, 6), subplot_kw=dict(aspect='equal'))

    xs = (0, 30)
    ys = (20, 150)
    ax.plot(xs, ys)
    p0, p1 = zip(xs, ys)

    xs = (0, 0, 20, 30)
    ys = (100, 150, 30, 200)
    ax.scatter(xs, ys)

    dist = proj3d._line2d_seg_dist(p0, p1, (xs[0], ys[0]))
    dist = proj3d._line2d_seg_dist(p0, p1, np.array((xs, ys)))
    for x, y, d in zip(xs, ys, dist):
        c = Circle((x, y), d, fill=0)
        ax.add_patch(c)

    ax.set_xlim(-50, 150)
    ax.set_ylim(0, 300)


def test_autoscale():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.margins(x=0, y=.1, z=.2)
    ax.plot([0, 1], [0, 1], [0, 1])
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.2, 1.2)
    ax.autoscale(False)
    ax.set_autoscalez_on(True)
    ax.plot([0, 2], [0, 2], [0, 2])
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.4, 2.4)


@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
@pytest.mark.parametrize('auto', (True, False, None))
def test_unautoscale(axis, auto):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = np.arange(100)
    y = np.linspace(-0.1, 0.1, 100)
    ax.scatter(x, y)

    get_autoscale_on = getattr(ax, f'get_autoscale{axis}_on')
    set_lim = getattr(ax, f'set_{axis}lim')
    get_lim = getattr(ax, f'get_{axis}lim')

    post_auto = get_autoscale_on() if auto is None else auto

    set_lim((-0.5, 0.5), auto=auto)
    assert post_auto == get_autoscale_on()
    fig.canvas.draw()
    np.testing.assert_array_equal(get_lim(), (-0.5, 0.5))


@mpl3d_image_comparison(['axes3d_ortho.png'], remove_text=False)
def test_axes3d_ortho():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')


@mpl3d_image_comparison(['axes3d_isometric.png'])
def test_axes3d_isometric():
    from itertools import combinations, product
    fig, ax = plt.subplots(subplot_kw=dict(
        projection='3d',
        proj_type='ortho',
        box_aspect=(4, 4, 4)
    ))
    r = (-1, 1)  # stackoverflow.com/a/11156353
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if abs(s - e).sum() == r[1] - r[0]:
            ax.plot3D(*zip(s, e), c='k')
    ax.view_init(elev=np.degrees(np.arctan(1. / np.sqrt(2))), azim=-45)
    ax.grid(True)


@pytest.mark.parametrize('value', [np.inf, np.nan])
@pytest.mark.parametrize(('setter', 'side'), [
    ('set_xlim3d', 'left'),
    ('set_xlim3d', 'right'),
    ('set_ylim3d', 'bottom'),
    ('set_ylim3d', 'top'),
    ('set_zlim3d', 'bottom'),
    ('set_zlim3d', 'top'),
])
def test_invalid_axes_limits(setter, side, value):
    limit = {side: value}
    fig = plt.figure()
    obj = fig.add_subplot(projection='3d')
    with pytest.raises(ValueError):
        getattr(obj, setter)(**limit)


class TestVoxels:
    @mpl3d_image_comparison(['voxels-simple.png'])
    def test_simple(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((5, 4, 3))
        voxels = (x == y) | (y == z)
        ax.voxels(voxels)

    @mpl3d_image_comparison(['voxels-edge-style.png'])
    def test_edge_style(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((5, 5, 4))
        voxels = ((x - 2)**2 + (y - 2)**2 + (z-1.5)**2) < 2.2**2
        v = ax.voxels(voxels, linewidths=3, edgecolor='C1')

        # change the edge color of one voxel
        v[max(v.keys())].set_edgecolor('C2')

    @mpl3d_image_comparison(['voxels-named-colors.png'])
    def test_named_colors(self):
        """Test with colors set to a 3D object array of strings."""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((10, 10, 10))
        voxels = (x == y) | (y == z)
        voxels = voxels & ~(x * y * z < 1)
        colors = np.full((10, 10, 10), 'C0', dtype=np.object_)
        colors[(x < 5) & (y < 5)] = '0.25'
        colors[(x + z) < 10] = 'cyan'
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-rgb-data.png'])
    def test_rgb_data(self):
        """Test with colors set to a 4d float array of rgb data."""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((10, 10, 10))
        voxels = (x == y) | (y == z)
        colors = np.zeros((10, 10, 10, 3))
        colors[..., 0] = x / 9
        colors[..., 1] = y / 9
        colors[..., 2] = z / 9
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-alpha.png'])
    def test_alpha(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x, y, z = np.indices((10, 10, 10))
        v1 = x == y
        v2 = np.abs(x - y) < 2
        voxels = v1 | v2
        colors = np.zeros((10, 10, 10, 4))
        colors[v2] = [1, 0, 0, 0.5]
        colors[v1] = [0, 1, 0, 0.5]
        v = ax.voxels(voxels, facecolors=colors)

        assert type(v) is dict
        for coord, poly in v.items():
            assert voxels[coord], "faces returned for absent voxel"
            assert isinstance(poly, art3d.Poly3DCollection)

    @mpl3d_image_comparison(['voxels-xyz.png'], tol=0.01, remove_text=False)
    def test_xyz(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        def midpoints(x):
            sl = ()
            for i in range(x.ndim):
                x = (x[sl + np.index_exp[:-1]] +
                     x[sl + np.index_exp[1:]]) / 2.0
                sl += np.index_exp[:]
            return x

        # prepare some coordinates, and attach rgb values to each
        r, g, b = np.indices((17, 17, 17)) / 16.0
        rc = midpoints(r)
        gc = midpoints(g)
        bc = midpoints(b)

        # define a sphere about [0.5, 0.5, 0.5]
        sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

        # combine the color components
        colors = np.zeros(sphere.shape + (3,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc

        # and plot everything
        ax.voxels(r, g, b, sphere,
                  facecolors=colors,
                  edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
                  linewidth=0.5)

    def test_calling_conventions(self):
        x, y, z = np.indices((3, 4, 5))
        filled = np.ones((2, 3, 4))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # all the valid calling conventions
        for kw in (dict(), dict(edgecolor='k')):
            ax.voxels(filled, **kw)
            ax.voxels(filled=filled, **kw)
            ax.voxels(x, y, z, filled, **kw)
            ax.voxels(x, y, z, filled=filled, **kw)

        # duplicate argument
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y, z, filled, filled=filled)
        # missing arguments
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y)
        # x, y, z are positional only - this passes them on as attributes of
        # Poly3DCollection
        with pytest.raises(AttributeError):
            ax.voxels(filled=filled, x=x, y=y, z=z)


def test_line3d_set_get_data_3d():
    x, y, z = [0, 1], [2, 3], [4, 5]
    x2, y2, z2 = [6, 7], [8, 9], [10, 11]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    lines = ax.plot(x, y, z)
    line = lines[0]
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    line.set_data_3d(x2, y2, z2)
    np.testing.assert_array_equal((x2, y2, z2), line.get_data_3d())
    line.set_xdata(x)
    line.set_ydata(y)
    line.set_3d_properties(zs=z, zdir='z')
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    line.set_3d_properties(zs=0, zdir='z')
    np.testing.assert_array_equal((x, y, np.zeros_like(z)), line.get_data_3d())


@check_figures_equal(extensions=["png"])
def test_inverted(fig_test, fig_ref):
    # Plot then invert.
    ax = fig_test.add_subplot(projection="3d")
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])
    ax.invert_yaxis()
    # Invert then plot.
    ax = fig_ref.add_subplot(projection="3d")
    ax.invert_yaxis()
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])


def test_inverted_cla():
    # GitHub PR #5450. Setting autoscale should reset
    # axes to be non-inverted.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # 1. test that a new axis is not inverted per default
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()
    ax.set_xlim(1, 0)
    ax.set_ylim(1, 0)
    ax.set_zlim(1, 0)
    assert ax.xaxis_inverted()
    assert ax.yaxis_inverted()
    assert ax.zaxis_inverted()
    ax.cla()
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()


def test_ax3d_tickcolour():
    fig = plt.figure()
    with pytest.warns(MatplotlibDeprecationWarning):
        ax = Axes3D(fig)

    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.tick_params(axis='z', colors='red')
    fig.canvas.draw()

    for tick in ax.xaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.yaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.zaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'


@check_figures_equal(extensions=["png"])
def test_ticklabel_format(fig_test, fig_ref):
    axs = fig_test.subplots(4, 5, subplot_kw={"projection": "3d"})
    for ax in axs.flat:
        ax.set_xlim(1e7, 1e7 + 10)
    for row, name in zip(axs, ["x", "y", "z", "both"]):
        row[0].ticklabel_format(
            axis=name, style="plain")
        row[1].ticklabel_format(
            axis=name, scilimits=(-2, 2))
        row[2].ticklabel_format(
            axis=name, useOffset=not mpl.rcParams["axes.formatter.useoffset"])
        row[3].ticklabel_format(
            axis=name, useLocale=not mpl.rcParams["axes.formatter.use_locale"])
        row[4].ticklabel_format(
            axis=name,
            useMathText=not mpl.rcParams["axes.formatter.use_mathtext"])

    def get_formatters(ax, names):
        return [getattr(ax, name).get_major_formatter() for name in names]

    axs = fig_ref.subplots(4, 5, subplot_kw={"projection": "3d"})
    for ax in axs.flat:
        ax.set_xlim(1e7, 1e7 + 10)
    for row, names in zip(
            axs, [["xaxis"], ["yaxis"], ["zaxis"], ["xaxis", "yaxis", "zaxis"]]
    ):
        for fmt in get_formatters(row[0], names):
            fmt.set_scientific(False)
        for fmt in get_formatters(row[1], names):
            fmt.set_powerlimits((-2, 2))
        for fmt in get_formatters(row[2], names):
            fmt.set_useOffset(not mpl.rcParams["axes.formatter.useoffset"])
        for fmt in get_formatters(row[3], names):
            fmt.set_useLocale(not mpl.rcParams["axes.formatter.use_locale"])
        for fmt in get_formatters(row[4], names):
            fmt.set_useMathText(
                not mpl.rcParams["axes.formatter.use_mathtext"])


@check_figures_equal(extensions=["png"])
def test_quiver3D_smoke(fig_test, fig_ref):
    pivot = "middle"
    # Make the grid
    x, y, z = np.meshgrid(
        np.arange(-0.8, 1, 0.2),
        np.arange(-0.8, 1, 0.2),
        np.arange(-0.8, 1, 0.8)
    )
    u = v = w = np.ones_like(x)

    for fig, length in zip((fig_ref, fig_test), (1, 1.0)):
        ax = fig.add_subplot(projection="3d")
        ax.quiver(x, y, z, u, v, w, length=length, pivot=pivot)


@image_comparison(["minor_ticks.png"], style="mpl20")
def test_minor_ticks():
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_xticks([0.25], minor=True)
    ax.set_xticklabels(["quarter"], minor=True)
    ax.set_yticks([0.33], minor=True)
    ax.set_yticklabels(["third"], minor=True)
    ax.set_zticks([0.50], minor=True)
    ax.set_zticklabels(["half"], minor=True)


@mpl3d_image_comparison(['errorbar3d_errorevery.png'])
def test_errorbar3d_errorevery():
    """Tests errorevery functionality for 3D errorbars."""
    t = np.arange(0, 2*np.pi+.1, 0.01)
    x, y, z = np.sin(t), np.cos(3*t), np.sin(5*t)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    estep = 15
    i = np.arange(t.size)
    zuplims = (i % estep == 0) & (i // estep % 3 == 0)
    zlolims = (i % estep == 0) & (i // estep % 3 == 2)

    ax.errorbar(x, y, z, 0.2, zuplims=zuplims, zlolims=zlolims,
                errorevery=estep)


@mpl3d_image_comparison(['errorbar3d.png'])
def test_errorbar3d():
    """Tests limits, color styling, and legend for 3D errorbars."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    d = [1, 2, 3, 4, 5]
    e = [.5, .5, .5, .5, .5]
    ax.errorbar(x=d, y=d, z=d, xerr=e, yerr=e, zerr=e, capsize=3,
                zuplims=[False, True, False, True, True],
                zlolims=[True, False, False, True, False],
                yuplims=True,
                ecolor='purple', label='Error lines')
    ax.legend()


@image_comparison(['stem3d.png'], style='mpl20',
                  tol=0.0 if platform.machine() == 'x86_64' else 0.003)
def test_stem3d():
    fig, axs = plt.subplots(2, 3, figsize=(8, 6),
                            constrained_layout=True,
                            subplot_kw={'projection': '3d'})

    theta = np.linspace(0, 2*np.pi)
    x = np.cos(theta - np.pi/2)
    y = np.sin(theta - np.pi/2)
    z = theta

    for ax, zdir in zip(axs[0], ['x', 'y', 'z']):
        ax.stem(x, y, z, orientation=zdir)
        ax.set_title(f'orientation={zdir}')

    x = np.linspace(-np.pi/2, np.pi/2, 20)
    y = np.ones_like(x)
    z = np.cos(x)

    for ax, zdir in zip(axs[1], ['x', 'y', 'z']):
        markerline, stemlines, baseline = ax.stem(
            x, y, z,
            linefmt='C4-.', markerfmt='C1D', basefmt='C2',
            orientation=zdir)
        ax.set_title(f'orientation={zdir}')
        markerline.set(markerfacecolor='none', markeredgewidth=2)
        baseline.set_linewidth(3)


@image_comparison(["equal_box_aspect.png"], style="mpl20")
def test_equal_box_aspect():
    from itertools import product, combinations

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z)

    # draw cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="b")

    # Make axes limits
    xyzlim = np.column_stack(
        [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    )
    XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.axis('off')
    ax.set_box_aspect((1, 1, 1))


def test_colorbar_pos():
    num_plots = 2
    fig, axs = plt.subplots(1, num_plots, figsize=(4, 5),
                            constrained_layout=True,
                            subplot_kw={'projection': '3d'})
    for ax in axs:
        p_tri = ax.plot_trisurf(np.random.randn(5), np.random.randn(5),
                                np.random.randn(5))

    cbar = plt.colorbar(p_tri, ax=axs, orientation='horizontal')

    fig.canvas.draw()
    # check that actually on the bottom
    assert cbar.ax.get_position().extents[1] < 0.2


def test_shared_axes_retick():
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection="3d")
    ax2 = fig.add_subplot(212, projection="3d", sharez=ax1)
    ax1.plot([0, 1], [0, 1], [0, 2])
    ax2.plot([0, 1], [0, 1], [0, 2])
    ax1.set_zticks([-0.5, 0, 2, 2.5])
    # check that setting ticks on a shared axis is synchronized
    assert ax1.get_zlim() == (-0.5, 2.5)
    assert ax2.get_zlim() == (-0.5, 2.5)


def test_pan():
    """Test mouse panning using the middle mouse button."""

    def convert_lim(dmin, dmax):
        """Convert min/max limits to center and range."""
        center = (dmin + dmax) / 2
        range_ = dmax - dmin
        return center, range_

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(0, 0, 0)
    ax.figure.canvas.draw()

    x_center0, x_range0 = convert_lim(*ax.get_xlim3d())
    y_center0, y_range0 = convert_lim(*ax.get_ylim3d())
    z_center0, z_range0 = convert_lim(*ax.get_zlim3d())

    # move mouse diagonally to pan along all axis.
    ax._button_press(
        mock_event(ax, button=MouseButton.MIDDLE, xdata=0, ydata=0))
    ax._on_move(
        mock_event(ax, button=MouseButton.MIDDLE, xdata=1, ydata=1))

    x_center, x_range = convert_lim(*ax.get_xlim3d())
    y_center, y_range = convert_lim(*ax.get_ylim3d())
    z_center, z_range = convert_lim(*ax.get_zlim3d())

    # Ranges have not changed
    assert x_range == pytest.approx(x_range0)
    assert y_range == pytest.approx(y_range0)
    assert z_range == pytest.approx(z_range0)

    # But center positions have
    assert x_center != pytest.approx(x_center0)
    assert y_center != pytest.approx(y_center0)
    assert z_center != pytest.approx(z_center0)


@mpl.style.context('default')
@check_figures_equal(extensions=["png"])
def test_scalarmap_update(fig_test, fig_ref):

    x, y, z = np.array((list(itertools.product(*[np.arange(0, 5, 1),
                                                 np.arange(0, 5, 1),
                                                 np.arange(0, 5, 1)])))).T
    c = x + y

    # test
    ax_test = fig_test.add_subplot(111, projection='3d')
    sc_test = ax_test.scatter(x, y, z, c=c, s=40, cmap='viridis')
    # force a draw
    fig_test.canvas.draw()
    # mark it as "stale"
    sc_test.changed()

    # ref
    ax_ref = fig_ref.add_subplot(111, projection='3d')
    sc_ref = ax_ref.scatter(x, y, z, c=c, s=40, cmap='viridis')


def test_subfigure_simple():
    # smoketest that subfigures can work...
    fig = plt.figure()
    sf = fig.subfigures(1, 2)
    ax = sf[0].add_subplot(1, 1, 1, projection='3d')
    ax = sf[1].add_subplot(1, 1, 1, projection='3d', label='other')


@image_comparison(baseline_images=['computed_zorder'], remove_text=True,
                  extensions=['png'])
def test_computed_zorder():
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.computed_zorder = False

    # create a horizontal plane
    corners = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    for ax in (ax1, ax2):
        tri = art3d.Poly3DCollection([corners],
                                     facecolors='white',
                                     edgecolors='black',
                                     zorder=1)
        ax.add_collection3d(tri)

        # plot a vector
        ax.plot((2, 2), (2, 2), (0, 4), c='red', zorder=2)

        # plot some points
        ax.scatter((3, 3), (1, 3), (1, 3), c='red', zorder=10)

        ax.set_xlim((0, 5.0))
        ax.set_ylim((0, 5.0))
        ax.set_zlim((0, 2.5))

    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.computed_zorder = False

    dim = 10
    X, Y = np.meshgrid((-dim, dim), (-dim, dim))
    Z = np.zeros((2, 2))

    angle = 0.5
    X2, Y2 = np.meshgrid((-dim, dim), (0, dim))
    Z2 = Y2 * angle
    X3, Y3 = np.meshgrid((-dim, dim), (-dim, 0))
    Z3 = Y3 * angle

    r = 7
    M = 1000
    th = np.linspace(0, 2 * np.pi, M)
    x, y, z = r * np.cos(th),  r * np.sin(th), angle * r * np.sin(th)
    for ax in (ax3, ax4):
        ax.plot_surface(X2, Y3, Z3,
                        color='blue',
                        alpha=0.5,
                        linewidth=0,
                        zorder=-1)
        ax.plot(x[y < 0], y[y < 0], z[y < 0],
                lw=5,
                linestyle='--',
                color='green',
                zorder=0)

        ax.plot_surface(X, Y, Z,
                        color='red',
                        alpha=0.5,
                        linewidth=0,
                        zorder=1)

        ax.plot(r * np.sin(th), r * np.cos(th), np.zeros(M),
                lw=5,
                linestyle='--',
                color='black',
                zorder=2)

        ax.plot_surface(X2, Y2, Z2,
                        color='blue',
                        alpha=0.5,
                        linewidth=0,
                        zorder=3)

        ax.plot(x[y > 0], y[y > 0], z[y > 0], lw=5,
                linestyle='--',
                color='green',
                zorder=4)
        ax.view_init(azim=-20, elev=20)
        ax.axis('off')


@image_comparison(baseline_images=['scatter_spiral.png'],
                  remove_text=True,
                  style='default')
def test_scatter_spiral():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    th = np.linspace(0, 2 * np.pi * 6, 256)
    sc = ax.scatter(np.sin(th), np.cos(th), th, s=(1 + th * 5), c=th ** 2)

    # force at least 1 draw!
    fig.canvas.draw()


@pytest.mark.parametrize(
    "vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected",
    [
        (
            "z",
            [
                [0.0, 1.142857, 0.0, -0.571429],
                [0.0, 0.0, 0.857143, -0.428571],
                [0.0, 0.0, 0.0, -10.0],
                [-1.142857, 0.0, 0.0, 10.571429],
            ],
            [
                ([0.05617978, 0.06329114], [-0.04213483, -0.04746835]),
                ([-0.06329114, 0.06329114], [-0.04746835, -0.04746835]),
                ([-0.06329114, -0.06329114], [-0.04746835, 0.04746835]),
            ],
            [1, 0, 0],
        ),
        (
            "y",
            [
                [1.142857, 0.0, 0.0, -0.571429],
                [0.0, 0.857143, 0.0, -0.428571],
                [0.0, 0.0, 0.0, -10.0],
                [0.0, 0.0, -1.142857, 10.571429],
            ],
            [
                ([0.06329114, -0.06329114], [-0.04746835, -0.04746835]),
                ([-0.06329114, -0.06329114], [0.04746835, -0.04746835]),
                ([0.05617978, 0.06329114], [-0.04213483, -0.04746835]),
            ],
            [2, 2, 0],
        ),
        (
            "x",
            [
                [0.0, 0.0, 1.142857, -0.571429],
                [0.857143, 0.0, 0.0, -0.428571],
                [0.0, 0.0, 0.0, -10.0],
                [0.0, -1.142857, 0.0, 10.571429],
            ],
            [
                ([-0.06329114, -0.06329114], [-0.04746835, 0.04746835]),
                ([0.06329114, 0.05617978], [-0.04746835, -0.04213483]),
                ([0.06329114, -0.06329114], [-0.04746835, -0.04746835]),
            ],
            [1, 2, 1],
        ),
    ],
)
def test_view_init_vertical_axis(
    vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected
):
    """
    Test the actual projection, axis lines and ticks matches expected values.

    Parameters
    ----------
    vertical_axis : str
        Axis to align vertically.
    proj_expected : ndarray
        Expected values from ax.get_proj().
    axis_lines_expected : tuple of arrays
        Edgepoints of the axis line. Expected values retrieved according
        to ``ax.get_[xyz]axis().line.get_data()``.
    tickdirs_expected : list of int
        indexes indicating which axis to create a tick line along.
    """
    rtol = 2e-06
    ax = plt.subplot(1, 1, 1, projection="3d")
    ax.view_init(azim=0, elev=0, vertical_axis=vertical_axis)
    ax.figure.canvas.draw()

    # Assert the projection matrix:
    proj_actual = ax.get_proj()
    np.testing.assert_allclose(proj_expected, proj_actual, rtol=rtol)

    for i, axis in enumerate([ax.get_xaxis(), ax.get_yaxis(), ax.get_zaxis()]):
        # Assert black lines are correctly aligned:
        axis_line_expected = axis_lines_expected[i]
        axis_line_actual = axis.line.get_data()
        np.testing.assert_allclose(axis_line_expected, axis_line_actual,
                                   rtol=rtol)

        # Assert ticks are correctly aligned:
        tickdir_expected = tickdirs_expected[i]
        tickdir_actual = axis._get_tickdir()
        np.testing.assert_array_equal(tickdir_expected, tickdir_actual)


def test_do_3d_projection_renderer_deprecation_warn_on_argument():
    """
    Test that an external artist with an old-style calling convention raises
    a suitable deprecation warning.
    """
    class DummyPatch(art3d.Patch3D):
        def do_3d_projection(self, renderer):
            return 0

        def draw(self, renderer):
            pass

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    artist = DummyPatch()
    ax.add_artist(artist)

    match = r"The 'renderer' parameter of do_3d_projection\(\) was deprecated"
    with pytest.warns(MatplotlibDeprecationWarning, match=match):
        fig.canvas.draw()


def test_do_3d_projection_renderer_deprecation_nowarn_on_optional_argument():
    """
    Test that an external artist with a calling convention compatible with
    both v3.3 and v3.4 does not raise a deprecation warning.
    """
    class DummyPatch(art3d.Patch3D):
        def do_3d_projection(self, renderer=None):
            return 0

        def draw(self, renderer):
            pass

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    artist = DummyPatch()
    ax.add_artist(artist)
    fig.canvas.draw()


def test_do_3d_projection_renderer_deprecation_nowarn_on_no_argument():
    """
    Test that an external artist with a calling convention compatible with
    only v3.4 does not raise a deprecation warning.
    """
    class DummyPatch(art3d.Patch3D):
        def do_3d_projection(self):
            return 0

        def draw(self, renderer):
            pass

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    artist = DummyPatch()
    ax.add_artist(artist)
    fig.canvas.draw()
