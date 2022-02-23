from datetime import datetime
import io
from pathlib import Path
import platform
from threading import Timer
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from PIL import Image

import matplotlib as mpl
from matplotlib import cbook, rcParams
from matplotlib._api.deprecation import MatplotlibDeprecationWarning
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec


@image_comparison(['figure_align_labels'], extensions=['png', 'svg'],
                  tol=0 if platform.machine() == 'x86_64' else 0.01)
def test_align_labels():
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(3, 3)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(np.arange(0, 1e6, 1000))
    ax.set_ylabel('Ylabel0 0')
    ax = fig.add_subplot(gs[0, -1])
    ax.plot(np.arange(0, 1e4, 100))

    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.set_ylabel('YLabel1 %d' % i)
        ax.set_xlabel('XLabel1 %d' % i)
        if i in [0, 2]:
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
        if i == 0:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        if i == 2:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.set_xlabel(f'XLabel2 {i}')
        ax.set_ylabel(f'YLabel2 {i}')

        if i == 2:
            ax.plot(np.arange(0, 1e4, 10))
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)

    fig.align_labels()


def test_align_labels_stray_axes():
    fig, axs = plt.subplots(2, 2)
    for nn, ax in enumerate(axs.flat):
        ax.set_xlabel('Boo')
        ax.set_xlabel('Who')
        ax.plot(np.arange(4)**nn, np.arange(4)**nn)
    fig.align_ylabels()
    fig.align_xlabels()
    fig.draw_without_rendering()
    xn = np.zeros(4)
    yn = np.zeros(4)
    for nn, ax in enumerate(axs.flat):
        yn[nn] = ax.xaxis.label.get_position()[1]
        xn[nn] = ax.yaxis.label.get_position()[0]
    np.testing.assert_allclose(xn[:2], xn[2:])
    np.testing.assert_allclose(yn[::2], yn[1::2])

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for nn, ax in enumerate(axs.flat):
        ax.set_xlabel('Boo')
        ax.set_xlabel('Who')
        pc = ax.pcolormesh(np.random.randn(10, 10))
    fig.colorbar(pc, ax=ax)
    fig.align_ylabels()
    fig.align_xlabels()
    fig.draw_without_rendering()
    xn = np.zeros(4)
    yn = np.zeros(4)
    for nn, ax in enumerate(axs.flat):
        yn[nn] = ax.xaxis.label.get_position()[1]
        xn[nn] = ax.yaxis.label.get_position()[0]
    np.testing.assert_allclose(xn[:2], xn[2:])
    np.testing.assert_allclose(yn[::2], yn[1::2])


def test_figure_label():
    # pyplot figure creation, selection, and closing with label/number/instance
    plt.close('all')
    fig_today = plt.figure('today')
    plt.figure(3)
    plt.figure('tomorrow')
    plt.figure()
    plt.figure(0)
    plt.figure(1)
    plt.figure(3)
    assert plt.get_fignums() == [0, 1, 3, 4, 5]
    assert plt.get_figlabels() == ['', 'today', '', 'tomorrow', '']
    plt.close(10)
    plt.close()
    plt.close(5)
    plt.close('tomorrow')
    assert plt.get_fignums() == [0, 1]
    assert plt.get_figlabels() == ['', 'today']
    plt.figure(fig_today)
    assert plt.gcf() == fig_today
    with pytest.raises(ValueError):
        plt.figure(Figure())


def test_fignum_exists():
    # pyplot figure creation, selection and closing with fignum_exists
    plt.figure('one')
    plt.figure(2)
    plt.figure('three')
    plt.figure()
    assert plt.fignum_exists('one')
    assert plt.fignum_exists(2)
    assert plt.fignum_exists('three')
    assert plt.fignum_exists(4)
    plt.close('one')
    plt.close(4)
    assert not plt.fignum_exists('one')
    assert not plt.fignum_exists(4)


def test_clf_keyword():
    # test if existing figure is cleared with figure() and subplots()
    text1 = 'A fancy plot'
    text2 = 'Really fancy!'

    fig0 = plt.figure(num=1)
    fig0.suptitle(text1)
    assert [t.get_text() for t in fig0.texts] == [text1]

    fig1 = plt.figure(num=1, clear=False)
    fig1.text(0.5, 0.5, text2)
    assert fig0 is fig1
    assert [t.get_text() for t in fig1.texts] == [text1, text2]

    fig2, ax2 = plt.subplots(2, 1, num=1, clear=True)
    assert fig0 is fig2
    assert [t.get_text() for t in fig2.texts] == []


@image_comparison(['figure_today'])
def test_figure():
    # named figure support
    fig = plt.figure('today')
    ax = fig.add_subplot()
    ax.set_title(fig.get_label())
    ax.plot(np.arange(5))
    # plot red line in a different figure.
    plt.figure('tomorrow')
    plt.plot([0, 1], [1, 0], 'r')
    # Return to the original; make sure the red line is not there.
    plt.figure('today')
    plt.close('tomorrow')


@image_comparison(['figure_legend'])
def test_figure_legend():
    fig, axs = plt.subplots(2)
    axs[0].plot([0, 1], [1, 0], label='x', color='g')
    axs[0].plot([0, 1], [0, 1], label='y', color='r')
    axs[0].plot([0, 1], [0.5, 0.5], label='y', color='k')

    axs[1].plot([0, 1], [1, 0], label='_y', color='r')
    axs[1].plot([0, 1], [0, 1], label='z', color='b')
    fig.legend()


def test_gca():
    fig = plt.figure()

    with pytest.raises(TypeError):
        assert fig.add_axes() is None

    ax0 = fig.add_axes([0, 0, 1, 1])
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        assert fig.gca(projection='rectilinear') is ax0
    assert fig.gca() is ax0

    ax1 = fig.add_axes(rect=[0.1, 0.1, 0.8, 0.8])
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        assert fig.gca(projection='rectilinear') is ax1
    assert fig.gca() is ax1

    ax2 = fig.add_subplot(121, projection='polar')
    assert fig.gca() is ax2
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        assert fig.gca(polar=True) is ax2

    ax3 = fig.add_subplot(122)
    assert fig.gca() is ax3

    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        assert fig.gca(polar=True) is ax3
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        assert fig.gca(polar=True) is not ax2
    assert fig.gca().get_subplotspec().get_geometry() == (1, 2, 1, 1)

    # add_axes on an existing Axes should not change stored order, but will
    # make it current.
    fig.add_axes(ax0)
    assert fig.axes == [ax0, ax1, ax2, ax3]
    assert fig.gca() is ax0

    # add_subplot on an existing Axes should not change stored order, but will
    # make it current.
    fig.add_subplot(ax2)
    assert fig.axes == [ax0, ax1, ax2, ax3]
    assert fig.gca() is ax2

    fig.sca(ax1)
    with pytest.warns(
            MatplotlibDeprecationWarning,
            match=r'Calling gca\(\) with keyword arguments was deprecated'):
        assert fig.gca(projection='rectilinear') is ax1
    assert fig.gca() is ax1

    # sca() should not change stored order of Axes, which is order added.
    assert fig.axes == [ax0, ax1, ax2, ax3]


def test_add_subplot_subclass():
    fig = plt.figure()
    fig.add_subplot(axes_class=Axes)
    with pytest.raises(ValueError):
        fig.add_subplot(axes_class=Axes, projection="3d")
    with pytest.raises(ValueError):
        fig.add_subplot(axes_class=Axes, polar=True)
    with pytest.raises(ValueError):
        fig.add_subplot(projection="3d", polar=True)
    with pytest.raises(TypeError):
        fig.add_subplot(projection=42)


def test_add_subplot_invalid():
    fig = plt.figure()
    with pytest.raises(ValueError,
                       match='Number of columns must be a positive integer'):
        fig.add_subplot(2, 0, 1)
    with pytest.raises(ValueError,
                       match='Number of rows must be a positive integer'):
        fig.add_subplot(0, 2, 1)
    with pytest.raises(ValueError, match='num must be 1 <= num <= 4'):
        fig.add_subplot(2, 2, 0)
    with pytest.raises(ValueError, match='num must be 1 <= num <= 4'):
        fig.add_subplot(2, 2, 5)

    with pytest.raises(ValueError, match='must be a three-digit integer'):
        fig.add_subplot(42)
    with pytest.raises(ValueError, match='must be a three-digit integer'):
        fig.add_subplot(1000)

    with pytest.raises(TypeError, match='takes 1 or 3 positional arguments '
                                        'but 2 were given'):
        fig.add_subplot(2, 2)
    with pytest.raises(TypeError, match='takes 1 or 3 positional arguments '
                                        'but 4 were given'):
        fig.add_subplot(1, 2, 3, 4)
    with pytest.raises(ValueError,
                       match="Number of rows must be a positive integer, "
                             "not '2'"):
        fig.add_subplot('2', 2, 1)
    with pytest.raises(ValueError,
                       match='Number of columns must be a positive integer, '
                             'not 2.0'):
        fig.add_subplot(2, 2.0, 1)
    _, ax = plt.subplots()
    with pytest.raises(ValueError,
                       match='The Subplot must have been created in the '
                             'present figure'):
        fig.add_subplot(ax)


@image_comparison(['figure_suptitle'])
def test_suptitle():
    fig, _ = plt.subplots()
    fig.suptitle('hello', color='r')
    fig.suptitle('title', color='g', rotation='30')


def test_suptitle_fontproperties():
    fig, ax = plt.subplots()
    fps = mpl.font_manager.FontProperties(size='large', weight='bold')
    txt = fig.suptitle('fontprops title', fontproperties=fps)
    assert txt.get_fontsize() == fps.get_size_in_points()
    assert txt.get_weight() == fps.get_weight()


@image_comparison(['alpha_background'],
                  # only test png and svg. The PDF output appears correct,
                  # but Ghostscript does not preserve the background color.
                  extensions=['png', 'svg'],
                  savefig_kwarg={'facecolor': (0, 1, 0.4),
                                 'edgecolor': 'none'})
def test_alpha():
    # We want an image which has a background color and an alpha of 0.4.
    fig = plt.figure(figsize=[2, 1])
    fig.set_facecolor((0, 1, 0.4))
    fig.patch.set_alpha(0.4)
    fig.patches.append(mpl.patches.CirclePolygon(
        [20, 20], radius=15, alpha=0.6, facecolor='red'))


def test_too_many_figures():
    with pytest.warns(RuntimeWarning):
        for i in range(rcParams['figure.max_open_warning'] + 1):
            plt.figure()


def test_iterability_axes_argument():

    # This is a regression test for matplotlib/matplotlib#3196. If one of the
    # arguments returned by _as_mpl_axes defines __getitem__ but is not
    # iterable, this would raise an exception. This is because we check
    # whether the arguments are iterable, and if so we try and convert them
    # to a tuple. However, the ``iterable`` function returns True if
    # __getitem__ is present, but some classes can define __getitem__ without
    # being iterable. The tuple conversion is now done in a try...except in
    # case it fails.

    class MyAxes(Axes):
        def __init__(self, *args, myclass=None, **kwargs):
            return Axes.__init__(self, *args, **kwargs)

    class MyClass:

        def __getitem__(self, item):
            if item != 'a':
                raise ValueError("item should be a")

        def _as_mpl_axes(self):
            return MyAxes, {'myclass': self}

    fig = plt.figure()
    fig.add_subplot(1, 1, 1, projection=MyClass())
    plt.close(fig)


def test_set_fig_size():
    fig = plt.figure()

    # check figwidth
    fig.set_figwidth(5)
    assert fig.get_figwidth() == 5

    # check figheight
    fig.set_figheight(1)
    assert fig.get_figheight() == 1

    # check using set_size_inches
    fig.set_size_inches(2, 4)
    assert fig.get_figwidth() == 2
    assert fig.get_figheight() == 4

    # check using tuple to first argument
    fig.set_size_inches((1, 3))
    assert fig.get_figwidth() == 1
    assert fig.get_figheight() == 3


def test_axes_remove():
    fig, axs = plt.subplots(2, 2)
    axs[-1, -1].remove()
    for ax in axs.ravel()[:-1]:
        assert ax in fig.axes
    assert axs[-1, -1] not in fig.axes
    assert len(fig.axes) == 3


def test_figaspect():
    w, h = plt.figaspect(np.float64(2) / np.float64(1))
    assert h / w == 2
    w, h = plt.figaspect(2)
    assert h / w == 2
    w, h = plt.figaspect(np.zeros((1, 2)))
    assert h / w == 0.5
    w, h = plt.figaspect(np.zeros((2, 2)))
    assert h / w == 1


@pytest.mark.parametrize('which', ['both', 'major', 'minor'])
def test_autofmt_xdate(which):
    date = ['3 Jan 2013', '4 Jan 2013', '5 Jan 2013', '6 Jan 2013',
            '7 Jan 2013', '8 Jan 2013', '9 Jan 2013', '10 Jan 2013',
            '11 Jan 2013', '12 Jan 2013', '13 Jan 2013', '14 Jan 2013']

    time = ['16:44:00', '16:45:00', '16:46:00', '16:47:00', '16:48:00',
            '16:49:00', '16:51:00', '16:52:00', '16:53:00', '16:55:00',
            '16:56:00', '16:57:00']

    angle = 60
    minors = [1, 2, 3, 4, 5, 6, 7]

    x = mdates.datestr2num(date)
    y = mdates.datestr2num(time)

    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.yaxis_date()
    ax.xaxis_date()

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            'FixedFormatter should only be used together with FixedLocator')
        ax.xaxis.set_minor_formatter(FixedFormatter(minors))

    fig.autofmt_xdate(0.2, angle, 'right', which)

    if which in ('both', 'major'):
        for label in fig.axes[0].get_xticklabels(False, 'major'):
            assert int(label.get_rotation()) == angle

    if which in ('both', 'minor'):
        for label in fig.axes[0].get_xticklabels(True, 'minor'):
            assert int(label.get_rotation()) == angle


@mpl.style.context('default')
def test_change_dpi():
    fig = plt.figure(figsize=(4, 4))
    fig.draw_without_rendering()
    assert fig.canvas.renderer.height == 400
    assert fig.canvas.renderer.width == 400
    fig.dpi = 50
    fig.draw_without_rendering()
    assert fig.canvas.renderer.height == 200
    assert fig.canvas.renderer.width == 200


@pytest.mark.parametrize('width, height', [
    (1, np.nan),
    (-1, 1),
    (np.inf, 1)
])
def test_invalid_figure_size(width, height):
    with pytest.raises(ValueError):
        plt.figure(figsize=(width, height))

    fig = plt.figure()
    with pytest.raises(ValueError):
        fig.set_size_inches(width, height)


def test_invalid_figure_add_axes():
    fig = plt.figure()
    with pytest.raises(ValueError):
        fig.add_axes((.1, .1, .5, np.nan))

    with pytest.raises(TypeError, match="multiple values for argument 'rect'"):
        fig.add_axes([0, 0, 1, 1], rect=[0, 0, 1, 1])

    _, ax = plt.subplots()
    with pytest.raises(ValueError,
                       match="The Axes must have been created in the present "
                             "figure"):
        fig.add_axes(ax)


def test_subplots_shareax_loglabels():
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False)
    for ax in axs.flat:
        ax.plot([10, 20, 30], [10, 20, 30])

    ax.set_yscale("log")
    ax.set_xscale("log")

    for ax in axs[0, :]:
        assert 0 == len(ax.xaxis.get_ticklabels(which='both'))

    for ax in axs[1, :]:
        assert 0 < len(ax.xaxis.get_ticklabels(which='both'))

    for ax in axs[:, 1]:
        assert 0 == len(ax.yaxis.get_ticklabels(which='both'))

    for ax in axs[:, 0]:
        assert 0 < len(ax.yaxis.get_ticklabels(which='both'))


def test_savefig():
    fig = plt.figure()
    msg = r"savefig\(\) takes 2 positional arguments but 3 were given"
    with pytest.raises(TypeError, match=msg):
        fig.savefig("fname1.png", "fname2.png")


def test_savefig_warns():
    fig = plt.figure()
    msg = r'savefig\(\) got unexpected keyword argument "non_existent_kwarg"'
    for format in ['png', 'pdf', 'svg', 'tif', 'jpg']:
        with pytest.warns(cbook.MatplotlibDeprecationWarning, match=msg):
            fig.savefig(io.BytesIO(), format=format, non_existent_kwarg=True)


def test_savefig_backend():
    fig = plt.figure()
    # Intentionally use an invalid module name.
    with pytest.raises(ModuleNotFoundError, match="No module named '@absent'"):
        fig.savefig("test", backend="module://@absent")
    with pytest.raises(ValueError,
                       match="The 'pdf' backend does not support png output"):
        fig.savefig("test.png", backend="pdf")


@pytest.mark.parametrize('backend', [
    pytest.param('Agg', marks=[pytest.mark.backend('Agg')]),
    pytest.param('Cairo', marks=[pytest.mark.backend('Cairo')]),
])
def test_savefig_pixel_ratio(backend):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        ratio1 = Image.open(buf)
        ratio1.load()

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    fig.canvas._set_device_pixel_ratio(2)
    with io.BytesIO() as buf:
        fig.savefig(buf, format='png')
        ratio2 = Image.open(buf)
        ratio2.load()

    assert ratio1 == ratio2


def test_figure_repr():
    fig = plt.figure(figsize=(10, 20), dpi=10)
    assert repr(fig) == "<Figure size 100x200 with 0 Axes>"


def test_valid_layouts():
    fig = Figure(layout=None)
    assert not fig.get_tight_layout()
    assert not fig.get_constrained_layout()

    fig = Figure(layout='tight')
    assert fig.get_tight_layout()
    assert not fig.get_constrained_layout()

    fig = Figure(layout='constrained')
    assert not fig.get_tight_layout()
    assert fig.get_constrained_layout()


def test_invalid_layouts():
    fig, ax = plt.subplots(constrained_layout=True)
    with pytest.warns(UserWarning):
        # this should warn,
        fig.subplots_adjust(top=0.8)
    assert not(fig.get_constrained_layout())

    # Using layout + (tight|constrained)_layout warns, but the former takes
    # precedence.
    with pytest.warns(UserWarning, match="Figure parameters 'layout' and "
                      "'tight_layout' cannot"):
        fig = Figure(layout='tight', tight_layout=False)
    assert fig.get_tight_layout()
    assert not fig.get_constrained_layout()
    with pytest.warns(UserWarning, match="Figure parameters 'layout' and "
                      "'constrained_layout' cannot"):
        fig = Figure(layout='constrained', constrained_layout=False)
    assert not fig.get_tight_layout()
    assert fig.get_constrained_layout()

    with pytest.raises(ValueError,
                       match="'foobar' is not a valid value for layout"):
        Figure(layout='foobar')


@check_figures_equal(extensions=["png", "pdf"])
def test_add_artist(fig_test, fig_ref):
    fig_test.set_dpi(100)
    fig_ref.set_dpi(100)

    fig_test.subplots()
    l1 = plt.Line2D([.2, .7], [.7, .7], gid='l1')
    l2 = plt.Line2D([.2, .7], [.8, .8], gid='l2')
    r1 = plt.Circle((20, 20), 100, transform=None, gid='C1')
    r2 = plt.Circle((.7, .5), .05, gid='C2')
    r3 = plt.Circle((4.5, .8), .55, transform=fig_test.dpi_scale_trans,
                    facecolor='crimson', gid='C3')
    for a in [l1, l2, r1, r2, r3]:
        fig_test.add_artist(a)
    l2.remove()

    ax2 = fig_ref.subplots()
    l1 = plt.Line2D([.2, .7], [.7, .7], transform=fig_ref.transFigure,
                    gid='l1', zorder=21)
    r1 = plt.Circle((20, 20), 100, transform=None, clip_on=False, zorder=20,
                    gid='C1')
    r2 = plt.Circle((.7, .5), .05, transform=fig_ref.transFigure, gid='C2',
                    zorder=20)
    r3 = plt.Circle((4.5, .8), .55, transform=fig_ref.dpi_scale_trans,
                    facecolor='crimson', clip_on=False, zorder=20, gid='C3')
    for a in [l1, r1, r2, r3]:
        ax2.add_artist(a)


@pytest.mark.parametrize("fmt", ["png", "pdf", "ps", "eps", "svg"])
def test_fspath(fmt, tmpdir):
    out = Path(tmpdir, "test.{}".format(fmt))
    plt.savefig(out)
    with out.open("rb") as file:
        # All the supported formats include the format name (case-insensitive)
        # in the first 100 bytes.
        assert fmt.encode("ascii") in file.read(100).lower()


def test_tightbbox():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    t = ax.text(1., 0.5, 'This dangles over end')
    renderer = fig.canvas.get_renderer()
    x1Nom0 = 9.035  # inches
    assert abs(t.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom0) < 0.05
    assert abs(fig.get_tightbbox(renderer).x0 - 0.679) < 0.05
    # now exclude t from the tight bbox so now the bbox is quite a bit
    # smaller
    t.set_in_layout(False)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom * fig.dpi) < 2
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom) < 0.05

    t.set_in_layout(True)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    # test bbox_extra_artists method...
    assert abs(ax.get_tightbbox(renderer, bbox_extra_artists=[]).x1
               - x1Nom * fig.dpi) < 2


def test_axes_removal():
    # Check that units can set the formatter after an Axes removal
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[1].remove()
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    assert isinstance(axs[0].xaxis.get_major_formatter(),
                      mdates.AutoDateFormatter)

    # Check that manually setting the formatter, then removing Axes keeps
    # the set formatter.
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[1].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].remove()
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    assert isinstance(axs[0].xaxis.get_major_formatter(),
                      ScalarFormatter)


def test_removed_axis():
    # Simple smoke test to make sure removing a shared axis works
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].remove()
    fig.canvas.draw()


@mpl.style.context('mpl20')
def test_picking_does_not_stale():
    fig, ax = plt.subplots()
    col = ax.scatter([0], [0], [1000], picker=True)
    fig.canvas.draw()
    assert not fig.stale

    mouse_event = SimpleNamespace(x=ax.bbox.x0 + ax.bbox.width / 2,
                                  y=ax.bbox.y0 + ax.bbox.height / 2,
                                  inaxes=ax, guiEvent=None)
    fig.pick(mouse_event)
    assert not fig.stale


def test_add_subplot_twotuple():
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, (3, 5))
    assert ax1.get_subplotspec().rowspan == range(1, 3)
    assert ax1.get_subplotspec().colspan == range(0, 1)
    ax2 = fig.add_subplot(3, 2, (4, 6))
    assert ax2.get_subplotspec().rowspan == range(1, 3)
    assert ax2.get_subplotspec().colspan == range(1, 2)
    ax3 = fig.add_subplot(3, 2, (3, 6))
    assert ax3.get_subplotspec().rowspan == range(1, 3)
    assert ax3.get_subplotspec().colspan == range(0, 2)
    ax4 = fig.add_subplot(3, 2, (4, 5))
    assert ax4.get_subplotspec().rowspan == range(1, 3)
    assert ax4.get_subplotspec().colspan == range(0, 2)
    with pytest.raises(IndexError):
        fig.add_subplot(3, 2, (6, 3))


@image_comparison(['tightbbox_box_aspect.svg'], style='mpl20',
                  savefig_kwarg={'bbox_inches': 'tight',
                                 'facecolor': 'teal'},
                  remove_text=True)
def test_tightbbox_box_aspect():
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax1.set_box_aspect(.5)
    ax2.set_box_aspect((2, 1, 1))


@check_figures_equal(extensions=["svg", "pdf", "eps", "png"])
def test_animated_with_canvas_change(fig_test, fig_ref):
    ax_ref = fig_ref.subplots()
    ax_ref.plot(range(5))

    ax_test = fig_test.subplots()
    ax_test.plot(range(5), animated=True)


class TestSubplotMosaic:
    @check_figures_equal(extensions=["png"])
    @pytest.mark.parametrize(
        "x", [[["A", "A", "B"], ["C", "D", "B"]], [[1, 1, 2], [3, 4, 2]]]
    )
    def test_basic(self, fig_test, fig_ref, x):
        grid_axes = fig_test.subplot_mosaic(x)

        for k, ax in grid_axes.items():
            ax.set_title(k)

        labels = sorted(np.unique(x))

        assert len(labels) == len(grid_axes)

        gs = fig_ref.add_gridspec(2, 3)
        axA = fig_ref.add_subplot(gs[:1, :2])
        axA.set_title(labels[0])

        axB = fig_ref.add_subplot(gs[:, 2])
        axB.set_title(labels[1])

        axC = fig_ref.add_subplot(gs[1, 0])
        axC.set_title(labels[2])

        axD = fig_ref.add_subplot(gs[1, 1])
        axD.set_title(labels[3])

    @check_figures_equal(extensions=["png"])
    def test_all_nested(self, fig_test, fig_ref):
        x = [["A", "B"], ["C", "D"]]
        y = [["E", "F"], ["G", "H"]]

        fig_ref.set_constrained_layout(True)
        fig_test.set_constrained_layout(True)

        grid_axes = fig_test.subplot_mosaic([[x, y]])
        for ax in grid_axes.values():
            ax.set_title(ax.get_label())

        gs = fig_ref.add_gridspec(1, 2)
        gs_left = gs[0, 0].subgridspec(2, 2)
        for j, r in enumerate(x):
            for k, label in enumerate(r):
                fig_ref.add_subplot(gs_left[j, k]).set_title(label)

        gs_right = gs[0, 1].subgridspec(2, 2)
        for j, r in enumerate(y):
            for k, label in enumerate(r):
                fig_ref.add_subplot(gs_right[j, k]).set_title(label)

    @check_figures_equal(extensions=["png"])
    def test_nested(self, fig_test, fig_ref):

        fig_ref.set_constrained_layout(True)
        fig_test.set_constrained_layout(True)

        x = [["A", "B"], ["C", "D"]]

        y = [["F"], [x]]

        grid_axes = fig_test.subplot_mosaic(y)

        for k, ax in grid_axes.items():
            ax.set_title(k)

        gs = fig_ref.add_gridspec(2, 1)

        gs_n = gs[1, 0].subgridspec(2, 2)

        axA = fig_ref.add_subplot(gs_n[0, 0])
        axA.set_title("A")

        axB = fig_ref.add_subplot(gs_n[0, 1])
        axB.set_title("B")

        axC = fig_ref.add_subplot(gs_n[1, 0])
        axC.set_title("C")

        axD = fig_ref.add_subplot(gs_n[1, 1])
        axD.set_title("D")

        axF = fig_ref.add_subplot(gs[0, 0])
        axF.set_title("F")

    @check_figures_equal(extensions=["png"])
    def test_nested_tuple(self, fig_test, fig_ref):
        x = [["A", "B", "B"], ["C", "C", "D"]]
        xt = (("A", "B", "B"), ("C", "C", "D"))

        fig_ref.subplot_mosaic([["F"], [x]])
        fig_test.subplot_mosaic([["F"], [xt]])

    @check_figures_equal(extensions=["png"])
    @pytest.mark.parametrize(
        "x, empty_sentinel",
        [
            ([["A", None], [None, "B"]], None),
            ([["A", "."], [".", "B"]], "SKIP"),
            ([["A", 0], [0, "B"]], 0),
            ([[1, None], [None, 2]], None),
            ([[1, "."], [".", 2]], "SKIP"),
            ([[1, 0], [0, 2]], 0),
        ],
    )
    def test_empty(self, fig_test, fig_ref, x, empty_sentinel):
        if empty_sentinel != "SKIP":
            kwargs = {"empty_sentinel": empty_sentinel}
        else:
            kwargs = {}
        grid_axes = fig_test.subplot_mosaic(x, **kwargs)

        for k, ax in grid_axes.items():
            ax.set_title(k)

        labels = sorted(
            {name for row in x for name in row} - {empty_sentinel, "."}
        )

        assert len(labels) == len(grid_axes)

        gs = fig_ref.add_gridspec(2, 2)
        axA = fig_ref.add_subplot(gs[0, 0])
        axA.set_title(labels[0])

        axB = fig_ref.add_subplot(gs[1, 1])
        axB.set_title(labels[1])

    def test_fail_list_of_str(self):
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic(['foo', 'bar'])
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic(['foo'])

    @check_figures_equal(extensions=["png"])
    @pytest.mark.parametrize("subplot_kw", [{}, {"projection": "polar"}, None])
    def test_subplot_kw(self, fig_test, fig_ref, subplot_kw):
        x = [[1, 2]]
        grid_axes = fig_test.subplot_mosaic(x, subplot_kw=subplot_kw)
        subplot_kw = subplot_kw or {}

        gs = fig_ref.add_gridspec(1, 2)
        axA = fig_ref.add_subplot(gs[0, 0], **subplot_kw)

        axB = fig_ref.add_subplot(gs[0, 1], **subplot_kw)

    def test_string_parser(self):
        normalize = Figure._normalize_grid_string
        assert normalize('ABC') == [['A', 'B', 'C']]
        assert normalize('AB;CC') == [['A', 'B'], ['C', 'C']]
        assert normalize('AB;CC;DE') == [['A', 'B'], ['C', 'C'], ['D', 'E']]
        assert normalize("""
                         ABC
                         """) == [['A', 'B', 'C']]
        assert normalize("""
                         AB
                         CC
                         """) == [['A', 'B'], ['C', 'C']]
        assert normalize("""
                         AB
                         CC
                         DE
                         """) == [['A', 'B'], ['C', 'C'], ['D', 'E']]

    @check_figures_equal(extensions=["png"])
    @pytest.mark.parametrize("str_pattern",
                             ["AAA\nBBB", "\nAAA\nBBB\n", "ABC\nDEF"]
                             )
    def test_single_str_input(self, fig_test, fig_ref, str_pattern):
        grid_axes = fig_test.subplot_mosaic(str_pattern)

        grid_axes = fig_ref.subplot_mosaic(
            [list(ln) for ln in str_pattern.strip().split("\n")]
        )

    @pytest.mark.parametrize(
        "x,match",
        [
            (
                [["A", "."], [".", "A"]],
                (
                    "(?m)we found that the label .A. specifies a "
                    + "non-rectangular or non-contiguous area."
                ),
            ),
            (
                [["A", "B"], [None, [["A", "B"], ["C", "D"]]]],
                "There are duplicate keys .* between the outer layout",
            ),
            ("AAA\nc\nBBB", "All of the rows must be the same length"),
            (
                [["A", [["B", "C"], ["D"]]], ["E", "E"]],
                "All of the rows must be the same length",
            ),
        ],
    )
    def test_fail(self, x, match):
        fig = plt.figure()
        with pytest.raises(ValueError, match=match):
            fig.subplot_mosaic(x)

    @check_figures_equal(extensions=["png"])
    def test_hashable_keys(self, fig_test, fig_ref):
        fig_test.subplot_mosaic([[object(), object()]])
        fig_ref.subplot_mosaic([["A", "B"]])

    @pytest.mark.parametrize('str_pattern',
                             ['abc', 'cab', 'bca', 'cba', 'acb', 'bac'])
    def test_user_order(self, str_pattern):
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(str_pattern)
        assert list(str_pattern) == list(ax_dict)
        assert list(fig.axes) == list(ax_dict.values())

    def test_nested_user_order(self):
        layout = [
            ["A", [["B", "C"],
                   ["D", "E"]]],
            ["F", "G"],
            [".", [["H", [["I"],
                          ["."]]]]]
        ]

        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(layout)
        assert list(ax_dict) == list("ABCDEFGHI")
        assert list(fig.axes) == list(ax_dict.values())

    def test_share_all(self):
        layout = [
            ["A", [["B", "C"],
                   ["D", "E"]]],
            ["F", "G"],
            [".", [["H", [["I"],
                          ["."]]]]]
        ]
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(layout, sharex=True, sharey=True)
        ax_dict["A"].set(xscale="log", yscale="logit")
        assert all(ax.get_xscale() == "log" and ax.get_yscale() == "logit"
                   for ax in ax_dict.values())


def test_reused_gridspec():
    """Test that these all use the same gridspec"""
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 2, (3, 5))
    ax2 = fig.add_subplot(3, 2, 4)
    ax3 = plt.subplot2grid((3, 2), (2, 1), colspan=2, fig=fig)

    gs1 = ax1.get_subplotspec().get_gridspec()
    gs2 = ax2.get_subplotspec().get_gridspec()
    gs3 = ax3.get_subplotspec().get_gridspec()

    assert gs1 == gs2
    assert gs1 == gs3


@image_comparison(['test_subfigure.png'], style='mpl20',
                  savefig_kwarg={'facecolor': 'teal'},
                  remove_text=False)
def test_subfigure():
    np.random.seed(19680801)
    fig = plt.figure(constrained_layout=True)
    sub = fig.subfigures(1, 2)

    axs = sub[0].subplots(2, 2)
    for ax in axs.flat:
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2, vmax=2)
    sub[0].colorbar(pc, ax=axs)
    sub[0].suptitle('Left Side')

    axs = sub[1].subplots(1, 3)
    for ax in axs.flat:
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2, vmax=2)
    sub[1].colorbar(pc, ax=axs, location='bottom')
    sub[1].suptitle('Right Side')

    fig.suptitle('Figure suptitle', fontsize='xx-large')


def test_subfigure_tightbbox():
    # test that we can get the tightbbox with a subfigure...
    fig = plt.figure(constrained_layout=True)
    sub = fig.subfigures(1, 2)

    np.testing.assert_allclose(
            fig.get_tightbbox(fig.canvas.get_renderer()).width,
            8.0)


@image_comparison(['test_subfigure_ss.png'], style='mpl20',
                  savefig_kwarg={'facecolor': 'teal'},
                  remove_text=False)
def test_subfigure_ss():
    # test assigning the subfigure via subplotspec
    np.random.seed(19680801)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    sub = fig.add_subfigure(gs[0], facecolor='pink')

    axs = sub.subplots(2, 2)
    for ax in axs.flat:
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2, vmax=2)
    sub.colorbar(pc, ax=axs)
    sub.suptitle('Left Side')

    ax = fig.add_subplot(gs[1])
    ax.plot(np.arange(20))
    ax.set_title('Axes')

    fig.suptitle('Figure suptitle', fontsize='xx-large')


@image_comparison(['test_subfigure_double.png'], style='mpl20',
                  savefig_kwarg={'facecolor': 'teal'},
                  remove_text=False)
def test_subfigure_double():
    # test assigning the subfigure via subplotspec
    np.random.seed(19680801)

    fig = plt.figure(constrained_layout=True, figsize=(10, 8))

    fig.suptitle('fig')

    subfigs = fig.subfigures(1, 2, wspace=0.07)

    subfigs[0].set_facecolor('coral')
    subfigs[0].suptitle('subfigs[0]')

    subfigs[1].set_facecolor('coral')
    subfigs[1].suptitle('subfigs[1]')

    subfigsnest = subfigs[0].subfigures(2, 1, height_ratios=[1, 1.4])
    subfigsnest[0].suptitle('subfigsnest[0]')
    subfigsnest[0].set_facecolor('r')
    axsnest0 = subfigsnest[0].subplots(1, 2, sharey=True)
    for ax in axsnest0:
        fontsize = 12
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)

    subfigsnest[0].colorbar(pc, ax=axsnest0)

    subfigsnest[1].suptitle('subfigsnest[1]')
    subfigsnest[1].set_facecolor('g')
    axsnest1 = subfigsnest[1].subplots(3, 1, sharex=True)
    for nn, ax in enumerate(axsnest1):
        ax.set_ylabel(f'ylabel{nn}')
    subfigsnest[1].supxlabel('supxlabel')
    subfigsnest[1].supylabel('supylabel')

    axsRight = subfigs[1].subplots(2, 2)


def test_subfigure_spanning():
    # test that subfigures get laid out properly...
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    sub_figs = [
        fig.add_subfigure(gs[0, 0]),
        fig.add_subfigure(gs[0:2, 1]),
        fig.add_subfigure(gs[2, 1:3]),
        fig.add_subfigure(gs[0:, 1:])
    ]

    w = 640
    h = 480
    np.testing.assert_allclose(sub_figs[0].bbox.min, [0., h * 2/3])
    np.testing.assert_allclose(sub_figs[0].bbox.max, [w / 3, h])

    np.testing.assert_allclose(sub_figs[1].bbox.min, [w / 3, h / 3])
    np.testing.assert_allclose(sub_figs[1].bbox.max, [w * 2/3, h])

    np.testing.assert_allclose(sub_figs[2].bbox.min, [w / 3, 0])
    np.testing.assert_allclose(sub_figs[2].bbox.max, [w, h / 3])

    # check here that slicing actually works.  Last sub_fig
    # with open slices failed, but only on draw...
    for i in range(4):
        sub_figs[i].add_subplot()
    fig.draw_without_rendering()


@mpl.style.context('mpl20')
def test_subfigure_ticks():
    # This tests a tick-spacing error that only seems applicable
    # when the subfigures are saved to file.  It is very hard to replicate
    fig = plt.figure(constrained_layout=True, figsize=(10, 3))
    # create left/right subfigs nested in bottom subfig
    (subfig_bl, subfig_br) = fig.subfigures(1, 2, wspace=0.01,
                                            width_ratios=[7, 2])

    # put ax1-ax3 in gridspec of bottom-left subfig
    gs = subfig_bl.add_gridspec(nrows=1, ncols=14)

    ax1 = subfig_bl.add_subplot(gs[0, :1])
    ax1.scatter(x=[-56.46881504821776, 24.179891162109396], y=[1500, 3600])

    ax2 = subfig_bl.add_subplot(gs[0, 1:3], sharey=ax1)
    ax2.scatter(x=[-126.5357270050049, 94.68456736755368], y=[1500, 3600])
    ax3 = subfig_bl.add_subplot(gs[0, 3:14], sharey=ax1)

    fig.set_dpi(120)
    fig.draw_without_rendering()
    ticks120 = ax2.get_xticks()
    fig.set_dpi(300)
    fig.draw_without_rendering()
    ticks300 = ax2.get_xticks()
    np.testing.assert_allclose(ticks120, ticks300)


@image_comparison(['test_subfigure_scatter_size.png'], style='mpl20',
                   remove_text=True)
def test_subfigure_scatter_size():
    # markers in the left- and right-most subplots should be the same
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(gs[1])
    ax0.scatter([1, 2, 3], [1, 2, 3], s=30, marker='s')
    ax0.scatter([3, 4, 5], [1, 2, 3], s=[20, 30, 40], marker='s')

    sfig = fig.add_subfigure(gs[0])
    axs = sfig.subplots(1, 2)
    for ax in [ax0, axs[0]]:
        ax.scatter([1, 2, 3], [1, 2, 3], s=30, marker='s', color='r')
        ax.scatter([3, 4, 5], [1, 2, 3], s=[20, 30, 40], marker='s', color='g')


def test_add_subplot_kwargs():
    # fig.add_subplot() always creates new axes, even if axes kwargs differ.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax1 = fig.add_subplot(1, 1, 1)
    assert ax is not None
    assert ax1 is not ax
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    ax1 = fig.add_subplot(1, 1, 1, projection='polar')
    assert ax is not None
    assert ax1 is not ax
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    ax1 = fig.add_subplot(1, 1, 1)
    assert ax is not None
    assert ax1.name == 'rectilinear'
    assert ax1 is not ax
    plt.close()


def test_add_axes_kwargs():
    # fig.add_axes() always creates new axes, even if axes kwargs differ.
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax1 = fig.add_axes([0, 0, 1, 1])
    assert ax is not None
    assert ax1 is not ax
    plt.close()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='polar')
    ax1 = fig.add_axes([0, 0, 1, 1], projection='polar')
    assert ax is not None
    assert ax1 is not ax
    plt.close()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='polar')
    ax1 = fig.add_axes([0, 0, 1, 1])
    assert ax is not None
    assert ax1.name == 'rectilinear'
    assert ax1 is not ax
    plt.close()


def test_ginput(recwarn):  # recwarn undoes warn filters at exit.
    warnings.filterwarnings("ignore", "cannot show the figure")
    fig, ax = plt.subplots()

    def single_press():
        fig.canvas.button_press_event(*ax.transData.transform((.1, .2)), 1)

    Timer(.1, single_press).start()
    assert fig.ginput() == [(.1, .2)]

    def multi_presses():
        fig.canvas.button_press_event(*ax.transData.transform((.1, .2)), 1)
        fig.canvas.key_press_event("backspace")
        fig.canvas.button_press_event(*ax.transData.transform((.3, .4)), 1)
        fig.canvas.button_press_event(*ax.transData.transform((.5, .6)), 1)
        fig.canvas.button_press_event(*ax.transData.transform((0, 0)), 2)

    Timer(.1, multi_presses).start()
    np.testing.assert_allclose(fig.ginput(3), [(.3, .4), (.5, .6)])


def test_waitforbuttonpress(recwarn):  # recwarn undoes warn filters at exit.
    warnings.filterwarnings("ignore", "cannot show the figure")
    fig = plt.figure()
    assert fig.waitforbuttonpress(timeout=.1) is None
    Timer(.1, fig.canvas.key_press_event, ("z",)).start()
    assert fig.waitforbuttonpress() is True
    Timer(.1, fig.canvas.button_press_event, (0, 0, 1)).start()
    assert fig.waitforbuttonpress() is False


def test_kwargs_pass():
    fig = Figure(label='whole Figure')
    sub_fig = fig.subfigures(1, 1, label='sub figure')

    assert fig.get_label() == 'whole Figure'
    assert sub_fig.get_label() == 'sub figure'
