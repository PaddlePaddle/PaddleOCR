from datetime import datetime
import io
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.text import Text


needs_usetex = pytest.mark.skipif(
    not mpl.checkdep_usetex(True),
    reason="This test needs a TeX installation")


@image_comparison(['font_styles'])
def test_font_styles():

    def find_matplotlib_font(**kw):
        prop = FontProperties(**kw)
        path = findfont(prop, directory=mpl.get_data_path())
        return FontProperties(fname=path)

    from matplotlib.font_manager import FontProperties, findfont
    warnings.filterwarnings(
        'ignore',
        r"findfont: Font family \[u?'Foo'\] not found. Falling back to .",
        UserWarning,
        module='matplotlib.font_manager')

    fig, ax = plt.subplots()

    normal_font = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        size=14)
    ax.annotate(
        "Normal Font",
        (0.1, 0.1),
        xycoords='axes fraction',
        fontproperties=normal_font)

    bold_font = find_matplotlib_font(
        family="Foo",
        style="normal",
        variant="normal",
        weight="bold",
        stretch=500,
        size=14)
    ax.annotate(
        "Bold Font",
        (0.1, 0.2),
        xycoords='axes fraction',
        fontproperties=bold_font)

    bold_italic_font = find_matplotlib_font(
        family="sans serif",
        style="italic",
        variant="normal",
        weight=750,
        stretch=500,
        size=14)
    ax.annotate(
        "Bold Italic Font",
        (0.1, 0.3),
        xycoords='axes fraction',
        fontproperties=bold_italic_font)

    light_font = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight=200,
        stretch=500,
        size=14)
    ax.annotate(
        "Light Font",
        (0.1, 0.4),
        xycoords='axes fraction',
        fontproperties=light_font)

    condensed_font = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight=500,
        stretch=100,
        size=14)
    ax.annotate(
        "Condensed Font",
        (0.1, 0.5),
        xycoords='axes fraction',
        fontproperties=condensed_font)

    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(['multiline'])
def test_multiline():
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("multiline\ntext alignment")

    plt.text(
        0.2, 0.5, "TpTpTp\n$M$\nTpTpTp", size=20, ha="center", va="top")

    plt.text(
        0.5, 0.5, "TpTpTp\n$M^{M^{M^{M}}}$\nTpTpTp", size=20,
        ha="center", va="top")

    plt.text(
        0.8, 0.5, "TpTpTp\n$M_{q_{q_{q}}}$\nTpTpTp", size=20,
        ha="center", va="top")

    plt.xlim(0, 1)
    plt.ylim(0, 0.8)

    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(['multiline2'], style='mpl20')
def test_multiline2():
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6

    fig, ax = plt.subplots()

    ax.set_xlim([0, 1.4])
    ax.set_ylim([0, 2])
    ax.axhline(0.5, color='C2', linewidth=0.3)
    sts = ['Line', '2 Lineg\n 2 Lg', '$\\sum_i x $', 'hi $\\sum_i x $\ntest',
           'test\n $\\sum_i x $', '$\\sum_i x $\n $\\sum_i x $']
    renderer = fig.canvas.get_renderer()

    def draw_box(ax, tt):
        r = mpatches.Rectangle((0, 0), 1, 1, clip_on=False,
                               transform=ax.transAxes)
        r.set_bounds(
            tt.get_window_extent(renderer)
            .transformed(ax.transAxes.inverted())
            .bounds)
        ax.add_patch(r)

    horal = 'left'
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 0.5, st, horizontalalignment=horal,
                     verticalalignment='bottom')
        draw_box(ax, tt)
    ax.text(1.2, 0.5, 'Bottom align', color='C2')

    ax.axhline(1.3, color='C2', linewidth=0.3)
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 1.3, st, horizontalalignment=horal,
                     verticalalignment='top')
        draw_box(ax, tt)
    ax.text(1.2, 1.3, 'Top align', color='C2')

    ax.axhline(1.8, color='C2', linewidth=0.3)
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 1.8, st, horizontalalignment=horal,
                     verticalalignment='baseline')
        draw_box(ax, tt)
    ax.text(1.2, 1.8, 'Baseline align', color='C2')

    ax.axhline(0.1, color='C2', linewidth=0.3)
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 0.1, st, horizontalalignment=horal,
                     verticalalignment='bottom', rotation=20)
        draw_box(ax, tt)
    ax.text(1.2, 0.1, 'Bot align, rot20', color='C2')


@image_comparison(['antialiased.png'])
def test_antialiasing():
    mpl.rcParams['text.antialiased'] = True

    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.75, "antialiased", horizontalalignment='center',
             verticalalignment='center')
    fig.text(0.5, 0.25, r"$\sqrt{x}$", horizontalalignment='center',
             verticalalignment='center')
    # NOTE: We don't need to restore the rcParams here, because the
    # test cleanup will do it for us.  In fact, if we do it here, it
    # will turn antialiasing back off before the images are actually
    # rendered.


def test_afm_kerning():
    fn = mpl.font_manager.findfont("Helvetica", fontext="afm")
    with open(fn, 'rb') as fh:
        afm = mpl.afm.AFM(fh)
    assert afm.string_width_height('VAVAVAVAVAVA') == (7174.0, 718)


@image_comparison(['text_contains.png'])
def test_contains():
    fig = plt.figure()
    ax = plt.axes()

    mevent = MouseEvent('button_press_event', fig.canvas, 0.5, 0.5, 1, None)

    xs = np.linspace(0.25, 0.75, 30)
    ys = np.linspace(0.25, 0.75, 30)
    xs, ys = np.meshgrid(xs, ys)

    txt = plt.text(
        0.5, 0.4, 'hello world', ha='center', fontsize=30, rotation=30)
    # uncomment to draw the text's bounding box
    # txt.set_bbox(dict(edgecolor='black', facecolor='none'))

    # draw the text. This is important, as the contains method can only work
    # when a renderer exists.
    fig.canvas.draw()

    for x, y in zip(xs.flat, ys.flat):
        mevent.x, mevent.y = plt.gca().transAxes.transform([x, y])
        contains, _ = txt.contains(mevent)
        color = 'yellow' if contains else 'red'

        # capture the viewLim, plot a point, and reset the viewLim
        vl = ax.viewLim.frozen()
        ax.plot(x, y, 'o', color=color)
        ax.viewLim.set(vl)


def test_annotation_contains():
    # Check that Annotation.contains looks at the bboxes of the text and the
    # arrow separately, not at the joint bbox.
    fig, ax = plt.subplots()
    ann = ax.annotate(
        "hello", xy=(.4, .4), xytext=(.6, .6), arrowprops={"arrowstyle": "->"})
    fig.canvas.draw()   # Needed for the same reason as in test_contains.
    event = MouseEvent(
        "button_press_event", fig.canvas, *ax.transData.transform((.5, .6)))
    assert ann.contains(event) == (False, {})


@image_comparison(['titles'])
def test_titles():
    # left and right side titles
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title("left title", loc="left")
    ax.set_title("right title", loc="right")
    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(['text_alignment'], style='mpl20')
def test_alignment():
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    x = 0.1
    for rotation in (0, 30):
        for alignment in ('top', 'bottom', 'baseline', 'center'):
            ax.text(
                x, 0.5, alignment + " Tj", va=alignment, rotation=rotation,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.text(
                x, 1.0, r'$\sum_{i=0}^{j}$', va=alignment, rotation=rotation)
            x += 0.1

    ax.plot([0, 1], [0.5, 0.5])
    ax.plot([0, 1], [1.0, 1.0])

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.5])
    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(['axes_titles.png'])
def test_axes_titles():
    # Related to issue #3327
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.set_title('center', loc='center', fontsize=20, fontweight=700)
    ax.set_title('left', loc='left', fontsize=12, fontweight=400)
    ax.set_title('right', loc='right', fontsize=12, fontweight=400)


def test_set_position():
    fig, ax = plt.subplots()

    # test set_position
    ann = ax.annotate(
        'test', (0, 0), xytext=(0, 0), textcoords='figure pixels')
    fig.canvas.draw()

    init_pos = ann.get_window_extent(fig.canvas.renderer)
    shift_val = 15
    ann.set_position((shift_val, shift_val))
    fig.canvas.draw()
    post_pos = ann.get_window_extent(fig.canvas.renderer)

    for a, b in zip(init_pos.min, post_pos.min):
        assert a + shift_val == b

    # test xyann
    ann = ax.annotate(
        'test', (0, 0), xytext=(0, 0), textcoords='figure pixels')
    fig.canvas.draw()

    init_pos = ann.get_window_extent(fig.canvas.renderer)
    shift_val = 15
    ann.xyann = (shift_val, shift_val)
    fig.canvas.draw()
    post_pos = ann.get_window_extent(fig.canvas.renderer)

    for a, b in zip(init_pos.min, post_pos.min):
        assert a + shift_val == b


@pytest.mark.parametrize('text', ['', 'O'], ids=['empty', 'non-empty'])
def test_non_default_dpi(text):
    fig, ax = plt.subplots()

    t1 = ax.text(0.5, 0.5, text, ha='left', va='bottom')
    fig.canvas.draw()
    dpi = fig.dpi

    bbox1 = t1.get_window_extent()
    bbox2 = t1.get_window_extent(dpi=dpi * 10)
    np.testing.assert_allclose(bbox2.get_points(), bbox1.get_points() * 10,
                               rtol=5e-2)
    # Text.get_window_extent should not permanently change dpi.
    assert fig.dpi == dpi


def test_get_rotation_string():
    assert mpl.text.get_rotation('horizontal') == 0.
    assert mpl.text.get_rotation('vertical') == 90.
    assert mpl.text.get_rotation('15.') == 15.


def test_get_rotation_float():
    for i in [15., 16.70, 77.4]:
        assert mpl.text.get_rotation(i) == i


def test_get_rotation_int():
    for i in [67, 16, 41]:
        assert mpl.text.get_rotation(i) == float(i)


def test_get_rotation_raises():
    with pytest.raises(ValueError):
        mpl.text.get_rotation('hozirontal')


def test_get_rotation_none():
    assert mpl.text.get_rotation(None) == 0.0


def test_get_rotation_mod360():
    for i, j in zip([360., 377., 720+177.2], [0., 17., 177.2]):
        assert_almost_equal(mpl.text.get_rotation(i), j)


@pytest.mark.parametrize("ha", ["center", "right", "left"])
@pytest.mark.parametrize("va", ["center", "top", "bottom",
                                "baseline", "center_baseline"])
def test_null_rotation_with_rotation_mode(ha, va):
    fig, ax = plt.subplots()
    kw = dict(rotation=0, va=va, ha=ha)
    t0 = ax.text(.5, .5, 'test', rotation_mode='anchor', **kw)
    t1 = ax.text(.5, .5, 'test', rotation_mode='default', **kw)
    fig.canvas.draw()
    assert_almost_equal(t0.get_window_extent(fig.canvas.renderer).get_points(),
                        t1.get_window_extent(fig.canvas.renderer).get_points())


@image_comparison(['text_bboxclip'])
def test_bbox_clipping():
    plt.text(0.9, 0.2, 'Is bbox clipped?', backgroundcolor='r', clip_on=True)
    t = plt.text(0.9, 0.5, 'Is fancy bbox clipped?', clip_on=True)
    t.set_bbox({"boxstyle": "round, pad=0.1"})


@image_comparison(['annotation_negative_ax_coords.png'])
def test_annotation_negative_ax_coords():
    fig, ax = plt.subplots()

    ax.annotate('+ pts',
                xytext=[30, 20], textcoords='axes points',
                xy=[30, 20], xycoords='axes points', fontsize=32)
    ax.annotate('- pts',
                xytext=[30, -20], textcoords='axes points',
                xy=[30, -20], xycoords='axes points', fontsize=32,
                va='top')
    ax.annotate('+ frac',
                xytext=[0.75, 0.05], textcoords='axes fraction',
                xy=[0.75, 0.05], xycoords='axes fraction', fontsize=32)
    ax.annotate('- frac',
                xytext=[0.75, -0.05], textcoords='axes fraction',
                xy=[0.75, -0.05], xycoords='axes fraction', fontsize=32,
                va='top')

    ax.annotate('+ pixels',
                xytext=[160, 25], textcoords='axes pixels',
                xy=[160, 25], xycoords='axes pixels', fontsize=32)
    ax.annotate('- pixels',
                xytext=[160, -25], textcoords='axes pixels',
                xy=[160, -25], xycoords='axes pixels', fontsize=32,
                va='top')


@image_comparison(['annotation_negative_fig_coords.png'])
def test_annotation_negative_fig_coords():
    fig, ax = plt.subplots()

    ax.annotate('+ pts',
                xytext=[10, 120], textcoords='figure points',
                xy=[10, 120], xycoords='figure points', fontsize=32)
    ax.annotate('- pts',
                xytext=[-10, 180], textcoords='figure points',
                xy=[-10, 180], xycoords='figure points', fontsize=32,
                va='top')
    ax.annotate('+ frac',
                xytext=[0.05, 0.55], textcoords='figure fraction',
                xy=[0.05, 0.55], xycoords='figure fraction', fontsize=32)
    ax.annotate('- frac',
                xytext=[-0.05, 0.5], textcoords='figure fraction',
                xy=[-0.05, 0.5], xycoords='figure fraction', fontsize=32,
                va='top')

    ax.annotate('+ pixels',
                xytext=[50, 50], textcoords='figure pixels',
                xy=[50, 50], xycoords='figure pixels', fontsize=32)
    ax.annotate('- pixels',
                xytext=[-50, 100], textcoords='figure pixels',
                xy=[-50, 100], xycoords='figure pixels', fontsize=32,
                va='top')


def test_text_stale():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.draw_all()
    assert not ax1.stale
    assert not ax2.stale
    assert not fig.stale

    txt1 = ax1.text(.5, .5, 'aardvark')
    assert ax1.stale
    assert txt1.stale
    assert fig.stale

    ann1 = ax2.annotate('aardvark', xy=[.5, .5])
    assert ax2.stale
    assert ann1.stale
    assert fig.stale

    plt.draw_all()
    assert not ax1.stale
    assert not ax2.stale
    assert not fig.stale


@image_comparison(['agg_text_clip.png'])
def test_agg_text_clip():
    np.random.seed(1)
    fig, (ax1, ax2) = plt.subplots(2)
    for x, y in np.random.rand(10, 2):
        ax1.text(x, y, "foo", clip_on=True)
        ax2.text(x, y, "foo")


def test_text_size_binding():
    mpl.rcParams['font.size'] = 10
    fp = mpl.font_manager.FontProperties(size='large')
    sz1 = fp.get_size_in_points()
    mpl.rcParams['font.size'] = 100

    assert sz1 == fp.get_size_in_points()


@image_comparison(['font_scaling.pdf'])
def test_font_scaling():
    mpl.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(6.4, 12.4))
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylim(-10, 600)

    for i, fs in enumerate(range(4, 43, 2)):
        ax.text(0.1, i*30, "{fs} pt font size".format(fs=fs), fontsize=fs)


@pytest.mark.parametrize('spacing1, spacing2', [(0.4, 2), (2, 0.4), (2, 2)])
def test_two_2line_texts(spacing1, spacing2):
    text_string = 'line1\nline2'
    fig = plt.figure()
    renderer = fig.canvas.get_renderer()

    text1 = plt.text(0.25, 0.5, text_string, linespacing=spacing1)
    text2 = plt.text(0.25, 0.5, text_string, linespacing=spacing2)
    fig.canvas.draw()

    box1 = text1.get_window_extent(renderer=renderer)
    box2 = text2.get_window_extent(renderer=renderer)

    # line spacing only affects height
    assert box1.width == box2.width
    if spacing1 == spacing2:
        assert box1.height == box2.height
    else:
        assert box1.height != box2.height


def test_nonfinite_pos():
    fig, ax = plt.subplots()
    ax.text(0, np.nan, 'nan')
    ax.text(np.inf, 0, 'inf')
    fig.canvas.draw()


def test_hinting_factor_backends():
    plt.rcParams['text.hinting_factor'] = 1
    fig = plt.figure()
    t = fig.text(0.5, 0.5, 'some text')

    fig.savefig(io.BytesIO(), format='svg')
    expected = t.get_window_extent().intervalx

    fig.savefig(io.BytesIO(), format='png')
    # Backends should apply hinting_factor consistently (within 10%).
    np.testing.assert_allclose(t.get_window_extent().intervalx, expected,
                               rtol=0.1)


@needs_usetex
def test_usetex_is_copied():
    # Indirectly tests that update_from (which is used to copy tick label
    # properties) copies usetex state.
    fig = plt.figure()
    plt.rcParams["text.usetex"] = False
    ax1 = fig.add_subplot(121)
    plt.rcParams["text.usetex"] = True
    ax2 = fig.add_subplot(122)
    fig.canvas.draw()
    for ax, usetex in [(ax1, False), (ax2, True)]:
        for t in ax.xaxis.majorTicks:
            assert t.label1.get_usetex() == usetex


@needs_usetex
def test_single_artist_usetex():
    # Check that a single artist marked with usetex does not get passed through
    # the mathtext parser at all (for the Agg backend) (the mathtext parser
    # currently fails to parse \frac12, requiring \frac{1}{2} instead).
    fig = plt.figure()
    fig.text(.5, .5, r"$\frac12$", usetex=True)
    fig.canvas.draw()


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_single_artist_usenotex(fmt):
    # Check that a single artist can be marked as not-usetex even though the
    # rcParam is on ("2_2_2" fails if passed to TeX).  This currently skips
    # postscript output as the ps renderer doesn't support mixing usetex and
    # non-usetex.
    plt.rcParams["text.usetex"] = True
    fig = plt.figure()
    fig.text(.5, .5, "2_2_2", usetex=False)
    fig.savefig(io.BytesIO(), format=fmt)


@image_comparison(['text_as_path_opacity.svg'])
def test_text_as_path_opacity():
    plt.figure()
    plt.gca().set_axis_off()
    plt.text(0.25, 0.25, 'c', color=(0, 0, 0, 0.5))
    plt.text(0.25, 0.5, 'a', alpha=0.5)
    plt.text(0.25, 0.75, 'x', alpha=0.5, color=(0, 0, 0, 1))


@image_comparison(['text_as_text_opacity.svg'])
def test_text_as_text_opacity():
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.figure()
    plt.gca().set_axis_off()
    plt.text(0.25, 0.25, '50% using `color`', color=(0, 0, 0, 0.5))
    plt.text(0.25, 0.5, '50% using `alpha`', alpha=0.5)
    plt.text(0.25, 0.75, '50% using `alpha` and 100% `color`', alpha=0.5,
             color=(0, 0, 0, 1))


def test_text_repr():
    # smoketest to make sure text repr doesn't error for category
    plt.plot(['A', 'B'], [1, 2])
    repr(plt.text(['A'], 0.5, 'Boo'))


def test_annotation_update():
    fig, ax = plt.subplots(1, 1)
    an = ax.annotate('annotation', xy=(0.5, 0.5))
    extent1 = an.get_window_extent(fig.canvas.get_renderer())
    fig.tight_layout()
    extent2 = an.get_window_extent(fig.canvas.get_renderer())

    assert not np.allclose(extent1.get_points(), extent2.get_points(),
                           rtol=1e-6)


@check_figures_equal(extensions=["png"])
def test_annotation_units(fig_test, fig_ref):
    ax = fig_test.add_subplot()
    ax.plot(datetime.now(), 1, "o")  # Implicitly set axes extents.
    ax.annotate("x", (datetime.now(), 0.5), xycoords=("data", "axes fraction"),
                # This used to crash before.
                xytext=(0, 0), textcoords="offset points")
    ax = fig_ref.add_subplot()
    ax.plot(datetime.now(), 1, "o")
    ax.annotate("x", (datetime.now(), 0.5), xycoords=("data", "axes fraction"))


@image_comparison(['large_subscript_title.png'], style='mpl20')
def test_large_subscript_title():
    # Remove this line when this test image is regenerated.
    plt.rcParams['text.kerning_factor'] = 6
    plt.rcParams['axes.titley'] = None

    fig, axs = plt.subplots(1, 2, figsize=(9, 2.5), constrained_layout=True)
    ax = axs[0]
    ax.set_title(r'$\sum_{i} x_i$')
    ax.set_title('New way', loc='left')
    ax.set_xticklabels([])

    ax = axs[1]
    ax.set_title(r'$\sum_{i} x_i$', y=1.01)
    ax.set_title('Old Way', loc='left')
    ax.set_xticklabels([])


def test_wrap():
    fig = plt.figure(figsize=(6, 4))
    s = 'This is a very long text that should be wrapped multiple times.'
    text = fig.text(0.7, 0.5, s, wrap=True)
    fig.canvas.draw()
    assert text._get_wrapped_text() == ('This is a very long\n'
                                        'text that should be\n'
                                        'wrapped multiple\n'
                                        'times.')


def test_long_word_wrap():
    fig = plt.figure(figsize=(6, 4))
    text = fig.text(9.5, 8, 'Alonglineoftexttowrap', wrap=True)
    fig.canvas.draw()
    assert text._get_wrapped_text() == 'Alonglineoftexttowrap'


def test_wrap_no_wrap():
    fig = plt.figure(figsize=(6, 4))
    text = fig.text(0, 0, 'non wrapped text', wrap=True)
    fig.canvas.draw()
    assert text._get_wrapped_text() == 'non wrapped text'


@check_figures_equal(extensions=["png"])
def test_buffer_size(fig_test, fig_ref):
    # On old versions of the Agg renderer, large non-ascii single-character
    # strings (here, "€") would be rendered clipped because the rendering
    # buffer would be set by the physical size of the smaller "a" character.
    ax = fig_test.add_subplot()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["€", "a"])
    ax.yaxis.majorTicks[1].label1.set_color("w")
    ax = fig_ref.add_subplot()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["€", ""])


def test_fontproperties_kwarg_precedence():
    """Test that kwargs take precedence over fontproperties defaults."""
    plt.figure()
    text1 = plt.xlabel("value", fontproperties='Times New Roman', size=40.0)
    text2 = plt.ylabel("counts", size=40.0, fontproperties='Times New Roman')
    assert text1.get_size() == 40.0
    assert text2.get_size() == 40.0


def test_transform_rotates_text():
    ax = plt.gca()
    transform = mtransforms.Affine2D().rotate_deg(30)
    text = ax.text(0, 0, 'test', transform=transform,
                   transform_rotates_text=True)
    result = text.get_rotation()
    assert_almost_equal(result, 30)


def test_update_mutate_input():
    inp = dict(fontproperties=FontProperties(weight="bold"),
               bbox=None)
    cache = dict(inp)
    t = Text()
    t.update(inp)
    assert inp['fontproperties'] == cache['fontproperties']
    assert inp['bbox'] == cache['bbox']


def test_invalid_color():
    with pytest.raises(ValueError):
        plt.figtext(.5, .5, "foo", c="foobar")


@image_comparison(['text_pdf_kerning.pdf'], style='mpl20')
def test_pdf_kerning():
    plt.figure()
    plt.figtext(0.1, 0.5, "ATATATATATATATATATA", size=30)


def test_unsupported_script(recwarn):
    fig = plt.figure()
    fig.text(.5, .5, "\N{BENGALI DIGIT ZERO}")
    fig.canvas.draw()
    assert all(isinstance(warn.message, UserWarning) for warn in recwarn)
    assert (
        [warn.message.args for warn in recwarn] ==
        [(r"Glyph 2534 (\N{BENGALI DIGIT ZERO}) missing from current font.",),
         (r"Matplotlib currently does not support Bengali natively.",)])


def test_parse_math():
    fig, ax = plt.subplots()
    ax.text(0, 0, r"$ \wrong{math} $", parse_math=False)
    fig.canvas.draw()

    ax.text(0, 0, r"$ \wrong{math} $", parse_math=True)
    with pytest.raises(ValueError, match='Unknown symbol'):
        fig.canvas.draw()


@image_comparison(['text_pdf_font42_kerning.pdf'], style='mpl20')
def test_pdf_font42_kerning():
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure()
    plt.figtext(0.1, 0.5, "ATAVATAVATAVATAVATA", size=30)


@image_comparison(['text_pdf_chars_beyond_bmp.pdf'], style='mpl20')
def test_pdf_chars_beyond_bmp():
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['mathtext.fontset'] = 'stixsans'
    plt.figure()
    plt.figtext(0.1, 0.5, "Mass $m$ \U00010308", size=30)
