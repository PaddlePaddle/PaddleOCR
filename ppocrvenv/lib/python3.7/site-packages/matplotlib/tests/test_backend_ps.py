from collections import Counter
from pathlib import Path
import io
import re
import tempfile

import pytest

from matplotlib import cbook, patheffects
from matplotlib.cbook import MatplotlibDeprecationWarning
from matplotlib.figure import Figure
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib as mpl
import matplotlib.pyplot as plt

needs_ghostscript = pytest.mark.skipif(
    "eps" not in mpl.testing.compare.converter,
    reason="This test needs a ghostscript installation")
needs_usetex = pytest.mark.skipif(
    not mpl.checkdep_usetex(True),
    reason="This test needs a TeX installation")


# This tests tends to hit a TeX cache lock on AppVeyor.
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize('orientation', ['portrait', 'landscape'])
@pytest.mark.parametrize('format, use_log, rcParams', [
    ('ps', False, {}),
    ('ps', False, {'ps.usedistiller': 'ghostscript'}),
    ('ps', False, {'ps.usedistiller': 'xpdf'}),
    ('ps', False, {'text.usetex': True}),
    ('eps', False, {}),
    ('eps', True, {'ps.useafm': True}),
    ('eps', False, {'text.usetex': True}),
], ids=[
    'ps',
    'ps with distiller=ghostscript',
    'ps with distiller=xpdf',
    'ps with usetex',
    'eps',
    'eps afm',
    'eps with usetex'
])
def test_savefig_to_stringio(format, use_log, rcParams, orientation):
    mpl.rcParams.update(rcParams)

    fig, ax = plt.subplots()

    with io.StringIO() as s_buf, io.BytesIO() as b_buf:

        if use_log:
            ax.set_yscale('log')

        ax.plot([1, 2], [1, 2])
        title = "Déjà vu"
        if not mpl.rcParams["text.usetex"]:
            title += " \N{MINUS SIGN}\N{EURO SIGN}"
        ax.set_title(title)
        allowable_exceptions = []
        if rcParams.get("ps.usedistiller"):
            allowable_exceptions.append(mpl.ExecutableNotFoundError)
        if rcParams.get("text.usetex"):
            allowable_exceptions.append(RuntimeError)
        if rcParams.get("ps.useafm"):
            allowable_exceptions.append(MatplotlibDeprecationWarning)
        try:
            fig.savefig(s_buf, format=format, orientation=orientation)
            fig.savefig(b_buf, format=format, orientation=orientation)
        except tuple(allowable_exceptions) as exc:
            pytest.skip(str(exc))

        assert not s_buf.closed
        assert not b_buf.closed
        s_val = s_buf.getvalue().encode('ascii')
        b_val = b_buf.getvalue()

        # Strip out CreationDate: ghostscript and cairo don't obey
        # SOURCE_DATE_EPOCH, and that environment variable is already tested in
        # test_determinism.
        s_val = re.sub(b"(?<=\n%%CreationDate: ).*", b"", s_val)
        b_val = re.sub(b"(?<=\n%%CreationDate: ).*", b"", b_val)

        assert s_val == b_val.replace(b'\r\n', b'\n')


def test_patheffects():
    mpl.rcParams['path.effects'] = [
        patheffects.withStroke(linewidth=4, foreground='w')]
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    with io.BytesIO() as ps:
        fig.savefig(ps, format='ps')


@needs_usetex
@needs_ghostscript
def test_tilde_in_tempfilename(tmpdir):
    # Tilde ~ in the tempdir path (e.g. TMPDIR, TMP or TEMP on windows
    # when the username is very long and windows uses a short name) breaks
    # latex before https://github.com/matplotlib/matplotlib/pull/5928
    base_tempdir = Path(tmpdir, "short-1")
    base_tempdir.mkdir()
    # Change the path for new tempdirs, which is used internally by the ps
    # backend to write a file.
    with cbook._setattr_cm(tempfile, tempdir=str(base_tempdir)):
        # usetex results in the latex call, which does not like the ~
        mpl.rcParams['text.usetex'] = True
        plt.plot([1, 2, 3, 4])
        plt.xlabel(r'\textbf{time} (s)')
        # use the PS backend to write the file...
        plt.savefig(base_tempdir / 'tex_demo.eps', format="ps")


@image_comparison(["empty.eps"])
def test_transparency():
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.plot([0, 1], color="r", alpha=0)
    ax.text(.5, .5, "foo", color="r", alpha=0)


@needs_usetex
@image_comparison(["empty.eps"])
def test_transparency_tex():
    mpl.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.plot([0, 1], color="r", alpha=0)
    ax.text(.5, .5, "foo", color="r", alpha=0)


def test_bbox():
    fig, ax = plt.subplots()
    with io.BytesIO() as buf:
        fig.savefig(buf, format='eps')
        buf = buf.getvalue()

    bb = re.search(b'^%%BoundingBox: (.+) (.+) (.+) (.+)$', buf, re.MULTILINE)
    assert bb
    hibb = re.search(b'^%%HiResBoundingBox: (.+) (.+) (.+) (.+)$', buf,
                     re.MULTILINE)
    assert hibb

    for i in range(1, 5):
        # BoundingBox must use integers, and be ceil/floor of the hi res.
        assert b'.' not in bb.group(i)
        assert int(bb.group(i)) == pytest.approx(float(hibb.group(i)), 1)


@needs_usetex
def test_failing_latex():
    """Test failing latex subprocess call"""
    mpl.rcParams['text.usetex'] = True
    # This fails with "Double subscript"
    plt.xlabel("$22_2_2$")
    with pytest.raises(RuntimeError):
        plt.savefig(io.BytesIO(), format="ps")


@needs_usetex
def test_partial_usetex(caplog):
    caplog.set_level("WARNING")
    plt.figtext(.1, .1, "foo", usetex=True)
    plt.figtext(.2, .2, "bar", usetex=True)
    plt.savefig(io.BytesIO(), format="ps")
    record, = caplog.records  # asserts there's a single record.
    assert "as if usetex=False" in record.getMessage()


@needs_usetex
def test_usetex_preamble(caplog):
    mpl.rcParams.update({
        "text.usetex": True,
        # Check that these don't conflict with the packages loaded by default.
        "text.latex.preamble": r"\usepackage{color,graphicx,textcomp}",
    })
    plt.figtext(.5, .5, "foo")
    plt.savefig(io.BytesIO(), format="ps")


@image_comparison(["useafm.eps"])
def test_useafm():
    mpl.rcParams["ps.useafm"] = True
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.axhline(.5)
    ax.text(.5, .5, "qk")


@image_comparison(["type3.eps"])
def test_type3_font():
    plt.figtext(.5, .5, "I/J")


@check_figures_equal(extensions=["eps"])
def test_text_clip(fig_test, fig_ref):
    ax = fig_test.add_subplot()
    # Fully clipped-out text should not appear.
    ax.text(0, 0, "hello", transform=fig_test.transFigure, clip_on=True)
    fig_ref.add_subplot()


@needs_ghostscript
def test_d_glyph(tmp_path):
    # Ensure that we don't have a procedure defined as /d, which would be
    # overwritten by the glyph definition for "d".
    fig = plt.figure()
    fig.text(.5, .5, "def")
    out = tmp_path / "test.eps"
    fig.savefig(out)
    mpl.testing.compare.convert(out, cache=False)  # Should not raise.


@image_comparison(["type42_without_prep.eps"], style='mpl20')
def test_type42_font_without_prep():
    # Test whether Type 42 fonts without prep table are properly embedded
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["mathtext.fontset"] = "stix"

    plt.figtext(0.5, 0.5, "Mass $m$")


@pytest.mark.parametrize('fonttype', ["3", "42"])
def test_fonttype(fonttype):
    mpl.rcParams["ps.fonttype"] = fonttype
    fig, ax = plt.subplots()

    ax.text(0.25, 0.5, "Forty-two is the answer to everything!")

    buf = io.BytesIO()
    fig.savefig(buf, format="ps")

    test = b'/FontType ' + bytes(f"{fonttype}", encoding='utf-8') + b' def'

    assert re.search(test, buf.getvalue(), re.MULTILINE)


def test_linedash():
    """Test that dashed lines do not break PS output"""
    fig, ax = plt.subplots()

    ax.plot([0, 1], linestyle="--")

    buf = io.BytesIO()
    fig.savefig(buf, format="ps")

    assert buf.tell() > 0


def test_no_duplicate_definition():

    fig = Figure()
    axs = fig.subplots(4, 4, subplot_kw=dict(projection="polar"))
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])
        ax.plot([1, 2])
    fig.suptitle("hello, world")

    buf = io.StringIO()
    fig.savefig(buf, format='eps')
    buf.seek(0)

    wds = [ln.partition(' ')[0] for
           ln in buf.readlines()
           if ln.startswith('/')]

    assert max(Counter(wds).values()) == 1
