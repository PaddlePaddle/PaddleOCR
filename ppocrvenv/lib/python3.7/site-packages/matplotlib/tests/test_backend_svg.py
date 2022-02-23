import datetime
from io import BytesIO
import tempfile
import xml.etree.ElementTree
import xml.parsers.expat

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import dviread
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison, check_figures_equal


needs_usetex = pytest.mark.skipif(
    not mpl.checkdep_usetex(True),
    reason="This test needs a TeX installation")


def test_visibility():
    fig, ax = plt.subplots()

    x = np.linspace(0, 4 * np.pi, 50)
    y = np.sin(x)
    yerr = np.ones_like(y)

    a, b, c = ax.errorbar(x, y, yerr=yerr, fmt='ko')
    for artist in b:
        artist.set_visible(False)

    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue()

    parser = xml.parsers.expat.ParserCreate()
    parser.Parse(buf)  # this will raise ExpatError if the svg is invalid


@image_comparison(['fill_black_with_alpha.svg'], remove_text=True)
def test_fill_black_with_alpha():
    fig, ax = plt.subplots()
    ax.scatter(x=[0, 0.1, 1], y=[0, 0, 0], c='k', alpha=0.1, s=10000)


@image_comparison(['noscale'], remove_text=True)
def test_noscale():
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    Z = np.sin(Y ** 2)

    fig, ax = plt.subplots()
    ax.imshow(Z, cmap='gray', interpolation='none')


def test_text_urls():
    fig = plt.figure()

    test_url = "http://test_text_urls.matplotlib.org"
    fig.suptitle("test_text_urls", url=test_url)

    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue().decode()

    expected = '<a xlink:href="{0}">'.format(test_url)
    assert expected in buf


@image_comparison(['bold_font_output.svg'])
def test_bold_font_output():
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10))
    ax.set_xlabel('nonbold-xlabel')
    ax.set_ylabel('bold-ylabel', fontweight='bold')
    ax.set_title('bold-title', fontweight='bold')


@image_comparison(['bold_font_output_with_none_fonttype.svg'])
def test_bold_font_output_with_none_fonttype():
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots()
    ax.plot(np.arange(10), np.arange(10))
    ax.set_xlabel('nonbold-xlabel')
    ax.set_ylabel('bold-ylabel', fontweight='bold')
    ax.set_title('bold-title', fontweight='bold')


@check_figures_equal(tol=20)
def test_rasterized(fig_test, fig_ref):
    t = np.arange(0, 100) * (2.3)
    x = np.cos(t)
    y = np.sin(t)

    ax_ref = fig_ref.subplots()
    ax_ref.plot(x, y, "-", c="r", lw=10)
    ax_ref.plot(x+1, y, "-", c="b", lw=10)

    ax_test = fig_test.subplots()
    ax_test.plot(x, y, "-", c="r", lw=10, rasterized=True)
    ax_test.plot(x+1, y, "-", c="b", lw=10, rasterized=True)


@check_figures_equal()
def test_rasterized_ordering(fig_test, fig_ref):
    t = np.arange(0, 100) * (2.3)
    x = np.cos(t)
    y = np.sin(t)

    ax_ref = fig_ref.subplots()
    ax_ref.set_xlim(0, 3)
    ax_ref.set_ylim(-1.1, 1.1)
    ax_ref.plot(x, y, "-", c="r", lw=10, rasterized=True)
    ax_ref.plot(x+1, y, "-", c="b", lw=10, rasterized=False)
    ax_ref.plot(x+2, y, "-", c="g", lw=10, rasterized=True)
    ax_ref.plot(x+3, y, "-", c="m", lw=10, rasterized=True)

    ax_test = fig_test.subplots()
    ax_test.set_xlim(0, 3)
    ax_test.set_ylim(-1.1, 1.1)
    ax_test.plot(x, y, "-", c="r", lw=10, rasterized=True, zorder=1.1)
    ax_test.plot(x+2, y, "-", c="g", lw=10, rasterized=True, zorder=1.3)
    ax_test.plot(x+3, y, "-", c="m", lw=10, rasterized=True, zorder=1.4)
    ax_test.plot(x+1, y, "-", c="b", lw=10, rasterized=False, zorder=1.2)


def test_count_bitmaps():
    def count_tag(fig, tag):
        with BytesIO() as fd:
            fig.savefig(fd, format='svg')
            buf = fd.getvalue().decode()
        return buf.count(f"<{tag}")

    # No rasterized elements
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_axis_off()
    for n in range(5):
        ax1.plot([0, 20], [0, n], "b-", rasterized=False)
    assert count_tag(fig1, "image") == 0
    assert count_tag(fig1, "path") == 6  # axis patch plus lines

    # rasterized can be merged
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_axis_off()
    for n in range(5):
        ax2.plot([0, 20], [0, n], "b-", rasterized=True)
    assert count_tag(fig2, "image") == 1
    assert count_tag(fig2, "path") == 1  # axis patch

    # rasterized can't be merged without affecting draw order
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_axis_off()
    for n in range(5):
        ax3.plot([0, 20], [n, 0], "b-", rasterized=False)
        ax3.plot([0, 20], [0, n], "b-", rasterized=True)
    assert count_tag(fig3, "image") == 5
    assert count_tag(fig3, "path") == 6

    # rasterized whole axes
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.set_axis_off()
    ax4.set_rasterized(True)
    for n in range(5):
        ax4.plot([0, 20], [n, 0], "b-", rasterized=False)
        ax4.plot([0, 20], [0, n], "b-", rasterized=True)
    assert count_tag(fig4, "image") == 1
    assert count_tag(fig4, "path") == 1

    # rasterized can be merged, but inhibited by suppressComposite
    fig5 = plt.figure()
    fig5.suppressComposite = True
    ax5 = fig5.add_subplot(1, 1, 1)
    ax5.set_axis_off()
    for n in range(5):
        ax5.plot([0, 20], [0, n], "b-", rasterized=True)
    assert count_tag(fig5, "image") == 5
    assert count_tag(fig5, "path") == 1  # axis patch


@needs_usetex
def test_missing_psfont(monkeypatch):
    """An error is raised if a TeX font lacks a Type-1 equivalent"""

    def psfont(*args, **kwargs):
        return dviread.PsFont(texname='texfont', psname='Some Font',
                              effects=None, encoding=None, filename=None)

    monkeypatch.setattr(dviread.PsfontsMap, '__getitem__', psfont)
    mpl.rc('text', usetex=True)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'hello')
    with tempfile.TemporaryFile() as tmpfile, pytest.raises(ValueError):
        fig.savefig(tmpfile, format='svg')


# Use Computer Modern Sans Serif, not Helvetica (which has no \textwon).
@mpl.style.context('default')
@needs_usetex
def test_unicode_won():
    fig = Figure()
    fig.text(.5, .5, r'\textwon', usetex=True)

    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue()

    tree = xml.etree.ElementTree.fromstring(buf)
    ns = 'http://www.w3.org/2000/svg'
    won_id = 'SFSS3583-8e'
    assert len(tree.findall(f'.//{{{ns}}}path[@d][@id="{won_id}"]')) == 1
    assert f'#{won_id}' in tree.find(f'.//{{{ns}}}use').attrib.values()


def test_svgnone_with_data_coordinates():
    plt.rcParams.update({'svg.fonttype': 'none', 'font.stretch': 'condensed'})
    expected = 'Unlikely to appear by chance'

    fig, ax = plt.subplots()
    ax.text(np.datetime64('2019-06-30'), 1, expected)
    ax.set_xlim(np.datetime64('2019-01-01'), np.datetime64('2019-12-31'))
    ax.set_ylim(0, 2)

    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        fd.seek(0)
        buf = fd.read().decode()

    assert expected in buf and "condensed" in buf


def test_gid():
    """Test that object gid appears in output svg."""
    from matplotlib.offsetbox import OffsetBox
    from matplotlib.axis import Tick

    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax1.imshow([[1., 2.], [2., 3.]], aspect="auto")
    ax1.scatter([1, 2, 3], [1, 2, 3], label="myscatter")
    ax1.plot([2, 3, 1], label="myplot")
    ax1.legend()
    ax1a = ax1.twinx()
    ax1a.bar([1, 2, 3], [1, 2, 3])

    ax2 = fig.add_subplot(132, projection="polar")
    ax2.plot([0, 1.5, 3], [1, 2, 3])

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot([1, 2], [1, 2], [1, 2])

    fig.canvas.draw()

    gdic = {}
    for idx, obj in enumerate(fig.findobj(include_self=True)):
        if obj.get_visible():
            gid = f"test123{obj.__class__.__name__}_{idx}"
            gdic[gid] = obj
            obj.set_gid(gid)

    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue().decode()

    def include(gid, obj):
        # we need to exclude certain objects which will not appear in the svg
        if isinstance(obj, OffsetBox):
            return False
        if isinstance(obj, plt.Text):
            if obj.get_text() == "":
                return False
            elif obj.axes is None:
                return False
        if isinstance(obj, plt.Line2D):
            xdata, ydata = obj.get_data()
            if len(xdata) == len(ydata) == 1:
                return False
            elif not hasattr(obj, "axes") or obj.axes is None:
                return False
        if isinstance(obj, Tick):
            loc = obj.get_loc()
            if loc == 0:
                return False
            vi = obj.get_view_interval()
            if loc < min(vi) or loc > max(vi):
                return False
        return True

    for gid, obj in gdic.items():
        if include(gid, obj):
            assert gid in buf


def test_savefig_tight():
    # Check that the draw-disabled renderer correctly disables open/close_group
    # as well.
    plt.savefig(BytesIO(), format="svgz", bbox_inches="tight")


def test_url():
    # Test that object url appears in output svg.

    fig, ax = plt.subplots()

    # collections
    s = ax.scatter([1, 2, 3], [4, 5, 6])
    s.set_urls(['http://example.com/foo', 'http://example.com/bar', None])

    # Line2D
    p, = plt.plot([1, 3], [6, 5])
    p.set_url('http://example.com/baz')

    b = BytesIO()
    fig.savefig(b, format='svg')
    b = b.getvalue()
    for v in [b'foo', b'bar', b'baz']:
        assert b'http://example.com/' + v in b


def test_url_tick(monkeypatch):
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')

    fig1, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_url(f'http://example.com/{i}')

    fig2, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label1.set_url(f'http://example.com/{i}')
        tick.label2.set_url(f'http://example.com/{i}')

    b1 = BytesIO()
    fig1.savefig(b1, format='svg')
    b1 = b1.getvalue()

    b2 = BytesIO()
    fig2.savefig(b2, format='svg')
    b2 = b2.getvalue()

    for i in range(len(ax.yaxis.get_major_ticks())):
        assert f'http://example.com/{i}'.encode('ascii') in b1
    assert b1 == b2


def test_svg_default_metadata(monkeypatch):
    # Values have been predefined for 'Creator', 'Date', 'Format', and 'Type'.
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')

    fig, ax = plt.subplots()
    with BytesIO() as fd:
        fig.savefig(fd, format='svg')
        buf = fd.getvalue().decode()

    # Creator
    assert mpl.__version__ in buf
    # Date
    assert '1970-08-16' in buf
    # Format
    assert 'image/svg+xml' in buf
    # Type
    assert 'StillImage' in buf

    # Now make sure all the default metadata can be cleared.
    with BytesIO() as fd:
        fig.savefig(fd, format='svg', metadata={'Date': None, 'Creator': None,
                                                'Format': None, 'Type': None})
        buf = fd.getvalue().decode()

    # Creator
    assert mpl.__version__ not in buf
    # Date
    assert '1970-08-16' not in buf
    # Format
    assert 'image/svg+xml' not in buf
    # Type
    assert 'StillImage' not in buf


def test_svg_clear_default_metadata(monkeypatch):
    # Makes sure that setting a default metadata to `None`
    # removes the corresponding tag from the metadata.
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')

    metadata_contains = {'creator': mpl.__version__, 'date': '1970-08-16',
                         'format': 'image/svg+xml', 'type': 'StillImage'}

    SVGNS = '{http://www.w3.org/2000/svg}'
    RDFNS = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    CCNS = '{http://creativecommons.org/ns#}'
    DCNS = '{http://purl.org/dc/elements/1.1/}'

    fig, ax = plt.subplots()
    for name in metadata_contains:
        with BytesIO() as fd:
            fig.savefig(fd, format='svg', metadata={name.title(): None})
            buf = fd.getvalue().decode()

        root = xml.etree.ElementTree.fromstring(buf)
        work, = root.findall(f'./{SVGNS}metadata/{RDFNS}RDF/{CCNS}Work')
        for key in metadata_contains:
            data = work.findall(f'./{DCNS}{key}')
            if key == name:
                # The one we cleared is not there
                assert not data
                continue
            # Everything else should be there
            data, = data
            xmlstr = xml.etree.ElementTree.tostring(data, encoding="unicode")
            assert metadata_contains[key] in xmlstr


def test_svg_clear_all_metadata():
    # Makes sure that setting all default metadata to `None`
    # removes the metadata tag from the output.

    fig, ax = plt.subplots()
    with BytesIO() as fd:
        fig.savefig(fd, format='svg', metadata={'Date': None, 'Creator': None,
                                                'Format': None, 'Type': None})
        buf = fd.getvalue().decode()

    SVGNS = '{http://www.w3.org/2000/svg}'

    root = xml.etree.ElementTree.fromstring(buf)
    assert not root.findall(f'./{SVGNS}metadata')


def test_svg_metadata():
    single_value = ['Coverage', 'Identifier', 'Language', 'Relation', 'Source',
                    'Title', 'Type']
    multi_value = ['Contributor', 'Creator', 'Keywords', 'Publisher', 'Rights']
    metadata = {
        'Date': [datetime.date(1968, 8, 1),
                 datetime.datetime(1968, 8, 2, 1, 2, 3)],
        'Description': 'description\ntext',
        **{k: f'{k} foo' for k in single_value},
        **{k: [f'{k} bar', f'{k} baz'] for k in multi_value},
    }

    fig, ax = plt.subplots()
    with BytesIO() as fd:
        fig.savefig(fd, format='svg', metadata=metadata)
        buf = fd.getvalue().decode()

    SVGNS = '{http://www.w3.org/2000/svg}'
    RDFNS = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    CCNS = '{http://creativecommons.org/ns#}'
    DCNS = '{http://purl.org/dc/elements/1.1/}'

    root = xml.etree.ElementTree.fromstring(buf)
    rdf, = root.findall(f'./{SVGNS}metadata/{RDFNS}RDF')

    # Check things that are single entries.
    titles = [node.text for node in root.findall(f'./{SVGNS}title')]
    assert titles == [metadata['Title']]
    types = [node.attrib[f'{RDFNS}resource']
             for node in rdf.findall(f'./{CCNS}Work/{DCNS}type')]
    assert types == [metadata['Type']]
    for k in ['Description', *single_value]:
        if k == 'Type':
            continue
        values = [node.text
                  for node in rdf.findall(f'./{CCNS}Work/{DCNS}{k.lower()}')]
        assert values == [metadata[k]]

    # Check things that are multi-value entries.
    for k in multi_value:
        if k == 'Keywords':
            continue
        values = [
            node.text
            for node in rdf.findall(
                f'./{CCNS}Work/{DCNS}{k.lower()}/{CCNS}Agent/{DCNS}title')]
        assert values == metadata[k]

    # Check special things.
    dates = [node.text for node in rdf.findall(f'./{CCNS}Work/{DCNS}date')]
    assert dates == ['1968-08-01/1968-08-02T01:02:03']

    values = [node.text for node in
              rdf.findall(f'./{CCNS}Work/{DCNS}subject/{RDFNS}Bag/{RDFNS}li')]
    assert values == metadata['Keywords']
