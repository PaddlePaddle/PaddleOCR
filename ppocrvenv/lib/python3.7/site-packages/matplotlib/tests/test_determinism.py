"""
Test output reproducibility.
"""

import os
import subprocess
import sys

import pytest

import matplotlib as mpl
import matplotlib.testing.compare
from matplotlib import pyplot as plt


needs_ghostscript = pytest.mark.skipif(
    "eps" not in mpl.testing.compare.converter,
    reason="This test needs a ghostscript installation")
needs_usetex = pytest.mark.skipif(
    not mpl.checkdep_usetex(True),
    reason="This test needs a TeX installation")


def _save_figure(objects='mhi', fmt="pdf", usetex=False):
    mpl.use(fmt)
    mpl.rcParams.update({'svg.hashsalt': 'asdf', 'text.usetex': usetex})

    fig = plt.figure()

    if 'm' in objects:
        # use different markers...
        ax1 = fig.add_subplot(1, 6, 1)
        x = range(10)
        ax1.plot(x, [1] * 10, marker='D')
        ax1.plot(x, [2] * 10, marker='x')
        ax1.plot(x, [3] * 10, marker='^')
        ax1.plot(x, [4] * 10, marker='H')
        ax1.plot(x, [5] * 10, marker='v')

    if 'h' in objects:
        # also use different hatch patterns
        ax2 = fig.add_subplot(1, 6, 2)
        bars = (ax2.bar(range(1, 5), range(1, 5)) +
                ax2.bar(range(1, 5), [6] * 4, bottom=range(1, 5)))
        ax2.set_xticks([1.5, 2.5, 3.5, 4.5])

        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

    if 'i' in objects:
        # also use different images
        A = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
        fig.add_subplot(1, 6, 3).imshow(A, interpolation='nearest')
        A = [[1, 3, 2], [1, 2, 3], [3, 1, 2]]
        fig.add_subplot(1, 6, 4).imshow(A, interpolation='bilinear')
        A = [[2, 3, 1], [1, 2, 3], [2, 1, 3]]
        fig.add_subplot(1, 6, 5).imshow(A, interpolation='bicubic')

    x = range(5)
    ax = fig.add_subplot(1, 6, 6)
    ax.plot(x, x)
    ax.set_title('A string $1+2+\\sigma$')
    ax.set_xlabel('A string $1+2+\\sigma$')
    ax.set_ylabel('A string $1+2+\\sigma$')

    stdout = getattr(sys.stdout, 'buffer', sys.stdout)
    fig.savefig(stdout, format=fmt)


@pytest.mark.parametrize(
    "objects, fmt, usetex", [
        ("", "pdf", False),
        ("m", "pdf", False),
        ("h", "pdf", False),
        ("i", "pdf", False),
        ("mhi", "pdf", False),
        ("mhi", "ps", False),
        pytest.param(
            "mhi", "ps", True, marks=[needs_usetex, needs_ghostscript]),
        ("mhi", "svg", False),
        pytest.param("mhi", "svg", True, marks=needs_usetex),
    ]
)
def test_determinism_check(objects, fmt, usetex):
    """
    Output three times the same graphs and checks that the outputs are exactly
    the same.

    Parameters
    ----------
    objects : str
        Objects to be included in the test document: 'm' for markers, 'h' for
        hatch patterns, 'i' for images.
    fmt : {"pdf", "ps", "svg"}
        Output format.
    """
    plots = [
        subprocess.check_output(
            [sys.executable, "-R", "-c",
             f"from matplotlib.tests.test_determinism import _save_figure;"
             f"_save_figure({objects!r}, {fmt!r}, {usetex})"],
            env={**os.environ, "SOURCE_DATE_EPOCH": "946684800",
                 "MPLBACKEND": "Agg"})
        for _ in range(3)
    ]
    for p in plots[1:]:
        if fmt == "ps" and usetex:
            if p != plots[0]:
                pytest.skip("failed, maybe due to ghostscript timestamps")
        else:
            assert p == plots[0]


@pytest.mark.parametrize(
    "fmt, string", [
        ("pdf", b"/CreationDate (D:20000101000000Z)"),
        # SOURCE_DATE_EPOCH support is not tested with text.usetex,
        # because the produced timestamp comes from ghostscript:
        # %%CreationDate: D:20000101000000Z00\'00\', and this could change
        # with another ghostscript version.
        ("ps", b"%%CreationDate: Sat Jan 01 00:00:00 2000"),
    ]
)
def test_determinism_source_date_epoch(fmt, string):
    """
    Test SOURCE_DATE_EPOCH support. Output a document with the environment
    variable SOURCE_DATE_EPOCH set to 2000-01-01 00:00 UTC and check that the
    document contains the timestamp that corresponds to this date (given as an
    argument).

    Parameters
    ----------
    fmt : {"pdf", "ps", "svg"}
        Output format.
    string : bytes
        Timestamp string for 2000-01-01 00:00 UTC.
    """
    buf = subprocess.check_output(
        [sys.executable, "-R", "-c",
         f"from matplotlib.tests.test_determinism import _save_figure; "
         f"_save_figure('', {fmt!r})"],
        env={**os.environ, "SOURCE_DATE_EPOCH": "946684800",
             "MPLBACKEND": "Agg"})
    assert string in buf
