"""Tests for tinypages build using sphinx extensions."""

import filecmp
import os
from pathlib import Path
import shutil
from subprocess import Popen, PIPE
import sys

import pytest


pytest.importorskip('sphinx',
                    minversion=None if sys.version_info < (3, 10) else '4.1.3')


def test_tinypages(tmpdir):
    source_dir = Path(tmpdir) / 'src'
    shutil.copytree(Path(__file__).parent / 'tinypages', source_dir)
    html_dir = source_dir / '_build' / 'html'
    doctree_dir = source_dir / 'doctrees'
    # Build the pages with warnings turned into errors
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html',
           '-d', str(doctree_dir),
           str(Path(__file__).parent / 'tinypages'), str(html_dir)]
    # On CI, gcov emits warnings (due to agg headers being included with the
    # same name in multiple extension modules -- but we don't care about their
    # coverage anyways); hide them using GCOV_ERROR_FILE.
    proc = Popen(
        cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True,
        env={**os.environ, "MPLBACKEND": "", "GCOV_ERROR_FILE": os.devnull})
    out, err = proc.communicate()

    # Build the pages with warnings turned into errors
    build_sphinx_html(source_dir, doctree_dir, html_dir)

    def plot_file(num):
        return html_dir / f'some_plots-{num}.png'

    def plot_directive_file(num):
        # This is always next to the doctree dir.
        return doctree_dir.parent / 'plot_directive' / f'some_plots-{num}.png'

    range_10, range_6, range_4 = [plot_file(i) for i in range(1, 4)]
    # Plot 5 is range(6) plot
    assert filecmp.cmp(range_6, plot_file(5))
    # Plot 7 is range(4) plot
    assert filecmp.cmp(range_4, plot_file(7))
    # Plot 11 is range(10) plot
    assert filecmp.cmp(range_10, plot_file(11))
    # Plot 12 uses the old range(10) figure and the new range(6) figure
    assert filecmp.cmp(range_10, plot_file('12_00'))
    assert filecmp.cmp(range_6, plot_file('12_01'))
    # Plot 13 shows close-figs in action
    assert filecmp.cmp(range_4, plot_file(13))
    # Plot 14 has included source
    html_contents = (html_dir / 'some_plots.html').read_bytes()

    assert b'# Only a comment' in html_contents
    # check plot defined in external file.
    assert filecmp.cmp(range_4, html_dir / 'range4.png')
    assert filecmp.cmp(range_6, html_dir / 'range6.png')
    # check if figure caption made it into html file
    assert b'This is the caption for plot 15.' in html_contents
    # check if figure caption using :caption: made it into html file
    assert b'Plot 17 uses the caption option.' in html_contents
    # check if figure caption made it into html file
    assert b'This is the caption for plot 18.' in html_contents
    # check if the custom classes made it into the html file
    assert b'plot-directive my-class my-other-class' in html_contents
    # check that the multi-image caption is applied twice
    assert html_contents.count(b'This caption applies to both plots.') == 2
    # Plot 21 is range(6) plot via an include directive. But because some of
    # the previous plots are repeated, the argument to plot_file() is only 17.
    assert filecmp.cmp(range_6, plot_file(17))

    # Modify the included plot
    contents = (source_dir / 'included_plot_21.rst').read_text()
    contents = contents.replace('plt.plot(range(6))', 'plt.plot(range(4))')
    (source_dir / 'included_plot_21.rst').write_text(contents)
    # Build the pages again and check that the modified file was updated
    modification_times = [plot_directive_file(i).stat().st_mtime
                          for i in (1, 2, 3, 5)]
    build_sphinx_html(source_dir, doctree_dir, html_dir)
    assert filecmp.cmp(range_4, plot_file(17))
    # Check that the plots in the plot_directive folder weren't changed.
    # (plot_directive_file(1) won't be modified, but it will be copied to html/
    # upon compilation, so plot_file(1) will be modified)
    assert plot_directive_file(1).stat().st_mtime == modification_times[0]
    assert plot_directive_file(2).stat().st_mtime == modification_times[1]
    assert plot_directive_file(3).stat().st_mtime == modification_times[2]
    assert filecmp.cmp(range_10, plot_file(1))
    assert filecmp.cmp(range_6, plot_file(2))
    assert filecmp.cmp(range_4, plot_file(3))
    # Make sure that figures marked with context are re-created (but that the
    # contents are the same)
    assert plot_directive_file(5).stat().st_mtime > modification_times[3]
    assert filecmp.cmp(range_6, plot_file(5))


def test_plot_html_show_source_link(tmpdir):
    source_dir = Path(tmpdir) / 'src'
    source_dir.mkdir()
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', source_dir / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', source_dir / '_static')
    doctree_dir = source_dir / 'doctrees'
    (source_dir / 'index.rst').write_text("""
.. plot::

    plt.plot(range(2))
""")
    # Make sure source scripts are created by default
    html_dir1 = source_dir / '_build' / 'html1'
    build_sphinx_html(source_dir, doctree_dir, html_dir1)
    assert "index-1.py" in [p.name for p in html_dir1.iterdir()]
    # Make sure source scripts are NOT created when
    # plot_html_show_source_link` is False
    html_dir2 = source_dir / '_build' / 'html2'
    build_sphinx_html(source_dir, doctree_dir, html_dir2,
                      extra_args=['-D', 'plot_html_show_source_link=0'])
    assert "index-1.py" not in [p.name for p in html_dir2.iterdir()]


def build_sphinx_html(source_dir, doctree_dir, html_dir, extra_args=None):
    # Build the pages with warnings turned into errors
    extra_args = [] if extra_args is None else extra_args
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html',
           '-d', str(doctree_dir), str(source_dir), str(html_dir), *extra_args]
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True,
                 env={**os.environ, "MPLBACKEND": ""})
    out, err = proc.communicate()

    assert proc.returncode == 0, \
        f"sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n"
    if err:
        pytest.fail(f"sphinx build emitted the following warnings:\n{err}")

    assert html_dir.is_dir()
