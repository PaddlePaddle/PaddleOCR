import os
import subprocess
import sys

import pytest

import matplotlib


@pytest.mark.parametrize('version_str, version_tuple', [
    ('3.5.0', (3, 5, 0, 'final', 0)),
    ('3.5.0rc2', (3, 5, 0, 'candidate', 2)),
    ('3.5.0.dev820+g6768ef8c4c', (3, 5, 0, 'alpha', 820)),
    ('3.5.0.post820+g6768ef8c4c', (3, 5, 1, 'alpha', 820)),
])
def test_parse_to_version_info(version_str, version_tuple):
    assert matplotlib._parse_to_version_info(version_str) == version_tuple


@pytest.mark.skipif(
    os.name == "nt", reason="chmod() doesn't work as is on Windows")
@pytest.mark.skipif(os.name != "nt" and os.geteuid() == 0,
                    reason="chmod() doesn't work as root")
def test_tmpconfigdir_warning(tmpdir):
    """Test that a warning is emitted if a temporary configdir must be used."""
    mode = os.stat(tmpdir).st_mode
    try:
        os.chmod(tmpdir, 0)
        proc = subprocess.run(
            [sys.executable, "-c", "import matplotlib"],
            env={**os.environ, "MPLCONFIGDIR": str(tmpdir)},
            stderr=subprocess.PIPE, universal_newlines=True, check=True)
        assert "set the MPLCONFIGDIR" in proc.stderr
    finally:
        os.chmod(tmpdir, mode)


def test_importable_with_no_home(tmpdir):
    subprocess.run(
        [sys.executable, "-c",
         "import pathlib; pathlib.Path.home = lambda *args: 1/0; "
         "import matplotlib.pyplot"],
        env={**os.environ, "MPLCONFIGDIR": str(tmpdir)}, check=True)


def test_use_doc_standard_backends():
    """
    Test that the standard backends mentioned in the docstring of
    matplotlib.use() are the same as in matplotlib.rcsetup.
    """
    def parse(key):
        backends = []
        for line in matplotlib.use.__doc__.split(key)[1].split('\n'):
            if not line.strip():
                break
            backends += [e.strip() for e in line.split(',') if e]
        return backends

    assert (set(parse('- interactive backends:\n')) ==
            set(matplotlib.rcsetup.interactive_bk))
    assert (set(parse('- non-interactive backends:\n')) ==
            set(matplotlib.rcsetup.non_interactive_bk))


def test_importable_with__OO():
    """
    When using -OO or export PYTHONOPTIMIZE=2, docstrings are discarded,
    this simple test may prevent something like issue #17970.
    """
    program = (
        "import matplotlib as mpl; "
        "import matplotlib.pyplot as plt; "
        "import matplotlib.cbook as cbook; "
        "import matplotlib.patches as mpatches"
    )
    cmd = [sys.executable, "-OO", "-c", program]
    assert subprocess.call(cmd, env={**os.environ, "MPLBACKEND": ""}) == 0
