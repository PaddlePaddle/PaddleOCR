from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager
import pytest


def test_fontconfig_preamble():
    """Test that the preamble is included in _fontconfig."""
    plt.rcParams['text.usetex'] = True

    tm1 = TexManager()
    font_config1 = tm1.get_font_config()

    plt.rcParams['text.latex.preamble'] = '\\usepackage{txfonts}'
    tm2 = TexManager()
    font_config2 = tm2.get_font_config()

    assert font_config1 != font_config2


@pytest.mark.parametrize(
    "rc, preamble, family", [
        ({"font.family": "sans-serif", "font.sans-serif": "helvetica"},
         r"\usepackage{helvet}", r"\sffamily"),
        ({"font.family": "serif", "font.serif": "palatino"},
         r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "cursive", "font.cursive": "zapf chancery"},
         r"\usepackage{chancery}", r"\rmfamily"),
        ({"font.family": "monospace", "font.monospace": "courier"},
         r"\usepackage{courier}", r"\ttfamily"),
        ({"font.family": "helvetica"}, r"\usepackage{helvet}", r"\sffamily"),
        ({"font.family": "palatino"}, r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "zapf chancery"},
         r"\usepackage{chancery}", r"\rmfamily"),
        ({"font.family": "courier"}, r"\usepackage{courier}", r"\ttfamily")
    ])
def test_font_selection(rc, preamble, family):
    plt.rcParams.update(rc)
    tm = TexManager()
    src = Path(tm.make_tex("hello, world", fontsize=12)).read_text()
    assert preamble in src
    assert [*re.findall(r"\\\w+family", src)] == [family]
