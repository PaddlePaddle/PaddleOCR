import sys
from scipy._lib._testutils import _parse_size, _get_mem_available
import pytest


def test__parse_size():
    expected = {
        '12': 12e6,
        '12 b': 12,
        '12k': 12e3,
        '  12  M  ': 12e6,
        '  12  G  ': 12e9,
        ' 12Tb ': 12e12,
        '12  Mib ': 12 * 1024.0**2,
        '12Tib': 12 * 1024.0**4,
    }

    for inp, outp in sorted(expected.items()):
        if outp is None:
            with pytest.raises(ValueError):
                _parse_size(inp)
        else:
            assert _parse_size(inp) == outp


def test__mem_available():
    # May return None on non-Linux platforms
    available = _get_mem_available()
    if sys.platform.startswith('linux'):
        assert available >= 0
    else:
        assert available is None or available >= 0
