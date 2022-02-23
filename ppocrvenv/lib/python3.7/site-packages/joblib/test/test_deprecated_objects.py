"""
Tests making sure that deprecated objects properly raise a deprecation warning
when imported/created.
"""
import sys

import pytest

from joblib.my_exceptions import _deprecated_names as _deprecated_exceptions
from joblib.format_stack import _deprecated_names as _deprecated_format_utils


@pytest.mark.xfail(sys.version_info < (3, 7), reason="no module-level getattr")
def test_deprecated_joblib_exceptions():
    assert 'JoblibException' in _deprecated_exceptions
    for name in _deprecated_exceptions:
        msg = ('{} is deprecated and will be removed from joblib in '
               '0.16'.format(name))
        with pytest.warns(DeprecationWarning, match=msg):
            exec('from joblib.my_exceptions import {}'.format(name))


@pytest.mark.xfail(sys.version_info < (3, 7), reason="no module-level getattr")
def test_deprecated_formatting_utilities(capsys):
    assert 'safe_repr' in _deprecated_format_utils
    assert 'eq_repr' in _deprecated_format_utils
    for name in _deprecated_format_utils:
        msg = ('{} is deprecated and will be removed from joblib in '
               '0.16'.format(name))
        with pytest.warns(DeprecationWarning, match=msg):
            exec('from joblib.format_stack import {}'.format(name))
