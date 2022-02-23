"""Tests for making sure experimental imports work as expected."""

import textwrap

from sklearn.utils._testing import assert_run_python_script


def test_import_raises_warning():
    code = """
    import pytest
    with pytest.warns(UserWarning, match="it is not needed to import"):
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    """
    assert_run_python_script(textwrap.dedent(code))
