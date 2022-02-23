import pytest


def test_cython_api_deprecation():
    match = ("`scipy._lib._test_deprecation_def.foo_deprecated` "
             "is deprecated, use `foo` instead!\n"
             "Deprecated in Scipy 42.0.0")
    with pytest.warns(DeprecationWarning, match=match):
        from .. import _test_deprecation_call
    assert _test_deprecation_call.call() == (1, 1)
