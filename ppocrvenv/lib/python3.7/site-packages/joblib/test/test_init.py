# Basic test case to test functioning of module's top-level

try:
    from joblib import *  # noqa
    _top_import_error = None
except Exception as ex:  # pragma: no cover
    _top_import_error = ex


def test_import_joblib():
    # Test either above import has failed for some reason
    # "import *" only allowed at module level, hence we
    # rely on setting up the variable above
    assert _top_import_error is None
