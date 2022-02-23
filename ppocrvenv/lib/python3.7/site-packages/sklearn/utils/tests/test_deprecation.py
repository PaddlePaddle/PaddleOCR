# Authors: Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause


import pickle

from sklearn.utils.deprecation import _is_deprecated
from sklearn.utils.deprecation import deprecated
import pytest


@deprecated("qwerty")
class MockClass1:
    pass


class MockClass2:
    @deprecated("mockclass2_method")
    def method(self):
        pass

    @deprecated("n_features_ is deprecated")  # type: ignore
    @property
    def n_features_(self):
        """Number of input features."""
        return 10


class MockClass3:
    @deprecated()
    def __init__(self):
        pass


class MockClass4:
    pass


@deprecated()
def mock_function():
    return 10


def test_deprecated():
    with pytest.warns(FutureWarning, match="qwerty"):
        MockClass1()
    with pytest.warns(FutureWarning, match="mockclass2_method"):
        MockClass2().method()
    with pytest.warns(FutureWarning, match="deprecated"):
        MockClass3()
    with pytest.warns(FutureWarning, match="deprecated"):
        val = mock_function()
    assert val == 10


def test_is_deprecated():
    # Test if _is_deprecated helper identifies wrapping via deprecated
    # NOTE it works only for class methods and functions
    assert _is_deprecated(MockClass1.__init__)
    assert _is_deprecated(MockClass2().method)
    assert _is_deprecated(MockClass3.__init__)
    assert not _is_deprecated(MockClass4.__init__)
    assert _is_deprecated(mock_function)


def test_pickle():
    pickle.loads(pickle.dumps(mock_function))


def test_deprecated_property_docstring_exists():
    """Deprecated property contains the original docstring."""
    mock_class_property = getattr(MockClass2, "n_features_")
    assert (
        "DEPRECATED: n_features_ is deprecated\n\n    Number of input features."
        == mock_class_property.__doc__
    )
