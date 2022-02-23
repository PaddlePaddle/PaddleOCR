import numpy as np
import pytest

from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.metaestimators import available_if


class Prefix:
    def func(self):
        pass


class MockMetaEstimator:
    """This is a mock meta estimator"""

    a_prefix = Prefix()

    @if_delegate_has_method(delegate="a_prefix")
    def func(self):
        """This is a mock delegated function"""
        pass


def test_delegated_docstring():
    assert "This is a mock delegated function" in str(
        MockMetaEstimator.__dict__["func"].__doc__
    )
    assert "This is a mock delegated function" in str(MockMetaEstimator.func.__doc__)
    assert "This is a mock delegated function" in str(MockMetaEstimator().func.__doc__)


class MetaEst:
    """A mock meta estimator"""

    def __init__(self, sub_est, better_sub_est=None):
        self.sub_est = sub_est
        self.better_sub_est = better_sub_est

    @if_delegate_has_method(delegate="sub_est")
    def predict(self):
        pass


class MetaEstTestTuple(MetaEst):
    """A mock meta estimator to test passing a tuple of delegates"""

    @if_delegate_has_method(delegate=("sub_est", "better_sub_est"))
    def predict(self):
        pass


class MetaEstTestList(MetaEst):
    """A mock meta estimator to test passing a list of delegates"""

    @if_delegate_has_method(delegate=["sub_est", "better_sub_est"])
    def predict(self):
        pass


class HasPredict:
    """A mock sub-estimator with predict method"""

    def predict(self):
        pass


class HasNoPredict:
    """A mock sub-estimator with no predict method"""

    pass


class HasPredictAsNDArray:
    """A mock sub-estimator where predict is a NumPy array"""

    predict = np.ones((10, 2), dtype=np.int64)


def test_if_delegate_has_method():
    assert hasattr(MetaEst(HasPredict()), "predict")
    assert not hasattr(MetaEst(HasNoPredict()), "predict")
    assert not hasattr(MetaEstTestTuple(HasNoPredict(), HasNoPredict()), "predict")
    assert hasattr(MetaEstTestTuple(HasPredict(), HasNoPredict()), "predict")
    assert not hasattr(MetaEstTestTuple(HasNoPredict(), HasPredict()), "predict")
    assert not hasattr(MetaEstTestList(HasNoPredict(), HasPredict()), "predict")
    assert hasattr(MetaEstTestList(HasPredict(), HasPredict()), "predict")


class AvailableParameterEstimator:
    """This estimator's `available` parameter toggles the presence of a method"""

    def __init__(self, available=True):
        self.available = available

    @available_if(lambda est: est.available)
    def available_func(self):
        """This is a mock available_if function"""
        pass


def test_available_if_docstring():
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.__dict__["available_func"].__doc__
    )
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.available_func.__doc__
    )
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator().available_func.__doc__
    )


def test_available_if():
    assert hasattr(AvailableParameterEstimator(), "available_func")
    assert not hasattr(AvailableParameterEstimator(available=False), "available_func")


def test_available_if_unbound_method():
    # This is a non regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/20614
    # to make sure that decorated functions can be used as an unbound method,
    # for instance when monkeypatching.
    est = AvailableParameterEstimator()
    AvailableParameterEstimator.available_func(est)

    est = AvailableParameterEstimator(available=False)
    with pytest.raises(
        AttributeError,
        match="This 'AvailableParameterEstimator' has no attribute 'available_func'",
    ):
        AvailableParameterEstimator.available_func(est)


def test_if_delegate_has_method_numpy_array():
    """Check that we can check for an attribute that is a NumPy array.

    This is a non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/21144
    """
    estimator = MetaEst(HasPredictAsNDArray())
    assert hasattr(estimator, "predict")
