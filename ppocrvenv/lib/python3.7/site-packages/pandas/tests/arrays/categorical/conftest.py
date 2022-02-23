import pytest


@pytest.fixture(params=[True, False])
def allow_fill(request):
    """Boolean 'allow_fill' parameter for Categorical.take"""
    return request.param
