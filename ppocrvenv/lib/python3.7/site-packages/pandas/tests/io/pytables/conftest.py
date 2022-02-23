import pytest

import pandas._testing as tm


@pytest.fixture
def setup_path():
    """Fixture for setup path"""
    return f"tmp.__{tm.rands(10)}__.h5"


@pytest.fixture(scope="module", autouse=True)
def setup_mode():
    """Reset testing mode fixture"""
    tm.reset_testing_mode()
    yield
    tm.set_testing_mode()
