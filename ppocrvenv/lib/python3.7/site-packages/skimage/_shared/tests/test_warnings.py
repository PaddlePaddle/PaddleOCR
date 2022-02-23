import os
from skimage._shared._warnings import expected_warnings
import pytest


@pytest.fixture(scope='function')
def setup():
    # Remove any environment variable if it exists
    old_strictness = os.environ.pop('SKIMAGE_TEST_STRICT_WARNINGS', None)
    yield
    # Add the user's desired strictness
    if old_strictness is not None:
        os.environ['SKIMAGE_TEST_STRICT_WARNINGS'] = old_strictness


def test_strict_warnigns_default(setup):
    # By default we should fail on missing expected warnings
    with pytest.raises(ValueError):
        with expected_warnings(['some warnings']):
            pass


@pytest.mark.parametrize('strictness', ['1', 'true', 'True', 'TRUE'])
def test_strict_warning_true(setup, strictness):
    os.environ['SKIMAGE_TEST_STRICT_WARNINGS'] = strictness
    with pytest.raises(ValueError):
        with expected_warnings(['some warnings']):
            pass


@pytest.mark.parametrize('strictness', ['0', 'false', 'False', 'FALSE'])
def test_strict_warning_false(setup, strictness):
    # If the user doesnn't wish to be strict about warnigns
    # the following shouldn't raise any error
    os.environ['SKIMAGE_TEST_STRICT_WARNINGS'] = strictness
    with expected_warnings(['some warnings']):
        pass
