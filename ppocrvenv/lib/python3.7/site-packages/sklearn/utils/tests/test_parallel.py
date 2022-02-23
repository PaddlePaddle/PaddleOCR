from distutils.version import LooseVersion

import pytest
from joblib import Parallel
import joblib

from numpy.testing import assert_array_equal

from sklearn._config import config_context, get_config
from sklearn.utils.fixes import delayed


def get_working_memory():
    return get_config()["working_memory"]


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("backend", ["loky", "threading", "multiprocessing"])
def test_configuration_passes_through_to_joblib(n_jobs, backend):
    # Tests that the global global configuration is passed to joblib jobs

    if joblib.__version__ < LooseVersion("0.12") and backend == "loky":
        pytest.skip("loky backend does not exist in joblib <0.12")

    with config_context(working_memory=123):
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(get_working_memory)() for _ in range(2)
        )

    assert_array_equal(results, [123] * 2)
