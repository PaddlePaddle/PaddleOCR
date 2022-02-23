from os import environ
from functools import wraps
import platform
import sys

import pytest
from threadpoolctl import threadpool_limits
from _pytest.doctest import DoctestItem

from sklearn.utils import _IS_32BIT
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.externals import _pilutil
from sklearn._min_dependencies import PYTEST_MIN_VERSION
from sklearn.utils.fixes import np_version, parse_version
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_kddcup99
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_rcv1


if parse_version(pytest.__version__) < parse_version(PYTEST_MIN_VERSION):
    raise ImportError(
        "Your version of pytest is too old, you should have "
        "at least pytest >= {} installed.".format(PYTEST_MIN_VERSION)
    )

dataset_fetchers = {
    "fetch_20newsgroups_fxt": fetch_20newsgroups,
    "fetch_20newsgroups_vectorized_fxt": fetch_20newsgroups_vectorized,
    "fetch_california_housing_fxt": fetch_california_housing,
    "fetch_covtype_fxt": fetch_covtype,
    "fetch_kddcup99_fxt": fetch_kddcup99,
    "fetch_olivetti_faces_fxt": fetch_olivetti_faces,
    "fetch_rcv1_fxt": fetch_rcv1,
}


def _fetch_fixture(f):
    """Fetch dataset (download if missing and requested by environment)."""
    download_if_missing = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"

    @wraps(f)
    def wrapped(*args, **kwargs):
        kwargs["download_if_missing"] = download_if_missing
        try:
            return f(*args, **kwargs)
        except IOError as e:
            if str(e) != "Data not found and `download_if_missing` is False":
                raise
            pytest.skip("test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0")

    return pytest.fixture(lambda: wrapped)


# Adds fixtures for fetching data
fetch_20newsgroups_fxt = _fetch_fixture(fetch_20newsgroups)
fetch_20newsgroups_vectorized_fxt = _fetch_fixture(fetch_20newsgroups_vectorized)
fetch_california_housing_fxt = _fetch_fixture(fetch_california_housing)
fetch_covtype_fxt = _fetch_fixture(fetch_covtype)
fetch_kddcup99_fxt = _fetch_fixture(fetch_kddcup99)
fetch_olivetti_faces_fxt = _fetch_fixture(fetch_olivetti_faces)
fetch_rcv1_fxt = _fetch_fixture(fetch_rcv1)


def pytest_collection_modifyitems(config, items):
    """Called after collect is completed.

    Parameters
    ----------
    config : pytest config
    items : list of collected items
    """
    run_network_tests = environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "0"
    skip_network = pytest.mark.skip(
        reason="test is enabled when SKLEARN_SKIP_NETWORK_TESTS=0"
    )

    # download datasets during collection to avoid thread unsafe behavior
    # when running pytest in parallel with pytest-xdist
    dataset_features_set = set(dataset_fetchers)
    datasets_to_download = set()

    for item in items:
        if not hasattr(item, "fixturenames"):
            continue
        item_fixtures = set(item.fixturenames)
        dataset_to_fetch = item_fixtures & dataset_features_set
        if not dataset_to_fetch:
            continue

        if run_network_tests:
            datasets_to_download |= dataset_to_fetch
        else:
            # network tests are skipped
            item.add_marker(skip_network)

    # Only download datasets on the first worker spawned by pytest-xdist
    # to avoid thread unsafe behavior. If pytest-xdist is not used, we still
    # download before tests run.
    worker_id = environ.get("PYTEST_XDIST_WORKER", "gw0")
    if worker_id == "gw0" and run_network_tests:
        for name in datasets_to_download:
            dataset_fetchers[name]()

    for item in items:
        # FeatureHasher is not compatible with PyPy
        if (
            item.name.endswith(("_hash.FeatureHasher", "text.HashingVectorizer"))
            and platform.python_implementation() == "PyPy"
        ):
            marker = pytest.mark.skip(
                reason="FeatureHasher is not compatible with PyPy"
            )
            item.add_marker(marker)
        # Known failure on with GradientBoostingClassifier on ARM64
        elif (
            item.name.endswith("GradientBoostingClassifier")
            and platform.machine() == "aarch64"
        ):

            marker = pytest.mark.xfail(
                reason=(
                    "know failure. See "
                    "https://github.com/scikit-learn/scikit-learn/issues/17797"  # noqa
                )
            )
            item.add_marker(marker)

    # numpy changed the str/repr formatting of numpy arrays in 1.14. We want to
    # run doctests only for numpy >= 1.14.
    skip_doctests = False
    try:
        import matplotlib  # noqa
    except ImportError:
        skip_doctests = True
        reason = "matplotlib is required to run the doctests"

    try:
        if np_version < parse_version("1.14"):
            reason = "doctests are only run for numpy >= 1.14"
            skip_doctests = True
        elif _IS_32BIT:
            reason = "doctest are only run when the default numpy int is 64 bits."
            skip_doctests = True
        elif sys.platform.startswith("win32"):
            reason = (
                "doctests are not run for Windows because numpy arrays "
                "repr is inconsistent across platforms."
            )
            skip_doctests = True
    except ImportError:
        pass

    # Normally doctest has the entire module's scope. Here we set globs to an empty dict
    # to remove the module's scope:
    # https://docs.python.org/3/library/doctest.html#what-s-the-execution-context
    for item in items:
        if isinstance(item, DoctestItem):
            item.dtest.globs = {}

    if skip_doctests:
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                # work-around an internal error with pytest if adding a skip
                # mark to a doctest in a contextmanager, see
                # https://github.com/pytest-dev/pytest/issues/8796 for more
                # details.
                if item.name != "sklearn._config.config_context":
                    item.add_marker(skip_marker)
    elif not _pilutil.pillow_installed:
        skip_marker = pytest.mark.skip(reason="pillow (or PIL) not installed!")
        for item in items:
            if item.name in [
                "sklearn.feature_extraction.image.PatchExtractor",
                "sklearn.feature_extraction.image.extract_patches_2d",
            ]:
                item.add_marker(skip_marker)


@pytest.fixture(scope="function")
def pyplot():
    """Setup and teardown fixture for matplotlib.

    This fixture checks if we can import matplotlib. If not, the tests will be
    skipped. Otherwise, we close the figures before and after running the
    functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close("all")
    yield pyplot
    pyplot.close("all")


def pytest_runtest_setup(item):
    """Set the number of openmp threads based on the number of workers
    xdist is using to prevent oversubscription.

    Parameters
    ----------
    item : pytest item
        item to be processed
    """
    xdist_worker_count = environ.get("PYTEST_XDIST_WORKER_COUNT")
    if xdist_worker_count is None:
        # returns if pytest-xdist is not installed
        return
    else:
        xdist_worker_count = int(xdist_worker_count)

    openmp_threads = _openmp_effective_n_threads()
    threads_per_worker = max(openmp_threads // xdist_worker_count, 1)
    threadpool_limits(threads_per_worker, user_api="openmp")


def pytest_configure(config):
    # Use matplotlib agg backend during the tests including doctests
    try:
        import matplotlib

        matplotlib.use("agg")
    except ImportError:
        pass
