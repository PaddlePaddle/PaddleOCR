"""Test  kddcup99 loader, if the data is available,
or if specifically requested via environment variable
(e.g. for travis cron job).

Only 'percent10' mode is tested, as the full data
is too big to use in unit-testing.
"""

from functools import partial
import pytest

from sklearn.datasets.tests.test_common import check_as_frame
from sklearn.datasets.tests.test_common import check_pandas_dependency_message
from sklearn.datasets.tests.test_common import check_return_X_y


@pytest.mark.parametrize("as_frame", [True, False])
@pytest.mark.parametrize(
    "subset, n_samples, n_features",
    [
        (None, 494021, 41),
        ("SA", 100655, 41),
        ("SF", 73237, 4),
        ("http", 58725, 3),
        ("smtp", 9571, 3),
    ],
)
def test_fetch_kddcup99_percent10(
    fetch_kddcup99_fxt, as_frame, subset, n_samples, n_features
):
    data = fetch_kddcup99_fxt(subset=subset, as_frame=as_frame)
    assert data.data.shape == (n_samples, n_features)
    assert data.target.shape == (n_samples,)
    if as_frame:
        assert data.frame.shape == (n_samples, n_features + 1)
    assert data.DESCR.startswith(".. _kddcup99_dataset:")


def test_fetch_kddcup99_return_X_y(fetch_kddcup99_fxt):
    fetch_func = partial(fetch_kddcup99_fxt, subset="smtp")
    data = fetch_func()
    check_return_X_y(data, fetch_func)


def test_fetch_kddcup99_as_frame(fetch_kddcup99_fxt):
    bunch = fetch_kddcup99_fxt()
    check_as_frame(bunch, fetch_kddcup99_fxt)


def test_fetch_kddcup99_shuffle(fetch_kddcup99_fxt):
    dataset = fetch_kddcup99_fxt(
        random_state=0,
        subset="SA",
        percent10=True,
    )
    dataset_shuffled = fetch_kddcup99_fxt(
        random_state=0,
        subset="SA",
        shuffle=True,
        percent10=True,
    )
    assert set(dataset["target"]) == set(dataset_shuffled["target"])
    assert dataset_shuffled.data.shape == dataset.data.shape
    assert dataset_shuffled.target.shape == dataset.target.shape


def test_pandas_dependency_message(fetch_kddcup99_fxt, hide_available_pandas):
    check_pandas_dependency_message(fetch_kddcup99_fxt)


def test_corrupted_file_error_message(fetch_kddcup99_fxt, tmp_path):
    """Check that a nice error message is raised when cache is corrupted."""
    kddcup99_dir = tmp_path / "kddcup99_10-py3"
    kddcup99_dir.mkdir()
    samples_path = kddcup99_dir / "samples"

    with samples_path.open("wb") as f:
        f.write(b"THIS IS CORRUPTED")

    msg = (
        "The cache for fetch_kddcup99 is invalid, please "
        f"delete {str(kddcup99_dir)} and run the fetch_kddcup99 again"
    )

    with pytest.raises(IOError, match=msg):
        fetch_kddcup99_fxt(data_home=str(tmp_path))
