from contextlib import ExitStack as does_not_raise
from datetime import datetime
import os
import platform
import random
import string

import numpy as np
import pytest
import pytz

import pandas as pd
from pandas import DataFrame
import pandas._testing as tm

api_exceptions = pytest.importorskip("google.api_core.exceptions")
bigquery = pytest.importorskip("google.cloud.bigquery")
service_account = pytest.importorskip("google.oauth2.service_account")
pandas_gbq = pytest.importorskip("pandas_gbq")

PROJECT_ID = None
PRIVATE_KEY_JSON_PATH = None
PRIVATE_KEY_JSON_CONTENTS = None

VERSION = platform.python_version()


def _skip_if_no_project_id():
    if not _get_project_id():
        pytest.skip("Cannot run integration tests without a project id")


def _skip_if_no_private_key_path():
    if not _get_private_key_path():
        pytest.skip("Cannot run integration tests without a private key json file path")


def _get_project_id():
    return PROJECT_ID or os.environ.get("GBQ_PROJECT_ID")


def _get_private_key_path():
    private_key_path = PRIVATE_KEY_JSON_PATH
    if not private_key_path:
        private_key_path = os.environ.get("GBQ_GOOGLE_APPLICATION_CREDENTIALS")
    return private_key_path


def _get_credentials():
    private_key_path = _get_private_key_path()
    if private_key_path:
        return service_account.Credentials.from_service_account_file(private_key_path)


def _get_client():
    project_id = _get_project_id()
    credentials = _get_credentials()
    return bigquery.Client(project=project_id, credentials=credentials)


def generate_rand_str(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def make_mixed_dataframe_v2(test_size):
    # create df to test for all BQ datatypes except RECORD
    bools = np.random.randint(2, size=(1, test_size)).astype(bool)
    flts = np.random.randn(1, test_size)
    ints = np.random.randint(1, 10, size=(1, test_size))
    strs = np.random.randint(1, 10, size=(1, test_size)).astype(str)
    times = [datetime.now(pytz.timezone("US/Arizona")) for t in range(test_size)]
    return DataFrame(
        {
            "bools": bools[0],
            "flts": flts[0],
            "ints": ints[0],
            "strs": strs[0],
            "times": times[0],
        },
        index=range(test_size),
    )


def test_read_gbq_without_deprecated_kwargs(monkeypatch):
    captured_kwargs = {}

    def mock_read_gbq(sql, **kwargs):
        captured_kwargs.update(kwargs)
        return DataFrame([[1.0]])

    monkeypatch.setattr("pandas_gbq.read_gbq", mock_read_gbq)
    pd.read_gbq("SELECT 1")

    assert "verbose" not in captured_kwargs
    assert "private_key" not in captured_kwargs


def test_read_gbq_with_new_kwargs(monkeypatch):
    captured_kwargs = {}

    def mock_read_gbq(sql, **kwargs):
        captured_kwargs.update(kwargs)
        return DataFrame([[1.0]])

    monkeypatch.setattr("pandas_gbq.read_gbq", mock_read_gbq)
    pd.read_gbq("SELECT 1", use_bqstorage_api=True, max_results=1)

    assert captured_kwargs["use_bqstorage_api"]
    assert captured_kwargs["max_results"]


def test_read_gbq_without_new_kwargs(monkeypatch):
    captured_kwargs = {}

    def mock_read_gbq(sql, **kwargs):
        captured_kwargs.update(kwargs)
        return DataFrame([[1.0]])

    monkeypatch.setattr("pandas_gbq.read_gbq", mock_read_gbq)
    pd.read_gbq("SELECT 1")

    assert "use_bqstorage_api" not in captured_kwargs
    assert "max_results" not in captured_kwargs


@pytest.mark.parametrize("progress_bar", [None, "foo"])
def test_read_gbq_progress_bar_type_kwarg(monkeypatch, progress_bar):
    # GH 29857
    captured_kwargs = {}

    def mock_read_gbq(sql, **kwargs):
        captured_kwargs.update(kwargs)
        return DataFrame([[1.0]])

    monkeypatch.setattr("pandas_gbq.read_gbq", mock_read_gbq)
    pd.read_gbq("SELECT 1", progress_bar_type=progress_bar)
    assert "progress_bar_type" in captured_kwargs


@pytest.mark.single
class TestToGBQIntegrationWithServiceAccountKeyPath:
    @pytest.fixture()
    def gbq_dataset(self):
        # Setup Dataset
        _skip_if_no_project_id()
        _skip_if_no_private_key_path()

        dataset_id = "pydata_pandas_bq_testing_" + generate_rand_str()

        self.client = _get_client()
        self.dataset = self.client.dataset(dataset_id)

        # Create the dataset
        self.client.create_dataset(bigquery.Dataset(self.dataset))

        table_name = generate_rand_str()
        destination_table = f"{dataset_id}.{table_name}"
        yield destination_table

        # Teardown Dataset
        self.client.delete_dataset(self.dataset, delete_contents=True)

    def test_roundtrip(self, gbq_dataset):
        destination_table = gbq_dataset

        test_size = 20001
        df = make_mixed_dataframe_v2(test_size)

        df.to_gbq(
            destination_table,
            _get_project_id(),
            chunksize=None,
            credentials=_get_credentials(),
        )

        result = pd.read_gbq(
            f"SELECT COUNT(*) AS num_rows FROM {destination_table}",
            project_id=_get_project_id(),
            credentials=_get_credentials(),
            dialect="standard",
        )
        assert result["num_rows"][0] == test_size

    @pytest.mark.parametrize(
        "if_exists, expected_num_rows, expectation",
        [
            ("append", 300, does_not_raise()),
            ("fail", 200, tm.external_error_raised(pandas_gbq.gbq.TableCreationError)),
            ("replace", 100, does_not_raise()),
        ],
    )
    def test_gbq_if_exists(
        self, if_exists, expected_num_rows, expectation, gbq_dataset
    ):
        # GH 29598
        destination_table = gbq_dataset

        test_size = 200
        df = make_mixed_dataframe_v2(test_size)

        df.to_gbq(
            destination_table,
            _get_project_id(),
            chunksize=None,
            credentials=_get_credentials(),
        )

        with expectation:
            df.iloc[:100].to_gbq(
                destination_table,
                _get_project_id(),
                if_exists=if_exists,
                chunksize=None,
                credentials=_get_credentials(),
            )

        result = pd.read_gbq(
            f"SELECT COUNT(*) AS num_rows FROM {destination_table}",
            project_id=_get_project_id(),
            credentials=_get_credentials(),
            dialect="standard",
        )
        assert result["num_rows"][0] == expected_num_rows
