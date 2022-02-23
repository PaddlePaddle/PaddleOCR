import logging
import os
import shlex
import subprocess
import time

import pytest

import pandas._testing as tm

from pandas.io.parsers import read_csv


@pytest.fixture
def tips_file(datapath):
    """Path to the tips dataset"""
    return datapath("io", "data", "csv", "tips.csv")


@pytest.fixture
def jsonl_file(datapath):
    """Path to a JSONL dataset"""
    return datapath("io", "parser", "data", "items.jsonl")


@pytest.fixture
def salaries_table(datapath):
    """DataFrame with the salaries dataset"""
    return read_csv(datapath("io", "parser", "data", "salaries.csv"), sep="\t")


@pytest.fixture
def feather_file(datapath):
    return datapath("io", "data", "feather", "feather-0_3_1.feather")


@pytest.fixture
def s3so(worker_id):
    worker_id = "5" if worker_id == "master" else worker_id.lstrip("gw")
    return {"client_kwargs": {"endpoint_url": f"http://127.0.0.1:555{worker_id}/"}}


@pytest.fixture(scope="session")
def s3_base(worker_id):
    """
    Fixture for mocking S3 interaction.

    Sets up moto server in separate process
    """
    pytest.importorskip("s3fs")
    pytest.importorskip("boto3")
    requests = pytest.importorskip("requests")
    logging.getLogger("requests").disabled = True

    with tm.ensure_safe_environment_variables():
        # temporary workaround as moto fails for botocore >= 1.11 otherwise,
        # see https://github.com/spulec/moto/issues/1924 & 1952
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "foobar_key")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foobar_secret")

        pytest.importorskip("moto", minversion="1.3.14")
        pytest.importorskip("flask")  # server mode needs flask too

        # Launching moto in server mode, i.e., as a separate process
        # with an S3 endpoint on localhost

        worker_id = "5" if worker_id == "master" else worker_id.lstrip("gw")
        endpoint_port = f"555{worker_id}"
        endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

        # pipe to null to avoid logging in terminal
        proc = subprocess.Popen(
            shlex.split(f"moto_server s3 -p {endpoint_port}"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        timeout = 5
        while timeout > 0:
            try:
                # OK to go once server is accepting connections
                r = requests.get(endpoint_uri)
                if r.ok:
                    break
            except Exception:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield endpoint_uri

        proc.terminate()
        proc.wait()


@pytest.fixture()
def s3_resource(s3_base, tips_file, jsonl_file, feather_file):
    """
    Sets up S3 bucket with contents

    The primary bucket name is "pandas-test". The following datasets
    are loaded.

    - tips.csv
    - tips.csv.gz
    - tips.csv.bz2
    - items.jsonl

    A private bucket "cant_get_it" is also created. The boto3 s3 resource
    is yielded by the fixture.
    """
    import boto3
    import s3fs

    test_s3_files = [
        ("tips#1.csv", tips_file),
        ("tips.csv", tips_file),
        ("tips.csv.gz", tips_file + ".gz"),
        ("tips.csv.bz2", tips_file + ".bz2"),
        ("items.jsonl", jsonl_file),
        ("simple_dataset.feather", feather_file),
    ]

    def add_tips_files(bucket_name):
        for s3_key, file_name in test_s3_files:
            with open(file_name, "rb") as f:
                cli.put_object(Bucket=bucket_name, Key=s3_key, Body=f)

    bucket = "pandas-test"
    conn = boto3.resource("s3", endpoint_url=s3_base)
    cli = boto3.client("s3", endpoint_url=s3_base)

    try:
        cli.create_bucket(Bucket=bucket)
    except:  # noqa
        # OK is bucket already exists
        pass
    try:
        cli.create_bucket(Bucket="cant_get_it", ACL="private")
    except:  # noqa
        # OK is bucket already exists
        pass
    timeout = 2
    while not cli.list_buckets()["Buckets"] and timeout > 0:
        time.sleep(0.1)
        timeout -= 0.1

    add_tips_files(bucket)
    add_tips_files("cant_get_it")
    s3fs.S3FileSystem.clear_instance_cache()
    yield conn

    s3 = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_base})

    try:
        s3.rm(bucket, recursive=True)
    except:  # noqa
        pass
    try:
        s3.rm("cant_get_it", recursive=True)
    except:  # noqa
        pass
    timeout = 2
    while cli.list_buckets()["Buckets"] and timeout > 0:
        time.sleep(0.1)
        timeout -= 0.1
