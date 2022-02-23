"""
Tests for the pandas custom headers in http(s) requests
"""
import gzip
import http.server
from io import BytesIO
import threading

import pytest

import pandas.util._test_decorators as td

import pandas as pd
import pandas._testing as tm


class BaseUserAgentResponder(http.server.BaseHTTPRequestHandler):
    """
    Base class for setting up a server that can be set up to respond
    with a particular file format with accompanying content-type headers.
    The interfaces on the different io methods are different enough
    that this seemed logical to do.
    """

    def start_processing_headers(self):
        """
        shared logic at the start of a GET request
        """
        self.send_response(200)
        self.requested_from_user_agent = self.headers["User-Agent"]
        response_df = pd.DataFrame(
            {
                "header": [self.requested_from_user_agent],
            }
        )
        return response_df

    def gzip_bytes(self, response_bytes):
        """
        some web servers will send back gzipped files to save bandwidth
        """
        bio = BytesIO()
        zipper = gzip.GzipFile(fileobj=bio, mode="w")
        zipper.write(response_bytes)
        zipper.close()
        response_bytes = bio.getvalue()
        return response_bytes

    def write_back_bytes(self, response_bytes):
        """
        shared logic at the end of a GET request
        """
        self.wfile.write(response_bytes)


class CSVUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()

        self.send_header("Content-Type", "text/csv")
        self.end_headers()

        response_bytes = response_df.to_csv(index=False).encode("utf-8")
        self.write_back_bytes(response_bytes)


class GzippedCSVUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Encoding", "gzip")
        self.end_headers()

        response_bytes = response_df.to_csv(index=False).encode("utf-8")
        response_bytes = self.gzip_bytes(response_bytes)

        self.write_back_bytes(response_bytes)


class JSONUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        response_bytes = response_df.to_json().encode("utf-8")

        self.write_back_bytes(response_bytes)


class GzippedJSONUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.end_headers()

        response_bytes = response_df.to_json().encode("utf-8")
        response_bytes = self.gzip_bytes(response_bytes)

        self.write_back_bytes(response_bytes)


class ParquetPyArrowUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        response_bytes = response_df.to_parquet(index=False, engine="pyarrow")

        self.write_back_bytes(response_bytes)


class ParquetFastParquetUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        # the fastparquet engine doesn't like to write to a buffer
        # it can do it via the open_with function being set appropriately
        # however it automatically calls the close method and wipes the buffer
        # so just overwrite that attribute on this instance to not do that

        # protected by an importorskip in the respective test
        import fsspec

        response_df.to_parquet(
            "memory://fastparquet_user_agent.parquet",
            index=False,
            engine="fastparquet",
            compression=None,
        )
        with fsspec.open("memory://fastparquet_user_agent.parquet", "rb") as f:
            response_bytes = f.read()

        self.write_back_bytes(response_bytes)


class PickleUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        bio = BytesIO()
        response_df.to_pickle(bio)
        response_bytes = bio.getvalue()

        self.write_back_bytes(response_bytes)


class StataUserAgentResponder(BaseUserAgentResponder):
    def do_GET(self):
        response_df = self.start_processing_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        bio = BytesIO()
        response_df.to_stata(bio, write_index=False)
        response_bytes = bio.getvalue()

        self.write_back_bytes(response_bytes)


class AllHeaderCSVResponder(http.server.BaseHTTPRequestHandler):
    """
    Send all request headers back for checking round trip
    """

    def do_GET(self):
        response_df = pd.DataFrame(self.headers.items())
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.end_headers()
        response_bytes = response_df.to_csv(index=False).encode("utf-8")
        self.wfile.write(response_bytes)


@pytest.mark.parametrize(
    "responder, read_method, parquet_engine",
    [
        (CSVUserAgentResponder, pd.read_csv, None),
        (JSONUserAgentResponder, pd.read_json, None),
        (ParquetPyArrowUserAgentResponder, pd.read_parquet, "pyarrow"),
        pytest.param(
            ParquetFastParquetUserAgentResponder,
            pd.read_parquet,
            "fastparquet",
            # TODO(ArrayManager) fastparquet
            marks=td.skip_array_manager_not_yet_implemented,
        ),
        (PickleUserAgentResponder, pd.read_pickle, None),
        (StataUserAgentResponder, pd.read_stata, None),
        (GzippedCSVUserAgentResponder, pd.read_csv, None),
        (GzippedJSONUserAgentResponder, pd.read_json, None),
    ],
)
def test_server_and_default_headers(responder, read_method, parquet_engine):
    if parquet_engine is not None:
        pytest.importorskip(parquet_engine)
        if parquet_engine == "fastparquet":
            pytest.importorskip("fsspec")

    # passing 0 for the port will let the system find an unused port
    with http.server.HTTPServer(("localhost", 0), responder) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()

        port = server.server_port
        if parquet_engine is None:
            df_http = read_method(f"http://localhost:{port}")
        else:
            df_http = read_method(f"http://localhost:{port}", engine=parquet_engine)
        server.shutdown()
        server.server_close()
        server_thread.join()
    assert not df_http.empty


@pytest.mark.parametrize(
    "responder, read_method, parquet_engine",
    [
        (CSVUserAgentResponder, pd.read_csv, None),
        (JSONUserAgentResponder, pd.read_json, None),
        (ParquetPyArrowUserAgentResponder, pd.read_parquet, "pyarrow"),
        pytest.param(
            ParquetFastParquetUserAgentResponder,
            pd.read_parquet,
            "fastparquet",
            # TODO(ArrayManager) fastparquet
            marks=td.skip_array_manager_not_yet_implemented,
        ),
        (PickleUserAgentResponder, pd.read_pickle, None),
        (StataUserAgentResponder, pd.read_stata, None),
        (GzippedCSVUserAgentResponder, pd.read_csv, None),
        (GzippedJSONUserAgentResponder, pd.read_json, None),
    ],
)
def test_server_and_custom_headers(responder, read_method, parquet_engine):
    if parquet_engine is not None:
        pytest.importorskip(parquet_engine)
        if parquet_engine == "fastparquet":
            pytest.importorskip("fsspec")

    custom_user_agent = "Super Cool One"
    df_true = pd.DataFrame({"header": [custom_user_agent]})

    # passing 0 for the port will let the system find an unused port
    with http.server.HTTPServer(("localhost", 0), responder) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()

        port = server.server_port
        if parquet_engine is None:
            df_http = read_method(
                f"http://localhost:{port}",
                storage_options={"User-Agent": custom_user_agent},
            )
        else:
            df_http = read_method(
                f"http://localhost:{port}",
                storage_options={"User-Agent": custom_user_agent},
                engine=parquet_engine,
            )
        server.shutdown()

        server.server_close()
        server_thread.join()

    tm.assert_frame_equal(df_true, df_http)


@pytest.mark.parametrize(
    "responder, read_method",
    [
        (AllHeaderCSVResponder, pd.read_csv),
    ],
)
def test_server_and_all_custom_headers(responder, read_method):
    custom_user_agent = "Super Cool One"
    custom_auth_token = "Super Secret One"
    storage_options = {
        "User-Agent": custom_user_agent,
        "Auth": custom_auth_token,
    }

    # passing 0 for the port will let the system find an unused port
    with http.server.HTTPServer(("localhost", 0), responder) as server:
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()

        port = server.server_port
        df_http = read_method(
            f"http://localhost:{port}",
            storage_options=storage_options,
        )
        server.shutdown()
        server.server_close()
        server_thread.join()

    df_http = df_http[df_http["0"].isin(storage_options.keys())]
    df_http = df_http.sort_values(["0"]).reset_index()
    df_http = df_http[["0", "1"]]

    keys = list(storage_options.keys())
    df_true = pd.DataFrame({"0": keys, "1": [storage_options[k] for k in keys]})
    df_true = df_true.sort_values(["0"])
    df_true = df_true.reset_index().drop(["index"], axis=1)

    tm.assert_frame_equal(df_true, df_http)


@pytest.mark.parametrize(
    "engine",
    [
        "pyarrow",
        "fastparquet",
    ],
)
def test_to_parquet_to_disk_with_storage_options(engine):
    headers = {
        "User-Agent": "custom",
        "Auth": "other_custom",
    }

    pytest.importorskip(engine)

    true_df = pd.DataFrame({"column_name": ["column_value"]})
    msg = (
        "storage_options passed with file object or non-fsspec file path|"
        "storage_options passed with buffer, or non-supported URL"
    )
    with pytest.raises(ValueError, match=msg):
        true_df.to_parquet("/tmp/junk.parquet", storage_options=headers, engine=engine)
