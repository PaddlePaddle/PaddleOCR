import pytest

import pandas.util._test_decorators as td

import pandas._testing as tm

from pandas.io.parsers import read_csv


@pytest.fixture
def frame(float_frame):
    """
    Returns the first ten items in fixture "float_frame".
    """
    return float_frame[:10]


@pytest.fixture
def tsframe():
    return tm.makeTimeDataFrame()[:5]


@pytest.fixture(params=[True, False])
def merge_cells(request):
    return request.param


@pytest.fixture
def df_ref(datapath):
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    filepath = datapath("io", "data", "csv", "test1.csv")
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine="python")
    return df_ref


@pytest.fixture(params=[".xls", ".xlsx", ".xlsm", ".ods", ".xlsb"])
def read_ext(request):
    """
    Valid extensions for reading Excel files.
    """
    return request.param


@pytest.fixture(autouse=True)
def check_for_file_leaks():
    """
    Fixture to run around every test to ensure that we are not leaking files.

    See also
    --------
    _test_decorators.check_file_leaks
    """
    # GH#30162
    psutil = td.safe_import("psutil")
    if not psutil:
        yield

    else:
        proc = psutil.Process()
        flist = proc.open_files()
        yield
        flist2 = proc.open_files()
        assert flist == flist2
