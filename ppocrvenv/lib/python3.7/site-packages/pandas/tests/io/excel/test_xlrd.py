import io

import pytest

from pandas.compat._optional import import_optional_dependency

import pandas as pd
import pandas._testing as tm
from pandas.tests.io.excel import xlrd_version
from pandas.util.version import Version

from pandas.io.excel import ExcelFile
from pandas.io.excel._base import inspect_excel_format

xlrd = pytest.importorskip("xlrd")
xlwt = pytest.importorskip("xlwt")


@pytest.fixture(autouse=True)
def skip_ods_and_xlsb_files(read_ext):
    if read_ext == ".ods":
        pytest.skip("Not valid for xlrd")
    if read_ext == ".xlsb":
        pytest.skip("Not valid for xlrd")
    if read_ext in (".xlsx", ".xlsm") and xlrd_version >= Version("2"):
        pytest.skip("Not valid for xlrd >= 2.0")


def test_read_xlrd_book(read_ext, frame):
    df = frame

    engine = "xlrd"
    sheet_name = "SheetA"

    with tm.ensure_clean(read_ext) as pth:
        df.to_excel(pth, sheet_name)
        book = xlrd.open_workbook(pth)

        with ExcelFile(book, engine=engine) as xl:
            result = pd.read_excel(xl, sheet_name=sheet_name, index_col=0)
            tm.assert_frame_equal(df, result)

        result = pd.read_excel(book, sheet_name=sheet_name, engine=engine, index_col=0)
        tm.assert_frame_equal(df, result)


def test_excel_file_warning_with_xlsx_file(datapath):
    # GH 29375
    path = datapath("io", "data", "excel", "test1.xlsx")
    has_openpyxl = import_optional_dependency("openpyxl", errors="ignore") is not None
    if not has_openpyxl:
        with tm.assert_produces_warning(
            FutureWarning,
            raise_on_extra_warnings=False,
            match="The xlrd engine is no longer maintained",
        ):
            ExcelFile(path, engine=None)
    else:
        with tm.assert_produces_warning(None):
            pd.read_excel(path, "Sheet1", engine=None)


def test_read_excel_warning_with_xlsx_file(datapath):
    # GH 29375
    path = datapath("io", "data", "excel", "test1.xlsx")
    has_openpyxl = import_optional_dependency("openpyxl", errors="ignore") is not None
    if not has_openpyxl:
        if xlrd_version >= Version("2"):
            with pytest.raises(
                ValueError,
                match="Your version of xlrd is ",
            ):
                pd.read_excel(path, "Sheet1", engine=None)
        else:
            with tm.assert_produces_warning(
                FutureWarning,
                raise_on_extra_warnings=False,
                match="The xlrd engine is no longer maintained",
            ):
                pd.read_excel(path, "Sheet1", engine=None)
    else:
        with tm.assert_produces_warning(None):
            pd.read_excel(path, "Sheet1", engine=None)


@pytest.mark.parametrize(
    "file_header",
    [
        b"\x09\x00\x04\x00\x07\x00\x10\x00",
        b"\x09\x02\x06\x00\x00\x00\x10\x00",
        b"\x09\x04\x06\x00\x00\x00\x10\x00",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    ],
)
def test_read_old_xls_files(file_header):
    # GH 41226
    f = io.BytesIO(file_header)
    assert inspect_excel_format(f) == "xls"
