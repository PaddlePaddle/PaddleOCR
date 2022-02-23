import re

import numpy as np
import pytest

from pandas import (
    DataFrame,
    MultiIndex,
    options,
)
import pandas._testing as tm

from pandas.io.excel import (
    ExcelWriter,
    _XlwtWriter,
)

xlwt = pytest.importorskip("xlwt")

pytestmark = pytest.mark.parametrize("ext,", [".xls"])


def test_excel_raise_error_on_multiindex_columns_and_no_index(ext):
    # MultiIndex as columns is not yet implemented 9794
    cols = MultiIndex.from_tuples(
        [("site", ""), ("2014", "height"), ("2014", "weight")]
    )
    df = DataFrame(np.random.randn(10, 3), columns=cols)

    msg = (
        "Writing to Excel with MultiIndex columns and no index "
        "\\('index'=False\\) is not yet implemented."
    )
    with pytest.raises(NotImplementedError, match=msg):
        with tm.ensure_clean(ext) as path:
            df.to_excel(path, index=False)


def test_excel_multiindex_columns_and_index_true(ext):
    cols = MultiIndex.from_tuples(
        [("site", ""), ("2014", "height"), ("2014", "weight")]
    )
    df = DataFrame(np.random.randn(10, 3), columns=cols)
    with tm.ensure_clean(ext) as path:
        df.to_excel(path, index=True)


def test_excel_multiindex_index(ext):
    # MultiIndex as index works so assert no error #9794
    cols = MultiIndex.from_tuples(
        [("site", ""), ("2014", "height"), ("2014", "weight")]
    )
    df = DataFrame(np.random.randn(3, 10), index=cols)
    with tm.ensure_clean(ext) as path:
        df.to_excel(path, index=False)


def test_to_excel_styleconverter(ext):
    hstyle = {
        "font": {"bold": True},
        "borders": {"top": "thin", "right": "thin", "bottom": "thin", "left": "thin"},
        "alignment": {"horizontal": "center", "vertical": "top"},
    }

    xls_style = _XlwtWriter._convert_to_style(hstyle)
    assert xls_style.font.bold
    assert xlwt.Borders.THIN == xls_style.borders.top
    assert xlwt.Borders.THIN == xls_style.borders.right
    assert xlwt.Borders.THIN == xls_style.borders.bottom
    assert xlwt.Borders.THIN == xls_style.borders.left
    assert xlwt.Alignment.HORZ_CENTER == xls_style.alignment.horz
    assert xlwt.Alignment.VERT_TOP == xls_style.alignment.vert


def test_write_append_mode_raises(ext):
    msg = "Append mode is not supported with xlwt!"

    with tm.ensure_clean(ext) as f:
        with pytest.raises(ValueError, match=msg):
            ExcelWriter(f, engine="xlwt", mode="a")


def test_to_excel_xlwt_warning(ext):
    # GH 26552
    df = DataFrame(np.random.randn(3, 10))
    with tm.ensure_clean(ext) as path:
        with tm.assert_produces_warning(
            FutureWarning,
            match="As the xlwt package is no longer maintained",
        ):
            df.to_excel(path)


def test_option_xls_writer_deprecated(ext):
    # GH 26552
    with tm.assert_produces_warning(
        FutureWarning,
        match="As the xlwt package is no longer maintained",
        check_stacklevel=False,
    ):
        options.io.excel.xls.writer = "xlwt"


@pytest.mark.parametrize("write_only", [True, False])
def test_kwargs(ext, write_only):
    # GH 42286
    # xlwt doesn't utilize kwargs, only test that supplying a kwarg works
    kwargs = {"write_only": write_only}
    with tm.ensure_clean(ext) as f:
        msg = re.escape("Use of **kwargs is deprecated")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with ExcelWriter(f, engine="openpyxl", **kwargs) as writer:
                # xlwt won't allow us to close without writing something
                DataFrame().to_excel(writer)


@pytest.mark.parametrize("write_only", [True, False])
def test_engine_kwargs(ext, write_only):
    # GH 42286
    # xlwt doesn't utilize kwargs, only test that supplying a engine_kwarg works
    engine_kwargs = {"write_only": write_only}
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f, engine="openpyxl", engine_kwargs=engine_kwargs) as writer:
            # xlwt won't allow us to close without writing something
            DataFrame().to_excel(writer)
