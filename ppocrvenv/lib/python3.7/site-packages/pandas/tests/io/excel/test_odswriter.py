import re

import pytest

import pandas._testing as tm

from pandas.io.excel import ExcelWriter

odf = pytest.importorskip("odf")

pytestmark = pytest.mark.parametrize("ext", [".ods"])


def test_write_append_mode_raises(ext):
    msg = "Append mode is not supported with odf!"

    with tm.ensure_clean(ext) as f:
        with pytest.raises(ValueError, match=msg):
            ExcelWriter(f, engine="odf", mode="a")


@pytest.mark.parametrize("nan_inf_to_errors", [True, False])
def test_kwargs(ext, nan_inf_to_errors):
    # GH 42286
    # odswriter doesn't utilize kwargs, nothing to check except that it works
    kwargs = {"options": {"nan_inf_to_errors": nan_inf_to_errors}}
    with tm.ensure_clean(ext) as f:
        msg = re.escape("Use of **kwargs is deprecated")
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with ExcelWriter(f, engine="odf", **kwargs) as _:
                pass


@pytest.mark.parametrize("nan_inf_to_errors", [True, False])
def test_engine_kwargs(ext, nan_inf_to_errors):
    # GH 42286
    # odswriter doesn't utilize engine_kwargs, nothing to check except that it works
    engine_kwargs = {"options": {"nan_inf_to_errors": nan_inf_to_errors}}
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f, engine="odf", engine_kwargs=engine_kwargs) as _:
            pass
