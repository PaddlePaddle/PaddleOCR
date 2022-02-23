import sys
import types

import pytest

import pandas.util._test_decorators as td


@pytest.mark.parametrize("name", ["foo", "hello123"])
def test_safe_import_non_existent(name):
    assert not td.safe_import(name)


def test_safe_import_exists():
    assert td.safe_import("pandas")


@pytest.mark.parametrize("min_version,valid", [("0.0.0", True), ("99.99.99", False)])
def test_safe_import_versions(min_version, valid):
    result = td.safe_import("pandas", min_version=min_version)
    result = result if valid else not result
    assert result


@pytest.mark.parametrize(
    "min_version,valid", [(None, False), ("1.0", True), ("2.0", False)]
)
def test_safe_import_dummy(monkeypatch, min_version, valid):
    mod_name = "hello123"

    mod = types.ModuleType(mod_name)
    mod.__version__ = "1.5"

    if min_version is not None:
        monkeypatch.setitem(sys.modules, mod_name, mod)

    result = td.safe_import(mod_name, min_version=min_version)
    result = result if valid else not result
    assert result
