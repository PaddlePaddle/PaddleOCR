from __future__ import annotations

import os

import pytest

from pandas import (
    read_csv,
    read_table,
)


class BaseParser:
    engine: str | None = None
    low_memory = True
    float_precision_choices: list[str | None] = []

    def update_kwargs(self, kwargs):
        kwargs = kwargs.copy()
        kwargs.update({"engine": self.engine, "low_memory": self.low_memory})

        return kwargs

    def read_csv(self, *args, **kwargs):
        kwargs = self.update_kwargs(kwargs)
        return read_csv(*args, **kwargs)

    def read_table(self, *args, **kwargs):
        kwargs = self.update_kwargs(kwargs)
        return read_table(*args, **kwargs)


class CParser(BaseParser):
    engine = "c"
    float_precision_choices = [None, "high", "round_trip"]


class CParserHighMemory(CParser):
    low_memory = False


class CParserLowMemory(CParser):
    low_memory = True


class PythonParser(BaseParser):
    engine = "python"
    float_precision_choices = [None]


@pytest.fixture
def csv_dir_path(datapath):
    """
    The directory path to the data files needed for parser tests.
    """
    return datapath("io", "parser", "data")


@pytest.fixture
def csv1(datapath):
    """
    The path to the data file "test1.csv" needed for parser tests.
    """
    return os.path.join(datapath("io", "data", "csv"), "test1.csv")


_cParserHighMemory = CParserHighMemory()
_cParserLowMemory = CParserLowMemory()
_pythonParser = PythonParser()

_py_parsers_only = [_pythonParser]
_c_parsers_only = [_cParserHighMemory, _cParserLowMemory]
_all_parsers = [*_c_parsers_only, *_py_parsers_only]

_py_parser_ids = ["python"]
_c_parser_ids = ["c_high", "c_low"]
_all_parser_ids = [*_c_parser_ids, *_py_parser_ids]


@pytest.fixture(params=_all_parsers, ids=_all_parser_ids)
def all_parsers(request):
    """
    Fixture all of the CSV parsers.
    """
    return request.param


@pytest.fixture(params=_c_parsers_only, ids=_c_parser_ids)
def c_parser_only(request):
    """
    Fixture all of the CSV parsers using the C engine.
    """
    return request.param


@pytest.fixture(params=_py_parsers_only, ids=_py_parser_ids)
def python_parser_only(request):
    """
    Fixture all of the CSV parsers using the Python engine.
    """
    return request.param


def _get_all_parser_float_precision_combinations():
    """
    Return all allowable parser and float precision
    combinations and corresponding ids.
    """
    params = []
    ids = []
    for parser, parser_id in zip(_all_parsers, _all_parser_ids):
        for precision in parser.float_precision_choices:
            params.append((parser, precision))
            ids.append(f"{parser_id}-{precision}")

    return {"params": params, "ids": ids}


@pytest.fixture(
    params=_get_all_parser_float_precision_combinations()["params"],
    ids=_get_all_parser_float_precision_combinations()["ids"],
)
def all_parsers_all_precisions(request):
    """
    Fixture for all allowable combinations of parser
    and float precision
    """
    return request.param


_utf_values = [8, 16, 32]

_encoding_seps = ["", "-", "_"]
_encoding_prefixes = ["utf", "UTF"]

_encoding_fmts = [
    f"{prefix}{sep}" + "{0}" for sep in _encoding_seps for prefix in _encoding_prefixes
]


@pytest.fixture(params=_utf_values)
def utf_value(request):
    """
    Fixture for all possible integer values for a UTF encoding.
    """
    return request.param


@pytest.fixture(params=_encoding_fmts)
def encoding_fmt(request):
    """
    Fixture for all possible string formats of a UTF encoding.
    """
    return request.param


@pytest.fixture(
    params=[
        ("-1,0", -1.0),
        ("-1,2e0", -1.2),
        ("-1e0", -1.0),
        ("+1e0", 1.0),
        ("+1e+0", 1.0),
        ("+1e-1", 0.1),
        ("+,1e1", 1.0),
        ("+1,e0", 1.0),
        ("-,1e1", -1.0),
        ("-1,e0", -1.0),
        ("0,1", 0.1),
        ("1,", 1.0),
        (",1", 0.1),
        ("-,1", -0.1),
        ("1_,", 1.0),
        ("1_234,56", 1234.56),
        ("1_234,56e0", 1234.56),
        # negative cases; must not parse as float
        ("_", "_"),
        ("-_", "-_"),
        ("-_1", "-_1"),
        ("-_1e0", "-_1e0"),
        ("_1", "_1"),
        ("_1,", "_1,"),
        ("_1,_", "_1,_"),
        ("_1e0", "_1e0"),
        ("1,2e_1", "1,2e_1"),
        ("1,2e1_0", "1,2e1_0"),
        ("1,_2", "1,_2"),
        (",1__2", ",1__2"),
        (",1e", ",1e"),
        ("-,1e", "-,1e"),
        ("1_000,000_000", "1_000,000_000"),
        ("1,e1_2", "1,e1_2"),
        ("e11,2", "e11,2"),
        ("1e11,2", "1e11,2"),
        ("1,2,2", "1,2,2"),
        ("1,2_1", "1,2_1"),
        ("1,2e-10e1", "1,2e-10e1"),
        ("--1,2", "--1,2"),
        ("1a_2,1", "1a_2,1"),
        ("1,2E-1", 0.12),
        ("1,2E1", 12.0),
    ]
)
def numeric_decimal(request):
    """
    Fixture for all numeric formats which should get recognized. The first entry
    represents the value to read while the second represents the expected result.
    """
    return request.param
