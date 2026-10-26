# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for delimiters with multiple OMML expressions."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

etree = pytest.importorskip("lxml.etree")
pytest.importorskip("pylatexenc")

MATH_DIR = Path(__file__).resolve().parents[2] / "paddleocr" / "_doc2md" / "math"
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _load_omml():
    package = types.ModuleType("test_doc2md_delimiter_math")
    package.__path__ = [str(MATH_DIR)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "test_doc2md_delimiter_math.omml", MATH_DIR / "omml.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("values", "separator", "expected"),
    [
        (["n", "k"], "|", r"\left(n\middle|k\right)"),
        (["a", "b", "c"], ";", r"\left(a;b;c\right)"),
        (["n", "k"], None, r"\left(n,k\right)"),
        (["x"], "|", r"\left(x\right)"),
    ],
)
def test_delimiter_keeps_all_expressions_and_separator(values, separator, expected):
    omml = _load_omml()
    prop = (
        f'<m:dPr><m:sepChr m:val="{separator}"/></m:dPr>'
        if separator is not None
        else "<m:dPr/>"
    )
    expressions = "".join(
        f"<m:e><m:r><m:t>{value}</m:t></m:r></m:e>" for value in values
    )
    xml = f'<m:oMath xmlns:m="{OMML_NS}"><m:d>{prop}{expressions}</m:d></m:oMath>'
    assert str(omml.oMath2Latex(etree.fromstring(xml))) == expected
