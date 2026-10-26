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

"""Regression tests for standalone OMML conversion."""

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
    """Load the converter without importing the full PaddleOCR model stack."""
    package = types.ModuleType("test_doc2md_math")
    package.__path__ = [str(MATH_DIR)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "test_doc2md_math.omml", MATH_DIR / "omml.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("acc", r"\hat{x}"),
        ("bar", r"\overline{x}"),
        ("d", r"\left(x\right)"),
        ("groupChr", r"\underbrace{x}"),
        ("func", "x"),
    ],
)
def test_optional_property_nodes_do_not_drop_formula(tag, expected):
    omml = _load_omml()
    xml = (
        f'<m:oMath xmlns:m="{OMML_NS}"><m:{tag}>'
        "<m:e><m:r><m:t>x</m:t></m:r></m:e>"
        f"</m:{tag}></m:oMath>"
    )
    assert str(omml.oMath2Latex(etree.fromstring(xml))) == expected


def test_explicit_bar_position_is_preserved():
    omml = _load_omml()
    xml = (
        f'<m:oMath xmlns:m="{OMML_NS}"><m:bar>'
        '<m:barPr><m:pos m:val="bot"/></m:barPr>'
        "<m:e><m:r><m:t>x</m:t></m:r></m:e>"
        "</m:bar></m:oMath>"
    )
    assert str(omml.oMath2Latex(etree.fromstring(xml))) == r"\underline{x}"


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("min", r"\min_{n}"),
        ("inf", r"\inf_{n}"),
        ("sup", r"\sup_{n}"),
        ("gcd", r"\gcd_{n}"),
        ("foo", r"\underset{n}{foo}"),
    ],
)
def test_limit_below_preserves_unsupported_base(base, expected):
    omml = _load_omml()
    xml = (
        f'<m:oMath xmlns:m="{OMML_NS}"><m:limLow>'
        f"<m:e><m:r><m:t>{base}</m:t></m:r></m:e>"
        "<m:lim><m:r><m:t>n</m:t></m:r></m:lim>"
        "</m:limLow></m:oMath>"
    )
    assert str(omml.oMath2Latex(etree.fromstring(xml))) == expected
