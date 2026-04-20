# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from .omml import OMML_NS, oMath2Latex

# XML namespace constants shared by converters for DrawingML math extraction
_M = OMML_NS  # already includes braces: "{http://...}"
_A14 = "{http://schemas.microsoft.com/office/drawing/2010/main}"


def convert_omath(omath_element) -> str:
    """Convert an m:oMath lxml element to LaTeX string. Returns empty string on failure."""
    try:
        return str(oMath2Latex(omath_element)).strip()
    except Exception:
        return ""


def paragraph_has_math(para_element) -> bool:
    """Check if an XML element contains OMML math (a14:m or m:oMath)."""
    return (
        para_element.find(f".//{_A14}m") is not None
        or para_element.find(f".//{_M}oMath") is not None
    )


def extract_math_from_paragraph(para_element) -> list:
    """Extract LaTeX strings from math elements in a DrawingML paragraph XML element.

    Handles three nesting patterns:
    1. a14:m → m:oMath (or m:oMathPara → m:oMath)
    2. Direct m:oMathPara → m:oMath (not wrapped in a14:m)
    3. Direct m:oMath (not inside a14:m or m:oMathPara)
    """
    results = []
    # a14:m wraps m:oMathPara or m:oMath
    for a14m in para_element.findall(f".//{_A14}m"):
        found_omath = False
        for omath in a14m.findall(f".//{_M}oMath"):
            latex = convert_omath(omath)
            if latex:
                results.append(latex)
                found_omath = True
        # No oMath inside this a14:m? Try the a14:m element itself
        if not found_omath:
            latex = convert_omath(a14m)
            if latex:
                results.append(latex)
    # Direct m:oMathPara / m:oMath not wrapped in a14:m
    for omath_para in para_element.findall(f".//{_M}oMathPara"):
        parent = omath_para.getparent()
        if parent is not None and parent.tag == f"{_A14}m":
            continue  # already handled above (oMathPara is inside a14:m)
        for omath in omath_para.findall(f"{_M}oMath"):
            latex = convert_omath(omath)
            if latex:
                results.append(latex)
    for omath in para_element.findall(f".//{_M}oMath"):
        parent = omath.getparent()
        if parent is not None and parent.tag in (f"{_A14}m", f"{_M}oMathPara"):
            continue  # already handled
        latex = convert_omath(omath)
        if latex:
            results.append(latex)
    return results


__all__ = [
    "oMath2Latex",
    "OMML_NS",
    "convert_omath",
    "paragraph_has_math",
    "extract_math_from_paragraph",
]
