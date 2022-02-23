# Copyright 2016-present MongoDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Some changes copyright 2021-present Matthias Valvekens,
# licensed under the license of the pyHanko project (see LICENSE file).


"""An implementation of RFC4013 SASLprep."""

__all__ = ['saslprep']

import stringprep
from typing import Callable, Tuple
import unicodedata

# RFC4013 section 2.3 prohibited output.
_PROHIBITED: Tuple[Callable[[str], bool], ...] = (
    # A strict reading of RFC 4013 requires table c12 here, but
    # characters from it are mapped to SPACE in the Map step. Can
    # normalization reintroduce them somehow?
    stringprep.in_table_c12,
    stringprep.in_table_c21_c22,
    stringprep.in_table_c3,
    stringprep.in_table_c4,
    stringprep.in_table_c5,
    stringprep.in_table_c6,
    stringprep.in_table_c7,
    stringprep.in_table_c8,
    stringprep.in_table_c9)


def saslprep(data: str, prohibit_unassigned_code_points: bool = True) -> str:
    """An implementation of RFC4013 SASLprep.
    :param data:
        The string to SASLprep.
    :param prohibit_unassigned_code_points:
        RFC 3454 and RFCs for various SASL mechanisms distinguish between
        `queries` (unassigned code points allowed) and
        `stored strings` (unassigned code points prohibited). Defaults
        to ``True`` (unassigned code points are prohibited).
    :return: The SASLprep'ed version of `data`.
    """
    if prohibit_unassigned_code_points:
        prohibited = _PROHIBITED + (stringprep.in_table_a1,)
    else:
        prohibited = _PROHIBITED

    # RFC3454 section 2, step 1 - Map
    # RFC4013 section 2.1 mappings
    # Map Non-ASCII space characters to SPACE (U+0020). Map
    # commonly mapped to nothing characters to, well, nothing.
    in_table_c12 = stringprep.in_table_c12
    in_table_b1 = stringprep.in_table_b1
    data = "".join(
        ["\u0020" if in_table_c12(elt) else elt
         for elt in data if not in_table_b1(elt)])

    # RFC3454 section 2, step 2 - Normalize
    # RFC4013 section 2.2 normalization
    data = unicodedata.ucd_3_2_0.normalize('NFKC', data)

    in_table_d1 = stringprep.in_table_d1
    if in_table_d1(data[0]):
        if not in_table_d1(data[-1]):
            # RFC3454, Section 6, #3. If a string contains any
            # RandALCat character, the first and last characters
            # MUST be RandALCat characters.
            raise ValueError("SASLprep: failed bidirectional check")
        # RFC3454, Section 6, #2. If a string contains any RandALCat
        # character, it MUST NOT contain any LCat character.
        prohibited = prohibited + (stringprep.in_table_d2,)
    else:
        # RFC3454, Section 6, #3. Following the logic of #3, if
        # the first character is not a RandALCat, no other character
        # can be either.
        prohibited = prohibited + (in_table_d1,)

    # RFC3454 section 2, step 3 and 4 - Prohibit and check bidi
    for char in data:
        if any(in_table(char) for in_table in prohibited):
            raise ValueError(
                "SASLprep: failed prohibited character check")

    return data
