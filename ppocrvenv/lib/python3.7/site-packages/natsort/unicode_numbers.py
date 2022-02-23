# -*- coding: utf-8 -*-
"""
Pre-determine the collection of unicode decimals, digits, and numerals.
"""

import unicodedata

from natsort.unicode_numeric_hex import numeric_hex

# Convert each hex into the literal Unicode character.
# Stop if a ValueError is raised in case of a narrow Unicode build.
# The extra check with unicodedata is in case this Python version
# does not support some characters.
numeric_chars = []
for a in numeric_hex:
    try:
        character = chr(a)
    except ValueError:  # pragma: no cover
        break
    if unicodedata.numeric(character, None) is None:
        continue  # pragma: no cover
    numeric_chars.append(character)

# The digit characters are a subset of the numerals.
digit_chars = [a for a in numeric_chars if unicodedata.digit(a, None) is not None]

# The decimal characters are a subset of the numerals
# (probably of the digits, but let's be safe).
decimal_chars = [a for a in numeric_chars if unicodedata.decimal(a, None) is not None]

# Create a single string with the above data.
decimals = "".join(decimal_chars)
digits = "".join(digit_chars)
numeric = "".join(numeric_chars)
digits_no_decimals = "".join([x for x in digits if x not in decimals])
numeric_no_decimals = "".join([x for x in numeric if x not in decimals])
