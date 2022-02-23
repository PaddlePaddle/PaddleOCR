# Copyright (c) 2010-2021 openpyxl

from openpyxl.cell.text import Text

from openpyxl.xml.functions import iterparse
from openpyxl.xml.constants import SHEET_MAIN_NS


def read_string_table(xml_source):
    """Read in all shared strings in the table"""

    strings = []
    STRING_TAG = '{%s}si' % SHEET_MAIN_NS

    for _, node in iterparse(xml_source):
        if node.tag == STRING_TAG:
            text = Text.from_tree(node).content
            text = text.replace('x005F_', '')
            node.clear()

            strings.append(text)

    return strings
