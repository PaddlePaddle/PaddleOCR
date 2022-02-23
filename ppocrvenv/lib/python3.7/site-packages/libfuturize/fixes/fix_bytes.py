"""Optional fixer that changes all unprefixed string literals "..." to b"...".

br'abcd' is a SyntaxError on Python 2 but valid on Python 3.
ur'abcd' is a SyntaxError on Python 3 but valid on Python 2.

"""
from __future__ import unicode_literals

import re
from lib2to3.pgen2 import token
from lib2to3 import fixer_base

_literal_re = re.compile(r"[^bBuUrR]?[\'\"]")

class FixBytes(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "STRING"

    def transform(self, node, results):
        if node.type == token.STRING:
            if _literal_re.match(node.value):
                new = node.clone()
                new.value = u'b' + new.value
                return new
