"""
Fixer that adds ``from builtins import object`` if there is a line
like this:
    class Foo(object):
"""

from lib2to3 import fixer_base

from libfuturize.fixer_util import touch_import_top


class FixObject(fixer_base.BaseFix):

    PATTERN = u"classdef< 'class' NAME '(' name='object' ')' colon=':' any >"

    def transform(self, node, results):
        touch_import_top(u'builtins', 'object', node)
