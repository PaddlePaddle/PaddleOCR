u"""
Fixer for:
it.__next__() -> it.next().
next(it) -> it.next().
"""

from lib2to3.pgen2 import token
from lib2to3.pygram import python_symbols as syms
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, Call, find_binding, Attr

bind_warning = u"Calls to builtin next() possibly shadowed by global binding"


class FixNext(fixer_base.BaseFix):

    PATTERN = u"""
    power< base=any+ trailer< '.' attr='__next__' > any* >
    |
    power< head='next' trailer< '(' arg=any ')' > any* >
    |
    classdef< 'class' base=any+ ':'
              suite< any*
                     funcdef< 'def'
                              attr='__next__'
                              parameters< '(' NAME ')' > any+ >
                     any* > >
    """

    def transform(self, node, results):
        assert results

        base = results.get(u"base")
        attr = results.get(u"attr")
        head = results.get(u"head")
        arg_ = results.get(u"arg")
        if arg_:
            arg = arg_.clone()
            head.replace(Attr(Name(unicode(arg),prefix=head.prefix),
                              Name(u"next")))
            arg_.remove()
        elif base:
            attr.replace(Name(u"next", prefix=attr.prefix))
