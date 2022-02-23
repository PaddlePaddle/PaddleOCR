u"""Fixer for 'raise E(V).with_traceback(T)' -> 'raise E, V, T'"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Comma, Node, Leaf, token, syms

class FixRaise(fixer_base.BaseFix):

    PATTERN = u"""
    raise_stmt< 'raise' (power< name=any [trailer< '(' val=any* ')' >]
        [trailer< '.' 'with_traceback' > trailer< '(' trc=any ')' >] > | any) ['from' chain=any] >"""

    def transform(self, node, results):
        name, val, trc = (results.get(u"name"), results.get(u"val"), results.get(u"trc"))
        chain = results.get(u"chain")
        if chain is not None:
            self.warning(node, u"explicit exception chaining is not supported in Python 2")
            chain.prev_sibling.remove()
            chain.remove()
        if trc is not None:
            val = val[0] if val else Leaf(token.NAME, u"None")
            val.prefix = trc.prefix = u" "
            kids = [Leaf(token.NAME, u"raise"), name.clone(), Comma(),
                    val.clone(), Comma(), trc.clone()]
            raise_stmt = Node(syms.raise_stmt, kids)
            node.replace(raise_stmt)
