u"""
Fixer for (metaclass=X) -> __metaclass__ = X
Some semantics (see PEP 3115) may be altered in the translation."""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, syms, Node, Leaf, Newline, find_root
from lib2to3.pygram import token
from libfuturize.fixer_util import indentation, suitify
# from ..fixer_util import Name, syms, Node, Leaf, Newline, find_root, indentation, suitify

def has_metaclass(parent):
    results = None
    for node in parent.children:
        kids = node.children
        if node.type == syms.argument:
            if kids[0] == Leaf(token.NAME, u"metaclass") and \
                kids[1] == Leaf(token.EQUAL, u"=") and \
                kids[2]:
                #Hack to avoid "class X(=):" with this case.
                results = [node] + kids
                break
        elif node.type == syms.arglist:
            # Argument list... loop through it looking for:
            # Node(*, [*, Leaf(token.NAME, u"metaclass"), Leaf(token.EQUAL, u"="), Leaf(*, *)]
            for child in node.children:
                if results: break
                if child.type == token.COMMA:
                    #Store the last comma, which precedes the metaclass
                    comma = child
                elif type(child) == Node:
                    meta = equal = name = None
                    for arg in child.children:
                        if arg == Leaf(token.NAME, u"metaclass"):
                            #We have the (metaclass) part
                            meta = arg
                        elif meta and arg == Leaf(token.EQUAL, u"="):
                            #We have the (metaclass=) part
                            equal = arg
                        elif meta and equal:
                            #Here we go, we have (metaclass=X)
                            name = arg
                            results = (comma, meta, equal, name)
                            break
    return results


class FixMetaclass(fixer_base.BaseFix):

    PATTERN = u"""
    classdef<any*>
    """

    def transform(self, node, results):
        meta_results = has_metaclass(node)
        if not meta_results: return
        for meta in meta_results:
            meta.remove()
        target = Leaf(token.NAME, u"__metaclass__")
        equal = Leaf(token.EQUAL, u"=", prefix=u" ")
        # meta is the last item in what was returned by has_metaclass(): name
        name = meta
        name.prefix = u" "
        stmt_node = Node(syms.atom, [target, equal, name])

        suitify(node)
        for item in node.children:
            if item.type == syms.suite:
                for stmt in item.children:
                    if stmt.type == token.INDENT:
                        # Insert, in reverse order, the statement, a newline,
                        # and an indent right after the first indented line
                        loc = item.children.index(stmt) + 1
                        # Keep consistent indentation form
                        ident = Leaf(token.INDENT, stmt.value)
                        item.insert_child(loc, ident)
                        item.insert_child(loc, Newline())
                        item.insert_child(loc, stmt_node)
                        break
