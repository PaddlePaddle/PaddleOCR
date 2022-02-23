u"""
Fixer for:
(a,)* *b (,c)* [,] = s
for (a,)* *b (,c)* [,] in d: ...
"""

from lib2to3 import fixer_base
from itertools import count
from lib2to3.fixer_util import (Assign, Comma, Call, Newline, Name,
                                Number, token, syms, Node, Leaf)
from libfuturize.fixer_util import indentation, suitify, commatize
# from libfuturize.fixer_util import Assign, Comma, Call, Newline, Name, Number, indentation, suitify, commatize, token, syms, Node, Leaf

def assignment_source(num_pre, num_post, LISTNAME, ITERNAME):
    u"""
    Accepts num_pre and num_post, which are counts of values
    before and after the starg (not including the starg)
    Returns a source fit for Assign() from fixer_util
    """
    children = []
    pre = unicode(num_pre)
    post = unicode(num_post)
    # This code builds the assignment source from lib2to3 tree primitives.
    # It's not very readable, but it seems like the most correct way to do it.
    if num_pre > 0:
        pre_part = Node(syms.power, [Name(LISTNAME), Node(syms.trailer, [Leaf(token.LSQB, u"["), Node(syms.subscript, [Leaf(token.COLON, u":"), Number(pre)]), Leaf(token.RSQB, u"]")])])
        children.append(pre_part)
        children.append(Leaf(token.PLUS, u"+", prefix=u" "))
    main_part = Node(syms.power, [Leaf(token.LSQB, u"[", prefix=u" "), Name(LISTNAME), Node(syms.trailer, [Leaf(token.LSQB, u"["), Node(syms.subscript, [Number(pre) if num_pre > 0 else Leaf(1, u""), Leaf(token.COLON, u":"), Node(syms.factor, [Leaf(token.MINUS, u"-"), Number(post)]) if num_post > 0 else Leaf(1, u"")]), Leaf(token.RSQB, u"]"), Leaf(token.RSQB, u"]")])])
    children.append(main_part)
    if num_post > 0:
        children.append(Leaf(token.PLUS, u"+", prefix=u" "))
        post_part = Node(syms.power, [Name(LISTNAME, prefix=u" "), Node(syms.trailer, [Leaf(token.LSQB, u"["), Node(syms.subscript, [Node(syms.factor, [Leaf(token.MINUS, u"-"), Number(post)]), Leaf(token.COLON, u":")]), Leaf(token.RSQB, u"]")])])
        children.append(post_part)
    source = Node(syms.arith_expr, children)
    return source

class FixUnpacking(fixer_base.BaseFix):

    PATTERN = u"""
    expl=expr_stmt< testlist_star_expr<
        pre=(any ',')*
            star_expr< '*' name=NAME >
        post=(',' any)* [','] > '=' source=any > |
    impl=for_stmt< 'for' lst=exprlist<
        pre=(any ',')*
            star_expr< '*' name=NAME >
        post=(',' any)* [','] > 'in' it=any ':' suite=any>"""

    def fix_explicit_context(self, node, results):
        pre, name, post, source = (results.get(n) for n in (u"pre", u"name", u"post", u"source"))
        pre = [n.clone() for n in pre if n.type == token.NAME]
        name.prefix = u" "
        post = [n.clone() for n in post if n.type == token.NAME]
        target = [n.clone() for n in commatize(pre + [name.clone()] + post)]
        # to make the special-case fix for "*z, = ..." correct with the least
        # amount of modification, make the left-side into a guaranteed tuple
        target.append(Comma())
        source.prefix = u""
        setup_line = Assign(Name(self.LISTNAME), Call(Name(u"list"), [source.clone()]))
        power_line = Assign(target, assignment_source(len(pre), len(post), self.LISTNAME, self.ITERNAME))
        return setup_line, power_line

    def fix_implicit_context(self, node, results):
        u"""
        Only example of the implicit context is
        a for loop, so only fix that.
        """
        pre, name, post, it = (results.get(n) for n in (u"pre", u"name", u"post", u"it"))
        pre = [n.clone() for n in pre if n.type == token.NAME]
        name.prefix = u" "
        post = [n.clone() for n in post if n.type == token.NAME]
        target = [n.clone() for n in commatize(pre + [name.clone()] + post)]
        # to make the special-case fix for "*z, = ..." correct with the least
        # amount of modification, make the left-side into a guaranteed tuple
        target.append(Comma())
        source = it.clone()
        source.prefix = u""
        setup_line = Assign(Name(self.LISTNAME), Call(Name(u"list"), [Name(self.ITERNAME)]))
        power_line = Assign(target, assignment_source(len(pre), len(post), self.LISTNAME, self.ITERNAME))
        return setup_line, power_line

    def transform(self, node, results):
        u"""
        a,b,c,d,e,f,*g,h,i = range(100) changes to
        _3to2list = list(range(100))
        a,b,c,d,e,f,g,h,i, = _3to2list[:6] + [_3to2list[6:-2]] + _3to2list[-2:]

        and

        for a,b,*c,d,e in iter_of_iters: do_stuff changes to
        for _3to2iter in iter_of_iters:
            _3to2list = list(_3to2iter)
            a,b,c,d,e, = _3to2list[:2] + [_3to2list[2:-2]] + _3to2list[-2:]
            do_stuff
        """
        self.LISTNAME = self.new_name(u"_3to2list")
        self.ITERNAME = self.new_name(u"_3to2iter")
        expl, impl = results.get(u"expl"), results.get(u"impl")
        if expl is not None:
            setup_line, power_line = self.fix_explicit_context(node, results)
            setup_line.prefix = expl.prefix
            power_line.prefix = indentation(expl.parent)
            setup_line.append_child(Newline())
            parent = node.parent
            i = node.remove()
            parent.insert_child(i, power_line)
            parent.insert_child(i, setup_line)
        elif impl is not None:
            setup_line, power_line = self.fix_implicit_context(node, results)
            suitify(node)
            suite = [k for k in node.children if k.type == syms.suite][0]
            setup_line.prefix = u""
            power_line.prefix = suite.children[1].value
            suite.children[2].prefix = indentation(suite.children[2])
            suite.insert_child(2, Newline())
            suite.insert_child(2, power_line)
            suite.insert_child(2, Newline())
            suite.insert_child(2, setup_line)
            results.get(u"lst").replace(Name(self.ITERNAME, prefix=u" "))
