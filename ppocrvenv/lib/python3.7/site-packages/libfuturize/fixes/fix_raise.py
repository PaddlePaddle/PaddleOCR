"""Fixer for 'raise E, V'

From Armin Ronacher's ``python-modernize``.

raise         -> raise
raise E       -> raise E
raise E, 5    -> raise E(5)
raise E, 5, T -> raise E(5).with_traceback(T)
raise E, None, T -> raise E.with_traceback(T)

raise (((E, E'), E''), E'''), 5 -> raise E(5)
raise "foo", V, T               -> warns about string exceptions

raise E, (V1, V2) -> raise E(V1, V2)
raise E, (V1, V2), T -> raise E(V1, V2).with_traceback(T)


CAVEATS:
1) "raise E, V, T" cannot be translated safely in general. If V
   is not a tuple or a (number, string, None) literal, then:

   raise E, V, T -> from future.utils import raise_
                    raise_(E, V, T)
"""
# Author: Collin Winter, Armin Ronacher, Mark Huang

# Local imports
from lib2to3 import pytree, fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import Name, Call, is_tuple, Comma, Attr, ArgList

from libfuturize.fixer_util import touch_import_top


class FixRaise(fixer_base.BaseFix):

    BM_compatible = True
    PATTERN = """
    raise_stmt< 'raise' exc=any [',' val=any [',' tb=any]] >
    """

    def transform(self, node, results):
        syms = self.syms

        exc = results["exc"].clone()
        if exc.type == token.STRING:
            msg = "Python 3 does not support string exceptions"
            self.cannot_convert(node, msg)
            return

        # Python 2 supports
        #  raise ((((E1, E2), E3), E4), E5), V
        # as a synonym for
        #  raise E1, V
        # Since Python 3 will not support this, we recurse down any tuple
        # literals, always taking the first element.
        if is_tuple(exc):
            while is_tuple(exc):
                # exc.children[1:-1] is the unparenthesized tuple
                # exc.children[1].children[0] is the first element of the tuple
                exc = exc.children[1].children[0].clone()
            exc.prefix = u" "

        if "tb" in results:
            tb = results["tb"].clone()
        else:
            tb = None

        if "val" in results:
            val = results["val"].clone()
            if is_tuple(val):
                # Assume that exc is a subclass of Exception and call exc(*val).
                args = [c.clone() for c in val.children[1:-1]]
                exc = Call(exc, args)
            elif val.type in (token.NUMBER, token.STRING):
                # Handle numeric and string literals specially, e.g.
                # "raise Exception, 5" -> "raise Exception(5)".
                val.prefix = u""
                exc = Call(exc, [val])
            elif val.type == token.NAME and val.value == u"None":
                # Handle None specially, e.g.
                # "raise Exception, None" -> "raise Exception".
                pass
            else:
                # val is some other expression. If val evaluates to an instance
                # of exc, it should just be raised. If val evaluates to None,
                # a default instance of exc should be raised (as above). If val
                # evaluates to a tuple, exc(*val) should be called (as
                # above). Otherwise, exc(val) should be called. We can only
                # tell what to do at runtime, so defer to future.utils.raise_(),
                # which handles all of these cases.
                touch_import_top(u"future.utils", u"raise_", node)
                exc.prefix = u""
                args = [exc, Comma(), val]
                if tb is not None:
                    args += [Comma(), tb]
                return Call(Name(u"raise_"), args)

        if tb is not None:
            tb.prefix = ""
            exc_list = Attr(exc, Name('with_traceback')) + [ArgList([tb])]
        else:
            exc_list = [exc]

        return pytree.Node(syms.raise_stmt,
                           [Name(u"raise")] + exc_list,
                           prefix=node.prefix)
