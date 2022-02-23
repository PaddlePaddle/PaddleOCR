u"""
Fixer to remove function annotations
"""

from lib2to3 import fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import syms

warning_text = u"Removing function annotations completely."

def param_without_annotations(node):
    return node.children[0]

class FixAnnotations(fixer_base.BaseFix):

    warned = False

    def warn_once(self, node, reason):
        if not self.warned:
            self.warned = True
            self.warning(node, reason=reason)

    PATTERN = u"""
              funcdef< 'def' any parameters< '(' [params=any] ')' > ['->' ret=any] ':' any* >
              """

    def transform(self, node, results):
        u"""
        This just strips annotations from the funcdef completely.
        """
        params = results.get(u"params")
        ret = results.get(u"ret")
        if ret is not None:
            assert ret.prev_sibling.type == token.RARROW, u"Invalid return annotation"
            self.warn_once(node, reason=warning_text)
            ret.prev_sibling.remove()
            ret.remove()
        if params is None: return
        if params.type == syms.typedargslist:
            # more than one param in a typedargslist
            for param in params.children:
                if param.type == syms.tname:
                    self.warn_once(node, reason=warning_text)
                    param.replace(param_without_annotations(param))
        elif params.type == syms.tname:
            # one param
            self.warn_once(node, reason=warning_text)
            params.replace(param_without_annotations(params))
