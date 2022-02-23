"""CSSComment is not defined in DOM Level 2 at all but a cssutils defined
class only.

Implements CSSRule which is also extended for a CSSComment rule type.
"""
__all__ = ['CSSComment']

from . import cssrule
import cssutils
import xml.dom


class CSSComment(cssrule.CSSRule):
    """
    Represents a CSS comment (cssutils only).

    Format::

        /*...*/
    """

    def __init__(
        self, cssText=None, parentRule=None, parentStyleSheet=None, readonly=False
    ):
        super(CSSComment, self).__init__(
            parentRule=parentRule, parentStyleSheet=parentStyleSheet
        )

        self._cssText = None
        if cssText:
            self._setCssText(cssText)

        self._readonly = readonly

    def __repr__(self):
        return "cssutils.css.%s(cssText=%r)" % (self.__class__.__name__, self.cssText)

    def __str__(self):
        return "<cssutils.css.%s object cssText=%r at 0x%x>" % (
            self.__class__.__name__,
            self.cssText,
            id(self),
        )

    def _getCssText(self):
        """Return serialized property cssText."""
        return cssutils.ser.do_CSSComment(self)

    def _setCssText(self, cssText):
        """
        :param cssText:
            textual text to set or tokenlist which is not tokenized
            anymore. May also be a single token for this rule

        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error and
              is unparsable.
            - :exc:`~xml.dom.InvalidModificationErr`:
              Raised if the specified CSS string value represents a different
              type of rule than the current one.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if the rule is readonly.
        """
        super(CSSComment, self)._setCssText(cssText)
        tokenizer = self._tokenize2(cssText)

        commenttoken = self._nexttoken(tokenizer)
        unexpected = self._nexttoken(tokenizer)

        if (
            not commenttoken
            or self._type(commenttoken) != self._prods.COMMENT
            or unexpected
        ):
            self._log.error(
                'CSSComment: Not a CSSComment: %r' % self._valuestr(cssText),
                error=xml.dom.InvalidModificationErr,
            )
        else:
            self._cssText = self._tokenvalue(commenttoken)

    cssText = property(
        _getCssText,
        _setCssText,
        doc="The parsable textual representation of this rule.",
    )

    type = property(
        lambda self: self.COMMENT,
        doc="The type of this rule, as defined by a CSSRule " "type constant.",
    )

    # constant but needed:
    wellformed = property(lambda self: True)
