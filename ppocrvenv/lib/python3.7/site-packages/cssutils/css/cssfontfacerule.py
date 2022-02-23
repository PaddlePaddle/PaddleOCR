"""CSSFontFaceRule implements DOM Level 2 CSS CSSFontFaceRule.

From cssutils 0.9.6 additions from CSS Fonts Module Level 3 are
added http://www.w3.org/TR/css3-fonts/.
"""
__all__ = ['CSSFontFaceRule']

from .cssstyledeclaration import CSSStyleDeclaration
from . import cssrule
import cssutils
import xml.dom


class CSSFontFaceRule(cssrule.CSSRule):
    """
    The CSSFontFaceRule interface represents a @font-face rule in a CSS
    style sheet. The @font-face rule is used to hold a set of font
    descriptions.

    Format::

        font_face
          : FONT_FACE_SYM S*
            '{' S* declaration [ ';' S* declaration ]* '}' S*
          ;

    cssutils uses a :class:`~cssutils.css.CSSStyleDeclaration`  to
    represent the font descriptions. For validation a specific profile
    is used though were some properties have other valid values than
    when used in e.g. a :class:`~cssutils.css.CSSStyleRule`.
    """

    def __init__(
        self, style=None, parentRule=None, parentStyleSheet=None, readonly=False
    ):
        """
        If readonly allows setting of properties in constructor only.

        :param style:
            CSSStyleDeclaration used to hold any font descriptions
            for this CSSFontFaceRule
        """
        super(CSSFontFaceRule, self).__init__(
            parentRule=parentRule, parentStyleSheet=parentStyleSheet
        )
        self._atkeyword = '@font-face'

        if style:
            self.style = style
        else:
            self.style = CSSStyleDeclaration()

        self._readonly = readonly

    def __repr__(self):
        return "cssutils.css.%s(style=%r)" % (
            self.__class__.__name__,
            self.style.cssText,
        )

    def __str__(self):
        return "<cssutils.css.%s object style=%r valid=%r at 0x%x>" % (
            self.__class__.__name__,
            self.style.cssText,
            self.valid,
            id(self),
        )

    def _getCssText(self):
        """Return serialized property cssText."""
        return cssutils.ser.do_CSSFontFaceRule(self)

    def _setCssText(self, cssText):
        """
        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error and
              is unparsable.
            - :exc:`~xml.dom.InvalidModificationErr`:
              Raised if the specified CSS string value represents a different
              type of rule than the current one.
            - :exc:`~xml.dom.HierarchyRequestErr`:
              Raised if the rule cannot be inserted at this point in the
              style sheet.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if the rule is readonly.
        """
        super(CSSFontFaceRule, self)._setCssText(cssText)

        tokenizer = self._tokenize2(cssText)
        attoken = self._nexttoken(tokenizer, None)
        if self._type(attoken) != self._prods.FONT_FACE_SYM:
            self._log.error(
                'CSSFontFaceRule: No CSSFontFaceRule found: %s'
                % self._valuestr(cssText),
                error=xml.dom.InvalidModificationErr,
            )
        else:
            newStyle = CSSStyleDeclaration(parentRule=self)
            ok = True

            beforetokens, brace = self._tokensupto2(
                tokenizer, blockstartonly=True, separateEnd=True
            )
            if self._tokenvalue(brace) != '{':
                ok = False
                self._log.error(
                    'CSSFontFaceRule: No start { of style '
                    'declaration found: %r' % self._valuestr(cssText),
                    brace,
                )

            # parse stuff before { which should be comments and S only
            new = {'wellformed': True}
            newseq = self._tempSeq()

            beforewellformed, expected = self._parse(
                expected=':',
                seq=newseq,
                tokenizer=self._tokenize2(beforetokens),
                productions={},
            )
            ok = ok and beforewellformed and new['wellformed']

            styletokens, braceorEOFtoken = self._tokensupto2(
                tokenizer, blockendonly=True, separateEnd=True
            )

            val, type_ = self._tokenvalue(braceorEOFtoken), self._type(braceorEOFtoken)
            if val != '}' and type_ != 'EOF':
                ok = False
                self._log.error(
                    'CSSFontFaceRule: No "}" after style '
                    'declaration found: %r' % self._valuestr(cssText)
                )

            nonetoken = self._nexttoken(tokenizer)
            if nonetoken:
                ok = False
                self._log.error(
                    'CSSFontFaceRule: Trailing content found.', token=nonetoken
                )

            if 'EOF' == type_:
                # add again as style needs it
                styletokens.append(braceorEOFtoken)

            # SET, may raise:
            newStyle.cssText = styletokens

            if ok:
                # contains probably comments only (upto ``{``)
                self._setSeq(newseq)
                self.style = newStyle

    cssText = property(
        _getCssText,
        _setCssText,
        doc="(DOM) The parsable textual representation of this " "rule.",
    )

    def _setStyle(self, style):
        """
        :param style:
            a CSSStyleDeclaration or string
        """
        self._checkReadonly()
        if isinstance(style, str):
            self._style = CSSStyleDeclaration(cssText=style, parentRule=self)
        else:
            style._parentRule = self
            self._style = style

    style = property(
        lambda self: self._style,
        _setStyle,
        doc="(DOM) The declaration-block of this rule set, "
        "a :class:`~cssutils.css.CSSStyleDeclaration`.",
    )

    type = property(
        lambda self: self.FONT_FACE_RULE,
        doc="The type of this rule, as defined by a CSSRule " "type constant.",
    )

    def _getValid(self):
        needed = ['font-family', 'src']
        for p in self.style.getProperties(all=True):
            if not p.valid:
                return False
            try:
                needed.remove(p.name)
            except ValueError:
                pass
        return not bool(needed)

    valid = property(
        _getValid,
        doc="CSSFontFace is valid if properties `font-family` "
        "and `src` are set and all properties are valid.",
    )

    # constant but needed:
    wellformed = property(lambda self: True)
