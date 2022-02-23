"""CSSPageRule implements DOM Level 2 CSS CSSPageRule."""
__all__ = ['CSSPageRule']

from itertools import chain
from .cssstyledeclaration import CSSStyleDeclaration
from .marginrule import MarginRule
from . import cssrule
import cssutils
import xml.dom


class CSSPageRule(cssrule.CSSRuleRules):
    """
    The CSSPageRule interface represents a @page rule within a CSS style
    sheet. The @page rule is used to specify the dimensions, orientation,
    margins, etc. of a page box for paged media.

    Format::

        page :
               PAGE_SYM S* IDENT? pseudo_page? S*
               '{' S* [ declaration | margin ]?
               [ ';' S* [ declaration | margin ]? ]* '}' S*
               ;

        pseudo_page :
               ':' [ "left" | "right" | "first" ]
               ;

        margin :
               margin_sym S* '{' declaration [ ';' S* declaration? ]* '}' S*
               ;

        margin_sym :
               TOPLEFTCORNER_SYM |
               TOPLEFT_SYM |
               TOPCENTER_SYM |
               TOPRIGHT_SYM |
               TOPRIGHTCORNER_SYM |
               BOTTOMLEFTCORNER_SYM |
               BOTTOMLEFT_SYM |
               BOTTOMCENTER_SYM |
               BOTTOMRIGHT_SYM |
               BOTTOMRIGHTCORNER_SYM |
               LEFTTOP_SYM |
               LEFTMIDDLE_SYM |
               LEFTBOTTOM_SYM |
               RIGHTTOP_SYM |
               RIGHTMIDDLE_SYM |
               RIGHTBOTTOM_SYM
               ;

    `cssRules` contains a list of `MarginRule` objects.
    """

    def __init__(
        self,
        selectorText=None,
        style=None,
        parentRule=None,
        parentStyleSheet=None,
        readonly=False,
    ):
        """
        If readonly allows setting of properties in constructor only.

        :param selectorText:
            type string
        :param style:
            CSSStyleDeclaration for this CSSStyleRule
        """
        super(CSSPageRule, self).__init__(
            parentRule=parentRule, parentStyleSheet=parentStyleSheet
        )
        self._atkeyword = '@page'
        self._specificity = (0, 0, 0)

        tempseq = self._tempSeq()

        if selectorText:
            self.selectorText = selectorText
            tempseq.append(self.selectorText, 'selectorText')
        else:
            self._selectorText = self._tempSeq()

        if style:
            self.style = style
        else:
            self.style = CSSStyleDeclaration()

        tempseq.append(self.style, 'style')

        self._setSeq(tempseq)
        self._readonly = readonly

    def __repr__(self):
        return "cssutils.css.%s(selectorText=%r, style=%r)" % (
            self.__class__.__name__,
            self.selectorText,
            self.style.cssText,
        )

    def __str__(self):
        return (
            "<cssutils.css.%s object selectorText=%r specificity=%r "
            + "style=%r cssRules=%r at 0x%x>"
        ) % (
            self.__class__.__name__,
            self.selectorText,
            self.specificity,
            self.style.cssText,
            len(self.cssRules),
            id(self),
        )

    def __contains__(self, margin):
        """Check if margin is set in the rule."""
        return margin in list(self.keys())

    def keys(self):
        "Return list of all set margins (MarginRule)."
        return list(r.margin for r in self.cssRules)

    def __getitem__(self, margin):
        """Retrieve the style (of MarginRule)
        for `margin` (which must be normalized).
        """
        for r in self.cssRules:
            if r.margin == margin:
                return r.style

    def __setitem__(self, margin, style):
        """Set the style (of MarginRule)
        for `margin` (which must be normalized).
        """
        for i, r in enumerate(self.cssRules):
            if r.margin == margin:
                r.style = style
                return i
        else:
            return self.add(MarginRule(margin, style))

    def __delitem__(self, margin):
        """Delete the style (the MarginRule)
        for `margin` (which must be normalized).
        """
        for r in self.cssRules:
            if r.margin == margin:
                self.deleteRule(r)

    def __parseSelectorText(self, selectorText):  # noqa: C901
        """
        Parse `selectorText` which may also be a list of tokens
        and returns (selectorText, seq).

        see _setSelectorText for details
        """
        # for closures: must be a mutable
        new = {'wellformed': True, 'last-S': False, 'name': 0, 'first': 0, 'lr': 0}

        def _char(expected, seq, token, tokenizer=None):
            # pseudo_page, :left, :right or :first
            val = self._tokenvalue(token)
            if not new['last-S'] and expected in ['page', ': or EOF'] and ':' == val:
                try:
                    identtoken = next(tokenizer)
                except StopIteration:
                    self._log.error('CSSPageRule selectorText: No IDENT found.', token)
                else:
                    ival, ityp = self._tokenvalue(identtoken), self._type(identtoken)
                    if self._prods.IDENT != ityp:
                        self._log.error(
                            'CSSPageRule selectorText: Expected '
                            'IDENT but found: %r' % ival,
                            token,
                        )
                    else:
                        if ival not in ('first', 'left', 'right'):
                            self._log.warn(
                                'CSSPageRule: Unknown @page '
                                'selector: %r' % (':' + ival,),
                                neverraise=True,
                            )
                        if ival == 'first':
                            new['first'] = 1
                        else:
                            new['lr'] = 1
                        seq.append(val + ival, 'pseudo')
                        return 'EOF'
                return expected
            else:
                new['wellformed'] = False
                self._log.error(
                    'CSSPageRule selectorText: Unexpected CHAR: %r' % val, token
                )
                return expected

        def S(expected, seq, token, tokenizer=None):
            "Does not raise if EOF is found."
            if expected == ': or EOF':
                # pseudo must directly follow IDENT if given
                new['last-S'] = True
            return expected

        def IDENT(expected, seq, token, tokenizer=None):
            """ """
            val = self._tokenvalue(token)
            if 'page' == expected:
                if self._normalize(val) == 'auto':
                    self._log.error(
                        'CSSPageRule selectorText: Invalid pagename.', token
                    )
                else:
                    new['name'] = 1
                    seq.append(val, 'IDENT')

                return ': or EOF'
            else:
                new['wellformed'] = False
                self._log.error(
                    'CSSPageRule selectorText: Unexpected IDENT: ' '%r' % val, token
                )
                return expected

        def COMMENT(expected, seq, token, tokenizer=None):
            "Does not raise if EOF is found."
            seq.append(cssutils.css.CSSComment([token]), 'COMMENT')
            return expected

        newseq = self._tempSeq()
        wellformed, expected = self._parse(
            expected='page',
            seq=newseq,
            tokenizer=self._tokenize2(selectorText),
            productions={'CHAR': _char, 'IDENT': IDENT, 'COMMENT': COMMENT, 'S': S},
            new=new,
        )
        wellformed = wellformed and new['wellformed']

        # post conditions
        if expected == 'ident':
            self._log.error(
                'CSSPageRule selectorText: No valid selector: %r'
                % self._valuestr(selectorText)
            )

        return wellformed, newseq, (new['name'], new['first'], new['lr'])

    def __parseMarginAndStyle(self, tokens):
        "tokens is a list, no generator (yet)"
        g = iter(tokens)
        styletokens = []

        # new rules until parse done
        cssRules = []

        for token in g:
            if (
                token[0] == 'ATKEYWORD'
                and self._normalize(token[1]) in MarginRule.margins
            ):

                # MarginRule
                m = MarginRule(parentRule=self, parentStyleSheet=self.parentStyleSheet)
                m.cssText = chain([token], g)

                # merge if margin set more than once
                for r in cssRules:
                    if r.margin == m.margin:
                        for p in m.style:
                            r.style.setProperty(p, replace=False)
                        break
                else:
                    cssRules.append(m)

                continue

            # TODO: Properties?
            styletokens.append(token)

        return cssRules, styletokens

    def _getCssText(self):
        """Return serialized property cssText."""
        return cssutils.ser.do_CSSPageRule(self)

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
        super(CSSPageRule, self)._setCssText(cssText)

        tokenizer = self._tokenize2(cssText)
        if self._type(self._nexttoken(tokenizer)) != self._prods.PAGE_SYM:
            self._log.error(
                'CSSPageRule: No CSSPageRule found: %s' % self._valuestr(cssText),
                error=xml.dom.InvalidModificationErr,
            )
        else:
            newStyle = CSSStyleDeclaration(parentRule=self)
            ok = True

            selectortokens, startbrace = self._tokensupto2(
                tokenizer, blockstartonly=True, separateEnd=True
            )
            styletokens, braceorEOFtoken = self._tokensupto2(
                tokenizer, blockendonly=True, separateEnd=True
            )
            nonetoken = self._nexttoken(tokenizer)
            if self._tokenvalue(startbrace) != '{':
                ok = False
                self._log.error(
                    'CSSPageRule: No start { of style declaration '
                    'found: %r' % self._valuestr(cssText),
                    startbrace,
                )
            elif nonetoken:
                ok = False
                self._log.error('CSSPageRule: Trailing content found.', token=nonetoken)

            selok, newselseq, specificity = self.__parseSelectorText(selectortokens)
            ok = ok and selok

            val, type_ = self._tokenvalue(braceorEOFtoken), self._type(braceorEOFtoken)

            if val != '}' and type_ != 'EOF':
                ok = False
                self._log.error(
                    'CSSPageRule: No "}" after style declaration found: %r'
                    % self._valuestr(cssText)
                )
            else:
                if 'EOF' == type_:
                    # add again as style needs it
                    styletokens.append(braceorEOFtoken)

                # filter pagemargin rules out first
                cssRules, styletokens = self.__parseMarginAndStyle(styletokens)

                # SET, may raise:
                newStyle.cssText = styletokens

            if ok:
                self._selectorText = newselseq
                self._specificity = specificity
                self.style = newStyle
                self.cssRules = cssutils.css.CSSRuleList()
                for r in cssRules:
                    self.cssRules.append(r)

    cssText = property(
        _getCssText,
        _setCssText,
        doc="(DOM) The parsable textual representation of this rule.",
    )

    def _getSelectorText(self):
        """Wrapper for cssutils Selector object."""
        return cssutils.ser.do_CSSPageRuleSelector(self._selectorText)

    def _setSelectorText(self, selectorText):
        """Wrapper for cssutils Selector object.

        :param selectorText:
            DOM String, in CSS 2.1 one of

            - :first
            - :left
            - :right
            - empty

        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error
              and is unparsable.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this rule is readonly.
        """
        self._checkReadonly()

        # may raise SYNTAX_ERR
        wellformed, newseq, specificity = self.__parseSelectorText(selectorText)
        if wellformed:
            self._selectorText = newseq
            self._specificity = specificity

    selectorText = property(
        _getSelectorText,
        _setSelectorText,
        doc="(DOM) The parsable textual representation of "
        "the page selector for the rule.",
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

    def insertRule(self, rule, index=None):
        """Implements base ``insertRule``."""
        rule, index = self._prepareInsertRule(rule, index)

        if rule is False or rule is True:
            # done or error
            return

        # check hierarchy
        if (
            isinstance(rule, cssutils.css.CSSCharsetRule)
            or isinstance(rule, cssutils.css.CSSFontFaceRule)
            or isinstance(rule, cssutils.css.CSSImportRule)
            or isinstance(rule, cssutils.css.CSSNamespaceRule)
            or isinstance(rule, CSSPageRule)
            or isinstance(rule, cssutils.css.CSSMediaRule)
        ):
            self._log.error(
                '%s: This type of rule is not allowed here: %s'
                % (self.__class__.__name__, rule.cssText),
                error=xml.dom.HierarchyRequestErr,
            )
            return

        return self._finishInsertRule(rule, index)

    specificity = property(
        lambda self: self._specificity,
        doc="""Specificity of this page rule (READONLY).
Tuple of (f, g, h) where:

 - if the page selector has a named page, f=1; else f=0
 - if the page selector has a ':first' pseudo-class, g=1; else g=0
 - if the page selector has a ':left' or ':right' pseudo-class, h=1; else h=0
""",
    )

    type = property(
        lambda self: self.PAGE_RULE,
        doc="The type of this rule, as defined by a CSSRule " "type constant.",
    )

    # constant but needed:
    wellformed = property(lambda self: True)
