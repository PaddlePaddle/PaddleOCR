"""CSSRule implements DOM Level 2 CSS CSSRule."""
__all__ = ['CSSRule']

import cssutils
import xml.dom


class CSSRule(cssutils.util.Base2):
    """Abstract base interface for any type of CSS statement. This includes
    both rule sets and at-rules. An implementation is expected to preserve
    all rules specified in a CSS style sheet, even if the rule is not
    recognized by the parser. Unrecognized rules are represented using the
    :class:`CSSUnknownRule` interface.
    """

    """
    CSSRule type constants.
    An integer indicating which type of rule this is.
    """
    UNKNOWN_RULE = 0
    ":class:`cssutils.css.CSSUnknownRule` (not used in CSSOM anymore)"
    STYLE_RULE = 1
    ":class:`cssutils.css.CSSStyleRule`"
    CHARSET_RULE = 2
    ":class:`cssutils.css.CSSCharsetRule` (not used in CSSOM anymore)"
    IMPORT_RULE = 3
    ":class:`cssutils.css.CSSImportRule`"
    MEDIA_RULE = 4
    ":class:`cssutils.css.CSSMediaRule`"
    FONT_FACE_RULE = 5
    ":class:`cssutils.css.CSSFontFaceRule`"
    PAGE_RULE = 6
    ":class:`cssutils.css.CSSPageRule`"
    NAMESPACE_RULE = 10
    """:class:`cssutils.css.CSSNamespaceRule`,
    Value has changed in 0.9.7a3 due to a change in the CSSOM spec."""
    COMMENT = 1001  # was -1, cssutils only
    """:class:`cssutils.css.CSSComment` - not in the offical spec,
    Value has changed in 0.9.7a3"""
    VARIABLES_RULE = 1008
    """:class:`cssutils.css.CSSVariablesRule` - experimental rule
    not in the offical spec"""

    MARGIN_RULE = 1006
    """:class:`cssutils.css.MarginRule` - experimental rule
    not in the offical spec"""

    _typestrings = {
        UNKNOWN_RULE: 'UNKNOWN_RULE',
        STYLE_RULE: 'STYLE_RULE',
        CHARSET_RULE: 'CHARSET_RULE',
        IMPORT_RULE: 'IMPORT_RULE',
        MEDIA_RULE: 'MEDIA_RULE',
        FONT_FACE_RULE: 'FONT_FACE_RULE',
        PAGE_RULE: 'PAGE_RULE',
        NAMESPACE_RULE: 'NAMESPACE_RULE',
        COMMENT: 'COMMENT',
        VARIABLES_RULE: 'VARIABLES_RULE',
        MARGIN_RULE: 'MARGIN_RULE',
    }

    def __init__(self, parentRule=None, parentStyleSheet=None, readonly=False):
        """Set common attributes for all rules."""
        super(CSSRule, self).__init__()
        self._parent = parentRule
        self._parentRule = parentRule
        self._parentStyleSheet = parentStyleSheet
        self._setSeq(self._tempSeq())
        # self._atkeyword = None
        # must be set after initialization of #inheriting rule is done
        self._readonly = False

    def _setAtkeyword(self, keyword):
        """Check if new keyword fits the rule it is used for."""
        atkeyword = self._normalize(keyword)
        if not self.atkeyword or (self.atkeyword == atkeyword):
            self._atkeyword = atkeyword
            self._keyword = keyword
        else:
            self._log.error(
                '%s: Invalid atkeyword for this rule: %r' % (self.atkeyword, keyword),
                error=xml.dom.InvalidModificationErr,
            )

    atkeyword = property(
        lambda self: self._atkeyword,
        _setAtkeyword,
        doc="Normalized  keyword of an @rule (e.g. ``@import``).",
    )

    def _setCssText(self, cssText):
        """
        :param cssText:
            A parsable DOMString.
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
        self._checkReadonly()

    cssText = property(
        lambda self: '',
        _setCssText,
        doc="(DOM) The parsable textual representation of the "
        "rule. This reflects the current state of the rule "
        "and not its initial value.",
    )

    parent = property(
        lambda self: self._parent, doc="The Parent Node of this CSSRule or None."
    )

    parentRule = property(
        lambda self: self._parentRule,
        doc="If this rule is contained inside another rule "
        "(e.g. a style rule inside an @media block), this "
        "is the containing rule. If this rule is not nested "
        "inside any other rules, this returns None.",
    )

    def _getParentStyleSheet(self):
        # rules contained in other rules (@media) use that rules parent
        if self.parentRule:
            return self.parentRule._parentStyleSheet
        else:
            return self._parentStyleSheet

    parentStyleSheet = property(
        _getParentStyleSheet, doc="The style sheet that contains this rule."
    )

    type = property(
        lambda self: self.UNKNOWN_RULE,
        doc="The type of this rule, as defined by a CSSRule " "type constant.",
    )

    typeString = property(
        lambda self: CSSRule._typestrings[self.type],
        doc="Descriptive name of this rule's type.",
    )

    wellformed = property(lambda self: False, doc="If the rule is wellformed.")


class CSSRuleRules(CSSRule):
    """Abstract base interface for rules that contain other rules
    like @media or @page. Methods may be overwritten if a rule
    has specific stuff to do like checking the order of insertion like
    @media does.
    """

    def __init__(self, parentRule=None, parentStyleSheet=None):

        super(CSSRuleRules, self).__init__(
            parentRule=parentRule, parentStyleSheet=parentStyleSheet
        )

        self.cssRules = cssutils.css.CSSRuleList()

    def __iter__(self):
        """Generator iterating over these rule's cssRules."""
        for rule in self._cssRules:
            yield rule

    def _setCssRules(self, cssRules):
        "Set new cssRules and update contained rules refs."
        cssRules.append = self.insertRule
        cssRules.extend = self.insertRule
        cssRules.__delitem__ == self.deleteRule

        for rule in cssRules:
            rule._parentRule = self
            rule._parentStyleSheet = None

        self._cssRules = cssRules

    cssRules = property(
        lambda self: self._cssRules,
        _setCssRules,
        "All Rules in this style sheet, a " ":class:`~cssutils.css.CSSRuleList`.",
    )

    def deleteRule(self, index):
        """
        Delete the rule at `index` from rules ``cssRules``.

        :param index:
            The `index` of the rule to be removed from the rules cssRules
            list. For an `index` < 0 **no** :exc:`~xml.dom.IndexSizeErr` is
            raised but rules for normal Python lists are used. E.g.
            ``deleteRule(-1)`` removes the last rule in cssRules.

            `index` may also be a CSSRule object which will then be removed.

        :Exceptions:
            - :exc:`~xml.dom.IndexSizeErr`:
              Raised if the specified index does not correspond to a rule in
              the media rule list.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this media rule is readonly.
        """
        self._checkReadonly()

        if isinstance(index, CSSRule):
            for i, r in enumerate(self.cssRules):
                if index == r:
                    index = i
                    break
            else:
                raise xml.dom.IndexSizeErr(
                    "%s: Not a rule in "
                    "this rule'a cssRules list: %s" % (self.__class__.__name__, index)
                )

        try:
            # detach
            self._cssRules[index]._parentRule = None
            del self._cssRules[index]

        except IndexError:
            raise xml.dom.IndexSizeErr(
                '%s: %s is not a valid index '
                'in the rulelist of length %i'
                % (self.__class__.__name__, index, self._cssRules.length)
            )

    def _prepareInsertRule(self, rule, index=None):
        "return checked `index` and optional parsed `rule`"
        self._checkReadonly()

        # check index
        if index is None:
            index = len(self._cssRules)

        elif index < 0 or index > self._cssRules.length:
            raise xml.dom.IndexSizeErr(
                '%s: Invalid index %s for '
                'CSSRuleList with a length of %s.'
                % (self.__class__.__name__, index, self._cssRules.length)
            )

        # check and optionally parse rule
        if isinstance(rule, str):
            tempsheet = cssutils.css.CSSStyleSheet()
            tempsheet.cssText = rule
            if len(tempsheet.cssRules) != 1 or (
                tempsheet.cssRules
                and not isinstance(tempsheet.cssRules[0], cssutils.css.CSSRule)
            ):
                self._log.error(
                    '%s: Invalid Rule: %s' % (self.__class__.__name__, rule)
                )
                return False, False
            rule = tempsheet.cssRules[0]

        elif isinstance(rule, cssutils.css.CSSRuleList):
            # insert all rules
            for i, r in enumerate(rule):
                self.insertRule(r, index + i)
            return True, True

        elif not isinstance(rule, cssutils.css.CSSRule):
            self._log.error('%s: Not a CSSRule: %s' % (rule, self.__class__.__name__))
            return False, False

        return rule, index

    def _finishInsertRule(self, rule, index):
        "add `rule` at `index`"
        rule._parentRule = self
        rule._parentStyleSheet = None
        self._cssRules.insert(index, rule)
        return index

    def add(self, rule):
        """Add `rule` to page rule. Same as ``insertRule(rule)``."""
        return self.insertRule(rule)

    def insertRule(self, rule, index=None):
        """
        Insert `rule` into the rules ``cssRules``.

        :param rule:
            the parsable text representing the `rule` to be inserted. For rule
            sets this contains both the selector and the style declaration.
            For at-rules, this specifies both the at-identifier and the rule
            content.

            cssutils also allows rule to be a valid
            :class:`~cssutils.css.CSSRule` object.

        :param index:
            before the `index` the specified `rule` will be inserted.
            If the specified `index` is equal to the length of the rules
            rule collection, the rule will be added to the end of the rule.
            If index is not given or None rule will be appended to rule
            list.

        :returns:
            the index of the newly inserted rule.

        :exceptions:
            - :exc:`~xml.dom.HierarchyRequestErr`:
              Raised if the `rule` cannot be inserted at the specified `index`,
              e.g., if an @import rule is inserted after a standard rule set
              or other at-rule.
            - :exc:`~xml.dom.IndexSizeErr`:
              Raised if the specified `index` is not a valid insertion point.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this rule is readonly.
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified `rule` has a syntax error and is
              unparsable.
        """
        return self._prepareInsertRule(rule, index)
