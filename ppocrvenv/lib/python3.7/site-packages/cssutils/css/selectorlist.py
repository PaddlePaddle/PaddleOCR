"""SelectorList is a list of CSS Selector objects.

TODO
    - remove duplicate Selectors. -> CSSOM canonicalize

    - ??? CSS2 gives a special meaning to the comma (,) in selectors.
        However, since it is not known if the comma may acquire other
        meanings in future versions of CSS, the whole statement should be
        ignored if there is an error anywhere in the selector, even though
        the rest of the selector may look reasonable in CSS2.

        Illegal example(s):

        For example, since the "&" is not a valid token in a CSS2 selector,
        a CSS2 user agent must ignore the whole second line, and not set
        the color of H3 to red:
"""
__all__ = ['SelectorList']

from .selector import Selector
import cssutils


class SelectorList(cssutils.util.Base, cssutils.util.ListSeq):
    """A list of :class:`~cssutils.css.Selector` objects
    of a :class:`~cssutils.css.CSSStyleRule`."""

    def __init__(self, selectorText=None, parentRule=None, readonly=False):
        """
        :Parameters:
            selectorText
                parsable list of Selectors
            parentRule
                the parent CSSRule if available
        """
        super(SelectorList, self).__init__()

        self._parentRule = parentRule

        if selectorText:
            self.selectorText = selectorText

        self._readonly = readonly

    def __repr__(self):
        if self._namespaces:
            st = (self.selectorText, self._namespaces)
        else:
            st = self.selectorText
        return "cssutils.css.%s(selectorText=%r)" % (self.__class__.__name__, st)

    def __str__(self):
        return "<cssutils.css.%s object selectorText=%r _namespaces=%r at " "0x%x>" % (
            self.__class__.__name__,
            self.selectorText,
            self._namespaces,
            id(self),
        )

    def __setitem__(self, index, newSelector):
        """Overwrite ListSeq.__setitem__

        Any duplicate Selectors are **not** removed.
        """
        newSelector = self.__prepareset(newSelector)
        if newSelector:
            self.seq[index] = newSelector

    def __prepareset(self, newSelector, namespaces=None):
        "Used by appendSelector and __setitem__"
        if not namespaces:
            namespaces = {}
        self._checkReadonly()
        if not isinstance(newSelector, Selector):
            newSelector = Selector((newSelector, namespaces), parent=self)
        if newSelector.wellformed:
            newSelector._parent = self  # maybe set twice but must be!
            return newSelector

    def __getNamespaces(self):
        """Use children namespaces if not attached to a sheet, else the sheet's
        ones.
        """
        try:
            return self.parentRule.parentStyleSheet.namespaces
        except AttributeError:
            namespaces = {}
            for selector in self.seq:
                namespaces.update(selector._namespaces)
            return namespaces

    def _getUsedUris(self):
        "Used by CSSStyleSheet to check if @namespace rules are needed"
        uris = set()
        for s in self:
            uris.update(s._getUsedUris())
        return uris

    _namespaces = property(
        __getNamespaces,
        doc="""If this SelectorList is
        attached to a CSSStyleSheet the namespaces of that sheet are mirrored
        here. While the SelectorList (or parentRule(s) are
        not attached the namespaces of all children Selectors are used.""",
    )

    def append(self, newSelector):
        "Same as :meth:`appendSelector`."
        self.appendSelector(newSelector)

    def appendSelector(self, newSelector):
        """
        Append `newSelector` to this list (a string will be converted to a
        :class:`~cssutils.css.Selector`).

        :param newSelector:
            comma-separated list of selectors (as a single string) or a tuple of
            `(newSelector, dict-of-namespaces)`
        :returns: New :class:`~cssutils.css.Selector` or ``None`` if
            `newSelector` is not wellformed.
        :exceptions:
            - :exc:`~xml.dom.NamespaceErr`:
              Raised if the specified selector uses an unknown namespace
              prefix.
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error
              and is unparsable.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this rule is readonly.
        """
        self._checkReadonly()

        # might be (selectorText, namespaces)
        newSelector, namespaces = self._splitNamespacesOff(newSelector)
        try:
            # use parent's only if available
            namespaces = self.parentRule.parentStyleSheet.namespaces
        except AttributeError:
            # use already present namespaces plus new given ones
            _namespaces = self._namespaces
            _namespaces.update(namespaces)
            namespaces = _namespaces

        newSelector = self.__prepareset(newSelector, namespaces)
        if newSelector:
            seq = self.seq[:]
            del self.seq[:]
            for s in seq:
                if s.selectorText != newSelector.selectorText:
                    self.seq.append(s)
            self.seq.append(newSelector)
            return newSelector

    def _getSelectorText(self):
        "Return serialized format."
        return cssutils.ser.do_css_SelectorList(self)

    def _setSelectorText(self, selectorText):
        """
        :param selectorText:
            comma-separated list of selectors or a tuple of
            (selectorText, dict-of-namespaces)
        :exceptions:
            - :exc:`~xml.dom.NamespaceErr`:
              Raised if the specified selector uses an unknown namespace
              prefix.
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error
              and is unparsable.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this rule is readonly.
        """
        self._checkReadonly()

        # might be (selectorText, namespaces)
        selectorText, namespaces = self._splitNamespacesOff(selectorText)
        try:
            # use parent's only if available
            namespaces = self.parentRule.parentStyleSheet.namespaces
        except AttributeError:
            pass

        wellformed = True
        tokenizer = self._tokenize2(selectorText)
        newseq = []

        expected = True
        while True:
            # find all upto and including next ",", EOF or nothing
            selectortokens = self._tokensupto2(tokenizer, listseponly=True)
            if selectortokens:
                if self._tokenvalue(selectortokens[-1]) == ',':
                    expected = selectortokens.pop()
                else:
                    expected = None

                selector = Selector((selectortokens, namespaces), parent=self)
                if selector.wellformed:
                    newseq.append(selector)
                else:
                    wellformed = False
                    self._log.error(
                        'SelectorList: Invalid Selector: %s'
                        % self._valuestr(selectortokens)
                    )
            else:
                break

        # post condition
        if ',' == expected:
            wellformed = False
            self._log.error(
                'SelectorList: Cannot end with ",": %r' % self._valuestr(selectorText)
            )
        elif expected:
            wellformed = False
            self._log.error(
                'SelectorList: Unknown Syntax: %r' % self._valuestr(selectorText)
            )
        if wellformed:
            self.seq = newseq

    selectorText = property(
        _getSelectorText,
        _setSelectorText,
        doc="(cssutils) The textual representation of the " "selector for a rule set.",
    )

    length = property(
        lambda self: len(self),
        doc="The number of :class:`~cssutils.css.Selector` " "objects in the list.",
    )

    parentRule = property(
        lambda self: self._parentRule,
        doc="(DOM) The CSS rule that contains this "
        "SelectorList or ``None`` if this SelectorList "
        "is not attached to a CSSRule.",
    )

    wellformed = property(lambda self: bool(len(self.seq)))
