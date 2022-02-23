"""CSSRuleList implements DOM Level 2 CSS CSSRuleList.
Partly also http://dev.w3.org/csswg/cssom/#the-cssrulelist."""
__all__ = ['CSSRuleList']


class CSSRuleList(list):
    """The CSSRuleList object represents an (ordered) list of statements.

    The items in the CSSRuleList are accessible via an integral index,
    starting from 0.

    Subclasses a standard Python list so theoretically all standard list
    methods are available. Setting methods like ``__init__``, ``append``,
    ``extend`` or ``__setslice__`` are added later on instances of this
    class if so desired.
    E.g. CSSStyleSheet adds ``append`` which is not available in a simple
    instance of this class!
    """

    def __init__(self, *ignored):
        "Nothing is set as this must also be defined later."
        pass

    def __notimplemented(self, *ignored):
        "Implemented in class using a CSSRuleList only."
        raise NotImplementedError(
            'Must be implemented by class using an instance of this class.'
        )

    append = extend = __setitem__ = __setslice__ = __notimplemented

    def item(self, index):
        """(DOM) Retrieve a CSS rule by ordinal `index`. The order in this
        collection represents the order of the rules in the CSS style
        sheet. If index is greater than or equal to the number of rules in
        the list, this returns None.

        Returns CSSRule, the style rule at the index position in the
        CSSRuleList, or None if that is not a valid index.
        """
        try:
            return self[index]
        except IndexError:
            return None

    length = property(
        lambda self: len(self), doc="(DOM) The number of CSSRules in the list."
    )

    def rulesOfType(self, type):
        """Yield the rules which have the given `type` only, one of the
        constants defined in :class:`cssutils.css.CSSRule`."""
        for r in self:
            if r.type == type:
                yield r
