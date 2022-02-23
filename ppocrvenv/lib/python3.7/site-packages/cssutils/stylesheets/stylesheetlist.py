"""StyleSheetList implements DOM Level 2 Style Sheets StyleSheetList."""
__all__ = ['StyleSheetList']


class StyleSheetList(list):
    """Interface `StyleSheetList` (introduced in DOM Level 2)

    The `StyleSheetList` interface provides the abstraction of an ordered
    collection of :class:`~cssutils.stylesheets.StyleSheet` objects.

    The items in the `StyleSheetList` are accessible via an integral index,
    starting from 0.

    This Python implementation is based on a standard Python list so e.g.
    allows ``examplelist[index]`` usage.
    """

    def item(self, index):
        """
        Used to retrieve a style sheet by ordinal `index`. If `index` is
        greater than or equal to the number of style sheets in the list,
        this returns ``None``.
        """
        try:
            return self[index]
        except IndexError:
            return None

    length = property(
        lambda self: len(self),
        doc="The number of :class:`StyleSheet` objects in the list. The range"
        "  of valid child stylesheet indices is 0 to length-1 inclusive.",
    )
