"""StyleSheet implements DOM Level 2 Style Sheets StyleSheet."""
__all__ = ['StyleSheet']

import cssutils


class StyleSheet(cssutils.util.Base2):
    """
    The StyleSheet interface is the abstract base interface
    for any type of style sheet. It represents a single style
    sheet associated with a structured document.

    In HTML, the StyleSheet interface represents either an
    external style sheet, included via the HTML LINK element,
    or an inline STYLE element (also an @import stylesheet?).

    In XML, this interface represents
    an external style sheet, included via a style sheet
    processing instruction.
    """

    def __init__(
        self,
        type='text/css',
        href=None,
        media=None,
        title='',
        ownerNode=None,
        parentStyleSheet=None,
        alternate=False,
        disabled=None,
        validating=True,
    ):
        """
        type
            readonly
        href: readonly
            If the style sheet is a linked style sheet, the value
            of this attribute is its location. For inline style
            sheets, the value of this attribute is None. See the
            href attribute definition for the LINK element in HTML
            4.0, and the href pseudo-attribute for the XML style
            sheet processing instruction.
        media: of type MediaList, readonly
            The intended destination media for style information.
            The media is often specified in the ownerNode. If no
            media has been specified, the MediaList will be empty.
            See the media attribute definition for the LINK element
            in HTML 4.0, and the media pseudo-attribute for the XML
            style sheet processing instruction. Modifying the media
            list may cause a change to the attribute disabled.
        title: readonly
            The advisory title. The title is often specified in
            the ownerNode. See the title attribute definition for
            the LINK element in HTML 4.0, and the title
            pseudo-attribute for the XML style sheet processing
            instruction.
        disabled: False if the style sheet is applied to the
            document. True if it is not. Modifying this attribute
            may cause a new resolution of style for the document.
            A stylesheet only applies if both an appropriate medium
            definition is present and the disabled attribute is False.
            So, if the media doesn't apply to the current user agent,
            the disabled attribute is ignored.
        ownerNode: of type Node, readonly
            The node that associates this style sheet with the
            document. For HTML, this may be the corresponding LINK
            or STYLE element. For XML, it may be the linking
            processing instruction. For style sheets that are
            included by other style sheets, the value of this
            attribute is None.
        parentStyleSheet: of type StyleSheet, readonly
            a StyleSheet or None
        alternate = False
            a flag stating if a style sheet is an alternate one or not.
            Currently not used in cssutils
        validating = True
            a flag defining if this sheet should be validate on change.

        """
        super(StyleSheet, self).__init__()

        self.validating = validating

        self._alternate = alternate
        self._href = href
        self._ownerNode = ownerNode
        self._parentStyleSheet = parentStyleSheet
        self._type = type

        self.disabled = bool(disabled)
        self.media = media
        self.title = title

    alternate = property(lambda self: self._alternate, doc="Not used in cssutils yet.")

    href = property(
        lambda self: self._href,
        doc="If the style sheet is a linked style sheet, the value "
        "of this attribute is its location. For inline style "
        "sheets, the value of this attribute is None. See the "
        "href attribute definition for the LINK element in HTML "
        "4.0, and the href pseudo-attribute for the XML style "
        "sheet processing instruction.",
    )

    ownerNode = property(lambda self: self._ownerNode, doc="Not used in cssutils yet.")

    parentStyleSheet = property(
        lambda self: self._parentStyleSheet,
        doc="For style sheet languages that support the concept "
        "of style sheet inclusion, this attribute represents "
        "the including style sheet, if one exists. If the style "
        "sheet is a top-level style sheet, or the style sheet "
        "language does not support inclusion, the value of this "
        "attribute is None.",
    )

    type = property(
        lambda self: self._type,
        doc="This specifies the style sheet language for this "
        "style sheet. The style sheet language is specified "
        "as a content type (e.g. ``text/css``). The content "
        "type is often specified in the ownerNode. Also see "
        "the type attribute definition for the LINK element "
        "in HTML 4.0, and the type pseudo-attribute for the "
        "XML style sheet processing instruction. "
        "For CSS this is always ``text/css``.",
    )
