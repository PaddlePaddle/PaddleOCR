"""MediaList implements DOM Level 2 Style Sheets MediaList.

TODO:
    - delete: maybe if deleting from all, replace *all* with all others?
    - is unknown media an exception?
"""
__all__ = ['MediaList']

from cssutils.prodparser import PreDef, Prod, ProdParser, Sequence
from cssutils.helper import normalize, pushtoken
from .mediaquery import MediaQuery
import cssutils
import xml.dom


# class MediaList(cssutils.util.Base, cssutils.util.ListSeq):
class MediaList(cssutils.util._NewListBase):
    """Provides the abstraction of an ordered collection of media,
    without defining or constraining how this collection is
    implemented.

    A single media in the list is an instance of :class:`MediaQuery`.
    An empty list is the same as a list that contains the medium "all".

    New format with :class:`MediaQuery`::

        : S* [media_query [ ',' S* media_query ]* ]?


    """

    def __init__(self, mediaText=None, parentRule=None, readonly=False):
        """
        :param mediaText:
            Unicodestring of parsable comma separared media
            or a (Python) list of media.
        :param parentRule:
            CSSRule this medialist is used in, e.g. an @import or @media.
        :param readonly:
            Not used yet.
        """
        super(MediaList, self).__init__()
        self._wellformed = False

        if isinstance(mediaText, list):
            mediaText = ','.join(mediaText)

        self._parentRule = parentRule

        if mediaText:
            self.mediaText = mediaText

        self._readonly = readonly

    def __repr__(self):
        return "cssutils.stylesheets.%s(mediaText=%r)" % (
            self.__class__.__name__,
            self.mediaText,
        )

    def __str__(self):
        return "<cssutils.stylesheets.%s object mediaText=%r at 0x%x>" % (
            self.__class__.__name__,
            self.mediaText,
            id(self),
        )

    def __iter__(self):
        for item in self._seq:
            if item.type == 'MediaQuery':
                yield item

    length = property(
        lambda self: len(list(self)),
        doc="The number of media in the list (DOM readonly).",
    )

    def _getMediaText(self):
        return cssutils.ser.do_stylesheets_medialist(self)

    def _setMediaText(self, mediaText):  # noqa: C901
        """
        :param mediaText:
            simple value or comma-separated list of media

        :exceptions:
            - - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified string value has a syntax error and is
              unparsable.
            - - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this media list is readonly.
        """
        self._checkReadonly()

        mediaquery = lambda: Prod(  # noqa
            name='MediaQueryStart',
            match=lambda t, v: t == 'IDENT' or v == '(',
            toSeq=lambda t, tokens: (
                'MediaQuery',
                MediaQuery(pushtoken(t, tokens), _partof=True),
            ),
        )
        prods = Sequence(
            Sequence(PreDef.comment(parent=self), minmax=lambda: (0, None)),
            mediaquery(),
            Sequence(PreDef.comma(toSeq=False), mediaquery(), minmax=lambda: (0, None)),
        )
        # parse
        ok, seq, store, unused = ProdParser().parse(
            mediaText, 'MediaList', prods, debug="ml"
        )

        # each mq must be valid
        atleastone = False

        for item in seq:
            v = item.value
            if isinstance(v, MediaQuery):
                if not v.wellformed:
                    ok = False
                    break
                else:
                    atleastone = True

        # must be at least one value!
        if not atleastone:
            ok = False
            self._wellformed = ok
            self._log.error('MediaQuery: No content.', error=xml.dom.SyntaxErr)

        self._wellformed = ok

        if ok:
            mediaTypes = []
            finalseq = cssutils.util.Seq(readonly=False)
            commentseqonly = cssutils.util.Seq(readonly=False)
            for item in seq:
                # filter for doubles?
                if item.type == 'MediaQuery':
                    mediaType = item.value.mediaType
                    if mediaType:
                        if mediaType == 'all':
                            # remove anthing else and keep all+comments(!) only
                            finalseq = commentseqonly
                            finalseq.append(item)
                            break
                        elif mediaType in mediaTypes:
                            continue
                        else:
                            mediaTypes.append(mediaType)
                elif isinstance(item.value, cssutils.css.csscomment.CSSComment):
                    commentseqonly.append(item)

                finalseq.append(item)

            self._setSeq(finalseq)

    mediaText = property(
        _getMediaText,
        _setMediaText,
        doc="The parsable textual representation of the media list.",
    )

    def __prepareset(self, newMedium):
        # used by appendSelector and __setitem__
        self._checkReadonly()

        if not isinstance(newMedium, MediaQuery):
            newMedium = MediaQuery(newMedium)

        if newMedium.wellformed:
            return newMedium

    def __setitem__(self, index, newMedium):
        """Overwriting ListSeq.__setitem__

        Any duplicate items are **not yet** removed.
        """
        # TODO: remove duplicates?
        newMedium = self.__prepareset(newMedium)
        if newMedium:
            self._seq[index] = (newMedium, 'MediaQuery', None, None)

    def appendMedium(self, newMedium):
        """Add the `newMedium` to the end of the list.
        If the `newMedium` is already used, it is first removed.

        :param newMedium:
            a string or a :class:`~cssutils.stylesheets.MediaQuery`
        :returns: Wellformedness of `newMedium`.
        :exceptions:
            - :exc:`~xml.dom.InvalidCharacterErr`:
              If the medium contains characters that are invalid in the
              underlying style language.
            - :exc:`~xml.dom.InvalidModificationErr`:
              If mediaText is "all" and a new medium is tried to be added.
              Exception is "handheld" which is set in any case (Opera does handle
              "all, handheld" special, this special case might be removed in the
              future).
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this list is readonly.
        """
        newMedium = self.__prepareset(newMedium)

        if newMedium:
            mts = [normalize(item.value.mediaType) for item in self]
            newmt = normalize(newMedium.mediaType)

            self._seq._readonly = False

            if 'all' in mts:
                self._log.info(
                    'MediaList: Ignoring new medium %r as already specified '
                    '"all" (set ``mediaText`` instead).' % newMedium,
                    error=xml.dom.InvalidModificationErr,
                )

            elif newmt and newmt in mts:
                # might be empty
                self.deleteMedium(newmt)
                self._seq.append(newMedium, 'MediaQuery')

            else:
                if 'all' == newmt:
                    self._clearSeq()

                self._seq.append(newMedium, 'MediaQuery')

            self._seq._readonly = True

            return True

        else:
            return False

    def append(self, newMedium):
        "Same as :meth:`appendMedium`."
        self.appendMedium(newMedium)

    def deleteMedium(self, oldMedium):
        """Delete a medium from the list.

        :param oldMedium:
            delete this medium from the list.
        :exceptions:
            - :exc:`~xml.dom.NotFoundErr`:
              Raised if `oldMedium` is not in the list.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this list is readonly.
        """
        self._checkReadonly()
        oldMedium = normalize(oldMedium)

        for i, mq in enumerate(self):
            if normalize(mq.value.mediaType) == oldMedium:
                del self[i]
                break
        else:
            self._log.error(
                '"%s" not in this MediaList' % oldMedium, error=xml.dom.NotFoundErr
            )

    def item(self, index):
        """Return the mediaType of the `index`'th element in the list.
        If `index` is greater than or equal to the number of media in the
        list, returns ``None``.
        """
        try:
            return self[index].mediaType
        except IndexError:
            return None

    parentRule = property(
        lambda self: self._parentRule,
        doc="The CSSRule (e.g. an @media or @import rule "
        "this list is part of or None",
    )

    wellformed = property(lambda self: self._wellformed)
