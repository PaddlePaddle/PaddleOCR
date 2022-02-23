"""Implements a DOM for MediaQuery, see
http://www.w3.org/TR/css3-mediaqueries/.

A cssutils implementation, not defined in official DOM.
"""
__all__ = ['MediaQuery']

from cssutils.prodparser import Choice, PreDef, Prod, ProdParser, Sequence
from cssutils.helper import normalize
import cssutils
import xml.dom


class MediaQuery(cssutils.util._NewBase):  # cssutils.util.Base):
    """
    A Media Query consists of one of :const:`MediaQuery.MEDIA_TYPES`
    and one or more expressions involving media features.

    Format::

        media_query
         : [ONLY | NOT]? S* media_type S* [ AND S* expression ]*
         | expression [ AND S* expression ]*
         ;
        media_type
         : IDENT
         ;
        expression
         : '(' S* media_feature S* [ ':' S* expr ]? ')' S*
         ;
        media_feature
         : IDENT
         ;

    """

    MEDIA_TYPES = [
        'all',
        'braille',
        'handheld',
        'print',
        'projection',
        'speech',
        'screen',
        'tty',
        'tv',
        'embossed',
    ]

    def __init__(self, mediaText=None, readonly=False, _partof=False):
        """
        :param mediaText:
            unicodestring of parsable media

        # _standalone: True if new from ML parser
        """
        super(MediaQuery, self).__init__()

        self._wellformed = False
        self._mediaType = ''
        self._partof = _partof
        if mediaText:
            self.mediaText = mediaText  # sets self._mediaType too
            self._partof = False

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

    def _getMediaText(self):
        return cssutils.ser.do_stylesheets_mediaquery(self)

    def _setMediaText(self, mediaText):
        """
        :param mediaText:
            a single media query string, e.g. ``print and (min-width: 25cm)``

        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified string value has a syntax error and is
              unparsable.
            - :exc:`~xml.dom.InvalidCharacterErr`:
              Raised if the given mediaType is unknown.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this media query is readonly.

        media_query
         : [ONLY | NOT]? S* media_type S* [ AND S* expression ]*
         | expression [ AND S* expression ]*
         ;
        media_type
         : IDENT
         ;
        expression
         : '(' S* media_feature S* [ ':' S* expr ]? ')' S*
         ;
        media_feature
         : IDENT
         ;

        """
        self._checkReadonly()

        expression = lambda: Sequence(  # noqa
            PreDef.char(name='expression', char='('),
            Prod(name='media_feature', match=lambda t, v: t == PreDef.types.IDENT),
            Sequence(
                PreDef.char(name='colon', char=':'),
                cssutils.css.value.MediaQueryValueProd(self),
                minmax=lambda: (0, 1),  # optional
            ),
            PreDef.char(
                name='expression END', char=')', stopIfNoMoreMatch=self._partof
            ),
        )

        prods = Choice(
            Sequence(
                Prod(
                    name='ONLY|NOT',  # media_query
                    match=lambda t, v: t == PreDef.types.IDENT
                    and normalize(v) in ('only', 'not'),
                    optional=True,
                    toStore='not simple',
                ),
                Prod(
                    name='media_type',
                    match=lambda t, v: t == PreDef.types.IDENT
                    and normalize(v) in self.MEDIA_TYPES,
                    stopIfNoMoreMatch=True,
                    toStore='media_type',
                ),
                Sequence(
                    Prod(
                        name='AND',
                        match=lambda t, v: t == PreDef.types.IDENT
                        and normalize(v) == 'and',
                        toStore='not simple',
                    ),
                    expression(),
                    minmax=lambda: (0, None),
                ),
            ),
            Sequence(
                expression(),
                Sequence(
                    Prod(
                        name='AND',
                        match=lambda t, v: t == PreDef.types.IDENT
                        and normalize(v) == 'and',
                    ),
                    expression(),
                    minmax=lambda: (0, None),
                ),
            ),
        )

        # parse
        ok, seq, store, unused = ProdParser().parse(mediaText, 'MediaQuery', prods)
        self._wellformed = ok
        if ok:
            try:
                media_type = store['media_type']
            except KeyError:
                pass
            else:
                if 'not simple' not in store:
                    self.mediaType = media_type.value

            # TODO: filter doubles!
            self._setSeq(seq)

    mediaText = property(
        _getMediaText,
        _setMediaText,
        doc="The parsable textual representation of the media list.",
    )

    def _setMediaType(self, mediaType):
        """
        :param mediaType:
            one of :attr:`MEDIA_TYPES`

        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified string value has a syntax error and is
              unparsable.
            - :exc:`~xml.dom.InvalidCharacterErr`:
              Raised if the given mediaType is unknown.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this media query is readonly.
        """
        self._checkReadonly()
        nmediaType = normalize(mediaType)

        if nmediaType not in self.MEDIA_TYPES:
            self._log.error(
                'MediaQuery: Syntax Error in media type "%s".' % mediaType,
                error=xml.dom.SyntaxErr,
            )
        else:
            # set
            self._mediaType = mediaType

            # update seq
            for i, x in enumerate(self._seq):
                if isinstance(x.value, str):
                    if normalize(x.value) in ('only', 'not'):
                        continue
                    else:
                        # TODO: simplify!
                        self._seq[i] = (mediaType, 'IDENT', None, None)
                        break
            else:
                self._seq.insert(0, mediaType, 'IDENT')

    mediaType = property(
        lambda self: self._mediaType,
        _setMediaType,
        doc="The media type of this MediaQuery (one of "
        ":attr:`MEDIA_TYPES`) but only if it is a simple MediaType!",
    )

    wellformed = property(lambda self: self._wellformed)
