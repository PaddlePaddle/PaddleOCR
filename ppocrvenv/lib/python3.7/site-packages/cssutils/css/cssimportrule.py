"""CSSImportRule implements DOM Level 2 CSS CSSImportRule plus the
``name`` property from http://www.w3.org/TR/css3-cascade/#cascading."""
__all__ = ['CSSImportRule']

from . import cssrule
import cssutils
import os
import urllib.parse
import xml.dom


class CSSImportRule(cssrule.CSSRule):
    """
    Represents an @import rule within a CSS style sheet.  The @import rule
    is used to import style rules from other style sheets.

    Format::

        import
          : IMPORT_SYM S*
          [STRING|URI] S* [ medium [ COMMA S* medium]* ]? S* STRING? S* ';' S*
          ;
    """

    def __init__(
        self,
        href=None,
        mediaText=None,
        name=None,
        parentRule=None,
        parentStyleSheet=None,
        readonly=False,
    ):
        """
        If readonly allows setting of properties in constructor only

        :param href:
            location of the style sheet to be imported.
        :param mediaText:
            A list of media types for which this style sheet may be used
            as a string
        :param name:
            Additional name of imported style sheet
        """
        super(CSSImportRule, self).__init__(
            parentRule=parentRule, parentStyleSheet=parentStyleSheet
        )
        self._atkeyword = '@import'
        self._styleSheet = None

        # string or uri used for reserialization
        self.hreftype = None

        # prepare seq
        seq = self._tempSeq()
        seq.append(None, 'href')
        # seq.append(None, 'media')
        seq.append(None, 'name')
        self._setSeq(seq)

        # 1. media
        if mediaText:
            self.media = mediaText
        else:
            # must be all for @import
            self.media = cssutils.stylesheets.MediaList(mediaText='all')
        # 2. name
        self.name = name
        # 3. href and styleSheet
        self.href = href

        self._readonly = readonly

    def __repr__(self):
        if self._usemedia:
            mediaText = self.media.mediaText
        else:
            mediaText = None
        return "cssutils.css.%s(href=%r, mediaText=%r, name=%r)" % (
            self.__class__.__name__,
            self.href,
            mediaText,
            self.name,
        )

    def __str__(self):
        if self._usemedia:
            mediaText = self.media.mediaText
        else:
            mediaText = None
        return "<cssutils.css.%s object href=%r mediaText=%r name=%r at 0x%x>" % (
            self.__class__.__name__,
            self.href,
            mediaText,
            self.name,
            id(self),
        )

    _usemedia = property(
        lambda self: self.media.mediaText not in ('', 'all'),
        doc="if self.media is used (or simply empty)",
    )

    def _getCssText(self):
        """Return serialized property cssText."""
        return cssutils.ser.do_CSSImportRule(self)

    def _setCssText(self, cssText):  # noqa: C901
        """
        :exceptions:
            - :exc:`~xml.dom.HierarchyRequestErr`:
              Raised if the rule cannot be inserted at this point in the
              style sheet.
            - :exc:`~xml.dom.InvalidModificationErr`:
              Raised if the specified CSS string value represents a different
              type of rule than the current one.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if the rule is readonly.
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error and
              is unparsable.
        """
        super(CSSImportRule, self)._setCssText(cssText)
        tokenizer = self._tokenize2(cssText)
        attoken = self._nexttoken(tokenizer, None)
        if self._type(attoken) != self._prods.IMPORT_SYM:
            self._log.error(
                'CSSImportRule: No CSSImportRule found: %s' % self._valuestr(cssText),
                error=xml.dom.InvalidModificationErr,
            )
        else:
            # for closures: must be a mutable
            new = {
                'keyword': self._tokenvalue(attoken),
                'href': None,
                'hreftype': None,
                'media': None,
                'name': None,
                'wellformed': True,
            }

            def __doname(seq, token):
                # called by _string or _ident
                new['name'] = self._stringtokenvalue(token)
                seq.append(new['name'], 'name')
                return ';'

            def _string(expected, seq, token, tokenizer=None):
                if 'href' == expected:
                    # href
                    new['href'] = self._stringtokenvalue(token)
                    new['hreftype'] = 'string'
                    seq.append(new['href'], 'href')
                    return 'media name ;'
                elif 'name' in expected:
                    # name
                    return __doname(seq, token)
                else:
                    new['wellformed'] = False
                    self._log.error('CSSImportRule: Unexpected string.', token)
                    return expected

            def _uri(expected, seq, token, tokenizer=None):
                # href
                if 'href' == expected:
                    uri = self._uritokenvalue(token)
                    new['hreftype'] = 'uri'
                    new['href'] = uri
                    seq.append(new['href'], 'href')
                    return 'media name ;'
                else:
                    new['wellformed'] = False
                    self._log.error('CSSImportRule: Unexpected URI.', token)
                    return expected

            def _ident(expected, seq, token, tokenizer=None):
                # medialist ending with ; which is checked upon too
                if expected.startswith('media'):
                    mediatokens = self._tokensupto2(
                        tokenizer, importmediaqueryendonly=True
                    )
                    mediatokens.insert(0, token)  # push found token

                    last = mediatokens.pop()  # retrieve ;
                    lastval, lasttyp = self._tokenvalue(last), self._type(last)
                    if lastval != ';' and lasttyp not in ('EOF', self._prods.STRING):
                        new['wellformed'] = False
                        self._log.error(
                            'CSSImportRule: No ";" found: %s' % self._valuestr(cssText),
                            token=token,
                        )

                    newMedia = cssutils.stylesheets.MediaList(parentRule=self)
                    newMedia.mediaText = mediatokens
                    if newMedia.wellformed:
                        new['media'] = newMedia
                        seq.append(newMedia, 'media')
                    else:
                        new['wellformed'] = False
                        self._log.error(
                            'CSSImportRule: Invalid MediaList: %s'
                            % self._valuestr(cssText),
                            token=token,
                        )

                    if lasttyp == self._prods.STRING:
                        # name
                        return __doname(seq, last)
                    else:
                        return 'EOF'  # ';' is token "last"
                else:
                    new['wellformed'] = False
                    self._log.error('CSSImportRule: Unexpected ident.', token)
                    return expected

            def _char(expected, seq, token, tokenizer=None):
                # final ;
                val = self._tokenvalue(token)
                if expected.endswith(';') and ';' == val:
                    return 'EOF'
                else:
                    new['wellformed'] = False
                    self._log.error('CSSImportRule: Unexpected char.', token)
                    return expected

            # import : IMPORT_SYM S* [STRING|URI]
            #            S* [ medium [ ',' S* medium]* ]? ';' S*
            #         STRING? # see http://www.w3.org/TR/css3-cascade/#cascading
            #        ;
            newseq = self._tempSeq()
            wellformed, expected = self._parse(
                expected='href',
                seq=newseq,
                tokenizer=tokenizer,
                productions={
                    'STRING': _string,
                    'URI': _uri,
                    'IDENT': _ident,
                    'CHAR': _char,
                },
                new=new,
            )

            # wellformed set by parse
            ok = wellformed and new['wellformed']

            # post conditions
            if not new['href']:
                ok = False
                self._log.error(
                    'CSSImportRule: No href found: %s' % self._valuestr(cssText)
                )

            if expected != 'EOF':
                ok = False
                self._log.error(
                    'CSSImportRule: No ";" found: %s' % self._valuestr(cssText)
                )

            # set all
            if ok:
                self._setSeq(newseq)

                self.atkeyword = new['keyword']
                self.hreftype = new['hreftype']
                self.name = new['name']

                if new['media']:
                    self.media = new['media']
                else:
                    # must be all for @import
                    self.media = cssutils.stylesheets.MediaList(mediaText='all')

                # needs new self.media
                self.href = new['href']

    cssText = property(
        fget=_getCssText,
        fset=_setCssText,
        doc="(DOM) The parsable textual representation of this rule.",
    )

    def _setHref(self, href):
        # set new href
        self._href = href
        # update seq
        for i, item in enumerate(self.seq):
            type_ = item.type
            if 'href' == type_:
                self._seq[i] = (href, type_, item.line, item.col)
                break

        importedSheet = cssutils.css.CSSStyleSheet(
            media=self.media, ownerRule=self, title=self.name
        )
        self.hrefFound = False
        # set styleSheet
        if href and self.parentStyleSheet:
            # loading errors are all catched!

            # relative href
            parentHref = self.parentStyleSheet.href
            if parentHref is None:
                # use cwd instead
                parentHref = cssutils.helper.path2url(os.getcwd()) + '/'

            fullhref = urllib.parse.urljoin(parentHref, self.href)

            # all possible exceptions are ignored
            try:
                usedEncoding, enctype, cssText = self.parentStyleSheet._resolveImport(
                    fullhref
                )

                if cssText is None:
                    # catched in next except below!
                    raise IOError('Cannot read Stylesheet.')

                # contentEncoding with parentStyleSheet.overrideEncoding,
                # HTTP or parent
                encodingOverride, encoding = None, None

                if enctype == 0:
                    encodingOverride = usedEncoding
                elif 0 < enctype < 5:
                    encoding = usedEncoding

                # inherit fetcher for @imports in styleSheet
                importedSheet._href = fullhref
                importedSheet._setFetcher(self.parentStyleSheet._fetcher)
                importedSheet._setCssTextWithEncodingOverride(
                    cssText, encodingOverride=encodingOverride, encoding=encoding
                )

            except (OSError, IOError, ValueError) as e:
                self._log.warn(
                    'CSSImportRule: While processing imported '
                    'style sheet href=%s: %r' % (self.href, e),
                    neverraise=True,
                )

            else:
                # used by resolveImports if to keep unprocessed href
                self.hrefFound = True

        self._styleSheet = importedSheet

    _href = None  # needs to be set
    href = property(
        lambda self: self._href,
        _setHref,
        doc="Location of the style sheet to be imported.",
    )

    def _setMedia(self, media):
        """
        :param media:
            a :class:`~cssutils.stylesheets.MediaList` or string
        """
        self._checkReadonly()
        if isinstance(media, str):
            self._media = cssutils.stylesheets.MediaList(
                mediaText=media, parentRule=self
            )
        else:
            media._parentRule = self
            self._media = media

        # update seq
        ihref = 0
        for i, item in enumerate(self.seq):
            if item.type == 'href':
                ihref = i
            elif item.type == 'media':
                self.seq[i] = (self._media, 'media', None, None)
                break
        else:
            # if no media until now add after href
            self.seq.insert(ihref + 1, self._media, 'media', None, None)

    media = property(
        lambda self: self._media,
        _setMedia,
        doc="(DOM) A list of media types for this rule "
        "of type :class:`~cssutils.stylesheets.MediaList`.",
    )

    def _setName(self, name=''):
        """Raises xml.dom.SyntaxErr if name is not a string."""
        if name is None or isinstance(name, str):
            # "" or '' handled as None
            if not name:
                name = None

            # save name
            self._name = name

            # update seq
            for i, item in enumerate(self.seq):
                typ = item.type
                if 'name' == typ:
                    self._seq[i] = (name, typ, item.line, item.col)
                    break

            # set title of imported sheet
            if self.styleSheet:
                self.styleSheet.title = name

        else:
            self._log.error('CSSImportRule: Not a valid name: %s' % name)

    name = property(
        lambda self: self._name,
        _setName,
        doc="An optional name for the imported sheet.",
    )

    styleSheet = property(
        lambda self: self._styleSheet,
        doc="(readonly) The style sheet referred to by this " "rule.",
    )

    type = property(
        lambda self: self.IMPORT_RULE,
        doc="The type of this rule, as defined by a CSSRule " "type constant.",
    )

    def _getWellformed(self):
        "Depending on if media is used at all."
        if self._usemedia:
            return bool(self.href and self.media.wellformed)
        else:
            return bool(self.href)

    wellformed = property(_getWellformed)
