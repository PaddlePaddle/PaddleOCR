"""CSSMediaRule implements DOM Level 2 CSS CSSMediaRule."""
__all__ = ['CSSMediaRule']

from . import cssrule
import cssutils
import xml.dom


class CSSMediaRule(cssrule.CSSRuleRules):
    """
    Objects implementing the CSSMediaRule interface can be identified by the
    MEDIA_RULE constant. On these objects the type attribute must return the
    value of that constant.

    Format::

      : MEDIA_SYM S* medium [ COMMA S* medium ]*

          STRING? # the name

      LBRACE S* ruleset* '}' S*;

    ``cssRules``
        All Rules in this media rule, a :class:`~cssutils.css.CSSRuleList`.
    """

    def __init__(
        self,
        mediaText='all',
        name=None,
        parentRule=None,
        parentStyleSheet=None,
        readonly=False,
    ):
        """constructor"""
        super(CSSMediaRule, self).__init__(
            parentRule=parentRule, parentStyleSheet=parentStyleSheet
        )
        self._atkeyword = '@media'

        # 1. media
        if mediaText:
            self.media = mediaText
        else:
            self.media = cssutils.stylesheets.MediaList()

        self.name = name
        self._readonly = readonly

    def __repr__(self):
        return "cssutils.css.%s(mediaText=%r)" % (
            self.__class__.__name__,
            self.media.mediaText,
        )

    def __str__(self):
        return "<cssutils.css.%s object mediaText=%r at 0x%x>" % (
            self.__class__.__name__,
            self.media.mediaText,
            id(self),
        )

    def _getCssText(self):
        """Return serialized property cssText."""
        return cssutils.ser.do_CSSMediaRule(self)

    def _setCssText(self, cssText):  # noqa: C901
        """
        :param cssText:
            a parseable string or a tuple of (cssText, dict-of-namespaces)
        :Exceptions:
            - :exc:`~xml.dom.NamespaceErr`:
              Raised if a specified selector uses an unknown namespace
              prefix.
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
        # media "name"? { cssRules }
        super(CSSMediaRule, self)._setCssText(cssText)

        # might be (cssText, namespaces)
        cssText, namespaces = self._splitNamespacesOff(cssText)

        tokenizer = self._tokenize2(cssText)
        attoken = self._nexttoken(tokenizer, None)
        if self._type(attoken) != self._prods.MEDIA_SYM:
            self._log.error(
                'CSSMediaRule: No CSSMediaRule found: %s' % self._valuestr(cssText),
                error=xml.dom.InvalidModificationErr,
            )

        else:
            # save if parse goes wrong
            oldMedia = self._media
            oldCssRules = self._cssRules

            ok = True

            # media
            mediatokens, end = self._tokensupto2(
                tokenizer, mediaqueryendonly=True, separateEnd=True
            )
            if '{' == self._tokenvalue(end) or self._prods.STRING == self._type(end):
                self.media = cssutils.stylesheets.MediaList(parentRule=self)
                # TODO: remove special case
                self.media.mediaText = mediatokens
                ok = ok and self.media.wellformed
            else:
                ok = False

            # name (optional)
            name = None
            nameseq = self._tempSeq()
            if self._prods.STRING == self._type(end):
                name = self._stringtokenvalue(end)
                # TODO: for now comments are lost after name
                nametokens, end = self._tokensupto2(
                    tokenizer, blockstartonly=True, separateEnd=True
                )
                wellformed, expected = self._parse(None, nameseq, nametokens, {})
                if not wellformed:
                    ok = False
                    self._log.error(
                        'CSSMediaRule: Syntax Error: %s' % self._valuestr(cssText)
                    )

            # check for {
            if '{' != self._tokenvalue(end):
                self._log.error(
                    'CSSMediaRule: No "{" found: %s' % self._valuestr(cssText)
                )
                return

            # cssRules
            cssrulestokens, braceOrEOF = self._tokensupto2(
                tokenizer, mediaendonly=True, separateEnd=True
            )
            nonetoken = self._nexttoken(tokenizer, None)
            if 'EOF' == self._type(braceOrEOF):
                # HACK!!!
                # TODO: Not complete, add EOF to rule and } to @media
                cssrulestokens.append(braceOrEOF)
                braceOrEOF = ('CHAR', '}', 0, 0)
                self._log.debug(
                    'CSSMediaRule: Incomplete, adding "}".',
                    token=braceOrEOF,
                    neverraise=True,
                )

            if '}' != self._tokenvalue(braceOrEOF):
                self._log.error('CSSMediaRule: No "}" found.', token=braceOrEOF)
            elif nonetoken:
                self._log.error(
                    'CSSMediaRule: Trailing content found.', token=nonetoken
                )
            else:
                # for closures: must be a mutable
                new = {'wellformed': True}

                def COMMENT(expected, seq, token, tokenizer=None):
                    self.insertRule(
                        cssutils.css.CSSComment(
                            [token],
                            parentRule=self,
                            parentStyleSheet=self.parentStyleSheet,
                        )
                    )
                    return expected

                def ruleset(expected, seq, token, tokenizer):
                    rule = cssutils.css.CSSStyleRule(
                        parentRule=self, parentStyleSheet=self.parentStyleSheet
                    )
                    rule.cssText = self._tokensupto2(tokenizer, token)
                    if rule.wellformed:
                        self.insertRule(rule)
                    return expected

                def atrule(expected, seq, token, tokenizer):
                    # TODO: get complete rule!
                    tokens = self._tokensupto2(tokenizer, token)
                    atval = self._tokenvalue(token)
                    factories = {
                        '@page': cssutils.css.CSSPageRule,
                        '@media': CSSMediaRule,
                    }
                    if atval in (
                        '@charset ',
                        '@font-face',
                        '@import',
                        '@namespace',
                        '@variables',
                    ):
                        self._log.error(
                            'CSSMediaRule: This rule is not '
                            'allowed in CSSMediaRule - ignored: '
                            '%s.' % self._valuestr(tokens),
                            token=token,
                            error=xml.dom.HierarchyRequestErr,
                        )
                    elif atval in factories:
                        rule = factories[atval](
                            parentRule=self, parentStyleSheet=self.parentStyleSheet
                        )
                        rule.cssText = tokens
                        if rule.wellformed:
                            self.insertRule(rule)
                    else:
                        rule = cssutils.css.CSSUnknownRule(
                            tokens,
                            parentRule=self,
                            parentStyleSheet=self.parentStyleSheet,
                        )
                        if rule.wellformed:
                            self.insertRule(rule)
                    return expected

                # save for possible reset
                oldCssRules = self.cssRules

                self.cssRules = cssutils.css.CSSRuleList()
                seq = []  # not used really

                tokenizer = iter(cssrulestokens)
                wellformed, expected = self._parse(
                    braceOrEOF,
                    seq,
                    tokenizer,
                    {
                        'COMMENT': COMMENT,
                        'CHARSET_SYM': atrule,
                        'FONT_FACE_SYM': atrule,
                        'IMPORT_SYM': atrule,
                        'NAMESPACE_SYM': atrule,
                        'PAGE_SYM': atrule,
                        'MEDIA_SYM': atrule,
                        'ATKEYWORD': atrule,
                    },
                    default=ruleset,
                    new=new,
                )
                ok = ok and wellformed

            if ok:
                self.name = name
                self._setSeq(nameseq)
            else:
                self._media = oldMedia
                self._cssRules = oldCssRules

    cssText = property(
        _getCssText,
        _setCssText,
        doc="(DOM) The parsable textual representation of this " "rule.",
    )

    def _setName(self, name):
        if isinstance(name, str) or name is None:
            # "" or ''
            if not name:
                name = None

            self._name = name
        else:
            self._log.error('CSSImportRule: Not a valid name: %s' % name)

    name = property(
        lambda self: self._name, _setName, doc="An optional name for this media rule."
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

        # NOT IN @media seq at all?!

    #        # update seq
    #        for i, item in enumerate(self.seq):
    #            if item.type == 'media':
    #                self._seq[i] = (self._media, 'media', None, None)
    #                break
    #        else:
    #            # insert after @media if not in seq at all
    #            self.seq.insert(0,
    #                             self._media, 'media', None, None)

    media = property(
        lambda self: self._media,
        _setMedia,
        doc="(DOM) A list of media types for this rule "
        "of type :class:`~cssutils.stylesheets.MediaList`.",
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
            or isinstance(rule, cssutils.css.MarginRule)
        ):
            self._log.error(
                '%s: This type of rule is not allowed here: %s'
                % (self.__class__.__name__, rule.cssText),
                error=xml.dom.HierarchyRequestErr,
            )
            return

        return self._finishInsertRule(rule, index)

    type = property(
        lambda self: self.MEDIA_RULE,
        doc="The type of this rule, as defined by a CSSRule " "type constant.",
    )

    wellformed = property(lambda self: self.media.wellformed)
