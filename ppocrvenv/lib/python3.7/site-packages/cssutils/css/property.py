"""Property is a single CSS property in a CSSStyleDeclaration."""
__all__ = ['Property']

from cssutils.helper import Deprecated
from .value import PropertyValue
import cssutils


class Property(cssutils.util.Base):
    """A CSS property in a StyleDeclaration of a CSSStyleRule (cssutils).

    Format::

        property = name
          : IDENT S*
          ;

        expr = value
          : term [ operator term ]*
          ;
        term
          : unary_operator?
            [ NUMBER S* | PERCENTAGE S* | LENGTH S* | EMS S* | EXS S* |
              ANGLE S* | TIME S* | FREQ S* | function ]
          | STRING S* | IDENT S* | URI S* | hexcolor
          ;
        function
          : FUNCTION S* expr ')' S*
          ;
        /*
         * There is a constraint on the color that it must
         * have either 3 or 6 hex-digits (i.e., [0-9a-fA-F])
         * after the "#"; e.g., "#000" is OK, but "#abcd" is not.
         */
        hexcolor
          : HASH S*
          ;

        prio
          : IMPORTANT_SYM S*
          ;

    """

    def __init__(
        self, name=None, value=None, priority='', _mediaQuery=False, parent=None
    ):
        """
        :param name:
            a property name string (will be normalized)
        :param value:
            a property value string
        :param priority:
            an optional priority string which currently must be u'',
            u'!important' or u'important'
        :param _mediaQuery:
            if ``True`` value is optional (used by MediaQuery)
        :param parent:
            the parent object, normally a
            :class:`cssutils.css.CSSStyleDeclaration`
        """
        super(Property, self).__init__()
        self.seqs = [[], None, []]
        self.wellformed = False
        self._mediaQuery = _mediaQuery
        self.parent = parent

        self.__nametoken = None
        self._name = ''
        self._literalname = ''
        self.seqs[1] = PropertyValue(parent=self)
        if name:
            self.name = name
            self.propertyValue = value

        self._priority = ''
        self._literalpriority = ''
        if priority:
            self.priority = priority

    def __repr__(self):
        return "cssutils.css.%s(name=%r, value=%r, priority=%r)" % (
            self.__class__.__name__,
            self.literalname,
            self.propertyValue.cssText,
            self.priority,
        )

    def __str__(self):
        return "<%s.%s object name=%r value=%r priority=%r valid=%r at 0x%x>" % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.name,
            self.propertyValue.cssText,
            self.priority,
            self.valid,
            id(self),
        )

    def _isValidating(self):
        """Return True if validation is enabled."""
        try:
            return self.parent.validating
        except AttributeError:
            # default (no parent)
            return True

    def _getCssText(self):
        """Return serialized property cssText."""
        return cssutils.ser.do_Property(self)

    def _setCssText(self, cssText):
        """
        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error and
              is unparsable.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if the rule is readonly.
        """
        # check and prepare tokenlists for setting
        tokenizer = self._tokenize2(cssText)
        nametokens = self._tokensupto2(tokenizer, propertynameendonly=True)
        if nametokens:
            wellformed = True

            valuetokens = self._tokensupto2(tokenizer, propertyvalueendonly=True)
            prioritytokens = self._tokensupto2(tokenizer, propertypriorityendonly=True)

            if self._mediaQuery and not valuetokens:
                # MediaQuery may consist of name only
                self.name = nametokens
                self.propertyValue = None
                self.priority = None
                return

            # remove colon from nametokens
            colontoken = nametokens.pop()
            if self._tokenvalue(colontoken) != ':':
                wellformed = False
                self._log.error(
                    'Property: No ":" after name found: %s' % self._valuestr(cssText),
                    colontoken,
                )
            elif not nametokens:
                wellformed = False
                self._log.error(
                    'Property: No property name found: %s' % self._valuestr(cssText),
                    colontoken,
                )

            if valuetokens:
                if self._tokenvalue(valuetokens[-1]) == '!':
                    # priority given, move "!" to prioritytokens
                    prioritytokens.insert(0, valuetokens.pop(-1))
            else:
                wellformed = False
                self._log.error(
                    'Property: No property value found: %s' % self._valuestr(cssText),
                    colontoken,
                )

            if wellformed:
                self.wellformed = True
                self.name = nametokens
                self.propertyValue = valuetokens
                self.priority = prioritytokens

                # also invalid values are set!

                if self._isValidating():
                    self.validate()

        else:
            self._log.error(
                'Property: No property name found: %s' % self._valuestr(cssText)
            )

    cssText = property(
        fget=_getCssText, fset=_setCssText, doc="A parsable textual representation."
    )

    def _setName(self, name):
        """
        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified name has a syntax error and is
              unparsable.
        """
        # for closures: must be a mutable
        new = {'literalname': None, 'wellformed': True}

        def _ident(expected, seq, token, tokenizer=None):
            # name
            if 'name' == expected:
                new['literalname'] = self._tokenvalue(token).lower()
                seq.append(new['literalname'])
                return 'EOF'
            else:
                new['wellformed'] = False
                self._log.error('Property: Unexpected ident.', token)
                return expected

        newseq = []
        wellformed, expected = self._parse(
            expected='name',
            seq=newseq,
            tokenizer=self._tokenize2(name),
            productions={'IDENT': _ident},
        )
        wellformed = wellformed and new['wellformed']

        # post conditions
        # define a token for error logging
        if isinstance(name, list):
            token = name[0]
            self.__nametoken = token
        else:
            token = None

        if not new['literalname']:
            wellformed = False
            self._log.error(
                'Property: No name found: %s' % self._valuestr(name), token=token
            )

        if wellformed:
            self.wellformed = True
            self._literalname = new['literalname']
            self._name = self._normalize(self._literalname)
            self.seqs[0] = newseq

            # validate
            if self._isValidating() and self._name not in cssutils.profile.knownNames:
                # self.valid = False
                self._log.warn(
                    'Property: Unknown Property name.', token=token, neverraise=True
                )
            else:
                pass
        #                self.valid = True
        #                if self.propertyValue:
        #                    self.propertyValue._propertyName = self._name
        #                    #self.valid = self.propertyValue.valid
        else:
            self.wellformed = False

    name = property(lambda self: self._name, _setName, doc="Name of this property.")

    literalname = property(
        lambda self: self._literalname,
        doc="Readonly literal (not normalized) name " "of this property",
    )

    def _setPropertyValue(self, cssText):
        """
        See css.PropertyValue

        :exceptions:
        - :exc:`~xml.dom.SyntaxErr`:
          Raised if the specified CSS string value has a syntax error
          (according to the attached property) or is unparsable.
        - :exc:`~xml.dom.InvalidModificationErr`:
          TODO: Raised if the specified CSS string value represents a different
          type of values than the values allowed by the CSS property.
        """
        if self._mediaQuery and not cssText:
            self.seqs[1] = PropertyValue(parent=self)
        else:
            self.seqs[1].cssText = cssText
            self.wellformed = self.wellformed and self.seqs[1].wellformed

    propertyValue = property(
        lambda self: self.seqs[1],
        _setPropertyValue,
        doc="(cssutils) PropertyValue object of property",
    )

    def _getValue(self):
        if self.propertyValue:
            # value without comments
            return self.propertyValue.value
        else:
            return ''

    def _setValue(self, value):
        self._setPropertyValue(value)

    value = property(
        _getValue, _setValue, doc="The textual value of this Properties propertyValue."
    )

    def _setPriority(self, priority):  # noqa: C901
        """
        priority
            a string, currently either u'', u'!important' or u'important'

        Format::

            prio
              : IMPORTANT_SYM S*
              ;

            "!"{w}"important"   {return IMPORTANT_SYM;}

        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified priority has a syntax error and is
              unparsable.
              In this case a priority not equal to None, "" or "!{w}important".
              As CSSOM defines CSSStyleDeclaration.getPropertyPriority resulting
              in u'important' this value is also allowed to set a Properties
              priority
        """
        if self._mediaQuery:
            self._priority = ''
            self._literalpriority = ''
            if priority:
                self._log.error('Property: No priority in a MediaQuery - ' 'ignored.')
            return

        if isinstance(priority, str) and 'important' == self._normalize(priority):
            priority = '!%s' % priority

        # for closures: must be a mutable
        new = {'literalpriority': '', 'wellformed': True}

        def _char(expected, seq, token, tokenizer=None):
            # "!"
            val = self._tokenvalue(token)
            if '!' == expected == val:
                seq.append(val)
                return 'important'
            else:
                new['wellformed'] = False
                self._log.error('Property: Unexpected char.', token)
                return expected

        def _ident(expected, seq, token, tokenizer=None):
            # "important"
            val = self._tokenvalue(token)
            if 'important' == expected:
                new['literalpriority'] = val
                seq.append(val)
                return 'EOF'
            else:
                new['wellformed'] = False
                self._log.error('Property: Unexpected ident.', token)
                return expected

        newseq = []
        wellformed, expected = self._parse(
            expected='!',
            seq=newseq,
            tokenizer=self._tokenize2(priority),
            productions={'CHAR': _char, 'IDENT': _ident},
        )
        wellformed = wellformed and new['wellformed']

        # post conditions
        if priority and not new['literalpriority']:
            wellformed = False
            self._log.info('Property: Invalid priority: %s' % self._valuestr(priority))

        if wellformed:
            self.wellformed = self.wellformed and wellformed
            self._literalpriority = new['literalpriority']
            self._priority = self._normalize(self.literalpriority)
            self.seqs[2] = newseq
            # validate priority
            if self._priority not in ('', 'important'):
                self._log.error('Property: No CSS priority value: %s' % self._priority)

    priority = property(
        lambda self: self._priority, _setPriority, doc="Priority of this property."
    )

    literalpriority = property(
        lambda self: self._literalpriority,
        doc="Readonly literal (not normalized) priority of this property",
    )

    def _setParent(self, parent):
        self._parent = parent

    parent = property(
        lambda self: self._parent,
        _setParent,
        doc="The Parent Node (normally a CSSStyledeclaration) of this " "Property",
    )

    def validate(self):  # noqa: C901
        """Validate value against `profiles` which are checked dynamically.
        properties in e.g. @font-face rules are checked against
        ``cssutils.profile.CSS3_FONT_FACE`` only.

        For each of the following cases a message is reported:

        - INVALID (so the property is known but not valid)
            ``ERROR    Property: Invalid value for "{PROFILE-1[/PROFILE-2...]"
            property: ...``

        - VALID but not in given profiles or defaultProfiles
            ``WARNING    Property: Not valid for profile "{PROFILE-X}" but valid
            "{PROFILE-Y}" property: ...``

        - VALID in current profile
            ``DEBUG    Found valid "{PROFILE-1[/PROFILE-2...]" property...``

        - UNKNOWN property
            ``WARNING    Unknown Property name...`` is issued

        so for example::

            cssutils.log.setLevel(logging.DEBUG)
            parser = cssutils.CSSParser()
            s = parser.parseString('''body {
                unknown-property: x;
                color: 4;
                color: rgba(1,2,3,4);
                color: red
            }''')

            # Log output:

            WARNING Property: Unknown Property name. [2:9: unknown-property]
            ERROR   Property: Invalid value for \
                "CSS Color Module Level 3/CSS Level 2.1" property: 4 [3:9: color]
            DEBUG   Property: Found valid \
                "CSS Color Module Level 3" value: rgba(1, 2, 3, 4) [4:9: color]
            DEBUG   Property: Found valid "CSS Level 2.1" value: red [5:9: color]


        and when setting an explicit default profile::

            cssutils.profile.defaultProfiles = cssutils.profile.CSS_LEVEL_2
            s = parser.parseString('''body {
                unknown-property: x;
                color: 4;
                color: rgba(1,2,3,4);
                color: red
            }''')

            # Log output:

            WARNING Property: Unknown Property name. [2:9: unknown-property]
            ERROR   Property: Invalid value for \
                "CSS Color Module Level 3/CSS Level 2.1" property: 4 [3:9: color]
            WARNING Property: Not valid for profile \
                "CSS Level 2.1" but valid "CSS Color Module Level 3" \
                value: rgba(1, 2, 3, 4)  [4:9: color]
            DEBUG   Property: Found valid "CSS Level 2.1" value: red [5:9: color]
        """
        valid = False

        profiles = None
        try:
            # if @font-face use that profile
            rule = self.parent.parentRule
        except AttributeError:
            pass
        else:
            if rule is not None:
                if rule.type == rule.FONT_FACE_RULE:
                    profiles = [cssutils.profile.CSS3_FONT_FACE]
                # TODO: same for @page

        if self.name and self.value:

            # TODO
            # cv = self.propertyValue
            # if cv.cssValueType == cv.CSS_VARIABLE and not cv.value:
            #     # TODO: false alarms too!
            #     cssutils.log.warn(u'No value for variable "%s" found, keeping '
            #                       u'variable.' % cv.name, neverraise=True)

            if self.name in cssutils.profile.knownNames:
                # add valid, matching, validprofiles...
                valid, matching, validprofiles = cssutils.profile.validateWithProfile(
                    self.name, self.value, profiles
                )

                if not valid:
                    self._log.error(
                        'Property: Invalid value for '
                        '"%s" property: %s' % ('/'.join(validprofiles), self.value),
                        token=self.__nametoken,
                        neverraise=True,
                    )

                # TODO: remove logic to profiles!
                elif (
                    valid and not matching
                ):  # (profiles and profiles not in validprofiles):
                    if not profiles:
                        notvalidprofiles = '/'.join(cssutils.profile.defaultProfiles)
                    else:
                        notvalidprofiles = profiles
                    self._log.warn(
                        'Property: Not valid for profile "%s" '
                        'but valid "%s" value: %s '
                        % (notvalidprofiles, '/'.join(validprofiles), self.value),
                        token=self.__nametoken,
                        neverraise=True,
                    )
                    valid = False

                elif valid:
                    self._log.debug(
                        'Property: Found valid "%s" value: %s'
                        % ('/'.join(validprofiles), self.value),
                        token=self.__nametoken,
                        neverraise=True,
                    )

        if self._priority not in ('', 'important'):
            valid = False

        return valid

    valid = property(
        validate,
        doc="Check if value of this property is valid " "in the properties context.",
    )

    @Deprecated('Use ``property.propertyValue`` instead.')
    def _getCSSValue(self):
        return self.propertyValue

    @Deprecated('Use ``property.propertyValue`` instead.')
    def _setCSSValue(self, cssText):
        self._setPropertyValue(cssText)

    cssValue = property(
        _getCSSValue,
        _setCSSValue,
        doc="(DEPRECATED) Use ``property.propertyValue`` instead.",
    )
