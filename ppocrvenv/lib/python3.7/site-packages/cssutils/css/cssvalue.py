"""CSSValue related classes

- CSSValue implements DOM Level 2 CSS CSSValue
- CSSPrimitiveValue implements DOM Level 2 CSS CSSPrimitiveValue
- CSSValueList implements DOM Level 2 CSS CSSValueList

"""
__all__ = ['CSSValue', 'CSSPrimitiveValue', 'CSSValueList', 'RGBColor', 'CSSVariable']

from cssutils.prodparser import Choice, PreDef, Prod, ProdParser, Sequence
import cssutils
import cssutils.helper
import math
import re
import xml.dom


class CSSValue(cssutils.util._NewBase):
    """The CSSValue interface represents a simple or a complex value.
    A CSSValue object only occurs in a context of a CSS property.
    """

    # The value is inherited and the cssText contains "inherit".
    CSS_INHERIT = 0
    # The value is a CSSPrimitiveValue.
    CSS_PRIMITIVE_VALUE = 1
    # The value is a CSSValueList.
    CSS_VALUE_LIST = 2
    # The value is a custom value.
    CSS_CUSTOM = 3
    # The value is a CSSVariable.
    CSS_VARIABLE = 4

    _typestrings = {
        0: 'CSS_INHERIT',
        1: 'CSS_PRIMITIVE_VALUE',
        2: 'CSS_VALUE_LIST',
        3: 'CSS_CUSTOM',
        4: 'CSS_VARIABLE',
    }

    def __init__(self, cssText=None, parent=None, readonly=False):
        """
        :param cssText:
            the parsable cssText of the value
        :param readonly:
            defaults to False
        """
        super(CSSValue, self).__init__()

        self._cssValueType = None
        self.wellformed = False
        self.parent = parent
        if cssText is not None:  # may be 0
            if isinstance(cssText, int):
                cssText = str(cssText)  # if it is an integer
            elif isinstance(cssText, float):
                cssText = '%f' % cssText  # if it is a floating point number

            self.cssText = cssText

        self._readonly = readonly

    def __repr__(self):
        return "cssutils.css.%s(%r)" % (self.__class__.__name__, self.cssText)

    def __str__(self):
        return (
            "<cssutils.css.%s object cssValueTypeString=%r cssText=%r at "
            "0x%x>"
            % (self.__class__.__name__, self.cssValueTypeString, self.cssText, id(self))
        )

    def _setCssText(self, cssText):  # noqa: C901
        """
        Format::

            unary_operator
              : '-' | '+'
              ;
            operator
              : '/' S* | ',' S* | /* empty */
              ;
            expr
              : term [ operator term ]*
              ;
            term
              : unary_operator?
                [ NUMBER S* | PERCENTAGE S* | LENGTH S* | EMS S* | EXS S* |
                  ANGLE S* | TIME S* | FREQ S* ]
              | STRING S* | IDENT S* | URI S* | hexcolor | function
              | UNICODE-RANGE S*
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

        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error
              (according to the attached property) or is unparsable.
            - :exc:`~xml.dom.InvalidModificationErr`:
              TODO: Raised if the specified CSS string value represents a
              different type of values than the values allowed by the CSS
              property.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this value is readonly.
        """
        self._checkReadonly()

        # used as operator is , / or S
        nextSor = ',/'

        term = Choice(
            Sequence(
                PreDef.unary(),
                Choice(
                    PreDef.number(nextSor=nextSor),
                    PreDef.percentage(nextSor=nextSor),
                    PreDef.dimension(nextSor=nextSor),
                ),
            ),
            PreDef.string(nextSor=nextSor),
            PreDef.ident(nextSor=nextSor),
            PreDef.uri(nextSor=nextSor),
            PreDef.hexcolor(nextSor=nextSor),
            PreDef.unicode_range(nextSor=nextSor),
            # special case IE only expression
            Prod(
                name='expression',
                match=lambda t, v: t == self._prods.FUNCTION
                and (
                    cssutils.helper.normalize(v)
                    in (
                        'expression(',
                        'alpha(',
                        'blur(',
                        'chroma(',
                        'dropshadow(',
                        'fliph(',
                        'flipv(',
                        'glow(',
                        'gray(',
                        'invert(',
                        'mask(',
                        'shadow(',
                        'wave(',
                        'xray(',
                    )
                    or v.startswith('progid:DXImageTransform.Microsoft.')
                ),
                nextSor=nextSor,
                toSeq=lambda t, tokens: (
                    ExpressionValue._functionName,
                    ExpressionValue(cssutils.helper.pushtoken(t, tokens), parent=self),
                ),
            ),
            # CSS Variable var(
            PreDef.variable(
                nextSor=nextSor,
                toSeq=lambda t, tokens: (
                    'CSSVariable',
                    CSSVariable(cssutils.helper.pushtoken(t, tokens), parent=self),
                ),
            ),
            # calc(
            PreDef.calc(
                nextSor=nextSor,
                toSeq=lambda t, tokens: (
                    CalcValue._functionName,
                    CalcValue(cssutils.helper.pushtoken(t, tokens), parent=self),
                ),
            ),
            # TODO:
            # # rgb/rgba(
            # Prod(name='RGBColor',
            #      match=lambda t, v: t == self._prods.FUNCTION and (
            #         cssutils.helper.normalize(v) in (u'rgb(',
            #                                          u'rgba('
            #                                          )
            #      ),
            #      nextSor=nextSor,
            #             toSeq=lambda t, tokens: (RGBColor._functionName,
            #                                      RGBColor(
            #                   cssutils.helper.pushtoken(t, tokens),
            #                   parent=self)
            #                                      )
            # ),
            # other functions like rgb( etc
            PreDef.function(
                nextSor=nextSor,
                toSeq=lambda t, tokens: (
                    'FUNCTION',
                    CSSFunction(cssutils.helper.pushtoken(t, tokens), parent=self),
                ),
            ),
        )
        operator = Choice(
            PreDef.S(),
            PreDef.char('comma', ',', toSeq=lambda t, tokens: ('operator', t[1])),
            PreDef.char('slash', '/', toSeq=lambda t, tokens: ('operator', t[1])),
            optional=True,
        )
        # CSSValue PRODUCTIONS
        valueprods = Sequence(
            term,
            Sequence(
                operator,  # mayEnd this Sequence if whitespace
                # TODO: only when setting via other class
                # used by variabledeclaration currently
                PreDef.char('END', ';', stopAndKeep=True, optional=True),
                term,
                minmax=lambda: (0, None),
            ),
        )
        # parse
        wellformed, seq, store, notused = ProdParser().parse(
            cssText, 'CSSValue', valueprods, keepS=True
        )
        if wellformed:
            # - count actual values and set firstvalue which is used later on
            # - combine comma separated list, e.g. font-family to a single item
            # - remove S which should be an operator but is no needed
            count, firstvalue = 0, ()
            newseq = self._tempSeq()
            i, end = 0, len(seq)
            while i < end:
                item = seq[i]
                if item.type == self._prods.S:
                    pass

                elif (item.value, item.type) == (',', 'operator'):
                    # , separared counts as a single STRING for now
                    # URI or STRING value might be a single CHAR too!
                    newseq.appendItem(item)
                    count -= 1
                    if firstvalue:
                        # list of IDENTs is handled as STRING!
                        if firstvalue[1] == self._prods.IDENT:
                            firstvalue = firstvalue[0], 'STRING'

                elif item.value == '/':
                    # / separated items count as one
                    newseq.appendItem(item)

                elif item.value == '-' or item.value == '+':
                    # combine +- and following number or other
                    i += 1
                    try:
                        next = seq[i]
                    except IndexError:
                        firstvalue = ()  # raised later
                        break

                    newval = item.value + next.value
                    newseq.append(newval, next.type, item.line, item.col)
                    if not firstvalue:
                        firstvalue = (newval, next.type)
                    count += 1

                elif item.type != cssutils.css.CSSComment:
                    newseq.appendItem(item)
                    if not firstvalue:
                        firstvalue = (item.value, item.type)
                    count += 1

                else:
                    newseq.appendItem(item)

                i += 1

            if not firstvalue:
                self._log.error(
                    'CSSValue: Unknown syntax or no value: %r.'
                    % self._valuestr(cssText)
                )
            else:
                # ok and set
                self._setSeq(newseq)
                self.wellformed = wellformed

                if hasattr(self, '_value'):
                    # only in case of CSSPrimitiveValue, else remove!
                    del self._value

                if count == 1:
                    # inherit, primitive or variable
                    if isinstance(
                        firstvalue[0], str
                    ) and 'inherit' == cssutils.helper.normalize(firstvalue[0]):
                        self.__class__ = CSSValue
                        self._cssValueType = CSSValue.CSS_INHERIT
                    elif 'CSSVariable' == firstvalue[1]:
                        self.__class__ = CSSVariable
                        self._value = firstvalue
                        # TODO: remove major hack!
                        self._name = firstvalue[0]._name
                    else:
                        self.__class__ = CSSPrimitiveValue
                        self._value = firstvalue

                elif count > 1:
                    # valuelist
                    self.__class__ = CSSValueList

                    # change items in list to specific type (primitive etc)
                    newseq = self._tempSeq()
                    commalist = []
                    nexttocommalist = False

                    def itemValue(item):
                        "Reserialized simple item.value"
                        if self._prods.STRING == item.type:
                            return cssutils.helper.string(item.value)
                        elif self._prods.URI == item.type:
                            return cssutils.helper.uri(item.value)
                        elif (
                            self._prods.FUNCTION == item.type
                            or 'CSSVariable' == item.type
                        ):
                            return item.value.cssText
                        else:
                            return item.value

                    def saveifcommalist(commalist, newseq):
                        """
                        saves items in commalist to seq and items
                        if anything in there
                        """
                        if commalist:
                            newseq.replace(
                                -1,
                                CSSPrimitiveValue(cssText=''.join(commalist)),
                                CSSPrimitiveValue,
                                newseq[-1].line,
                                newseq[-1].col,
                            )
                            del commalist[:]

                    for i, item in enumerate(self._seq):
                        if issubclass(type(item.value), CSSValue):
                            # set parent of CSSValueList items to the lists
                            # parent
                            item.value.parent = self.parent

                        if item.type in (
                            self._prods.DIMENSION,
                            self._prods.FUNCTION,
                            self._prods.HASH,
                            self._prods.IDENT,
                            self._prods.NUMBER,
                            self._prods.PERCENTAGE,
                            self._prods.STRING,
                            self._prods.URI,
                            self._prods.UNICODE_RANGE,
                            'CSSVariable',
                        ):
                            if nexttocommalist:
                                # wait until complete
                                commalist.append(itemValue(item))
                            else:
                                saveifcommalist(commalist, newseq)
                                # append new item
                                if hasattr(item.value, 'cssText'):
                                    newseq.append(
                                        item.value,
                                        item.value.__class__,
                                        item.line,
                                        item.col,
                                    )

                                else:
                                    newseq.append(
                                        CSSPrimitiveValue(itemValue(item)),
                                        CSSPrimitiveValue,
                                        item.line,
                                        item.col,
                                    )

                            nexttocommalist = False

                        elif ',' == item.value:
                            if not commalist:
                                # save last item to commalist
                                commalist.append(itemValue(self._seq[i - 1]))
                            commalist.append(',')
                            nexttocommalist = True

                        else:
                            if nexttocommalist:
                                commalist.append(item.value.cssText)
                            else:
                                newseq.appendItem(item)

                    saveifcommalist(commalist, newseq)
                    self._setSeq(newseq)

                else:
                    # should not happen...
                    self.__class__ = CSSValue
                    self._cssValueType = CSSValue.CSS_CUSTOM

    cssText = property(
        lambda self: cssutils.ser.do_css_CSSValue(self),
        _setCssText,
        doc="A string representation of the current value.",
    )

    cssValueType = property(
        lambda self: self._cssValueType,
        doc="A (readonly) code defining the type of the value.",
    )

    cssValueTypeString = property(
        lambda self: CSSValue._typestrings.get(self.cssValueType, None),
        doc="(readonly) Name of cssValueType.",
    )


class CSSPrimitiveValue(CSSValue):
    """Represents a single CSS Value.  May be used to determine the value of a
    specific style property currently set in a block or to set a specific
    style property explicitly within the block. Might be obtained from the
    getPropertyCSSValue method of CSSStyleDeclaration.

    Conversions are allowed between absolute values (from millimeters to
    centimeters, from degrees to radians, and so on) but not between
    relative values. (For example, a pixel value cannot be converted to a
    centimeter value.) Percentage values can't be converted since they are
    relative to the parent value (or another property value). There is one
    exception for color percentage values: since a color percentage value
    is relative to the range 0-255, a color percentage value can be
    converted to a number; (see also the RGBColor interface).
    """

    # constant: type of this CSSValue class
    cssValueType = CSSValue.CSS_PRIMITIVE_VALUE

    __types = cssutils.cssproductions.CSSProductions

    # An integer indicating which type of unit applies to the value.
    CSS_UNKNOWN = 0  # only obtainable via cssText
    CSS_NUMBER = 1
    CSS_PERCENTAGE = 2
    CSS_EMS = 3
    CSS_EXS = 4
    CSS_PX = 5
    CSS_CM = 6
    CSS_MM = 7
    CSS_IN = 8
    CSS_PT = 9
    CSS_PC = 10
    CSS_DEG = 11
    CSS_RAD = 12
    CSS_GRAD = 13
    CSS_MS = 14
    CSS_S = 15
    CSS_HZ = 16
    CSS_KHZ = 17
    CSS_DIMENSION = 18
    CSS_STRING = 19
    CSS_URI = 20
    CSS_IDENT = 21
    CSS_ATTR = 22
    CSS_COUNTER = 23
    CSS_RECT = 24
    CSS_RGBCOLOR = 25
    # NOT OFFICIAL:
    CSS_RGBACOLOR = 26
    CSS_UNICODE_RANGE = 27

    _floattypes = (
        CSS_NUMBER,
        CSS_PERCENTAGE,
        CSS_EMS,
        CSS_EXS,
        CSS_PX,
        CSS_CM,
        CSS_MM,
        CSS_IN,
        CSS_PT,
        CSS_PC,
        CSS_DEG,
        CSS_RAD,
        CSS_GRAD,
        CSS_MS,
        CSS_S,
        CSS_HZ,
        CSS_KHZ,
        CSS_DIMENSION,
    )
    _stringtypes = (CSS_ATTR, CSS_IDENT, CSS_STRING, CSS_URI)
    _countertypes = (CSS_COUNTER,)
    _recttypes = (CSS_RECT,)
    _rbgtypes = (CSS_RGBCOLOR, CSS_RGBACOLOR)
    _lengthtypes = (
        CSS_NUMBER,
        CSS_EMS,
        CSS_EXS,
        CSS_PX,
        CSS_CM,
        CSS_MM,
        CSS_IN,
        CSS_PT,
        CSS_PC,
    )

    # oldtype: newType: converterfunc
    _converter = {
        # cm <-> mm <-> in, 1 inch is equal to 2.54 centimeters.
        # pt <-> pc, the points used by CSS 2.1 are equal to 1/72nd of an inch.
        # pc: picas - 1 pica is equal to 12 points
        (CSS_CM, CSS_MM): lambda x: x * 10,
        (CSS_MM, CSS_CM): lambda x: x / 10,
        (CSS_PT, CSS_PC): lambda x: x * 12,
        (CSS_PC, CSS_PT): lambda x: x / 12,
        (CSS_CM, CSS_IN): lambda x: x / 2.54,
        (CSS_IN, CSS_CM): lambda x: x * 2.54,
        (CSS_MM, CSS_IN): lambda x: x / 25.4,
        (CSS_IN, CSS_MM): lambda x: x * 25.4,
        (CSS_IN, CSS_PT): lambda x: x / 72,
        (CSS_PT, CSS_IN): lambda x: x * 72,
        (CSS_CM, CSS_PT): lambda x: x / 2.54 / 72,
        (CSS_PT, CSS_CM): lambda x: x * 72 * 2.54,
        (CSS_MM, CSS_PT): lambda x: x / 25.4 / 72,
        (CSS_PT, CSS_MM): lambda x: x * 72 * 25.4,
        (CSS_IN, CSS_PC): lambda x: x / 72 / 12,
        (CSS_PC, CSS_IN): lambda x: x * 12 * 72,
        (CSS_CM, CSS_PC): lambda x: x / 2.54 / 72 / 12,
        (CSS_PC, CSS_CM): lambda x: x * 12 * 72 * 2.54,
        (CSS_MM, CSS_PC): lambda x: x / 25.4 / 72 / 12,
        (CSS_PC, CSS_MM): lambda x: x * 12 * 72 * 25.4,
        # hz <-> khz
        (CSS_KHZ, CSS_HZ): lambda x: x * 1000,
        (CSS_HZ, CSS_KHZ): lambda x: x / 1000,
        # s <-> ms
        (CSS_S, CSS_MS): lambda x: x * 1000,
        (CSS_MS, CSS_S): lambda x: x / 1000,
        (CSS_RAD, CSS_DEG): lambda x: math.degrees(x),
        (CSS_DEG, CSS_RAD): lambda x: math.radians(x),
        # TODO: convert grad <-> deg or rad
        # (CSS_RAD, CSS_GRAD): lambda x: math.degrees(x),
        # (CSS_DEG, CSS_GRAD): lambda x: math.radians(x),
        # (CSS_GRAD, CSS_RAD): lambda x: math.radians(x),
        # (CSS_GRAD, CSS_DEG): lambda x: math.radians(x)
    }

    def __init__(self, cssText=None, parent=None, readonly=False):
        """See CSSPrimitiveValue.__init__()"""
        super(CSSPrimitiveValue, self).__init__(
            cssText=cssText, parent=parent, readonly=readonly
        )

    def __str__(self):
        return "<cssutils.css.%s object primitiveType=%s cssText=%r at 0x%x>" % (
            self.__class__.__name__,
            self.primitiveTypeString,
            self.cssText,
            id(self),
        )

    _unitnames = [
        'CSS_UNKNOWN',
        'CSS_NUMBER',
        'CSS_PERCENTAGE',
        'CSS_EMS',
        'CSS_EXS',
        'CSS_PX',
        'CSS_CM',
        'CSS_MM',
        'CSS_IN',
        'CSS_PT',
        'CSS_PC',
        'CSS_DEG',
        'CSS_RAD',
        'CSS_GRAD',
        'CSS_MS',
        'CSS_S',
        'CSS_HZ',
        'CSS_KHZ',
        'CSS_DIMENSION',
        'CSS_STRING',
        'CSS_URI',
        'CSS_IDENT',
        'CSS_ATTR',
        'CSS_COUNTER',
        'CSS_RECT',
        'CSS_RGBCOLOR',
        'CSS_RGBACOLOR',
        'CSS_UNICODE_RANGE',
    ]

    _reNumDim = re.compile(r'([+-]?\d*\.\d+|[+-]?\d+)(.*)$', re.I | re.U | re.X)

    def _unitDIMENSION(value):
        """Check val for dimension name."""
        units = {
            'em': 'CSS_EMS',
            'ex': 'CSS_EXS',
            'px': 'CSS_PX',
            'cm': 'CSS_CM',
            'mm': 'CSS_MM',
            'in': 'CSS_IN',
            'pt': 'CSS_PT',
            'pc': 'CSS_PC',
            'deg': 'CSS_DEG',
            'rad': 'CSS_RAD',
            'grad': 'CSS_GRAD',
            'ms': 'CSS_MS',
            's': 'CSS_S',
            'hz': 'CSS_HZ',
            'khz': 'CSS_KHZ',
        }
        val, dim = CSSPrimitiveValue._reNumDim.findall(
            cssutils.helper.normalize(value)
        )[0]
        return units.get(dim, 'CSS_DIMENSION')

    def _unitFUNCTION(value):
        """Check val for function name."""
        units = {
            'attr(': 'CSS_ATTR',
            'counter(': 'CSS_COUNTER',
            'rect(': 'CSS_RECT',
            'rgb(': 'CSS_RGBCOLOR',
            'rgba(': 'CSS_RGBACOLOR',
        }
        return units.get(
            re.findall(r'^(.*?\()', cssutils.helper.normalize(value.cssText), re.U)[0],
            'CSS_UNKNOWN',
        )

    __unitbytype = {
        __types.NUMBER: 'CSS_NUMBER',
        __types.PERCENTAGE: 'CSS_PERCENTAGE',
        __types.STRING: 'CSS_STRING',
        __types.UNICODE_RANGE: 'CSS_UNICODE_RANGE',
        __types.URI: 'CSS_URI',
        __types.IDENT: 'CSS_IDENT',
        __types.HASH: 'CSS_RGBCOLOR',
        __types.DIMENSION: _unitDIMENSION,
        __types.FUNCTION: _unitFUNCTION,
    }

    def __set_primitiveType(self):
        """primitiveType is readonly but is set lazy if accessed"""
        # TODO: check unary and font-family STRING a, b, "c"
        val, type_ = self._value
        # try get by type_
        pt = self.__unitbytype.get(type_, 'CSS_UNKNOWN')
        if callable(pt):
            # multiple options, check value too
            pt = pt(val)
        self._primitiveType = getattr(self, pt)

    def _getPrimitiveType(self):
        if not hasattr(self, '_primitivetype'):
            self.__set_primitiveType()
        return self._primitiveType

    primitiveType = property(
        _getPrimitiveType,
        doc="(readonly) The type of the value as defined "
        "by the constants in this class.",
    )

    def _getPrimitiveTypeString(self):
        return self._unitnames[self.primitiveType]

    primitiveTypeString = property(
        _getPrimitiveTypeString, doc="Name of primitive type of this value."
    )

    def _getCSSPrimitiveTypeString(self, type):
        "get TypeString by given type which may be unknown, used by setters"
        try:
            return self._unitnames[type]
        except (IndexError, TypeError):
            return '%r (UNKNOWN TYPE)' % type

    def _getNumDim(self, value=None):
        "Split self._value in numerical and dimension part."
        if value is None:
            value = cssutils.helper.normalize(self._value[0])

        try:
            val, dim = CSSPrimitiveValue._reNumDim.findall(value)[0]
        except IndexError:
            val, dim = value, ''
        try:
            val = float(val)
            if val == int(val):
                val = int(val)
        except ValueError:
            raise xml.dom.InvalidAccessErr(
                'CSSPrimitiveValue: No float value %r' % self._value[0]
            )

        return val, dim

    def getFloatValue(self, unitType=None):
        """(DOM) This method is used to get a float value in a
        specified unit. If this CSS value doesn't contain a float value
        or can't be converted into the specified unit, a DOMException
        is raised.

        :param unitType:
            to get the float value. The unit code can only be a float unit type
            (i.e. CSS_NUMBER, CSS_PERCENTAGE, CSS_EMS, CSS_EXS, CSS_PX, CSS_CM,
            CSS_MM, CSS_IN, CSS_PT, CSS_PC, CSS_DEG, CSS_RAD, CSS_GRAD, CSS_MS,
            CSS_S, CSS_HZ, CSS_KHZ, CSS_DIMENSION) or None in which case
            the current dimension is used.

        :returns:
            not necessarily a float but some cases just an integer
            e.g. if the value is ``1px`` it return ``1`` and **not** ``1.0``

            Conversions might return strange values like 1.000000000001
        """
        if unitType is not None and unitType not in self._floattypes:
            raise xml.dom.InvalidAccessErr('unitType Parameter is not a float type')

        val, dim = self._getNumDim()

        if unitType is not None and self.primitiveType != unitType:
            # convert if needed
            try:
                val = self._converter[self.primitiveType, unitType](val)
            except KeyError:
                raise xml.dom.InvalidAccessErr(
                    'CSSPrimitiveValue: Cannot coerce primitiveType %r to %r'
                    % (
                        self.primitiveTypeString,
                        self._getCSSPrimitiveTypeString(unitType),
                    )
                )

        if val == int(val):
            val = int(val)

        return val

    def setFloatValue(self, unitType, floatValue):
        """(DOM) A method to set the float value with a specified unit.
        If the property attached with this value can not accept the
        specified unit or the float value, the value will be unchanged and
        a DOMException will be raised.

        :param unitType:
            a unit code as defined above. The unit code can only be a float
            unit type
        :param floatValue:
            the new float value which does not have to be a float value but
            may simple be an int e.g. if setting::

                setFloatValue(CSS_PX, 1)

        :exceptions:
            - :exc:`~xml.dom.InvalidAccessErr`:
              Raised if the attached property doesn't
              support the float value or the unit type.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this property is readonly.
        """
        self._checkReadonly()
        if unitType not in self._floattypes:
            raise xml.dom.InvalidAccessErr(
                'CSSPrimitiveValue: unitType %r is not a float type'
                % self._getCSSPrimitiveTypeString(unitType)
            )
        try:
            val = float(floatValue)
        except ValueError:
            raise xml.dom.InvalidAccessErr(
                'CSSPrimitiveValue: floatValue %r is not a float' % floatValue
            )

        oldval, dim = self._getNumDim()
        if self.primitiveType != unitType:
            # convert if possible
            try:
                val = self._converter[unitType, self.primitiveType](val)
            except KeyError:
                raise xml.dom.InvalidAccessErr(
                    'CSSPrimitiveValue: Cannot coerce primitiveType %r to %r'
                    % (
                        self.primitiveTypeString,
                        self._getCSSPrimitiveTypeString(unitType),
                    )
                )

        if val == int(val):
            val = int(val)

        self.cssText = '%s%s' % (val, dim)

    def getStringValue(self):
        """(DOM) This method is used to get the string value. If the
        CSS value doesn't contain a string value, a DOMException is raised.

        Some properties (like 'font-family' or 'voice-family')
        convert a whitespace separated list of idents to a string.

        Only the actual value is returned so e.g. all the following return the
        actual value ``a``: url(a), attr(a), "a", 'a'
        """
        if self.primitiveType not in self._stringtypes:
            raise xml.dom.InvalidAccessErr(
                'CSSPrimitiveValue %r is not a string type' % self.primitiveTypeString
            )

        if CSSPrimitiveValue.CSS_ATTR == self.primitiveType:
            return self._value[0].cssText[5:-1]
        else:
            return self._value[0]

    def setStringValue(self, stringType, stringValue):
        """(DOM) A method to set the string value with the specified
        unit. If the property attached to this value can't accept the
        specified unit or the string value, the value will be unchanged and
        a DOMException will be raised.

        :param stringType:
            a string code as defined above. The string code can only be a
            string unit type (i.e. CSS_STRING, CSS_URI, CSS_IDENT, and
            CSS_ATTR).
        :param stringValue:
            the new string value
            Only the actual value is expected so for (CSS_URI, "a") the
            new value will be ``url(a)``. For (CSS_STRING, "'a'")
            the new value will be ``"\\'a\\'"`` as the surrounding ``'`` are
            not part of the string value

        :exceptions:
            - :exc:`~xml.dom.InvalidAccessErr`:
              Raised if the CSS value doesn't contain a
              string value or if the string value can't be converted into
              the specified unit.

            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this property is readonly.
        """
        self._checkReadonly()
        # self not stringType
        if self.primitiveType not in self._stringtypes:
            raise xml.dom.InvalidAccessErr(
                'CSSPrimitiveValue %r is not a string type' % self.primitiveTypeString
            )
        # given stringType is no StringType
        if stringType not in self._stringtypes:
            raise xml.dom.InvalidAccessErr(
                'CSSPrimitiveValue: stringType %s is not a string type'
                % self._getCSSPrimitiveTypeString(stringType)
            )

        if self._primitiveType != stringType:
            raise xml.dom.InvalidAccessErr(
                'CSSPrimitiveValue: Cannot coerce primitiveType %r to %r'
                % (
                    self.primitiveTypeString,
                    self._getCSSPrimitiveTypeString(stringType),
                )
            )

        if CSSPrimitiveValue.CSS_STRING == self._primitiveType:
            self.cssText = cssutils.helper.string(stringValue)
        elif CSSPrimitiveValue.CSS_URI == self._primitiveType:
            self.cssText = cssutils.helper.uri(stringValue)
        elif CSSPrimitiveValue.CSS_ATTR == self._primitiveType:
            self.cssText = 'attr(%s)' % stringValue
        else:
            self.cssText = stringValue
        self._primitiveType = stringType

    def getCounterValue(self):
        """(DOM) This method is used to get the Counter value. If
        this CSS value doesn't contain a counter value, a DOMException
        is raised. Modification to the corresponding style property
        can be achieved using the Counter interface.

        **Not implemented.**
        """
        if not self.CSS_COUNTER == self.primitiveType:
            raise xml.dom.InvalidAccessErr('Value is not a counter type')
        # TODO: use Counter class
        raise NotImplementedError()

    def getRGBColorValue(self):
        """(DOM) This method is used to get the RGB color. If this
        CSS value doesn't contain a RGB color value, a DOMException
        is raised. Modification to the corresponding style property
        can be achieved using the RGBColor interface.
        """
        if self.primitiveType not in self._rbgtypes:
            raise xml.dom.InvalidAccessErr('Value is not a RGBColor value')
        return RGBColor(self._value[0])

    def getRectValue(self):
        """(DOM) This method is used to get the Rect value. If this CSS
        value doesn't contain a rect value, a DOMException is raised.
        Modification to the corresponding style property can be achieved
        using the Rect interface.

        **Not implemented.**
        """
        if self.primitiveType not in self._recttypes:
            raise xml.dom.InvalidAccessErr('value is not a Rect value')
        # TODO: use Rect class
        raise NotImplementedError()

    def _getCssText(self):
        """Overwrites CSSValue."""
        return cssutils.ser.do_css_CSSPrimitiveValue(self)

    def _setCssText(self, cssText):
        """Use CSSValue."""
        return super(CSSPrimitiveValue, self)._setCssText(cssText)

    cssText = property(
        _getCssText, _setCssText, doc="A string representation of the current value."
    )


class CSSValueList(CSSValue):
    """The CSSValueList interface provides the abstraction of an ordered
    collection of CSS values.

    Some properties allow an empty list into their syntax. In that case,
    these properties take the none identifier. So, an empty list means
    that the property has the value none.

    The items in the CSSValueList are accessible via an integral index,
    starting from 0.
    """

    cssValueType = CSSValue.CSS_VALUE_LIST

    def __init__(self, cssText=None, parent=None, readonly=False):
        """Init a new CSSValueList"""
        super(CSSValueList, self).__init__(
            cssText=cssText, parent=parent, readonly=readonly
        )
        self._items = []

    def __iter__(self):
        "CSSValueList is iterable."
        for item in self.__items():
            yield item.value

    def __str__(self):
        return (
            "<cssutils.css.%s object cssValueType=%r cssText=%r length=%r "
            "at 0x%x>"
            % (
                self.__class__.__name__,
                self.cssValueTypeString,
                self.cssText,
                self.length,
                id(self),
            )
        )

    def __items(self):
        return [item for item in self._seq if isinstance(item.value, CSSValue)]

    def item(self, index):
        """(DOM) Retrieve a CSSValue by ordinal `index`. The
        order in this collection represents the order of the values in the
        CSS style property. If `index` is greater than or equal to the number
        of values in the list, this returns ``None``.
        """
        try:
            return self.__items()[index].value
        except IndexError:
            return None

    length = property(
        lambda self: len(self.__items()),
        doc="(DOM attribute) The number of CSSValues in the " "list.",
    )


class CSSFunction(CSSPrimitiveValue):
    """A CSS function value like rect() etc."""

    _functionName = 'CSSFunction'
    primitiveType = CSSPrimitiveValue.CSS_UNKNOWN

    def __init__(self, cssText=None, parent=None, readonly=False):
        """
        Init a new CSSFunction

        :param cssText:
            the parsable cssText of the value
        :param readonly:
            defaults to False
        """
        super(CSSFunction, self).__init__(parent=parent)
        self._funcType = None
        self.valid = False
        self.wellformed = False
        if cssText is not None:
            self.cssText = cssText
        self._readonly = readonly

    def _productiondefinition(self):
        """Return definition used for parsing."""
        types = self._prods  # rename!

        value = Sequence(
            PreDef.unary(),
            Prod(
                name='PrimitiveValue',
                match=lambda t, v: t
                in (
                    types.DIMENSION,
                    types.HASH,
                    types.IDENT,
                    types.NUMBER,
                    types.PERCENTAGE,
                    types.STRING,
                ),
                toSeq=lambda t, tokens: (t[0], CSSPrimitiveValue(t[1])),
            ),
        )
        valueOrFunc = Choice(
            value,
            # FUNC is actually not in spec but used in e.g. Prince
            PreDef.function(
                toSeq=lambda t, tokens: (
                    'FUNCTION',
                    CSSFunction(cssutils.helper.pushtoken(t, tokens)),
                )
            ),
        )
        funcProds = Sequence(
            Prod(
                name='FUNC',
                match=lambda t, v: t == types.FUNCTION,
                toSeq=lambda t, tokens: (t[0], cssutils.helper.normalize(t[1])),
            ),
            Choice(
                Sequence(
                    valueOrFunc,
                    # more values starting with Comma
                    # should use store where colorType is saved to
                    # define min and may, closure?
                    Sequence(PreDef.comma(), valueOrFunc, minmax=lambda: (0, None)),
                    PreDef.funcEnd(stop=True),
                ),
                PreDef.funcEnd(stop=True),
            ),
        )
        return funcProds

    def _setCssText(self, cssText):
        self._checkReadonly()
        # store: colorType, parts
        wellformed, seq, store, unusedtokens = ProdParser().parse(
            cssText, self._functionName, self._productiondefinition(), keepS=True
        )
        if wellformed:
            # combine +/- and following CSSPrimitiveValue, remove S
            newseq = self._tempSeq()
            i, end = 0, len(seq)
            while i < end:
                item = seq[i]
                if item.type == self._prods.S:
                    pass
                elif item.value == '+' or item.value == '-':
                    i += 1
                    next = seq[i]
                    newval = next.value
                    if isinstance(newval, CSSPrimitiveValue):
                        newval.setFloatValue(
                            newval.primitiveType,
                            float(item.value + str(newval.getFloatValue())),
                        )
                        newseq.append(newval, next.type, item.line, item.col)
                    else:
                        # expressions only?
                        newseq.appendItem(item)
                        newseq.appendItem(next)
                else:
                    newseq.appendItem(item)

                i += 1

            self.wellformed = True
            self._setSeq(newseq)
            self._funcType = newseq[0].value

    cssText = property(
        lambda self: cssutils.ser.do_css_FunctionValue(self), _setCssText
    )

    funcType = property(lambda self: self._funcType)


class RGBColor(CSSFunction):
    """A CSS color like RGB, RGBA or a simple value like `#000` or `red`."""

    _functionName = 'Function rgb()'

    def __init__(self, cssText=None, parent=None, readonly=False):
        """
        Init a new RGBColor

        :param cssText:
            the parsable cssText of the value
        :param readonly:
            defaults to False
        """
        super(CSSFunction, self).__init__(parent=parent)
        self._colorType = None
        self.valid = False
        self.wellformed = False
        if cssText is not None:
            try:
                # if it is a Function object
                cssText = cssText.cssText
            except AttributeError:
                pass
            self.cssText = cssText

        self._readonly = readonly

    def __repr__(self):
        return "cssutils.css.%s(%r)" % (self.__class__.__name__, self.cssText)

    def __str__(self):
        return "<cssutils.css.%s object colorType=%r cssText=%r at 0x%x>" % (
            self.__class__.__name__,
            self.colorType,
            self.cssText,
            id(self),
        )

    def _setCssText(self, cssText):
        self._checkReadonly()
        types = self._prods  # rename!
        valueProd = Prod(
            name='value',
            match=lambda t, v: t in (types.NUMBER, types.PERCENTAGE),
            toSeq=lambda t, v: (CSSPrimitiveValue, CSSPrimitiveValue(v)),
            toStore='parts',
        )
        # COLOR PRODUCTION
        funccolor = Sequence(
            Prod(
                name='FUNC',
                match=lambda t, v: t == types.FUNCTION
                and cssutils.helper.normalize(v) in ('rgb(', 'rgba(', 'hsl(', 'hsla('),
                toSeq=lambda t, v: (t, v),  # cssutils.helper.normalize(v)),
                toStore='colorType',
            ),
            PreDef.unary(),
            valueProd,
            # 2 or 3 more values starting with Comma
            Sequence(PreDef.comma(), PreDef.unary(), valueProd, minmax=lambda: (2, 3)),
            PreDef.funcEnd(),
        )
        colorprods = Choice(
            funccolor,
            PreDef.hexcolor('colorType'),
            Prod(
                name='named color',
                match=lambda t, v: t == types.IDENT,
                toStore='colorType',
            ),
        )
        # store: colorType, parts
        wellformed, seq, store, unusedtokens = ProdParser().parse(
            cssText, 'RGBColor', colorprods, keepS=True, store={'parts': []}
        )

        if wellformed:
            self.wellformed = True
            if store['colorType'].type == self._prods.HASH:
                self._colorType = 'HEX'
            elif store['colorType'].type == self._prods.IDENT:
                self._colorType = 'Named Color'
            else:
                self._colorType = store['colorType'].value[:-1]
                # self._colorType = \
                # cssutils.helper.normalize(store['colorType'].value)[:-1]

            self._setSeq(seq)

    cssText = property(lambda self: cssutils.ser.do_css_RGBColor(self), _setCssText)

    colorType = property(lambda self: self._colorType)


class CalcValue(CSSFunction):
    """Calc Function"""

    _functionName = 'Function calc()'

    def _productiondefinition(self):
        """Return defintion used for parsing."""
        types = self._prods  # rename!

        def toSeq(t, tokens):
            "Do not normalize function name!"
            return t[0], t[1]

        funcProds = Sequence(
            Prod(name='calc', match=lambda t, v: t == types.FUNCTION, toSeq=toSeq),
            Sequence(
                Choice(
                    Prod(
                        name='nested function',
                        match=lambda t, v: t == self._prods.FUNCTION,
                        toSeq=lambda t, tokens: (
                            CSSFunction._functionName,
                            CSSFunction(cssutils.helper.pushtoken(t, tokens)),
                        ),
                    ),
                    Prod(
                        name='part',
                        match=lambda t, v: v != ')',
                        toSeq=lambda t, tokens: (t[0], t[1]),
                    ),
                ),
                minmax=lambda: (0, None),
            ),
            PreDef.funcEnd(stop=True),
        )
        return funcProds

    def _getCssText(self):
        return cssutils.ser.do_css_CalcValue(self)

    def _setCssText(self, cssText):
        return super(CalcValue, self)._setCssText(cssText)

    cssText = property(
        _getCssText, _setCssText, doc="A string representation of the current value."
    )


class ExpressionValue(CSSFunction):
    """Special IE only CSSFunction which may contain *anything*.
    Used for expressions and ``alpha(opacity=100)`` currently."""

    _functionName = 'Expression (IE only)'

    def _productiondefinition(self):
        """Return defintion used for parsing."""
        types = self._prods  # rename!

        def toSeq(t, tokens):
            "Do not normalize function name!"
            return t[0], t[1]

        funcProds = Sequence(
            Prod(
                name='expression', match=lambda t, v: t == types.FUNCTION, toSeq=toSeq
            ),
            Sequence(
                Choice(
                    Prod(
                        name='nested function',
                        match=lambda t, v: t == self._prods.FUNCTION,
                        toSeq=lambda t, tokens: (
                            ExpressionValue._functionName,
                            ExpressionValue(cssutils.helper.pushtoken(t, tokens)),
                        ),
                    ),
                    Prod(
                        name='part',
                        match=lambda t, v: v != ')',
                        toSeq=lambda t, tokens: (t[0], t[1]),
                    ),
                ),
                minmax=lambda: (0, None),
            ),
            PreDef.funcEnd(stop=True),
        )
        return funcProds

    def _getCssText(self):
        return cssutils.ser.do_css_ExpressionValue(self)

    def _setCssText(self, cssText):
        # self._log.warn(u'CSSValue: Unoffial and probably invalid MS value used!')
        return super(ExpressionValue, self)._setCssText(cssText)

    cssText = property(
        _getCssText, _setCssText, doc="A string representation of the current value."
    )


class CSSVariable(CSSValue):
    """The CSSVariable represents a call to CSS Variable."""

    def __init__(self, cssText=None, parent=None, readonly=False):
        """Init a new CSSVariable.

        :param cssText:
            the parsable cssText of the value, e.g. ``var(x)``
        :param readonly:
            defaults to False
        """
        self._name = None
        super(CSSVariable, self).__init__(
            cssText=cssText, parent=parent, readonly=readonly
        )

    def __repr__(self):
        return "cssutils.css.%s(%r)" % (self.__class__.__name__, self.cssText)

    def __str__(self):
        return "<cssutils.css.%s object name=%r value=%r at 0x%x>" % (
            self.__class__.__name__,
            self.name,
            self.value,
            id(self),
        )

    def _setCssText(self, cssText):
        self._checkReadonly()

        types = self._prods  # rename!

        funcProds = Sequence(
            Prod(name='var', match=lambda t, v: t == types.FUNCTION),
            PreDef.ident(toStore='ident'),
            PreDef.funcEnd(stop=True),
        )

        # store: name of variable
        store = {'ident': None}
        wellformed, seq, store, unusedtokens = ProdParser().parse(
            cssText, 'CSSVariable', funcProds, keepS=True
        )
        if wellformed:
            self._name = store['ident'].value
            self._setSeq(seq)
            self.wellformed = True

    cssText = property(
        lambda self: cssutils.ser.do_css_CSSVariable(self),
        _setCssText,
        doc="A string representation of the current variable.",
    )

    cssValueType = CSSValue.CSS_VARIABLE

    # TODO: writable? check if var (value) available?
    name = property(lambda self: self._name)

    def _getValue(self):
        "Find contained sheet and @variables there"
        try:
            variables = self.parent.parent.parentRule.parentStyleSheet.variables
        except AttributeError:
            return None
        else:
            try:
                return variables[self.name]
            except KeyError:
                return None

    value = property(_getValue)
