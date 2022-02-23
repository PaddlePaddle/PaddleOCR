"""CSSVariablesDeclaration
http://disruptive-innovations.com/zoo/cssvariables/#mozTocId496530
"""
__all__ = ['CSSVariablesDeclaration']

from cssutils.prodparser import PreDef, Prod, ProdParser, Sequence
from cssutils.helper import normalize
from .value import PropertyValue
import cssutils
import itertools


class CSSVariablesDeclaration(cssutils.util._NewBase):
    """The CSSVariablesDeclaration interface represents a single block of
    variable declarations.
    """

    def __init__(self, cssText='', parentRule=None, readonly=False):
        """
        :param cssText:
            Shortcut, sets CSSVariablesDeclaration.cssText
        :param parentRule:
            The CSS rule that contains this declaration block or
            None if this CSSVariablesDeclaration is not attached to a CSSRule.
        :param readonly:
            defaults to False

        Format::

            variableset
                : vardeclaration [ ';' S* vardeclaration ]* S*
                ;

            vardeclaration
                : varname ':' S* term
                ;

            varname
                : IDENT S*
                ;
        """
        super(CSSVariablesDeclaration, self).__init__()
        self._parentRule = parentRule
        self._vars = {}
        if cssText:
            self.cssText = cssText

        self._readonly = readonly

    def __repr__(self):
        return "cssutils.css.%s(cssText=%r)" % (self.__class__.__name__, self.cssText)

    def __str__(self):
        return "<cssutils.css.%s object length=%r at 0x%x>" % (
            self.__class__.__name__,
            self.length,
            id(self),
        )

    def __contains__(self, variableName):
        """Check if a variable is in variable declaration block.

        :param variableName:
            a string
        """
        return normalize(variableName) in list(self.keys())

    def __getitem__(self, variableName):
        """Retrieve the value of variable ``variableName`` from this
        declaration.
        """
        return self.getVariableValue(variableName)

    def __setitem__(self, variableName, value):
        self.setVariable(variableName, value)

    def __delitem__(self, variableName):
        return self.removeVariable(variableName)

    def __iter__(self):
        """Iterator of names of set variables."""
        for name in list(self.keys()):
            yield name

    def keys(self):
        """Analoguous to standard dict returns variable names which are set in
        this declaration."""
        return list(self._vars.keys())

    def _getCssText(self):
        """Return serialized property cssText."""
        return cssutils.ser.do_css_CSSVariablesDeclaration(self)

    def _setCssText(self, cssText):
        """Setting this attribute will result in the parsing of the new value
        and resetting of all the properties in the declaration block
        including the removal or addition of properties.

        :exceptions:
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this declaration is readonly or a property is readonly.
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified CSS string value has a syntax error and
              is unparsable.

        Format::

            variableset
            : vardeclaration [ ';' S* vardeclaration ]*
            ;

            vardeclaration
            : varname ':' S* term
            ;

            varname
            : IDENT S*
            ;

            expr
            : [ VARCALL | term ] [ operator [ VARCALL | term ] ]*
            ;

        """
        self._checkReadonly()

        vardeclaration = Sequence(
            PreDef.ident(),
            PreDef.char(':', ':', toSeq=False, optional=True),
            # PreDef.S(toSeq=False, optional=True),
            Prod(
                name='term',
                match=lambda t, v: True,
                toSeq=lambda t, tokens: (
                    'value',
                    PropertyValue(itertools.chain([t], tokens), parent=self),
                ),
            ),
        )
        prods = Sequence(
            vardeclaration,
            Sequence(
                PreDef.S(optional=True),
                PreDef.char(';', ';', toSeq=False, optional=True),
                PreDef.S(optional=True),
                vardeclaration,
                minmax=lambda: (0, None),
            ),
            PreDef.S(optional=True),
            PreDef.char(';', ';', toSeq=False, optional=True),
        )
        # parse
        wellformed, seq, store, notused = ProdParser().parse(
            cssText, 'CSSVariableDeclaration', prods, emptyOk=True
        )
        if wellformed:
            newseq = self._tempSeq()
            newvars = {}

            # seq contains only name: value pairs plus comments etc
            nameitem = None
            for item in seq:
                if 'IDENT' == item.type:
                    nameitem = item
                elif 'value' == item.type:
                    nname = normalize(nameitem.value)
                    if nname in newvars:
                        # replace var with same name
                        for i, it in enumerate(newseq):
                            if normalize(it.value[0]) == nname:
                                newseq.replace(
                                    i,
                                    (nameitem.value, item.value),
                                    'var',
                                    nameitem.line,
                                    nameitem.col,
                                )
                    else:
                        # saved non normalized name for reserialization
                        newseq.append(
                            (nameitem.value, item.value),
                            'var',
                            nameitem.line,
                            nameitem.col,
                        )

                    #                    newseq.append((nameitem.value, item.value),
                    #                                  'var',
                    #                                  nameitem.line, nameitem.col)

                    newvars[nname] = item.value

                else:
                    newseq.appendItem(item)

            self._setSeq(newseq)
            self._vars = newvars
            self.wellformed = True

    cssText = property(
        _getCssText,
        _setCssText,
        doc="(DOM) A parsable textual representation of the declaration "
        "block excluding the surrounding curly braces.",
    )

    def _setParentRule(self, parentRule):
        self._parentRule = parentRule

    parentRule = property(
        lambda self: self._parentRule,
        _setParentRule,
        doc="(DOM) The CSS rule that contains this"
        " declaration block or None if this block"
        " is not attached to a CSSRule.",
    )

    def getVariableValue(self, variableName):
        """Used to retrieve the value of a variable if it has been explicitly
        set within this variable declaration block.

        :param variableName:
            The name of the variable.
        :returns:
            the value of the variable if it has been explicitly set in this
            variable declaration block. Returns the empty string if the
            variable has not been set.
        """
        try:
            return self._vars[normalize(variableName)].cssText
        except KeyError:
            return ''

    def removeVariable(self, variableName):
        """Used to remove a variable if it has been explicitly set within this
        variable declaration block.

        :param variableName:
            The name of the variable.
        :returns:
            the value of the variable if it has been explicitly set for this
            variable declaration block. Returns the empty string if the
            variable has not been set.

        :exceptions:
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this declaration is readonly is readonly.
        """
        normalname = variableName
        try:
            r = self._vars[normalname]
        except KeyError:
            return ''
        else:
            self.seq._readonly = False
            if normalname in self._vars:
                for i, x in enumerate(self.seq):
                    if x.value[0] == variableName:
                        del self.seq[i]
            self.seq._readonly = True
            del self._vars[normalname]

        return r.cssText

    def setVariable(self, variableName, value):
        """Used to set a variable value within this variable declaration block.

        :param variableName:
            The name of the CSS variable.
        :param value:
            The new value of the variable, may also be a PropertyValue object.

        :exceptions:
            - :exc:`~xml.dom.SyntaxErr`:
              Raised if the specified value has a syntax error and is
              unparsable.
            - :exc:`~xml.dom.NoModificationAllowedErr`:
              Raised if this declaration is readonly or the property is
              readonly.
        """
        self._checkReadonly()

        # check name
        wellformed, seq, store, unused = ProdParser().parse(
            normalize(variableName), 'variableName', Sequence(PreDef.ident())
        )
        if not wellformed:
            self._log.error('Invalid variableName: %r: %r' % (variableName, value))
        else:
            # check value
            if isinstance(value, PropertyValue):
                v = value
            else:
                v = PropertyValue(cssText=value, parent=self)

            if not v.wellformed:
                self._log.error(
                    'Invalid variable value: %r: %r' % (variableName, value)
                )
            else:
                # update seq
                self.seq._readonly = False

                variableName = normalize(variableName)

                if variableName in self._vars:
                    for i, x in enumerate(self.seq):
                        if x.value[0] == variableName:
                            self.seq.replace(
                                i, [variableName, v], x.type, x.line, x.col
                            )
                            break
                else:
                    self.seq.append([variableName, v], 'var')
                self.seq._readonly = True
                self._vars[variableName] = v

    def item(self, index):
        """Used to retrieve the variables that have been explicitly set in
        this variable declaration block. The order of the variables
        retrieved using this method does not have to be the order in which
        they were set. This method can be used to iterate over all variables
        in this variable declaration block.

        :param index:
            of the variable name to retrieve, negative values behave like
            negative indexes on Python lists, so -1 is the last element

        :returns:
            The name of the variable at this ordinal position. The empty
            string if no variable exists at this position.
        """
        try:
            return list(self.keys())[index]
        except IndexError:
            return ''

    length = property(
        lambda self: len(self._vars),
        doc="The number of variables that have been explicitly set in this"
        " variable declaration block. The range of valid indices is 0"
        " to length-1 inclusive.",
    )
