"""CSS2Properties (partly!) implements DOM Level 2 CSS CSS2Properties used
by CSSStyleDeclaration

TODO: CSS2Properties
    If an implementation does implement this interface, it is expected to
    understand the specific syntax of the shorthand properties, and apply
    their semantics; when the margin property is set, for example, the
    marginTop, marginRight, marginBottom and marginLeft properties are
    actually being set by the underlying implementation.

    When dealing with CSS "shorthand" properties, the shorthand properties
    should be decomposed into their component longhand properties as
    appropriate, and when querying for their value, the form returned
    should be the shortest form exactly equivalent to the declarations made
    in the ruleset. However, if there is no shorthand declaration that
    could be added to the ruleset without changing in any way the rules
    already declared in the ruleset (i.e., by adding longhand rules that
    were previously not declared in the ruleset), then the empty string
    should be returned for the shorthand property.

    For example, querying for the font property should not return
    "normal normal normal 14pt/normal Arial, sans-serif", when
    "14pt Arial, sans-serif" suffices. (The normals are initial values, and
    are implied by use of the longhand property.)

    If the values for all the longhand properties that compose a particular
    string are the initial values, then a string consisting of all the
    initial values should be returned (e.g. a border-width value of
    "medium" should be returned as such, not as "").

    For some shorthand properties that take missing values from other
    sides, such as the margin, padding, and border-[width|style|color]
    properties, the minimum number of sides possible should be used; i.e.,
    "0px 10px" will be returned instead of "0px 10px 0px 10px".

    If the value of a shorthand property can not be decomposed into its
    component longhand properties, as is the case for the font property
    with a value of "menu", querying for the values of the component
    longhand properties should return the empty string.

TODO: CSS2Properties DOMImplementation
    The interface found within this section are not mandatory. A DOM
    application can use the hasFeature method of the DOMImplementation
    interface to determine whether it is supported or not. The feature
    string for this extended interface listed in this section is "CSS2"
    and the version is "2.0".

"""
__all__ = ['CSS2Properties']

import cssutils.profiles
import re


class CSS2Properties(object):
    """The CSS2Properties interface represents a convenience mechanism
    for retrieving and setting properties within a CSSStyleDeclaration.
    The attributes of this interface correspond to all the properties
    specified in CSS2. Getting an attribute of this interface is
    equivalent to calling the getPropertyValue method of the
    CSSStyleDeclaration interface. Setting an attribute of this
    interface is equivalent to calling the setProperty method of the
    CSSStyleDeclaration interface.

    cssutils actually also allows usage of ``del`` to remove a CSS property
    from a CSSStyleDeclaration.

    This is an abstract class, the following functions need to be present
    in inheriting class:

    - ``_getP``
    - ``_setP``
    - ``_delP``
    """

    # actual properties are set after the class definition!
    def _getP(self, CSSname):
        pass

    def _setP(self, CSSname, value):
        pass

    def _delP(self, CSSname):
        pass


_reCSStoDOMname = re.compile('-[a-z]', re.I)


def _toDOMname(CSSname):
    """Returns DOMname for given CSSname e.g. for CSSname 'font-style' returns
    'fontStyle'.
    """

    def _doCSStoDOMname2(m):
        return m.group(0)[1].capitalize()

    return _reCSStoDOMname.sub(_doCSStoDOMname2, CSSname)


_reDOMtoCSSname = re.compile('([A-Z])[a-z]+')


def _toCSSname(DOMname):
    """Return CSSname for given DOMname e.g. for DOMname 'fontStyle' returns
    'font-style'.
    """

    def _doDOMtoCSSname2(m):
        return '-' + m.group(0).lower()

    return _reDOMtoCSSname.sub(_doDOMtoCSSname2, DOMname)


# add list of DOMname properties to CSS2Properties
# used for CSSStyleDeclaration to check if allowed properties
# but somehow doubled, any better way?
CSS2Properties._properties = []
for group in cssutils.profiles.properties:
    for name in cssutils.profiles.properties[group]:
        CSS2Properties._properties.append(_toDOMname(name))


# add CSS2Properties to CSSStyleDeclaration:
def __named_property_def(DOMname):
    """
    Closure to keep name known in each properties accessor function
    DOMname is converted to CSSname here, so actual calls use CSSname.
    """
    CSSname = _toCSSname(DOMname)

    def _get(self):
        return self._getP(CSSname)

    def _set(self, value):
        self._setP(CSSname, value)

    def _del(self):
        self._delP(CSSname)

    return _get, _set, _del


# add all CSS2Properties to CSSStyleDeclaration
for DOMname in CSS2Properties._properties:
    setattr(CSS2Properties, DOMname, property(*__named_property_def(DOMname)))
