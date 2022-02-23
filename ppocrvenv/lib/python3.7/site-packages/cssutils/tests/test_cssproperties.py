"""Testcases for cssutils.css.cssproperties."""

from . import basetest
import cssutils.css
import cssutils.profiles


class CSSPropertiesTestCase(basetest.BaseTestCase):

    #    def test_cssvalues(self):
    #        "cssproperties cssvalues"
    #        # does actually return match object, so a very simplified test...
    #        match = cssutils.css.cssproperties.cssvalues
    #
    #        self.assertEqual(True, bool(match['color']('red')))
    #        self.assertEqual(False, bool(match['top']('red')))
    #
    #        self.assertEqual(True, bool(match['left']('0')))
    #        self.assertEqual(True, bool(match['left']('1px')))
    #        self.assertEqual(True, bool(match['left']('.1px')))
    #        self.assertEqual(True, bool(match['left']('-1px')))
    #        self.assertEqual(True, bool(match['left']('-.1px')))
    #        self.assertEqual(True, bool(match['left']('-0.1px')))

    def test_toDOMname(self):
        "cssproperties _toDOMname(CSSname)"
        _toDOMname = cssutils.css.cssproperties._toDOMname

        self.assertEqual('color', _toDOMname('color'))
        self.assertEqual('fontStyle', _toDOMname('font-style'))
        self.assertEqual('MozOpacity', _toDOMname('-moz-opacity'))
        self.assertEqual('UNKNOWN', _toDOMname('UNKNOWN'))
        self.assertEqual('AnUNKNOWN', _toDOMname('-anUNKNOWN'))

    def test_toCSSname(self):
        "cssproperties _toCSSname(DOMname)"
        _toCSSname = cssutils.css.cssproperties._toCSSname

        self.assertEqual('color', _toCSSname('color'))
        self.assertEqual('font-style', _toCSSname('fontStyle'))
        self.assertEqual('-moz-opacity', _toCSSname('MozOpacity'))
        self.assertEqual('UNKNOWN', _toCSSname('UNKNOWN'))
        self.assertEqual('-anUNKNOWN', _toCSSname('AnUNKNOWN'))

    def test_CSS2Properties(self):
        "CSS2Properties"
        CSS2Properties = cssutils.css.cssproperties.CSS2Properties
        self.assertEqual(type(property()), type(CSS2Properties.color))
        self.assertEqual(
            sum([len(x) for x in list(cssutils.profiles.properties.values())]),
            len(CSS2Properties._properties),
        )

        c2 = CSS2Properties()
        # CSS2Properties has simplified implementation return always None
        self.assertEqual(None, c2.color)
        self.assertEqual(None, c2.__setattr__('color', 1))
        self.assertEqual(None, c2.__delattr__('color'))
        # only defined properties
        self.assertRaises(AttributeError, c2.__getattribute__, 'UNKNOWN')


if __name__ == '__main__':
    import unittest

    unittest.main()
