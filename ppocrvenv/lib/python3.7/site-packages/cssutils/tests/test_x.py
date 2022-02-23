"""Testcases for cssutils.css.CSSValue and CSSPrimitiveValue."""

# from decimal import Decimal # maybe for later tests?
from . import basetest
import cssutils


class XTestCase(basetest.BaseTestCase):
    def setUp(self):
        cssutils.ser.prefs.useDefaults()

    def tearDown(self):
        cssutils.ser.prefs.useDefaults()

    def test_prioriy(self):
        "Property.priority"
        s = cssutils.parseString('a { color: red }')
        self.assertEqual(s.cssText, 'a {\n    color: red\n    }'.encode())


#        self.assertEqual(u'', s.cssRules[0].style.getPropertyPriority('color'))
#
#        s = cssutils.parseString('a { color: red !important }')
#        self.assertEqual(u'a {\n    color: red !important\n    }', s.cssText)
#        self.assertEqual(
#            u'important', s.cssRules[0].style.getPropertyPriority('color'))
#
#        cssutils.log.raiseExceptions = True
#        p = cssutils.css.Property(u'color', u'red', u'')
#        self.assertEqual(p.priority, u'')
#        p = cssutils.css.Property(u'color', u'red', u'!important')
#        self.assertEqual(p.priority, u'important')
#        self.assertRaisesMsg(xml.dom.SyntaxErr,
#                             u'',
#                             cssutils.css.Property, u'color', u'red', u'x')
#
#        cssutils.log.raiseExceptions = False
#        p = cssutils.css.Property(u'color', u'red', u'!x')
#        self.assertEqual(p.priority, u'x')
#        p = cssutils.css.Property(u'color', u'red', u'!x')
#        self.assertEqual(p.priority, u'x')
#        cssutils.log.raiseExceptions = True
#
#
#        # invalid but kept!
#        #cssutils.log.raiseExceptions = False
#        s = cssutils.parseString('a { color: red !x }')
#        self.assertEqual(u'a {\n    color: red !x\n    }', s.cssText)
#        self.assertEqual(u'x', s.cssRules[0].style.getPropertyPriority('color'))
#

if __name__ == '__main__':
    import unittest

    unittest.main()
