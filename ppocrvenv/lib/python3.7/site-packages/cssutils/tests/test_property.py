"""Testcases for cssutils.css.property._Property."""

import xml.dom
from . import basetest
import cssutils


class PropertyTestCase(basetest.BaseTestCase):
    def setUp(self):
        self.r = cssutils.css.property.Property('top', '1px')  # , 'important')

    def test_init(self):
        "Property.__init__()"
        p = cssutils.css.property.Property('top', '1px')
        self.assertEqual('top: 1px', p.cssText)
        self.assertEqual('top', p.literalname)
        self.assertEqual('top', p.name)
        self.assertEqual('1px', p.value)
        # self.assertEqual('1px', p.cssValue.cssText)
        self.assertEqual('1px', p.propertyValue.cssText)
        self.assertEqual('', p.priority)
        self.assertEqual(True, p.valid)
        self.assertEqual(True, p.wellformed)

        self.assertEqual(['top'], p.seqs[0])
        self.assertEqual(
            type(cssutils.css.PropertyValue(cssText="2px")), type(p.seqs[1])
        )
        self.assertEqual([], p.seqs[2])

        self.assertEqual(True, p.valid)

        # Prop of MediaQuery
        p = cssutils.css.property.Property('top', _mediaQuery=True)
        self.assertEqual('top', p.cssText)
        self.assertEqual('top', p.literalname)
        self.assertEqual('top', p.name)
        self.assertEqual('', p.value)
        # self.assertEqual('', p.cssValue.cssText)
        self.assertEqual('', p.propertyValue.cssText)
        self.assertEqual('', p.priority)
        self.assertEqual(False, p.valid)
        # p.cssValue.cssText = '1px'
        p.propertyValue.cssText = '1px'
        self.assertEqual('top: 1px', p.cssText)
        # p.cssValue = ''
        p.propertyValue = ''
        self.assertEqual('top', p.cssText)

        self.assertRaises(xml.dom.SyntaxErr, cssutils.css.property.Property, 'top', '')
        self.assertRaises(xml.dom.SyntaxErr, cssutils.css.property.Property, 'top')
        p = cssutils.css.property.Property('top', '0')
        self.assertEqual('0', p.value)
        self.assertEqual(True, p.wellformed)
        self.assertRaises(xml.dom.SyntaxErr, p._setValue, '')
        self.assertEqual('0', p.value)
        self.assertEqual(True, p.wellformed)

    #        self.assertEqual(True, p.valid)

    #    def test_valid(self):
    #        "Property.valid"
    #        # context property must be set
    #        tests = [
    #            ('color', r'INHe\rIT', True),
    #            ('color', '1', False),
    #            ('color', 'red', True),
    #            ('left', '1', False),
    #            ('left', '1px', True),
    #            ('font', 'normal 1em/1.5 serif', True),
    #            ('background', 'url(x.gif) 1 0', False)
    #            ]
    #        for n, v, exp in tests:
    #            v = cssutils.css.CSSValue(cssText=v)
    #            self.assertTrue(v.wellformed, True)

    def test_cssText(self):
        "Property.cssText"
        p = cssutils.css.property.Property()

        tests = {
            'a: 1': None,
            'a: 1px 2px': None,
            'a: 1 !important': None,
            'a: 1 !IMPORTANT': 'a: 1 !important',
            'a: 1 !impor\\tant': 'a: 1 !important',
            # TODO: important with unicode escapes!
            'font: normal 1em/1.5 serif': None,
            'font: normal 1em/serif': None,
        }
        self.do_equal_r(tests)

        tests = {
            '': (xml.dom.SyntaxErr, 'Property: No property name found: '),
            ':': (xml.dom.SyntaxErr, 'Property: No property name found: : [1:1: :]'),
            'a': (xml.dom.SyntaxErr, 'Property: No ":" after name found: a [1:1: a]'),
            'b !': (
                xml.dom.SyntaxErr,
                'Property: No ":" after name found: b ! [1:3: !]',
            ),
            '/**/x': (
                xml.dom.SyntaxErr,
                'Property: No ":" after name found: /**/x [1:5: x]',
            ),
            'c:': (xml.dom.SyntaxErr, "Property: No property value found: c: [1:2: :]"),
            'd: ': (xml.dom.SyntaxErr, "No content to parse."),
            'e:!important': (xml.dom.SyntaxErr, "No content to parse."),
            'f: 1!': (xml.dom.SyntaxErr, 'Property: Invalid priority: !'),
            'g: 1!importantX': (
                xml.dom.SyntaxErr,
                "Property: No CSS priority value: importantx",
            ),
            # TODO?
            # u'a: 1;': (xml.dom.SyntaxErr,
            #       u'''CSSValue: No match: ('CHAR', u';', 1, 5)''')
        }
        for test in tests:
            ecp, msg = tests[test]
            self.assertRaisesMsg(ecp, msg, p._setCssText, test)

    def test_name(self):
        "Property.name"
        p = cssutils.css.property.Property('top', '1px')
        p.name = 'left'
        self.assertEqual('left', p.name)

        tests = {
            'top': None,
            ' top': 'top',
            'top ': 'top',
            ' top ': 'top',
            '/*x*/ top ': 'top',
            ' top /*x*/': 'top',
            '/*x*/top/*x*/': 'top',
            '\\x': 'x',
            'a\\010': 'a\x10',
            'a\\01': 'a\x01',
        }
        self.do_equal_r(tests, att='name')

        tests = {
            '': xml.dom.SyntaxErr,
            ' ': xml.dom.SyntaxErr,
            '"\n': xml.dom.SyntaxErr,
            '/*x*/': xml.dom.SyntaxErr,
            ':': xml.dom.SyntaxErr,
            ';': xml.dom.SyntaxErr,
            'top:': xml.dom.SyntaxErr,
            'top;': xml.dom.SyntaxErr,
            'color: #xyz': xml.dom.SyntaxErr,
        }
        self.do_raise_r(tests, att='_setName')

        p = cssutils.css.property.Property(r'c\olor', 'red')
        self.assertEqual(r'c\olor', p.literalname)
        self.assertEqual('color', p.name)

    def test_literalname(self):
        "Property.literalname"
        p = cssutils.css.property.Property(r'c\olor', 'red')
        self.assertEqual(r'c\olor', p.literalname)
        self.assertRaisesMsg(
            AttributeError, "can't set attribute", p.__setattr__, 'literalname', 'color'
        )

    def test_validate(self):
        "Property.valid"
        p = cssutils.css.property.Property('left', '1px', '')

        self.assertEqual(p.valid, True)

        p.name = 'color'
        self.assertEqual(p.valid, False)

        p.name = 'top'
        self.assertEqual(p.valid, True)

        p.value = 'red'
        self.assertEqual(p.valid, False)

    # def test_cssValue(self):
    #    "Property.cssValue"
    #    pass
    #    # DEPRECATED

    def test_priority(self):
        "Property.priority"
        p = cssutils.css.property.Property('top', '1px', 'important')

        for prio in (None, ''):
            p.priority = prio
            self.assertEqual('', p.priority)
            self.assertEqual('', p.literalpriority)

        for prio in (
            '!important',
            '! important',
            '!/* x */ important',
            '!/* x */ important /**/',
            'important',
            'IMPORTANT',
            r'im\portant',
        ):
            p.priority = prio
            self.assertEqual('important', p.priority)
            if prio.startswith('!'):
                prio = prio[1:].strip()
            if '/*' in prio:
                check = 'important'
            else:
                check = prio
            self.assertEqual(check, p.literalpriority)

        tests = {
            ' ': xml.dom.SyntaxErr,
            '"\n': xml.dom.SyntaxErr,
            # u'important': xml.dom.SyntaxErr,
            ';': xml.dom.SyntaxErr,
            '!important !important': xml.dom.SyntaxErr,
        }
        self.do_raise_r(tests, att='_setPriority')

    def test_value(self):
        "Property.value"
        p = cssutils.css.property.Property('top', '1px')
        self.assertEqual('1px', p.value)
        p.value = '2px'
        self.assertEqual('2px', p.value)

        tests = {
            '1px': None,
            ' 2px': '2px',
            '3px ': '3px',
            ' 4px ': '4px',
            '5px 1px': '5px 1px',
        }
        self.do_equal_r(tests, att='value')

        tests = {
            # no value
            None: xml.dom.SyntaxErr,
            '': xml.dom.SyntaxErr,
            ' ': xml.dom.SyntaxErr,
            '"\n': xml.dom.SyntaxErr,
            '/*x*/': xml.dom.SyntaxErr,
            # not allowed:
            ':': xml.dom.SyntaxErr,
            ';': xml.dom.SyntaxErr,
            '!important': xml.dom.SyntaxErr,
        }
        self.do_raise_r(tests, att='_setValue')

    def test_reprANDstr(self):
        "Property.__repr__(), .__str__()"
        name = "color"
        value = "red"
        priority = "important"

        s = cssutils.css.property.Property(name=name, value=value, priority=priority)

        self.assertTrue(name in str(s))
        self.assertTrue(value in str(s))
        self.assertTrue(priority in str(s))

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(name == s2.name)
        self.assertTrue(value == s2.value)
        self.assertTrue(priority == s2.priority)


if __name__ == '__main__':
    import unittest

    unittest.main()
