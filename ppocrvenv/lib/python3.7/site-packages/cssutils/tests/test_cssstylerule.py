"""Testcases for cssutils.css.CSSStyleRuleTestCase"""

import xml.dom
from . import test_cssrule
import cssutils


class CSSStyleRuleTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(CSSStyleRuleTestCase, self).setUp()
        self.r = cssutils.css.CSSStyleRule()
        self.rRO = cssutils.css.CSSStyleRule(readonly=True)
        self.r_type = cssutils.css.CSSStyleRule.STYLE_RULE
        self.r_typeString = 'STYLE_RULE'

    def test_init(self):
        "CSSStyleRule.type and init"
        super(CSSStyleRuleTestCase, self).test_init()
        self.assertEqual('', self.r.cssText)
        self.assertEqual(
            cssutils.css.selectorlist.SelectorList, type(self.r.selectorList)
        )
        self.assertEqual('', self.r.selectorText)
        self.assertEqual(cssutils.css.CSSStyleDeclaration, type(self.r.style))
        self.assertEqual(self.r, self.r.style.parentRule)

    def test_refs(self):
        "CSSStyleRule references"
        s = cssutils.css.CSSStyleRule()
        sel, style = s.selectorList, s.style

        self.assertEqual(s, sel.parentRule)
        self.assertEqual(s, style.parentRule)

        s.cssText = 'a { x:1 }'
        self.assertNotEqual(sel, s.selectorList)
        self.assertEqual('a', s.selectorList.selectorText)
        self.assertNotEqual(style, s.style)
        self.assertEqual('1', s.style.getPropertyValue('x'))

        sel, style = s.selectorList, s.style

        invalids = (
            '$b { x:2 }',  # invalid selector
            'c { $x3 }',  # invalid style
            '/b { 2 }',  # both invalid
        )
        for invalid in invalids:
            try:
                s.cssText = invalid
            except xml.dom.DOMException:
                pass
            self.assertEqual(sel, s.selectorList)
            self.assertEqual('a', s.selectorList.selectorText)
            self.assertEqual(style, s.style)
            self.assertEqual('1', s.style.getPropertyValue('x'))

        # CHANGING
        s = cssutils.parseString('a {s1: 1}')
        r = s.cssRules[0]
        sel1 = r.selectorList
        st1 = r.style

        # selectorList
        r.selectorText = 'b'
        self.assertNotEqual(sel1, r.selectorList)
        self.assertEqual('b', r.selectorList.selectorText)
        self.assertEqual('b', r.selectorText)
        sel1b = r.selectorList

        sel1b.selectorText = 'c'
        self.assertEqual(sel1b, r.selectorList)
        self.assertEqual('c', r.selectorList.selectorText)
        self.assertEqual('c', r.selectorText)

        sel2 = cssutils.css.SelectorList('sel2')
        s.selectorList = sel2
        self.assertEqual(sel2, s.selectorList)
        self.assertEqual('sel2', s.selectorList.selectorText)

        sel2.selectorText = 'sel2b'
        self.assertEqual('sel2b', sel2.selectorText)
        self.assertEqual('sel2b', s.selectorList.selectorText)

        s.selectorList.selectorText = 'sel2c'
        self.assertEqual('sel2c', sel2.selectorText)
        self.assertEqual('sel2c', s.selectorList.selectorText)

        # style
        r.style = 's1: 2'
        self.assertNotEqual(st1, r.style)
        self.assertEqual('s1: 2', r.style.cssText)

        st2 = cssutils.parseStyle('s2: 1')
        r.style = st2
        self.assertEqual(st2, r.style)
        self.assertEqual('s2: 1', r.style.cssText)

        # cssText
        sl, st = r.selectorList, r.style
        # fails
        try:
            r.cssText = '$ {content: "new"}'
        except xml.dom.SyntaxErr:
            pass
        self.assertEqual(sl, r.selectorList)
        self.assertEqual(st, r.style)

        r.cssText = 'a {content: "new"}'
        self.assertNotEqual(sl, r.selectorList)
        self.assertNotEqual(st, r.style)

    def test_cssText(self):
        "CSSStyleRule.cssText"
        tests = {
            '* {}': '',
            'a {}': '',
        }
        self.do_equal_p(tests)  # parse
        # self.do_equal_r(tests) # set cssText # TODO: WHY?

        cssutils.ser.prefs.keepEmptyRules = True
        tests = {
            # u'''a{;display:block;float:left}''': 'a {\n    display:block;\n    float:left\n    }', # issue 28
            '''a\n{color: #000}''': 'a {\n    color: #000\n    }',  # issue 4
            '''a\n{color: #000000}''': 'a {\n    color: #000\n    }',  # issue 4
            '''a\n{color: #abc}''': 'a {\n    color: #abc\n    }',  # issue 4
            '''a\n{color: #abcdef}''': 'a {\n    color: #abcdef\n    }',  # issue 4
            '''a\n{color: #00a}''': 'a {\n    color: #00a\n    }',  # issue 4
            '''a\n{color: #1a1a1a}''': 'a {\n    color: #1a1a1a\n    }',  # issue 4
            '''#id\n{ color: red }''': '#id {\n    color: red\n    }',  # issue 3
            '''* {}''': None,
            'a {}': None,
            'b { a: 1; }': 'b {\n    a: 1\n    }',
            # mix of comments and properties
            'c1 {/*1*/a:1;}': 'c1 {\n    /*1*/\n    a: 1\n    }',
            'c2 {a:1;/*2*/}': 'c2 {\n    a: 1;\n    /*2*/\n    }',
            'd1 {/*0*/}': 'd1 {\n    /*0*/\n    }',
            'd2 {/*0*//*1*/}': 'd2 {\n    /*0*/\n    /*1*/\n    }',
            # comments
            # TODO: spaces?
            '''a/*1*//*2*/,/*3*//*4*/b/*5*//*6*/{color: #000}''': 'a/*1*//*2*/, /*3*//*4*/b/*5*//*6*/ {\n    color: #000\n    }',
            '''a,b{color: #000}''': 'a, b {\n    color: #000\n    }',  # issue 4
            '''a\n\r\t\f ,\n\r\t\f b\n\r\t\f {color: #000}''': 'a, b {\n    color: #000\n    }',  # issue 4
        }
        self.do_equal_p(tests)  # parse
        self.do_equal_r(tests)  # set cssText

        tests = {
            '''a;''': xml.dom.SyntaxErr,
            '''a {{}''': xml.dom.SyntaxErr,
            '''a }''': xml.dom.SyntaxErr,
        }
        self.do_raise_p(tests)  # parse
        tests.update(
            {
                '''/*x*/''': xml.dom.SyntaxErr,
                '''a {''': xml.dom.SyntaxErr,
                # trailing
                '''a {}x''': xml.dom.SyntaxErr,
                '''a {/**/''': xml.dom.SyntaxErr,
                '''a {} ''': xml.dom.SyntaxErr,
            }
        )
        self.do_raise_r(tests)  # set cssText
        cssutils.ser.prefs.useDefaults()

    def test_selectorList(self):
        "CSSStyleRule.selectorList"
        r = cssutils.css.CSSStyleRule()

        r.selectorList.appendSelector('a')
        self.assertEqual(1, r.selectorList.length)
        self.assertEqual('a', r.selectorText)

        r.selectorList.appendSelector(' b  ')
        # only simple selector!
        self.assertRaises(
            xml.dom.InvalidModificationErr, r.selectorList.appendSelector, '  h1, x '
        )

        self.assertEqual(2, r.selectorList.length)
        self.assertEqual('a, b', r.selectorText)

    def test_selectorText(self):
        "CSSStyleRule.selectorText"
        r = cssutils.css.CSSStyleRule()

        r.selectorText = 'a'
        self.assertEqual(1, r.selectorList.length)
        self.assertEqual('a', r.selectorText)

        r.selectorText = ' b, h1  '
        self.assertEqual(2, r.selectorList.length)
        self.assertEqual('b, h1', r.selectorText)

    def test_style(self):
        "CSSStyleRule.style"
        d = cssutils.css.CSSStyleDeclaration()
        self.r.style = d
        self.assertEqual(d.cssText, self.r.style.cssText)

        # check if parentRule of d is set
        self.assertEqual(self.r, d.parentRule)

    def test_incomplete(self):
        "CSSStyleRule (incomplete)"
        cssutils.ser.prefs.keepEmptyRules = True
        tests = {
            'a {': 'a {}',  # no }
            'a { font-family: "arial sans': 'a {\n    font-family: "arial sans"\n    }',  # no "}
            'a { font-family: "arial sans";': 'a {\n    font-family: "arial sans"\n    }',  # no }
            '''p {
                color: green;
                font-family: 'Courier New Times
                color: red;
                color: green;
                }''': '''p {\n    color: green;\n    color: green\n    }''',
            # no ;
            '''p {
                color: green;
                font-family: 'Courier New Times'
                color: red;
                color: green;
                ''': '''p {\n    color: green;\n    color: green\n    }''',
        }
        self.do_equal_p(tests, raising=False)  # parse
        cssutils.ser.prefs.useDefaults()

    # TODO:   def test_InvalidModificationErr(self):
    #        "CSSStyleRule.cssText InvalidModificationErr"
    #        self._test_InvalidModificationErr(u'@a a {}')

    def test_reprANDstr(self):
        "CSSStyleRule.__repr__(), .__str__()"
        sel = 'a > b + c'

        s = cssutils.css.CSSStyleRule(selectorText=sel)

        self.assertTrue(sel in str(s))

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(sel == s2.selectorText)

    def test_valid(self):
        "CSSStyleRule.valid"
        rule = cssutils.css.CSSStyleRule(selectorText='*', style='color: red')
        self.assertTrue(rule.valid)
        rule.style = 'color: foobar'
        self.assertFalse(rule.valid)
        rule.style = 'foobar: red'
        self.assertFalse(rule.valid)


if __name__ == '__main__':
    import unittest

    unittest.main()
