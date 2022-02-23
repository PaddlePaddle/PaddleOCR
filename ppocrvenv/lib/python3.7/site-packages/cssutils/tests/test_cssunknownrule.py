"""testcases for cssutils.css.CSSUnkownRule"""

import xml.dom
from . import test_cssrule
import cssutils


class CSSUnknownRuleTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(CSSUnknownRuleTestCase, self).setUp()
        self.r = cssutils.css.CSSUnknownRule()
        self.rRO = cssutils.css.CSSUnknownRule(readonly=True)
        self.r_type = cssutils.css.CSSUnknownRule.UNKNOWN_RULE
        self.r_typeString = 'UNKNOWN_RULE'

    def tearDown(self):
        cssutils.ser.prefs.useDefaults()

    def test_init(self):
        "CSSUnknownRule.type and init"
        super(CSSUnknownRuleTestCase, self).test_init()

        self.assertFalse(self.r.wellformed)

        # only name
        r = cssutils.css.CSSUnknownRule(cssText='@init;')
        self.assertEqual('@init', r.atkeyword)
        self.assertEqual('@init;', r.cssText)
        self.assertTrue(r.wellformed)

        # @-... not allowed?
        r = cssutils.css.CSSUnknownRule(cssText='@-init;')
        self.assertEqual('@-init;', r.cssText)
        self.assertEqual('@-init', r.atkeyword)
        self.assertTrue(r.wellformed)

        r = cssutils.css.CSSUnknownRule(cssText='@_w-h-a-012;')
        self.assertEqual('@_w-h-a-012;', r.cssText)
        self.assertEqual('@_w-h-a-012', r.atkeyword)
        self.assertTrue(r.wellformed)

        # name and content
        r = cssutils.css.CSSUnknownRule(cssText='@init xxx;')
        self.assertEqual('@init', r.atkeyword)
        self.assertEqual('@init xxx;', r.cssText)
        self.assertTrue(r.wellformed)

        # name and block
        r = cssutils.css.CSSUnknownRule(cssText='@init { xxx }')
        self.assertEqual('@init', r.atkeyword)
        self.assertEqual('@init {\n    xxx\n    }', r.cssText)
        self.assertTrue(r.wellformed)

        # name and content and block
        r = cssutils.css.CSSUnknownRule(cssText='@init xxx { yyy }')
        self.assertEqual('@init', r.atkeyword)
        self.assertEqual('@init xxx {\n    yyy\n    }', r.cssText)
        self.assertTrue(r.wellformed)

    def test_cssText(self):
        "CSSUnknownRule.cssText"
        tests = {
            # not normal rules!
            '@font-facex{}': '@font-facex {\n    }',
            '@importurl(x.css);': '@importurl (x . css);',
            '@mediaAll{}': '@mediaall {\n    }',
            '@namespacep"x";': '@namespacep "x";',
            '@pageX{}': '@pagex {\n    }',
            '@xbottom { content: counter(page) }': '@xbottom {\n    content: counter(page)\n    }',
            '@xbottom { content: "x" counter(page) "y"}': '@xbottom {\n    content: "x" counter(page) "y"\n    }',
        }
        self.do_equal_p(tests)

        # expects the same atkeyword for self.r so do a new one each test
        oldr = self.r
        for t, e in list(tests.items()):
            self.r = cssutils.css.CSSUnknownRule()
            self.do_equal_r({t: e})
        self.r = oldr

        tests = {
            '@x;': None,
            '@x {}': '@x {\n    }',
            '@x{ \n \t \f\r}': '@x {\n    }',
            '@x {\n    [()]([ {\n        }]) {\n        }\n    }': None,
            '@x {\n    @b;\n    }': None,
            '''@x {
    @b {
        x: 1x;
        y: 2y;
        }
    }''': None,
            '@x "string" url(x);': None,
            # comments
            '@x/*1*//*2*/"str"/*3*//*4*/url("x");': '@x /*1*/ /*2*/ "str" /*3*/ /*4*/ url(x);',
            # WS
            '@x"string"url("x");': '@x "string" url(x);',
            '@x\n\r\t\f "string"\n\r\t\f url(\n\r\t\f "x"\n\r\t\f )\n\r\t\f ;': '@x "string" url(x);',
        }
        self.do_equal_p(tests)
        self.do_equal_r(tests)

        tests = {
            '@;': xml.dom.InvalidModificationErr,
            '@{}': xml.dom.InvalidModificationErr,
            '@ ;': xml.dom.InvalidModificationErr,
            '@ {};': xml.dom.InvalidModificationErr,
            '@x ;{}': xml.dom.SyntaxErr,
            '@x ;;': xml.dom.SyntaxErr,
            '@x }  ': xml.dom.SyntaxErr,
            '@x }  ;': xml.dom.SyntaxErr,
            '@x {  ': xml.dom.SyntaxErr,
            '@x {  ;': xml.dom.SyntaxErr,
            '@x ': xml.dom.SyntaxErr,
            '@x (;': xml.dom.SyntaxErr,
            '@x );': xml.dom.SyntaxErr,
            '@x [;': xml.dom.SyntaxErr,
            '@x ];': xml.dom.SyntaxErr,
            '@x {[(]()}': xml.dom.SyntaxErr,
            # trailing
            '@x{}{}': xml.dom.SyntaxErr,
            '@x{};': xml.dom.SyntaxErr,
            '@x{}1': xml.dom.SyntaxErr,
            '@x{} ': xml.dom.SyntaxErr,
            '@x{}/**/': xml.dom.SyntaxErr,
            '@x;1': xml.dom.SyntaxErr,
            '@x; ': xml.dom.SyntaxErr,
            '@x;/**/': xml.dom.SyntaxErr,
        }
        self.do_raise_r(tests)

    def test_InvalidModificationErr(self):
        "CSSUnknownRule.cssText InvalidModificationErr"
        self._test_InvalidModificationErr('@unknown')

    def test_reprANDstr(self):
        "CSSUnknownRule.__repr__(), .__str__()"
        s = cssutils.css.CSSUnknownRule(cssText='@x;')

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))


if __name__ == '__main__':
    import unittest

    unittest.main()
