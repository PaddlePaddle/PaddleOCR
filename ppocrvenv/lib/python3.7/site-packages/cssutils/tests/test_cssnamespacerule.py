"""Testcases for cssutils.css.CSSImportRule"""

import xml.dom
from . import test_cssrule
import cssutils


class CSSNamespaceRuleTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(CSSNamespaceRuleTestCase, self).setUp()
        self.r = cssutils.css.CSSNamespaceRule(namespaceURI='x')
        # self.rRO = cssutils.css.CSSNamespaceRule(namespaceURI='x',
        #                                         readonly=True)
        self.r_type = cssutils.css.CSSRule.NAMESPACE_RULE
        self.r_typeString = 'NAMESPACE_RULE'

    def test_init(self):
        "CSSNamespaceRule.__init__()"
        # cannot use here as self.r and self rRO and not useful
        # super(CSSNamespaceRuleTestCase, self).test_init()
        tests = [
            (None, None),
            ('', ''),
            (None, ''),
            ('', None),
            ('', 'no-uri'),
        ]
        for uri, p in tests:
            r = cssutils.css.CSSNamespaceRule(namespaceURI=uri, prefix=p)
            self.assertEqual(None, r.namespaceURI)
            self.assertEqual('', r.prefix)
            self.assertEqual('', r.cssText)
            self.assertEqual(None, r.parentStyleSheet)
            self.assertEqual(None, r.parentRule)

        r = cssutils.css.CSSNamespaceRule(namespaceURI='example')
        self.assertEqual('example', r.namespaceURI)
        self.assertEqual('', r.prefix)
        self.assertEqual('@namespace "example";', r.cssText)
        self.sheet.add(r)
        self.assertEqual(self.sheet, r.parentStyleSheet)

        r = cssutils.css.CSSNamespaceRule(namespaceURI='example', prefix='p')
        self.assertEqual('example', r.namespaceURI)
        self.assertEqual('p', r.prefix)
        self.assertEqual('@namespace p "example";', r.cssText)

        css = '@namespace p "u";'
        r = cssutils.css.CSSNamespaceRule(cssText=css)
        self.assertEqual(r.cssText, css)

        # only possible to set @... similar name
        self.assertRaises(xml.dom.InvalidModificationErr, self.r._setAtkeyword, 'x')

    def test_cssText(self):
        "CSSNamespaceRule.cssText"
        # cssText may only be set initalially
        r = cssutils.css.CSSNamespaceRule()
        css = '@namespace p "u";'
        r.cssText = css
        self.assertEqual(r.cssText, css)
        self.assertRaises(
            xml.dom.NoModificationAllowedErr, r._setCssText, '@namespace p "OTHER";'
        )

        tests = {
            '@namespace "";': None,
            '@namespace "u";': None,
            '@namespace empty "";': None,
            '@namespace p "p";': None,
            "@namespace p 'u';": '@namespace p "u";',
            '@\\namespace p "u";': '@namespace p "u";',
            '@NAMESPACE p "u";': '@namespace p "u";',
            '@namespace  p  "u"  ;': '@namespace p "u";',
            '@namespace p"u";': '@namespace p "u";',
            '@namespace p "u";': '@namespace p "u";',
            '@namespace/*1*/"u"/*2*/;': '@namespace /*1*/ "u" /*2*/;',
            '@namespace/*1*/p/*2*/"u"/*3*/;': '@namespace /*1*/ p /*2*/ "u" /*3*/;',
            '@namespace p url(u);': '@namespace p "u";',
            '@namespace p url(\'u\');': '@namespace p "u";',
            '@namespace p url(\"u\");': '@namespace p "u";',
            '@namespace p url( \"u\" );': '@namespace p "u";',
            # comments
            '@namespace/*1*//*2*/p/*3*//*4*/url(u)/*5*//*6*/;': '@namespace /*1*/ /*2*/ p /*3*/ /*4*/ "u" /*5*/ /*6*/;',
            '@namespace/*1*//*2*/p/*3*//*4*/"u"/*5*//*6*/;': '@namespace /*1*/ /*2*/ p /*3*/ /*4*/ "u" /*5*/ /*6*/;',
            '@namespace/*1*//*2*/p/*3*//*4*/url("u")/*5*//*6*/;': '@namespace /*1*/ /*2*/ p /*3*/ /*4*/ "u" /*5*/ /*6*/;',
            '@namespace/*1*//*2*/url(u)/*5*//*6*/;': '@namespace /*1*/ /*2*/ "u" /*5*/ /*6*/;',
            # WS
            '@namespace\n\r\t\f p\n\r\t\f url(\n\r\t\f u\n\r\t\f )\n\r\t\f ;': '@namespace p "u";',
            '@namespace\n\r\t\f p\n\r\t\f url(\n\r\t\f "u"\n\r\t\f )\n\r\t\f ;': '@namespace p "u";',
            '@namespace\n\r\t\f p\n\r\t\f "str"\n\r\t\f ;': '@namespace p "str";',
            '@namespace\n\r\t\f "str"\n\r\t\f ;': '@namespace "str";',
        }
        self.do_equal_p(tests)
        # self.do_equal_r(tests) # cannot use here as always new r is needed
        for test, expected in list(tests.items()):
            r = cssutils.css.CSSNamespaceRule(cssText=test)
            if expected is None:
                expected = test
            self.assertEqual(expected, r.cssText)

        tests = {
            '@namespace;': xml.dom.SyntaxErr,  # nothing
            '@namespace p;': xml.dom.SyntaxErr,  # no namespaceURI
            '@namespace "u" p;': xml.dom.SyntaxErr,  # order
            '@namespace "u";EXTRA': xml.dom.SyntaxErr,
            '@namespace p "u";EXTRA': xml.dom.SyntaxErr,
        }
        self.do_raise_p(tests)  # parse
        tests.update(
            {
                '@namespace p url(x)': xml.dom.SyntaxErr,  # missing ;
                '@namespace p "u"': xml.dom.SyntaxErr,  # missing ;
                # trailing
                '@namespace "u"; ': xml.dom.SyntaxErr,
                '@namespace "u";/**/': xml.dom.SyntaxErr,
                '@namespace p "u"; ': xml.dom.SyntaxErr,
                '@namespace p "u";/**/': xml.dom.SyntaxErr,
            }
        )

        def _do(test):
            cssutils.css.CSSNamespaceRule(cssText=test)

        for test, expected in list(tests.items()):
            self.assertRaises(expected, _do, test)

    def test_namespaceURI(self):
        "CSSNamespaceRule.namespaceURI"
        # set only initially
        r = cssutils.css.CSSNamespaceRule(namespaceURI='x')
        self.assertEqual('x', r.namespaceURI)
        self.assertEqual('@namespace "x";', r.cssText)

        r = cssutils.css.CSSNamespaceRule(namespaceURI='"')
        self.assertEqual('@namespace "\\"";', r.cssText)

        self.assertRaises(xml.dom.NoModificationAllowedErr, r._setNamespaceURI, 'x')

        self.assertRaises(
            xml.dom.NoModificationAllowedErr, r._setCssText, '@namespace "u";'
        )

        r._replaceNamespaceURI('http://example.com/new')
        self.assertEqual('http://example.com/new', r.namespaceURI)

    def test_prefix(self):
        "CSSNamespaceRule.prefix"
        r = cssutils.css.CSSNamespaceRule(namespaceURI='u')
        r.prefix = 'p'
        self.assertEqual('p', r.prefix)
        self.assertEqual('@namespace p "u";', r.cssText)

        r = cssutils.css.CSSNamespaceRule(cssText='@namespace x "u";')
        r.prefix = 'p'
        self.assertEqual('p', r.prefix)
        self.assertEqual('@namespace p "u";', r.cssText)

        valid = (None, '')
        for prefix in valid:
            r.prefix = prefix
            self.assertEqual(r.prefix, '')
            self.assertEqual('@namespace "u";', r.cssText)

        valid = ('a', '_x', 'a1', 'a-1')
        for prefix in valid:
            r.prefix = prefix
            self.assertEqual(r.prefix, prefix)
            self.assertEqual('@namespace %s "u";' % prefix, r.cssText)

        invalid = ('1', ' x', ' ', ',')
        for prefix in invalid:
            self.assertRaises(xml.dom.SyntaxErr, r._setPrefix, prefix)

    def test_InvalidModificationErr(self):
        "CSSNamespaceRule.cssText InvalidModificationErr"
        self._test_InvalidModificationErr('@namespace')

    def test_incomplete(self):
        "CSSNamespaceRule (incomplete)"
        tests = {
            '@namespace "uri': '@namespace "uri";',
            "@namespace url(x": '@namespace "x";',
            "@namespace url('x": '@namespace "x";',
            '@namespace url("x;': '@namespace "x;";',
            '@namespace url( "x;': '@namespace "x;";',
            '@namespace url("x ': '@namespace "x ";',
            '@namespace url(x ': '@namespace "x";',
        }
        self.do_equal_p(tests)  # parse
        tests = {
            '@namespace "uri': xml.dom.SyntaxErr,
            "@namespace url(x": xml.dom.SyntaxErr,
            "@namespace url('x": xml.dom.SyntaxErr,
            '@namespace url("x;': xml.dom.SyntaxErr,
            '@namespace url( "x;': xml.dom.SyntaxErr,
            '@namespace url("x ': xml.dom.SyntaxErr,
            '@namespace url(x ': xml.dom.SyntaxErr,
        }
        self.do_raise_r(tests)  # set cssText

    def test_reprANDstr(self):
        "CSSNamespaceRule.__repr__(), .__str__()"
        namespaceURI = 'http://example.com'
        prefix = 'prefix'

        s = cssutils.css.CSSNamespaceRule(namespaceURI=namespaceURI, prefix=prefix)

        self.assertTrue(namespaceURI in str(s))
        self.assertTrue(prefix in str(s))

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(namespaceURI == s2.namespaceURI)
        self.assertTrue(prefix == s2.prefix)


if __name__ == '__main__':
    import unittest

    unittest.main()
