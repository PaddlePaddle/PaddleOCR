"""Testcases for cssutils.css.CSSImportRule"""

import re
import xml.dom

import pytest

import cssutils
from . import test_cssrule


class CSSImportRuleTestCase(test_cssrule.CSSRuleTestCase):
    def setUp(self):
        super(CSSImportRuleTestCase, self).setUp()
        self.r = cssutils.css.CSSImportRule()
        self.rRO = cssutils.css.CSSImportRule(readonly=True)
        self.r_type = cssutils.css.CSSImportRule.IMPORT_RULE
        self.r_typeString = 'IMPORT_RULE'

    def test_init(self):
        "CSSImportRule.__init__()"
        super(CSSImportRuleTestCase, self).test_init()

        # no init param
        self.assertEqual(None, self.r.href)
        self.assertEqual(None, self.r.hreftype)
        self.assertEqual(False, self.r.hrefFound)
        self.assertEqual('all', self.r.media.mediaText)
        self.assertEqual(cssutils.stylesheets.MediaList, type(self.r.media))
        self.assertEqual(None, self.r.name)
        self.assertEqual(cssutils.css.CSSStyleSheet, type(self.r.styleSheet))
        self.assertEqual(0, self.r.styleSheet.cssRules.length)
        self.assertEqual('', self.r.cssText)

        # all
        r = cssutils.css.CSSImportRule(href='href', mediaText='tv', name='name')
        self.assertEqual('@import url(href) tv "name";', r.cssText)
        self.assertEqual("href", r.href)
        self.assertEqual(None, r.hreftype)
        self.assertEqual('tv', r.media.mediaText)
        self.assertEqual(cssutils.stylesheets.MediaList, type(r.media))
        self.assertEqual('name', r.name)
        self.assertEqual(None, r.parentRule)  # see CSSRule
        self.assertEqual(None, r.parentStyleSheet)  # see CSSRule
        self.assertEqual(cssutils.css.CSSStyleSheet, type(self.r.styleSheet))
        self.assertEqual(0, self.r.styleSheet.cssRules.length)

        # href
        r = cssutils.css.CSSImportRule('x')
        self.assertEqual('@import url(x);', r.cssText)
        self.assertEqual('x', r.href)
        self.assertEqual(None, r.hreftype)

        # href + mediaText
        r = cssutils.css.CSSImportRule('x', 'print')
        self.assertEqual('@import url(x) print;', r.cssText)
        self.assertEqual('x', r.href)
        self.assertEqual('print', r.media.mediaText)

        # href + name
        r = cssutils.css.CSSImportRule('x', name='n')
        self.assertEqual('@import url(x) "n";', r.cssText)
        self.assertEqual('x', r.href)
        self.assertEqual('n', r.name)

        # href + mediaText + name
        r = cssutils.css.CSSImportRule('x', 'print', 'n')
        self.assertEqual('@import url(x) print "n";', r.cssText)
        self.assertEqual('x', r.href)
        self.assertEqual('print', r.media.mediaText)
        self.assertEqual('n', r.name)

        # media +name only
        self.r = cssutils.css.CSSImportRule(mediaText='print', name="n")
        self.assertEqual(cssutils.stylesheets.MediaList, type(self.r.media))
        self.assertEqual('', self.r.cssText)
        self.assertEqual('print', self.r.media.mediaText)
        self.assertEqual('n', self.r.name)

        # only possible to set @... similar name
        self.assertRaises(xml.dom.InvalidModificationErr, self.r._setAtkeyword, 'x')

    def test_cssText(self):
        "CSSImportRule.cssText"
        tests = {
            # href string
            '''@import "str";''': None,
            '''@import"str";''': '''@import "str";''',
            '''@\\import "str";''': '''@import "str";''',
            '''@IMPORT "str";''': '''@import "str";''',
            '''@import 'str';''': '''@import "str";''',
            '''@import 'str' ;''': '''@import "str";''',
            '''@import "str";''': None,
            '''@import "str"  ;''': '''@import "str";''',
            r'''@import "\""  ;''': r'''@import "\"";''',
            '''@import '\\'';''': r'''@import "'";''',
            '''@import '"';''': r'''@import "\"";''',
            # href url
            '''@import url(x.css);''': None,
            # nospace
            '''@import url(")");''': '''@import url(")");''',
            '''@import url("\\"");''': '''@import url("\\"");''',
            '''@import url('\\'');''': '''@import url("'");''',
            # href + media
            # all is removed
            '''@import "str" all;''': '''@import "str";''',
            '''@import "str" tv, print;''': None,
            '''@import"str"tv,print;''': '''@import "str" tv, print;''',
            '''@import "str" tv, print, all;''': '''@import "str";''',
            '''@import "str" handheld, all;''': '''@import "str";''',
            '''@import "str" all, handheld;''': '''@import "str";''',
            '''@import "str" not tv;''': None,
            '''@import "str" only tv;''': None,
            '''@import "str" only tv and (color: 2);''': None,
            # href + name
            '''@import "str" "name";''': None,
            '''@import "str" 'name';''': '''@import "str" "name";''',
            '''@import url(x) "name";''': None,
            '''@import "str" "\\"";''': None,
            '''@import "str" '\\'';''': '''@import "str" "'";''',
            # href + media + name
            '''@import"str"tv"name";''': '''@import "str" tv "name";''',
            '''@import\t\r\f\n"str"\t\t\r\f\ntv\t\t\r\f\n"name"\t;''': '''@import "str" tv "name";''',
            # comments
            '''@import /*1*/ "str" /*2*/;''': None,
            '@import/*1*//*2*/"str"/*3*//*4*/all/*5*//*6*/"name"/*7*//*8*/ ;': '@import /*1*/ /*2*/ "str" /*3*/ /*4*/ all /*5*/ /*6*/ "name" /*7*/ /*8*/;',
            '@import/*1*//*2*/url(u)/*3*//*4*/all/*5*//*6*/"name"/*7*//*8*/ ;': '@import /*1*/ /*2*/ url(u) /*3*/ /*4*/ all /*5*/ /*6*/ "name" /*7*/ /*8*/;',
            '@import/*1*//*2*/url("u")/*3*//*4*/all/*5*//*6*/"name"/*7*//*8*/ ;': '@import /*1*/ /*2*/ url(u) /*3*/ /*4*/ all /*5*/ /*6*/ "name" /*7*/ /*8*/;',
            # WS
            '@import\n\t\f "str"\n\t\f tv\n\t\f "name"\n\t\f ;': '@import "str" tv "name";',
            '@import\n\t\f url(\n\t\f u\n\t\f )\n\t\f tv\n\t\f "name"\n\t\f ;': '@import url(u) tv "name";',
            '@import\n\t\f url("u")\n\t\f tv\n\t\f "name"\n\t\f ;': '@import url(u) tv "name";',
            '@import\n\t\f url(\n\t\f "u"\n\t\f )\n\t\f tv\n\t\f "name"\n\t\f ;': '@import url(u) tv "name";',
        }
        self.do_equal_r(tests)  # set cssText
        tests.update(
            {
                '@import "x.css" tv': '@import "x.css" tv;',
                '@import "x.css"': '@import "x.css";',  # no ;
                "@import 'x.css'": '@import "x.css";',  # no ;
                '@import url(x.css)': '@import url(x.css);',  # no ;
                '@import "x;': '@import "x;";',  # no "!
            }
        )
        self.do_equal_p(tests)  # parse

        tests = {
            '''@import;''': xml.dom.SyntaxErr,
            '''@import all;''': xml.dom.SyntaxErr,
            '''@import all"name";''': xml.dom.SyntaxErr,
            '''@import x";''': xml.dom.SyntaxErr,
            '''@import "str" ,all;''': xml.dom.SyntaxErr,
            '''@import "str" all,;''': xml.dom.SyntaxErr,
            '''@import "str" all tv;''': xml.dom.SyntaxErr,
            '''@import "str" "name" all;''': xml.dom.SyntaxErr,
        }
        self.do_raise_p(tests)  # parse
        tests.update(
            {
                '@import "x.css"': xml.dom.SyntaxErr,
                "@import 'x.css'": xml.dom.SyntaxErr,
                '@import url(x.css)': xml.dom.SyntaxErr,
                '@import "x.css" tv': xml.dom.SyntaxErr,
                '@import "x;': xml.dom.SyntaxErr,
                '''@import url("x);''': xml.dom.SyntaxErr,
                # trailing
                '''@import "x";"a"''': xml.dom.SyntaxErr,
                # trailing S or COMMENT
                '''@import "x";/**/''': xml.dom.SyntaxErr,
                '''@import "x"; ''': xml.dom.SyntaxErr,
            }
        )
        self.do_raise_r(tests)  # set cssText

    def test_href(self):
        "CSSImportRule.href"
        # set
        self.r.href = 'x'
        self.assertEqual('x', self.r.href)
        self.assertEqual('@import url(x);', self.r.cssText)

        # http
        self.r.href = 'http://www.example.com/x?css=z&v=1'
        self.assertEqual('http://www.example.com/x?css=z&v=1', self.r.href)
        self.assertEqual(
            '@import url(http://www.example.com/x?css=z&v=1);', self.r.cssText
        )

        # also if hreftype changed
        self.r.hreftype = 'string'
        self.assertEqual('http://www.example.com/x?css=z&v=1', self.r.href)
        self.assertEqual(
            '@import "http://www.example.com/x?css=z&v=1";', self.r.cssText
        )

        # string escaping?
        self.r.href = '"'
        self.assertEqual('@import "\\"";', self.r.cssText)
        self.r.hreftype = 'url'
        self.assertEqual('@import url("\\"");', self.r.cssText)

        # url escaping?
        self.r.href = ')'
        self.assertEqual('@import url(")");', self.r.cssText)

        self.r.hreftype = 'NOT VALID'  # using default
        self.assertEqual('@import url(")");', self.r.cssText)

    def test_hrefFound(self):
        "CSSImportRule.hrefFound"

        def fetcher(url):
            if url == 'http://example.com/yes':
                return None, '/**/'
            else:
                return None, None

        parser = cssutils.CSSParser(fetcher=fetcher)
        sheet = parser.parseString('@import "http://example.com/yes" "name"')

        r = sheet.cssRules[0]
        self.assertEqual('/**/'.encode(), r.styleSheet.cssText)
        self.assertEqual(True, r.hrefFound)
        self.assertEqual('name', r.name)

        r.cssText = '@import url(http://example.com/none) "name2";'
        self.assertEqual(''.encode(), r.styleSheet.cssText)
        self.assertEqual(False, r.hrefFound)
        self.assertEqual('name2', r.name)

        sheet.cssText = '@import url(http://example.com/none);'
        self.assertNotEqual(r, sheet.cssRules[0])

    def test_hreftype(self):
        "CSSImportRule.hreftype"
        self.r = cssutils.css.CSSImportRule()

        self.r.cssText = '@import /*1*/url(org) /*2*/;'
        self.assertEqual('uri', self.r.hreftype)
        self.assertEqual('@import /*1*/ url(org) /*2*/;', self.r.cssText)

        self.r.cssText = '@import /*1*/"org" /*2*/;'
        self.assertEqual('string', self.r.hreftype)
        self.assertEqual('@import /*1*/ "org" /*2*/;', self.r.cssText)

        self.r.href = 'new'
        self.assertEqual('@import /*1*/ "new" /*2*/;', self.r.cssText)

        self.r.hreftype = 'uri'
        self.assertEqual('@import /*1*/ url(new) /*2*/;', self.r.cssText)

    def test_media(self):
        "CSSImportRule.media"
        self.r.href = 'x'  # @import url(x)

        # media is readonly
        self.assertRaises(AttributeError, self.r.__setattr__, 'media', None)

        # but not static
        self.r.media.mediaText = 'print'
        self.assertEqual('@import url(x) print;', self.r.cssText)
        self.r.media.appendMedium('tv')
        self.assertEqual('@import url(x) print, tv;', self.r.cssText)

        tv_msg = re.escape(
            '''MediaList: Ignoring new medium '''
            '''cssutils.stylesheets.MediaQuery(mediaText='tv') '''
            '''as already specified "all" (set ``mediaText`` instead).'''
        )

        # for generated rule
        r = cssutils.css.CSSImportRule(href='x')
        with pytest.raises(xml.dom.InvalidModificationErr, match=tv_msg):
            r.media.appendMedium('tv')
        self.assertEqual('@import url(x);', r.cssText)
        with pytest.raises(xml.dom.InvalidModificationErr, match=tv_msg):
            r.media.appendMedium('tv')
        self.assertEqual('@import url(x);', r.cssText)
        r.media.mediaText = 'tv'
        self.assertEqual('@import url(x) tv;', r.cssText)
        r.media.appendMedium('print')  # all + tv = all!
        self.assertEqual('@import url(x) tv, print;', r.cssText)

        # for parsed rule without initial media
        s = cssutils.parseString('@import url(x);')
        r = s.cssRules[0]

        with pytest.raises(xml.dom.InvalidModificationErr, match=tv_msg):
            r.media.appendMedium('tv')
        self.assertEqual('@import url(x);', r.cssText)
        with pytest.raises(xml.dom.InvalidModificationErr, match=tv_msg):
            r.media.appendMedium('tv')
        self.assertEqual('@import url(x);', r.cssText)
        r.media.mediaText = 'tv'
        self.assertEqual('@import url(x) tv;', r.cssText)
        r.media.appendMedium('print')  # all + tv = all!
        self.assertEqual('@import url(x) tv, print;', r.cssText)

    def test_name(self):
        "CSSImportRule.name"
        r = cssutils.css.CSSImportRule('x', name='a000000')
        self.assertEqual('a000000', r.name)
        self.assertEqual('@import url(x) "a000000";', r.cssText)

        r.name = "n"
        self.assertEqual('n', r.name)
        self.assertEqual('@import url(x) "n";', r.cssText)
        r.name = '"'
        self.assertEqual('"', r.name)
        self.assertEqual('@import url(x) "\\"";', r.cssText)

        r.hreftype = 'string'
        self.assertEqual('@import "x" "\\"";', r.cssText)
        r.name = "123"
        self.assertEqual('@import "x" "123";', r.cssText)

        r.name = None
        self.assertEqual(None, r.name)
        self.assertEqual('@import "x";', r.cssText)

        r.name = ""
        self.assertEqual(None, r.name)
        self.assertEqual('@import "x";', r.cssText)

        self.assertRaises(xml.dom.SyntaxErr, r._setName, 0)
        self.assertRaises(xml.dom.SyntaxErr, r._setName, 123)

    def test_styleSheet(self):
        "CSSImportRule.styleSheet"

        def fetcher(url):
            if url == "/root/level1/anything.css":
                return None, '@import "level2/css.css" "title2";'
            else:
                return None, 'a { color: red }'

        parser = cssutils.CSSParser(fetcher=fetcher)
        sheet = parser.parseString(
            '''@charset "ascii";
                                   @import "level1/anything.css" tv "title";''',
            href='/root/',
        )

        self.assertEqual(sheet.href, '/root/')

        ir = sheet.cssRules[1]
        self.assertEqual(ir.href, 'level1/anything.css')
        self.assertEqual(ir.styleSheet.href, '/root/level1/anything.css')
        # inherits ascii as no self charset is set
        self.assertEqual(ir.styleSheet.encoding, 'ascii')
        self.assertEqual(ir.styleSheet.ownerRule, ir)
        self.assertEqual(ir.styleSheet.media.mediaText, 'tv')
        self.assertEqual(ir.styleSheet.parentStyleSheet, None)  # sheet
        self.assertEqual(ir.styleSheet.title, 'title')
        self.assertEqual(
            ir.styleSheet.cssText,
            '@charset "ascii";\n@import "level2/css.css" "title2";'.encode(),
        )

        ir2 = ir.styleSheet.cssRules[1]
        self.assertEqual(ir2.href, 'level2/css.css')
        self.assertEqual(ir2.styleSheet.href, '/root/level1/level2/css.css')
        # inherits ascii as no self charset is set
        self.assertEqual(ir2.styleSheet.encoding, 'ascii')
        self.assertEqual(ir2.styleSheet.ownerRule, ir2)
        self.assertEqual(ir2.styleSheet.media.mediaText, 'all')
        self.assertEqual(ir2.styleSheet.parentStyleSheet, None)  # ir.styleSheet
        self.assertEqual(ir2.styleSheet.title, 'title2')
        self.assertEqual(
            ir2.styleSheet.cssText,
            '@charset "ascii";\na {\n    color: red\n    }'.encode(),
        )

        sheet = cssutils.parseString('@import "CANNOT-FIND.css";')
        ir = sheet.cssRules[0]
        self.assertEqual(ir.href, "CANNOT-FIND.css")
        self.assertEqual(type(ir.styleSheet), cssutils.css.CSSStyleSheet)

        def fetcher(url):
            if url.endswith('level1.css'):
                return None, '@charset "ascii"; @import "level2.css";'.encode()
            else:
                return None, 'a { color: red }'.encode()

        parser = cssutils.CSSParser(fetcher=fetcher)

        sheet = parser.parseString('@charset "iso-8859-1";@import "level1.css";')
        self.assertEqual(sheet.encoding, 'iso-8859-1')

        sheet = sheet.cssRules[1].styleSheet
        self.assertEqual(sheet.encoding, 'ascii')

        sheet = sheet.cssRules[1].styleSheet
        self.assertEqual(sheet.encoding, 'ascii')

    def test_incomplete(self):
        "CSSImportRule (incomplete)"
        tests = {
            '@import "x.css': '@import "x.css";',
            "@import 'x": '@import "x";',
            # TODO:
            "@import url(x": '@import url(x);',
            "@import url('x": '@import url(x);',
            '@import url("x;': '@import url("x;");',
            '@import url( "x;': '@import url("x;");',
            '@import url("x ': '@import url("x ");',
            '@import url(x ': '@import url(x);',
            '''@import "a
                @import "b";
                @import "c";''': '@import "c";',
        }
        self.do_equal_p(tests, raising=False)  # parse

    def test_InvalidModificationErr(self):
        "CSSImportRule.cssText InvalidModificationErr"
        self._test_InvalidModificationErr('@import')

    def test_reprANDstr(self):
        "CSSImportRule.__repr__(), .__str__()"
        href = 'x.css'
        mediaText = 'tv, print'
        name = 'name'
        s = cssutils.css.CSSImportRule(href=href, mediaText=mediaText, name=name)

        # str(): mediaText nor name are present here
        self.assertTrue(href in str(s))

        # repr()
        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(href == s2.href)
        self.assertTrue(mediaText == s2.media.mediaText)
        self.assertTrue(name == s2.name)


if __name__ == '__main__':
    import unittest

    unittest.main()
