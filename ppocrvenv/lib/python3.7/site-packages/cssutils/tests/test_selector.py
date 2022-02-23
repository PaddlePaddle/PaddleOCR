"""Testcases for cssutils.css.selector.Selector.

what should happen here?
    - star 7 hack::
        x*
        does not validate but works in IE>5 and FF, does it???

"""
import xml.dom
from . import basetest
import cssutils


class SelectorTestCase(basetest.BaseTestCase):
    def setUp(self):
        self.r = cssutils.css.Selector('*')

    def test_init(self):
        "Selector.__init__()"
        s = cssutils.css.Selector('*')
        self.assertEqual((None, '*'), s.element)
        self.assertEqual({}, s._namespaces.namespaces)
        self.assertEqual(None, s.parent)
        self.assertEqual('*', s.selectorText)
        self.assertEqual((0, 0, 0, 0), s.specificity)
        self.assertEqual(True, s.wellformed)

        s = cssutils.css.Selector(('p|b', {'p': 'URI'}))
        self.assertEqual(('URI', 'b'), s.element)
        self.assertEqual({'p': 'URI'}, s._namespaces.namespaces)
        self.assertEqual(None, s.parent)
        self.assertEqual('p|b', s.selectorText)
        self.assertEqual((0, 0, 0, 1), s.specificity)
        self.assertEqual(True, s.wellformed)

        self.assertRaisesEx(xml.dom.NamespaceErr, cssutils.css.Selector, 'p|b')

    def test_element(self):
        "Selector.element (TODO: RESOLVE)"
        tests = {
            '*': (None, '*'),
            'x': (None, 'x'),
            '\\x': (None, '\\x'),
            '|x': ('', 'x'),
            '*|x': (cssutils._ANYNS, 'x'),
            'ex|x': ('example', 'x'),
            'a x': (None, 'x'),
            'a+x': (None, 'x'),
            'a>x': (None, 'x'),
            'a~x': (None, 'x'),
            'a+b~c x': (None, 'x'),
            'x[href]': (None, 'x'),
            'x[href="123"]': (None, 'x'),
            'x:hover': (None, 'x'),
            'x:first-letter': (None, 'x'),  # TODO: Really?
            'x::first-line': (None, 'x'),  # TODO: Really?
            'x:not(href)': (None, 'x'),  # TODO: Really?
            '#id': None,
            '.c': None,
            'x#id': (None, 'x'),
            'x.c': (None, 'x'),
        }
        for test, ele in list(tests.items()):
            s = cssutils.css.Selector((test, {'ex': 'example'}))
            self.assertEqual(ele, s.element)

    def test_namespaces(self):
        "Selector.namespaces"
        namespaces = [
            {'p': 'other'},  # no default
            {'': 'default', 'p': 'other'},  # with default
            {'': 'default', 'p': 'default'},  # same default
        ]
        tests = {
            # selector: with default, no default, same default
            '*': ('*', '*', '*'),
            'x': ('x', 'x', 'x'),
            '|*': ('|*', '|*', '|*'),
            '|x': ('|x', '|x', '|x'),
            '*|*': ('*|*', '*|*', '*|*'),
            '*|x': ('*|x', '*|x', '*|x'),
            'p|*': ('p|*', 'p|*', '*'),
            'p|x': ('p|x', 'p|x', 'x'),
            'x[a][|a][*|a][p|a]': (
                'x[a][a][*|a][p|a]',
                'x[a][a][*|a][p|a]',
                'x[a][a][*|a][a]',
            ),
        }
        for sel, exp in list(tests.items()):
            for i, result in enumerate(exp):
                s = cssutils.css.Selector((sel, namespaces[i]))
                self.assertEqual(result, s.selectorText)

        # add to CSSStyleSheet
        sheet = cssutils.css.CSSStyleSheet()
        sheet.cssText = '@namespace p "u"; a { color: green }'

        r = sheet.cssRules[1]

        self.assertEqual(r.selectorText, 'a')

        # add default namespace
        sheet.namespaces[''] = 'a'
        self.assertEqual(r.selectorText, '|a')

        del sheet.namespaces['']
        self.assertEqual(r.selectorText, 'a')

    #        r.selectorList.append('a')
    #        self.assertEqual(r.selectorText, u'|a, a')
    #        r.selectorList.append('*|a')
    #        self.assertEqual(r.selectorText, u'|a, a, *|a')

    def test_default_namespace(self):
        "Selector.namespaces default"
        css = '''@namespace "default";
                a[att] { color:green; }
        '''
        sheet = cssutils.css.CSSStyleSheet()
        sheet.cssText = css
        self.assertEqual(
            sheet.cssText,
            '@namespace "default";\na[att] {\n    color: green\n    }'.encode(),
        )
        # use a prefix for default namespace, does not goes for atts!
        sheet.namespaces['p'] = 'default'
        self.assertEqual(
            sheet.cssText,
            '@namespace p "default";\np|a[att] {\n    color: green\n    }'.encode(),
        )

    def test_parent(self):
        "Selector.parent"
        sl = cssutils.css.SelectorList('a, b')
        for sel in sl:
            self.assertEqual(sl, sel.parent)

        newsel = cssutils.css.Selector('x')
        sl.append(newsel)
        self.assertEqual(sl, newsel.parent)

        newsel = cssutils.css.Selector('y')
        sl.appendSelector(newsel)
        self.assertEqual(sl, newsel.parent)

    def test_selectorText(self):
        "Selector.selectorText"
        tests = {
            # combinators
            'a+b>c~e f': 'a + b > c ~ e f',
            'a  +  b  >  c  ~  e   f': 'a + b > c ~ e f',
            'a+b': 'a + b',
            'a  +  b': 'a + b',
            'a\n  +\t  b': 'a + b',
            'a~b': 'a ~ b',
            'a b': None,
            'a   b': 'a b',
            'a\nb': 'a b',
            'a\tb': 'a b',
            'a   #b': 'a #b',
            'a   .b': 'a .b',
            'a * b': None,
            # >
            'a>b': 'a > b',
            'a> b': 'a > b',
            'a >b': 'a > b',
            'a > b': 'a > b',
            # +
            'a+b': 'a + b',
            'a+ b': 'a + b',
            'a +b': 'a + b',
            'a + b': 'a + b',
            # ~
            'a~b': 'a ~ b',
            'a~ b': 'a ~ b',
            'a ~b': 'a ~ b',
            'a ~ b': 'a ~ b',
            # type selector
            'a': None,
            'h1-a_x__--': None,
            'a-a': None,
            'a_a': None,
            '-a': None,
            '_': None,
            '-_': None,
            r'-\72': '-r',
            # ur'\25': u'%', # TODO: should be escaped!
            '.a a': None,
            'a1': None,
            'a1-1': None,
            '.a1-1': None,
            # universal
            '*': None,
            '*/*x*/': None,
            '* /*x*/': None,
            '*:hover': None,
            '* :hover': None,
            '*:lang(fr)': None,
            '* :lang(fr)': None,
            '*::first-line': None,
            '* ::first-line': None,
            '*[lang=fr]': None,
            '[lang=fr]': None,
            # HASH
            '''#a''': None,
            '''#a1''': None,
            '''#1a''': None,  # valid to grammar but not for HTML
            '''#1''': None,  # valid to grammar but not for HTML
            '''a#b''': None,
            '''a #b''': None,
            '''a#b.c''': None,
            '''a.c#b''': None,
            '''a #b.c''': None,
            '''a .c#b''': None,
            # class
            'ab': 'ab',
            'a.b': None,
            'a.b.c': None,
            '.a1._1': None,
            # attrib
            '''[x]''': None,
            '''*[x]''': None,
            '''a[x]''': None,
            '''a[ x]''': 'a[x]',
            '''a[x ]''': 'a[x]',
            '''a [x]''': 'a [x]',
            '''* [x]''': None,  # is really * *[x]
            '''a[x="1"]''': None,
            '''a[x ="1"]''': 'a[x="1"]',
            '''a[x= "1"]''': 'a[x="1"]',
            '''a[x = "1"]''': 'a[x="1"]',
            '''a[ x = "1"]''': 'a[x="1"]',
            '''a[x = "1" ]''': 'a[x="1"]',
            '''a[ x = "1" ]''': 'a[x="1"]',
            '''a [ x = "1" ]''': 'a [x="1"]',
            '''a[x~=a1]''': None,
            '''a[x ~=a1]''': 'a[x~=a1]',
            '''a[x~= a1]''': 'a[x~=a1]',
            '''a[x ~= a1]''': 'a[x~=a1]',
            '''a[ x ~= a1]''': 'a[x~=a1]',
            '''a[x ~= a1 ]''': 'a[x~=a1]',
            '''a[ x ~= a1 ]''': 'a[x~=a1]',
            '''a [ x ~= a1 ]''': 'a [x~=a1]',  # same as next!
            '''a *[ x ~= a1 ]''': 'a *[x~=a1]',
            '''a[x|=en]''': None,
            '''a[x|= en]''': 'a[x|=en]',
            '''a[x |=en]''': 'a[x|=en]',
            '''a[x |= en]''': 'a[x|=en]',
            '''a[ x |= en]''': 'a[x|=en]',
            '''a[x |= en ]''': 'a[x|=en]',
            '''a[ x |= en]''': 'a[x|=en]',
            '''a [ x |= en]''': 'a [x|=en]',
            # CSS3
            '''a[x^=en]''': None,
            '''a[x$=en]''': None,
            '''a[x*=en]''': None,
            '''a[/*1*/x/*2*/]''': None,
            '''a[/*1*/x/*2*/=/*3*/a/*4*/]''': None,
            '''a[/*1*/x/*2*/~=/*3*/a/*4*/]''': None,
            '''a[/*1*/x/*2*/|=/*3*/a/*4*/]''': None,
            # pseudo-elements
            'a x:first-line': None,
            'a x:first-letter': None,
            'a x:before': None,
            'a x:after': None,
            'a x::selection': None,
            'a:hover+b:hover>c:hover~e:hover f:hover': 'a:hover + b:hover > c:hover ~ e:hover f:hover',
            'a:hover  +  b:hover  >  c:hover  ~  e:hover   f:hover': 'a:hover + b:hover > c:hover ~ e:hover f:hover',
            'a::selection+b::selection>c::selection~e::selection f::selection': 'a::selection + b::selection > c::selection ~ e::selection f::selection',
            'a::selection  +  b::selection  >  c::selection  ~  e::selection   '
            'f::selection': 'a::selection + b::selection > c::selection ~ e::selection f::selection',
            'x:lang(de) y': None,
            'x:nth-child(odd) y': None,
            # functional pseudo
            'x:func(a + b-2px22.3"s"i)': None,
            'x:func(1 + 1)': None,
            'x:func(1+1)': 'x:func(1+1)',
            'x:func(1   +   1)': 'x:func(1 + 1)',
            'x:func(1-1)': 'x:func(1-1)',
            'x:func(1  -  1)': 'x:func(1 -1)',  # TODO: FIX!
            'x:func(a-1)': 'x:func(a-1)',
            'x:func(a -1px)': 'x:func(a -1px)',
            'x:func(1px)': None,
            'x:func(23.4)': None,
            'x:func("s")': None,
            'x:func(i)': None,
            # negation
            ':not(y)': None,
            ':not(   y  \t\n)': ':not(y)',
            '*:not(y)': None,
            'x:not(y)': None,
            '.x:not(y)': None,
            ':not(*)': None,
            ':not(#a)': None,
            ':not(.a)': None,
            ':not([a])': None,
            ':not(:first-letter)': None,
            ':not(::first-letter)': None,
            # escapes
            r'\74\72 td': 'trtd',
            r'\74\72  td': 'tr td',
            r'\74\000072 td': 'trtd',
            r'\74\000072  td': 'tr td',
            # comments
            'a/**/ b': None,
            'a /**/b': None,
            'a /**/ b': None,
            'a  /**/ b': 'a /**/ b',
            'a /**/  b': 'a /**/ b',
            # namespaces
            '|e': None,
            '*|e': None,
            '*|*': None,
            ('p|*', (('p', 'uri'),)): 'p|*',
            ('p|e', (('p', 'uri'),)): 'p|e',
            ('-a_x12|e', (('-a_x12', 'uri'),)): '-a_x12|e',
            ('*|b[p|a]', (('p', 'uri'),)): '*|b[p|a]',
            # case
            'elemenT.clasS#iD[atT="valuE"]:noT(x)::firsT-linE': 'elemenT.clasS#iD[atT="valuE"]:not(x)::first-line',
        }
        # do not parse as not complete
        self.do_equal_r(tests, att='selectorText')

        tests = {
            'x|a': xml.dom.NamespaceErr,
            ('p|*', (('x', 'uri'),)): xml.dom.NamespaceErr,
            '': xml.dom.SyntaxErr,
            '1': xml.dom.SyntaxErr,
            '-1': xml.dom.SyntaxErr,
            'a*b': xml.dom.SyntaxErr,
            'a *b': xml.dom.SyntaxErr,
            'a* b': xml.dom.SyntaxErr,
            'a/**/b': xml.dom.SyntaxErr,
            '#': xml.dom.SyntaxErr,
            '|': xml.dom.SyntaxErr,
            ':': xml.dom.SyntaxErr,
            '::': xml.dom.SyntaxErr,
            ': a': xml.dom.SyntaxErr,
            ':: a': xml.dom.SyntaxErr,
            ':a()': xml.dom.SyntaxErr,  # no value
            '::a()': xml.dom.SyntaxErr,  # no value
            ':::a': xml.dom.SyntaxErr,
            ':1': xml.dom.SyntaxErr,
            '#.x': xml.dom.SyntaxErr,
            '.': xml.dom.SyntaxErr,
            '.1': xml.dom.SyntaxErr,
            '.a.1': xml.dom.SyntaxErr,
            '[a': xml.dom.SyntaxErr,
            'a]': xml.dom.SyntaxErr,
            '[a b]': xml.dom.SyntaxErr,
            '[=b]': xml.dom.SyntaxErr,
            '[a=]': xml.dom.SyntaxErr,
            '[a|=]': xml.dom.SyntaxErr,
            '[a~=]': xml.dom.SyntaxErr,
            '[a=1]': xml.dom.SyntaxErr,
            'a +': xml.dom.SyntaxErr,
            'a >': xml.dom.SyntaxErr,
            'a ++ b': xml.dom.SyntaxErr,
            'a + > b': xml.dom.SyntaxErr,
            # functional pseudo
            '*:lang(': xml.dom.SyntaxErr,
            '*:lang()': xml.dom.SyntaxErr,  # no arg
            # negation
            'not(x)': xml.dom.SyntaxErr,  # no valid function
            ':not()': xml.dom.SyntaxErr,  # no arg
            ':not(x': xml.dom.SyntaxErr,  # no )
            ':not(-': xml.dom.SyntaxErr,  # not allowed
            ':not(+': xml.dom.SyntaxErr,  # not allowed
            # only one selector!
            ',': xml.dom.InvalidModificationErr,
            ',a': xml.dom.InvalidModificationErr,
            'a,': xml.dom.InvalidModificationErr,
            # @
            'p @here': xml.dom.SyntaxErr,  # not allowed
        }
        # only set as not complete
        self.do_raise_r(tests, att='_setSelectorText')

    def test_specificity(self):
        "Selector.specificity"
        selector = cssutils.css.Selector()

        # readonly
        def _set():
            selector.specificity = 1

        self.assertRaisesMsg(AttributeError, "can't set attribute", _set)

        tests = {
            '*': (0, 0, 0, 0),
            'li': (0, 0, 0, 1),
            'li:first-line': (0, 0, 0, 2),
            'ul li': (0, 0, 0, 2),
            'ul ol+li': (0, 0, 0, 3),
            'h1 + *[rel=up]': (0, 0, 1, 1),
            'ul ol li.red': (0, 0, 1, 3),
            'li.red.level': (0, 0, 2, 1),
            '#x34y': (0, 1, 0, 0),
            'UL OL LI.red': (0, 0, 1, 3),
            'LI.red.level': (0, 0, 2, 1),
            '#s12:not(FOO)': (0, 1, 0, 1),
            'button:not([DISABLED])': (0, 0, 1, 1),  # ?
            '*:not(FOO)': (0, 0, 0, 1),
            # elements
            'a+b': (0, 0, 0, 2),
            'a>b': (0, 0, 0, 2),
            'a b': (0, 0, 0, 2),
            '* a': (0, 0, 0, 1),
            'a *': (0, 0, 0, 1),
            'a * b': (0, 0, 0, 2),
            'a:hover': (0, 0, 0, 1),
            'a:first-line': (0, 0, 0, 2),
            'a:first-letter': (0, 0, 0, 2),
            'a:before': (0, 0, 0, 2),
            'a:after': (0, 0, 0, 2),
            # classes and attributes
            '.a': (0, 0, 1, 0),
            '*.a': (0, 0, 1, 0),
            'a.a': (0, 0, 1, 1),
            '.a.a': (0, 0, 2, 0),  # IE<7 False (0,0,1,0)
            'a.a.a': (0, 0, 2, 1),
            '.a.b': (0, 0, 2, 0),
            'a.a.b': (0, 0, 2, 1),
            '.a .a': (0, 0, 2, 0),
            '*[x]': (0, 0, 1, 0),
            '*[x]': (0, 0, 1, 0),
            '*[x]': (0, 0, 1, 0),
            '*[x=a]': (0, 0, 1, 0),
            '*[x~=a]': (0, 0, 1, 0),
            '*[x|=a]': (0, 0, 1, 0),
            '*[x^=a]': (0, 0, 1, 0),
            '*[x*=a]': (0, 0, 1, 0),
            '*[x$=a]': (0, 0, 1, 0),
            '*[x][y]': (0, 0, 2, 0),
            # ids
            '#a': (0, 1, 0, 0),
            '*#a': (0, 1, 0, 0),
            'x#a': (0, 1, 0, 1),
            '.x#a': (0, 1, 1, 0),
            'a.x#a': (0, 1, 1, 1),
            '#a#a': (0, 2, 0, 0),  # e.g. html:id + xml:id
            '#a#b': (0, 2, 0, 0),
            '#a #b': (0, 2, 0, 0),
        }
        for text in tests:
            selector.selectorText = text
            self.assertEqual(tests[text], selector.specificity)

    def test_reprANDstr(self):
        "Selector.__repr__(), .__str__()"
        sel = 'a + b'

        s = cssutils.css.Selector(selectorText=sel)

        self.assertTrue(sel in str(s))

        s2 = eval(repr(s))
        self.assertTrue(isinstance(s2, s.__class__))
        self.assertTrue(sel == s2.selectorText)


if __name__ == '__main__':
    import unittest

    unittest.main()
