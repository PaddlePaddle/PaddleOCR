"""Testcases for cssutils.util"""


import cgi
import re
import urllib.request
import urllib.error
import urllib.parse

try:
    import mock
except ImportError:
    mock = None
    print("install mock library to run all tests")

from . import basetest

from cssutils.util import Base, ListSeq, _readUrl, _defaultFetcher, LazyRegex


class ListSeqTestCase(basetest.BaseTestCase):
    def test_all(self):
        "util.ListSeq"
        ls = ListSeq()
        self.assertEqual(0, len(ls))
        # append()
        self.assertRaises(NotImplementedError, ls.append, 1)
        # set
        self.assertRaises(NotImplementedError, ls.__setitem__, 0, 1)

        # hack:
        ls.seq.append(1)
        ls.seq.append(2)

        # len
        self.assertEqual(2, len(ls))
        # __contains__
        self.assertEqual(True, 1 in ls)
        # get
        self.assertEqual(1, ls[0])
        self.assertEqual(2, ls[1])
        # del
        del ls[0]
        self.assertEqual(1, len(ls))
        self.assertEqual(False, 1 in ls)
        # for in
        for x in ls:
            self.assertEqual(2, x)


class BaseTestCase(basetest.BaseTestCase):
    def test_normalize(self):
        "Base._normalize()"
        b = Base()
        tests = {
            'abcdefg ABCDEFG äöüß€ AÖÜ': 'abcdefg abcdefg äöüß€ aöü',
            r'\ga\Ga\\\ ': r'gaga\ ',
            r'0123456789': '0123456789',
            # unicode escape seqs should have been done by
            # the tokenizer...
        }
        for test, exp in list(tests.items()):
            self.assertEqual(b._normalize(test), exp)
            # static too
            self.assertEqual(Base._normalize(test), exp)

    def test_tokenupto(self):  # noqa: C901
        "Base._tokensupto2()"

        # tests nested blocks of {} [] or ()
        b = Base()

        tests = [
            ('default', 'a[{1}]({2}) { } NOT', 'a[{1}]({2}) { }', False),
            ('default', 'a[{1}]({2}) { } NOT', 'a[{1}]func({2}) { }', True),
            ('blockstartonly', 'a[{1}]({2}) { NOT', 'a[{1}]({2}) {', False),
            ('blockstartonly', 'a[{1}]({2}) { NOT', 'a[{1}]func({2}) {', True),
            ('propertynameendonly', 'a[(2)1] { }2 : a;', 'a[(2)1] { }2 :', False),
            ('propertynameendonly', 'a[(2)1] { }2 : a;', 'a[func(2)1] { }2 :', True),
            (
                'propertyvalueendonly',
                'a{;{;}[;](;)}[;{;}[;](;)](;{;}[;](;)) 1; NOT',
                'a{;{;}[;](;)}[;{;}[;](;)](;{;}[;](;)) 1;',
                False,
            ),
            (
                'propertyvalueendonly',
                'a{;{;}[;](;)}[;{;}[;](;)](;{;}[;](;)) 1; NOT',
                'a{;{;}[;]func(;)}[;{;}[;]func(;)]func(;{;}[;]func(;)) 1;',
                True,
            ),
            (
                'funcendonly',
                'a{[1]}([3])[{[1]}[2]([3])]) NOT',
                'a{[1]}([3])[{[1]}[2]([3])])',
                False,
            ),
            (
                'funcendonly',
                'a{[1]}([3])[{[1]}[2]([3])]) NOT',
                'a{[1]}func([3])[{[1]}[2]func([3])])',
                True,
            ),
            (
                'selectorattendonly',
                '[a[()]{()}([()]{()}())] NOT',
                '[a[()]{()}([()]{()}())]',
                False,
            ),
            (
                'selectorattendonly',
                '[a[()]{()}([()]{()}())] NOT',
                '[a[func()]{func()}func([func()]{func()}func())]',
                True,
            ),
            # issue 50
            ('withstarttoken [', 'a];x', '[a];', False),
        ]

        for typ, values, exp, paransasfunc in tests:

            def maketokens(valuelist):
                # returns list of tuples
                return [('TYPE', v, 0, 0) for v in valuelist]

            tokens = maketokens(list(values))
            if paransasfunc:
                for i, t in enumerate(tokens):
                    if '(' == t[1]:
                        tokens[i] = ('FUNCTION', 'func(', t[2], t[3])

            if 'default' == typ:
                restokens = b._tokensupto2(tokens)
            elif 'blockstartonly' == typ:
                restokens = b._tokensupto2(tokens, blockstartonly=True)
            elif 'propertynameendonly' == typ:
                restokens = b._tokensupto2(tokens, propertynameendonly=True)
            elif 'propertyvalueendonly' == typ:
                restokens = b._tokensupto2(tokens, propertyvalueendonly=True)
            elif 'funcendonly' == typ:
                restokens = b._tokensupto2(tokens, funcendonly=True)
            elif 'selectorattendonly' == typ:
                restokens = b._tokensupto2(tokens, selectorattendonly=True)
            elif 'withstarttoken [' == typ:
                restokens = b._tokensupto2(tokens, ('CHAR', '[', 0, 0))

            res = ''.join([t[1] for t in restokens])
            self.assertEqual(exp, res)


class _readUrl_TestCase(basetest.BaseTestCase):
    """needs mock"""

    def test_readUrl(self):
        """util._readUrl()"""
        # for additional tests see test_parse.py
        url = 'http://example.com/test.css'

        def make_fetcher(r):
            # normally r == encoding, content
            def fetcher(url):
                return r

            return fetcher

        tests = {
            # defaultFetcher returns: readUrl returns
            None: (None, None, None),
            (None, ''): ('utf-8', 5, ''),
            (None, '€'.encode('utf-8')): ('utf-8', 5, '€'),
            ('utf-8', '€'.encode('utf-8')): ('utf-8', 1, '€'),
            ('ISO-8859-1', 'ä'.encode('iso-8859-1')): ('ISO-8859-1', 1, 'ä'),
            ('ASCII', 'a'.encode('ascii')): ('ASCII', 1, 'a'),
        }

        for r, exp in list(tests.items()):
            self.assertEqual(_readUrl(url, fetcher=make_fetcher(r)), exp)

        tests = {
            # (overrideEncoding, parentEncoding, (httpencoding, content)):
            #                        readUrl returns
            # ===== 0. OVERRIDE WINS =====
            # override + parent + http
            ('latin1', 'ascii', ('utf-16', ''.encode())): ('latin1', 0, ''),
            ('latin1', 'ascii', ('utf-16', '123'.encode())): ('latin1', 0, '123'),
            ('latin1', 'ascii', ('utf-16', 'ä'.encode('iso-8859-1'))): (
                'latin1',
                0,
                'ä',
            ),
            ('latin1', 'ascii', ('utf-16', 'a'.encode('ascii'))): ('latin1', 0, 'a'),
            # + @charset
            ('latin1', 'ascii', ('utf-16', '@charset "ascii";'.encode())): (
                'latin1',
                0,
                '@charset "latin1";',
            ),
            ('latin1', 'ascii', ('utf-16', '@charset "utf-8";ä'.encode('latin1'))): (
                'latin1',
                0,
                '@charset "latin1";ä',
            ),
            ('latin1', 'ascii', ('utf-16', '@charset "utf-8";ä'.encode('utf-8'))): (
                'latin1',
                0,
                '@charset "latin1";\xc3\xa4',
            ),  # read as latin1!
            # override only
            ('latin1', None, None): (None, None, None),
            ('latin1', None, (None, ''.encode())): ('latin1', 0, ''),
            ('latin1', None, (None, '123'.encode())): ('latin1', 0, '123'),
            ('latin1', None, (None, 'ä'.encode('iso-8859-1'))): ('latin1', 0, 'ä'),
            ('latin1', None, (None, 'a'.encode('ascii'))): ('latin1', 0, 'a'),
            # + @charset
            ('latin1', None, (None, '@charset "ascii";'.encode())): (
                'latin1',
                0,
                '@charset "latin1";',
            ),
            ('latin1', None, (None, '@charset "utf-8";ä'.encode('latin1'))): (
                'latin1',
                0,
                '@charset "latin1";ä',
            ),
            ('latin1', None, (None, '@charset "utf-8";ä'.encode('utf-8'))): (
                'latin1',
                0,
                '@charset "latin1";\xc3\xa4',
            ),  # read as latin1!
            # override + parent
            ('latin1', 'ascii', None): (None, None, None),
            ('latin1', 'ascii', (None, ''.encode())): ('latin1', 0, ''),
            ('latin1', 'ascii', (None, '123'.encode())): ('latin1', 0, '123'),
            ('latin1', 'ascii', (None, 'ä'.encode('iso-8859-1'))): ('latin1', 0, 'ä'),
            ('latin1', 'ascii', (None, 'a'.encode('ascii'))): ('latin1', 0, 'a'),
            # + @charset
            ('latin1', 'ascii', (None, '@charset "ascii";'.encode())): (
                'latin1',
                0,
                '@charset "latin1";',
            ),
            ('latin1', 'ascii', (None, '@charset "utf-8";ä'.encode('latin1'))): (
                'latin1',
                0,
                '@charset "latin1";ä',
            ),
            ('latin1', 'ascii', (None, '@charset "utf-8";ä'.encode('utf-8'))): (
                'latin1',
                0,
                '@charset "latin1";\xc3\xa4',
            ),  # read as latin1!
            # override + http
            ('latin1', None, ('utf-16', ''.encode())): ('latin1', 0, ''),
            ('latin1', None, ('utf-16', '123'.encode())): ('latin1', 0, '123'),
            ('latin1', None, ('utf-16', 'ä'.encode('iso-8859-1'))): ('latin1', 0, 'ä'),
            ('latin1', None, ('utf-16', 'a'.encode('ascii'))): ('latin1', 0, 'a'),
            # + @charset
            ('latin1', None, ('utf-16', '@charset "ascii";'.encode())): (
                'latin1',
                0,
                '@charset "latin1";',
            ),
            ('latin1', None, ('utf-16', '@charset "utf-8";ä'.encode('latin1'))): (
                'latin1',
                0,
                '@charset "latin1";ä',
            ),
            ('latin1', None, ('utf-16', '@charset "utf-8";ä'.encode('utf-8'))): (
                'latin1',
                0,
                '@charset "latin1";\xc3\xa4',
            ),  # read as latin1!
            # override ü @charset
            ('latin1', None, (None, '@charset "ascii";'.encode())): (
                'latin1',
                0,
                '@charset "latin1";',
            ),
            ('latin1', None, (None, '@charset "utf-8";ä'.encode('latin1'))): (
                'latin1',
                0,
                '@charset "latin1";ä',
            ),
            ('latin1', None, (None, '@charset "utf-8";ä'.encode('utf-8'))): (
                'latin1',
                0,
                '@charset "latin1";\xc3\xa4',
            ),  # read as latin1!
            # ===== 1. HTTP WINS =====
            (None, 'ascii', ('latin1', ''.encode())): ('latin1', 1, ''),
            (None, 'ascii', ('latin1', '123'.encode())): ('latin1', 1, '123'),
            (None, 'ascii', ('latin1', 'ä'.encode('iso-8859-1'))): ('latin1', 1, 'ä'),
            (None, 'ascii', ('latin1', 'a'.encode('ascii'))): ('latin1', 1, 'a'),
            # + @charset
            (None, 'ascii', ('latin1', '@charset "ascii";'.encode())): (
                'latin1',
                1,
                '@charset "latin1";',
            ),
            (None, 'ascii', ('latin1', '@charset "utf-8";ä'.encode('latin1'))): (
                'latin1',
                1,
                '@charset "latin1";ä',
            ),
            (None, 'ascii', ('latin1', '@charset "utf-8";ä'.encode('utf-8'))): (
                'latin1',
                1,
                '@charset "latin1";\xc3\xa4',
            ),  # read as latin1!
            # ===== 2. @charset WINS =====
            (None, 'ascii', (None, '@charset "latin1";'.encode())): (
                'latin1',
                2,
                '@charset "latin1";',
            ),
            (None, 'ascii', (None, '@charset "latin1";ä'.encode('latin1'))): (
                'latin1',
                2,
                '@charset "latin1";ä',
            ),
            (None, 'ascii', (None, '@charset "latin1";ä'.encode('utf-8'))): (
                'latin1',
                2,
                '@charset "latin1";\xc3\xa4',
            ),  # read as latin1!
            # ===== 2. BOM WINS =====
            (None, 'ascii', (None, 'ä'.encode('utf-8-sig'))): (
                'utf-8-sig',
                2,
                '\xe4',
            ),  # read as latin1!
            (None, 'ascii', (None, '@charset "utf-8";ä'.encode('utf-8-sig'))): (
                'utf-8-sig',
                2,
                '@charset "utf-8";\xe4',
            ),  # read as latin1!
            (None, 'ascii', (None, '@charset "latin1";ä'.encode('utf-8-sig'))): (
                'utf-8-sig',
                2,
                '@charset "utf-8";\xe4',
            ),  # read as latin1!
            # ===== 4. parentEncoding WINS =====
            (None, 'latin1', (None, ''.encode())): ('latin1', 4, ''),
            (None, 'latin1', (None, '123'.encode())): ('latin1', 4, '123'),
            (None, 'latin1', (None, 'ä'.encode('iso-8859-1'))): ('latin1', 4, 'ä'),
            (None, 'latin1', (None, 'a'.encode('ascii'))): ('latin1', 4, 'a'),
            (None, 'latin1', (None, 'ä'.encode('utf-8'))): (
                'latin1',
                4,
                '\xc3\xa4',
            ),  # read as latin1!
            # ===== 5. default WINS which in this case is None! =====
            (None, None, (None, ''.encode())): ('utf-8', 5, ''),
            (None, None, (None, '123'.encode())): ('utf-8', 5, '123'),
            (None, None, (None, 'a'.encode('ascii'))): ('utf-8', 5, 'a'),
            (None, None, (None, 'ä'.encode('utf-8'))): (
                'utf-8',
                5,
                'ä',
            ),  # read as utf-8
            (
                None,
                None,
                (None, 'ä'.encode('iso-8859-1')),
            ): (  # trigger UnicodeDecodeError!
                'utf-8',
                5,
                None,
            ),
        }
        for (override, parent, r), exp in list(tests.items()):
            self.assertEqual(
                _readUrl(
                    url,
                    overrideEncoding=override,
                    parentEncoding=parent,
                    fetcher=make_fetcher(r),
                ),
                exp,
            )

    def test_defaultFetcher(self):  # noqa: C901
        """util._defaultFetcher"""
        if mock:

            class Response(object):
                """urllib2.Reponse mock"""

                def __init__(
                    self, url, contenttype, content, exception=None, args=None
                ):
                    self.url = url

                    mt, params = cgi.parse_header(contenttype)
                    self.mimetype = mt
                    self.charset = params.get('charset', None)

                    self.text = content

                    self.exception = exception
                    self.args = args

                def geturl(self):
                    return self.url

                def info(self):
                    mimetype, charset = self.mimetype, self.charset

                    class Info(object):

                        # py2x
                        def gettype(self):
                            return mimetype

                        def getparam(self, name=None):
                            return charset

                        # py 3x
                        get_content_type = gettype
                        get_content_charset = getparam  # here always charset!

                    return Info()

                def read(self):
                    # returns fake text or raises fake exception
                    if not self.exception:
                        return self.text
                    else:
                        raise self.exception(*self.args)

            def urlopen(url, contenttype=None, content=None, exception=None, args=None):
                # return an mock which returns parameterized Response
                def x(*ignored):
                    if exception:
                        raise exception(*args)
                    else:
                        return Response(
                            url, contenttype, content, exception=exception, args=args
                        )

                return x

            urlopenpatch = 'urllib.request.urlopen'

            # positive tests
            tests = {
                # content-type, contentstr: encoding, contentstr
                ('text/css', '€'.encode('utf-8')): (None, '€'.encode('utf-8')),
                ('text/css;charset=utf-8', '€'.encode('utf-8')): (
                    'utf-8',
                    '€'.encode('utf-8'),
                ),
                ('text/css;charset=ascii', 'a'): ('ascii', 'a'),
            }
            url = 'http://example.com/test.css'
            for (contenttype, content), exp in list(tests.items()):

                @mock.patch(urlopenpatch, new=urlopen(url, contenttype, content))
                def do(url):
                    return _defaultFetcher(url)

                self.assertEqual(exp, do(url))

            # wrong mimetype
            @mock.patch(urlopenpatch, new=urlopen(url, 'text/html', 'a'))
            def do(url):
                return _defaultFetcher(url)

            self.assertRaises(ValueError, do, url)

            # calling url results in fake exception

            # py2 ~= py3 raises error earlier than urlopen!
            tests = {
                '1': (ValueError, ['invalid value for url']),
                # _readUrl('mailto:a.css')
                'mailto:e4': (urllib.error.URLError, ['urlerror']),
                # cannot resolve x, IOError
                'http://x': (urllib.error.URLError, ['ioerror']),
            }
            for url, (exception, args) in list(tests.items()):

                @mock.patch(
                    urlopenpatch, new=urlopen(url, exception=exception, args=args)
                )
                def do(url):
                    return _defaultFetcher(url)

                self.assertRaises(exception, do, url)

            urlrequestpatch = 'urllib.request.Request'
            tests = {
                # _readUrl('http://cthedot.de/__UNKNOWN__.css')
                'e2': (urllib.error.HTTPError, ['u', 500, 'server error', {}, None]),
                'e3': (urllib.error.HTTPError, ['u', 404, 'not found', {}, None]),
            }
            for url, (exception, args) in list(tests.items()):

                @mock.patch(
                    urlrequestpatch, new=urlopen(url, exception=exception, args=args)
                )
                def do(url):
                    return _defaultFetcher(url)

                self.assertRaises(exception, do, url)

        else:
            self.assertEqual(False, 'Mock needed for this test')


class TestLazyRegex(basetest.BaseTestCase):
    """Tests for cssutils.util.LazyRegex."""

    def setUp(self):
        self.lazyre = LazyRegex('f.o')

    def test_public_interface(self):
        methods = [
            'search',
            'match',
            'split',
            'sub',
            'subn',
            'findall',
            'finditer',
            'pattern',
            'flags',
            'groups',
            'groupindex',
        ]
        for method in methods:
            self.assertTrue(
                hasattr(self.lazyre, method), 'expected %r public attribute' % method
            )

    def test_ensure(self):
        self.assertIsNone(self.lazyre.matcher)
        self.lazyre.ensure()
        self.assertIsNotNone(self.lazyre.matcher)

    def test_calling(self):
        self.assertIsNone(self.lazyre('bar'))
        match = self.lazyre('foobar')
        self.assertEqual(match.group(), 'foo')

    def test_matching(self):
        self.assertIsNone(self.lazyre.match('bar'))
        match = self.lazyre.match('foobar')
        self.assertEqual(match.group(), 'foo')

    def test_matching_with_position_parameters(self):
        self.assertIsNone(self.lazyre.match('foo', 1))
        self.assertIsNone(self.lazyre.match('foo', 0, 2))

    def test_searching(self):
        self.assertIsNone(self.lazyre.search('rafuubar'))
        match = self.lazyre.search('rafoobar')
        self.assertEqual(match.group(), 'foo')

    def test_searching_with_position_parameters(self):
        self.assertIsNone(self.lazyre.search('rafoobar', 3))
        self.assertIsNone(self.lazyre.search('rafoobar', 0, 4))
        match = self.lazyre.search('rafoofuobar', 4)
        self.assertEqual(match.group(), 'fuo')

    def test_split(self):
        self.assertEqual(self.lazyre.split('rafoobarfoobaz'), ['ra', 'bar', 'baz'])
        self.assertEqual(self.lazyre.split('rafoobarfoobaz', 1), ['ra', 'barfoobaz'])

    def test_findall(self):
        self.assertEqual(self.lazyre.findall('rafoobarfuobaz'), ['foo', 'fuo'])

    def test_finditer(self):
        result = self.lazyre.finditer('rafoobarfuobaz')
        self.assertEqual([m.group() for m in result], ['foo', 'fuo'])

    def test_sub(self):
        self.assertEqual(self.lazyre.sub('bar', 'foofoo'), 'barbar')
        self.assertEqual(self.lazyre.sub(lambda x: 'baz', 'foofoo'), 'bazbaz')

    def test_subn(self):
        subbed = self.lazyre.subn('bar', 'foofoo')
        self.assertEqual(subbed, ('barbar', 2))
        subbed = self.lazyre.subn(lambda x: 'baz', 'foofoo')
        self.assertEqual(subbed, ('bazbaz', 2))

    def test_groups(self):
        lazyre = LazyRegex('(.)(.)')
        self.assertIsNone(lazyre.groups)
        lazyre.ensure()
        self.assertEqual(lazyre.groups, 2)

    def test_groupindex(self):
        lazyre = LazyRegex('(?P<foo>.)')
        self.assertIsNone(lazyre.groupindex)
        lazyre.ensure()
        self.assertEqual(lazyre.groupindex, {'foo': 1})

    def test_flags(self):
        self.lazyre.ensure()
        self.assertEqual(self.lazyre.flags, re.compile('.').flags)

    def test_pattern(self):
        self.assertEqual(self.lazyre.pattern, 'f.o')


if __name__ == '__main__':
    import unittest

    unittest.main()
