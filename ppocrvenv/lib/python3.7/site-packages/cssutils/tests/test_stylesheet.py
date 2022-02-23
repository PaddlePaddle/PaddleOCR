"""Testcases for cssutils.stylesheets.StyleSheet"""

from . import basetest
import cssutils


class StyleSheetTestCase(basetest.BaseTestCase):
    def test_init(self):
        "StyleSheet.__init__()"
        s = cssutils.stylesheets.StyleSheet()

        self.assertEqual(s.type, 'text/css')
        self.assertEqual(s.href, None)
        self.assertEqual(s.media, None)
        self.assertEqual(s.title, '')
        self.assertEqual(s.ownerNode, None)
        self.assertEqual(s.parentStyleSheet, None)
        self.assertEqual(s.alternate, False)
        self.assertEqual(s.disabled, False)

        s = cssutils.stylesheets.StyleSheet(
            type='unknown',
            href='test.css',
            media=None,
            title='title',
            ownerNode=None,
            parentStyleSheet=None,
            alternate=True,
            disabled=True,
        )

        self.assertEqual(s.type, 'unknown')
        self.assertEqual(s.href, 'test.css')
        self.assertEqual(s.media, None)
        self.assertEqual(s.title, 'title')
        self.assertEqual(s.ownerNode, None)
        self.assertEqual(s.parentStyleSheet, None)
        self.assertEqual(s.alternate, True)
        self.assertEqual(s.disabled, True)


if __name__ == '__main__':
    import unittest

    unittest.main()
