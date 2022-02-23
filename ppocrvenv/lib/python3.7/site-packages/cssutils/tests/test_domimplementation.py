"""Testcases for cssutils.css.DOMImplementation"""

import xml.dom
import xml.dom.minidom
import unittest
import warnings

import cssutils


class DOMImplementationTestCase(unittest.TestCase):
    def setUp(self):
        self.domimpl = cssutils.DOMImplementationCSS()

    def test_createCSSStyleSheet(self):
        "DOMImplementationCSS.createCSSStyleSheet()"
        title, media = 'Test Title', cssutils.stylesheets.MediaList('all')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sheet = self.domimpl.createCSSStyleSheet(title, media)
        self.assertEqual(True, isinstance(sheet, cssutils.css.CSSStyleSheet))
        self.assertEqual(title, sheet.title)
        self.assertEqual(media, sheet.media)

    def test_createDocument(self):
        "DOMImplementationCSS.createDocument()"
        doc = self.domimpl.createDocument(None, None, None)
        self.assertTrue(isinstance(doc, xml.dom.minidom.Document))

    def test_createDocumentType(self):
        "DOMImplementationCSS.createDocumentType()"
        doctype = self.domimpl.createDocumentType('foo', 'bar', 'raboof')
        self.assertTrue(isinstance(doctype, xml.dom.minidom.DocumentType))

    def test_hasFeature(self):
        "DOMImplementationCSS.hasFeature()"
        tests = [
            ('css', '1.0'),
            ('css', '2.0'),
            ('stylesheets', '1.0'),
            ('stylesheets', '2.0'),
        ]
        for name, version in tests:
            self.assertEqual(True, self.domimpl.hasFeature(name, version))


if __name__ == '__main__':
    unittest.main()
