import logging
from pdfminer.utils import Rect
from typing import BinaryIO, Container, Dict, Iterator, List, Optional, Tuple
import warnings
from . import settings
from .psparser import LIT
from .pdftypes import PDFObjectNotFound
from .pdftypes import resolve1
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import dict_value
from .pdfparser import PDFParser
from .pdfdocument import PDFDocument, PDFTextExtractionNotAllowed
from .pdfdocument import PDFTextExtractionNotAllowedWarning


log = logging.getLogger(__name__)

# some predefined literals and keywords.
LITERAL_PAGE = LIT('Page')
LITERAL_PAGES = LIT('Pages')


class PDFPage:
    """An object that holds the information about a page.

    A PDFPage object is merely a convenience class that has a set
    of keys and values, which describe the properties of a page
    and point to its contents.

    Attributes:
      doc: a PDFDocument object.
      pageid: any Python object that can uniquely identify the page.
      attrs: a dictionary of page attributes.
      contents: a list of PDFStream objects that represents the page content.
      lastmod: the last modified time of the page.
      resources: a dictionary of resources used by the page.
      mediabox: the physical size of the page.
      cropbox: the crop rectangle of the page.
      rotate: the page rotation (in degree).
      annots: the page annotations.
      beads: a chain that represents natural reading order.
    """

    def __init__(
        self,
        doc: PDFDocument,
        pageid: object,
        attrs: object
    ) -> None:
        """Initialize a page object.

        doc: a PDFDocument object.
        pageid: any Python object that can uniquely identify the page.
        attrs: a dictionary of page attributes.
        """
        self.doc = doc
        self.pageid = pageid
        self.attrs = dict_value(attrs)
        self.lastmod = resolve1(self.attrs.get('LastModified'))
        self.resources: Dict[object, object] = \
            resolve1(self.attrs.get('Resources', dict()))
        self.mediabox: Rect = resolve1(self.attrs['MediaBox'])
        if 'CropBox' in self.attrs:
            self.cropbox: Rect = resolve1(self.attrs['CropBox'])
        else:
            self.cropbox = self.mediabox
        self.rotate = (int_value(self.attrs.get('Rotate', 0))+360) % 360
        self.annots = self.attrs.get('Annots')
        self.beads = self.attrs.get('B')
        if 'Contents' in self.attrs:
            contents = resolve1(self.attrs['Contents'])
        else:
            contents = []
        if not isinstance(contents, list):
            contents = [contents]
        self.contents: List[object] = contents
        return

    def __repr__(self) -> str:
        return '<PDFPage: Resources={!r}, MediaBox={!r}>'\
            .format(self.resources, self.mediabox)

    INHERITABLE_ATTRS = {'Resources', 'MediaBox', 'CropBox', 'Rotate'}

    @classmethod
    def create_pages(cls, document: PDFDocument) -> Iterator["PDFPage"]:
        def search(
            obj: object,
            parent: Dict[str, object]
        ) -> Iterator[Tuple[int, Dict[object, Dict[object, object]]]]:
            if isinstance(obj, int):
                objid = obj
                tree = dict_value(document.getobj(objid)).copy()
            else:
                # This looks broken. obj.objid means obj could be either
                # PDFObjRef or PDFStream, but neither is valid for dict_value.
                objid = obj.objid  # type: ignore[attr-defined]
                tree = dict_value(obj).copy()
            for (k, v) in parent.items():
                if k in cls.INHERITABLE_ATTRS and k not in tree:
                    tree[k] = v

            tree_type = tree.get('Type')
            if tree_type is None and not settings.STRICT:  # See #64
                tree_type = tree.get('type')

            if tree_type is LITERAL_PAGES and 'Kids' in tree:
                log.info('Pages: Kids=%r', tree['Kids'])
                for c in list_value(tree['Kids']):
                    yield from search(c, tree)
            elif tree_type is LITERAL_PAGE:
                log.info('Page: %r', tree)
                yield (objid, tree)
        pages = False
        if 'Pages' in document.catalog:
            objects = search(document.catalog['Pages'], document.catalog)
            for (objid, tree) in objects:
                yield cls(document, objid, tree)
                pages = True
        if not pages:
            # fallback when /Pages is missing.
            for xref in document.xrefs:
                for objid in xref.get_objids():
                    try:
                        obj = document.getobj(objid)
                        if isinstance(obj, dict) \
                                and obj.get('Type') is LITERAL_PAGE:
                            yield cls(document, objid, obj)
                    except PDFObjectNotFound:
                        pass
        return

    @classmethod
    def get_pages(
        cls,
        fp: BinaryIO,
        pagenos: Optional[Container[int]] = None,
        maxpages: int = 0,
        password: str = '',
        caching: bool = True,
        check_extractable: bool = False
    ) -> Iterator["PDFPage"]:
        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        doc = PDFDocument(parser, password=password, caching=caching)
        # Check if the document allows text extraction.
        # If not, warn the user and proceed.
        if not doc.is_extractable:
            if check_extractable:
                error_msg = 'Text extraction is not allowed: %r' % fp
                raise PDFTextExtractionNotAllowed(error_msg)
            else:
                warning_msg = 'The PDF %r contains a metadata field '\
                            'indicating that it should not allow '   \
                            'text extraction. Ignoring this field '  \
                            'and proceeding.' % fp
                warnings.warn(warning_msg, PDFTextExtractionNotAllowedWarning)
        # Process each page contained in the document.
        for (pageno, page) in enumerate(cls.create_pages(doc)):
            if pagenos and (pageno not in pagenos):
                continue
            yield page
            if maxpages and maxpages <= pageno+1:
                break
        return
