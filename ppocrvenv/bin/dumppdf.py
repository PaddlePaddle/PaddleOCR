#!/Users/vx/Documents/GitHub/PaddleOCR/ppocrvenv/bin/python3
"""Extract pdf structure in XML format"""
import logging
import os.path
import re
import sys
from typing import Any, Container, Dict, Iterable, List, Optional, TextIO, \
    Union, cast
import warnings
from argparse import ArgumentParser

import pdfminer
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines, PDFXRefFallback, \
    PDFNoValidXRefWarning
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjectNotFound, PDFValueError
from pdfminer.pdftypes import PDFStream, PDFObjRef, resolve1, stream_value
from pdfminer.psparser import PSKeyword, PSLiteral, LIT
from pdfminer.utils import isnumber

logging.basicConfig()

ESC_PAT = re.compile(r'[\000-\037&<>()"\042\047\134\177-\377]')


def escape(s: Union[str, bytes]) -> str:
    if isinstance(s, bytes):
        us = str(s, 'latin-1')
    else:
        us = s
    return ESC_PAT.sub(lambda m: '&#%d;' % ord(m.group(0)), us)


def dumpxml(out: TextIO, obj: object, codec: Optional[str] = None) -> None:
    if obj is None:
        out.write('<null />')
        return

    if isinstance(obj, dict):
        out.write('<dict size="%d">\n' % len(obj))
        for (k, v) in obj.items():
            out.write('<key>%s</key>\n' % k)
            out.write('<value>')
            dumpxml(out, v)
            out.write('</value>\n')
        out.write('</dict>')
        return

    if isinstance(obj, list):
        out.write('<list size="%d">\n' % len(obj))
        for v in obj:
            dumpxml(out, v)
            out.write('\n')
        out.write('</list>')
        return

    if isinstance(obj, (str, bytes)):
        out.write('<string size="%d">%s</string>' % (len(obj), escape(obj)))
        return

    if isinstance(obj, PDFStream):
        if codec == 'raw':
            # Bug: writing bytes to text I/O. This will raise TypeError.
            out.write(obj.get_rawdata())  # type: ignore [arg-type]
        elif codec == 'binary':
            # Bug: writing bytes to text I/O. This will raise TypeError.
            out.write(obj.get_data())  # type: ignore [arg-type]
        else:
            out.write('<stream>\n<props>\n')
            dumpxml(out, obj.attrs)
            out.write('\n</props>\n')
            if codec == 'text':
                data = obj.get_data()
                out.write('<data size="%d">%s</data>\n'
                          % (len(data), escape(data)))
            out.write('</stream>')
        return

    if isinstance(obj, PDFObjRef):
        out.write('<ref id="%d" />' % obj.objid)
        return

    if isinstance(obj, PSKeyword):
        # Likely bug: obj.name is bytes, not str
        out.write('<keyword>%s</keyword>'
                  % obj.name)  # type: ignore [str-bytes-safe]
        return

    if isinstance(obj, PSLiteral):
        # Likely bug: obj.name may be bytes, not str
        out.write('<literal>%s</literal>'
                  % obj.name)  # type: ignore [str-bytes-safe]
        return

    if isnumber(obj):
        out.write('<number>%s</number>' % obj)
        return

    raise TypeError(obj)


def dumptrailers(
    out: TextIO,
    doc: PDFDocument,
    show_fallback_xref: bool = False
) -> None:
    for xref in doc.xrefs:
        if not isinstance(xref, PDFXRefFallback) or show_fallback_xref:
            out.write('<trailer>\n')
            dumpxml(out, xref.get_trailer())
            out.write('\n</trailer>\n\n')
    no_xrefs = all(isinstance(xref, PDFXRefFallback) for xref in doc.xrefs)
    if no_xrefs and not show_fallback_xref:
        msg = 'This PDF does not have an xref. Use --show-fallback-xref if ' \
              'you want to display the content of a fallback xref that ' \
              'contains all objects.'
        warnings.warn(msg, PDFNoValidXRefWarning)
    return


def dumpallobjs(
    out: TextIO,
    doc: PDFDocument,
    codec: Optional[str] = None,
    show_fallback_xref: bool = False
) -> None:
    visited = set()
    out.write('<pdf>')
    for xref in doc.xrefs:
        for objid in xref.get_objids():
            if objid in visited:
                continue
            visited.add(objid)
            try:
                obj = doc.getobj(objid)
                if obj is None:
                    continue
                out.write('<object id="%d">\n' % objid)
                dumpxml(out, obj, codec=codec)
                out.write('\n</object>\n\n')
            except PDFObjectNotFound as e:
                print('not found: %r' % e)
    dumptrailers(out, doc, show_fallback_xref)
    out.write('</pdf>')
    return


def dumpoutline(
    outfp: TextIO,
    fname: str,
    objids: Any,
    pagenos: Container[int],
    password: str = '',
    dumpall: bool = False,
    codec: Optional[str] = None,
    extractdir: Optional[str] = None
) -> None:
    fp = open(fname, 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser, password)
    pages = {page.pageid: pageno for (pageno, page)
             in enumerate(PDFPage.create_pages(doc), 1)}

    def resolve_dest(dest: object) -> Any:
        if isinstance(dest, (str, bytes)):
            dest = resolve1(doc.get_dest(dest))
        elif isinstance(dest, PSLiteral):
            dest = resolve1(doc.get_dest(dest.name))
        if isinstance(dest, dict):
            dest = dest['D']
        if isinstance(dest, PDFObjRef):
            dest = dest.resolve()
        return dest

    try:
        outlines = doc.get_outlines()
        outfp.write('<outlines>\n')
        for (level, title, dest, a, se) in outlines:
            pageno = None
            if dest:
                dest = resolve_dest(dest)
                pageno = pages[dest[0].objid]
            elif a:
                action = a
                if isinstance(action, dict):
                    subtype = action.get('S')
                    if subtype and repr(subtype) == '/\'GoTo\'' and action.get(
                            'D'):
                        dest = resolve_dest(action['D'])
                        pageno = pages[dest[0].objid]
            s = escape(title)
            outfp.write('<outline level="{!r}" title="{}">\n'.format(level, s))
            if dest is not None:
                outfp.write('<dest>')
                dumpxml(outfp, dest)
                outfp.write('</dest>\n')
            if pageno is not None:
                outfp.write('<pageno>%r</pageno>\n' % pageno)
            outfp.write('</outline>\n')
        outfp.write('</outlines>\n')
    except PDFNoOutlines:
        pass
    parser.close()
    fp.close()
    return


LITERAL_FILESPEC = LIT('Filespec')
LITERAL_EMBEDDEDFILE = LIT('EmbeddedFile')


def extractembedded(fname: str, password: str, extractdir: str) -> None:
    def extract1(objid: int, obj: Dict[str, Any]) -> None:
        filename = os.path.basename(obj.get('UF') or
                                    cast(bytes, obj.get('F')).decode())
        fileref = obj['EF'].get('UF') or obj['EF'].get('F')
        fileobj = doc.getobj(fileref.objid)
        if not isinstance(fileobj, PDFStream):
            error_msg = 'unable to process PDF: reference for %r is not a ' \
                        'PDFStream' % filename
            raise PDFValueError(error_msg)
        if fileobj.get('Type') is not LITERAL_EMBEDDEDFILE:
            raise PDFValueError(
                'unable to process PDF: reference for %r '
                'is not an EmbeddedFile' % (filename))
        path = os.path.join(extractdir, '%.6d-%s' % (objid, filename))
        if os.path.exists(path):
            raise IOError('file exists: %r' % path)
        print('extracting: %r' % path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = open(path, 'wb')
        out.write(fileobj.get_data())
        out.close()
        return

    with open(fname, 'rb') as fp:
        parser = PDFParser(fp)
        doc = PDFDocument(parser, password)
        extracted_objids = set()
        for xref in doc.xrefs:
            for objid in xref.get_objids():
                obj = doc.getobj(objid)
                if objid not in extracted_objids and isinstance(obj, dict) \
                        and obj.get('Type') is LITERAL_FILESPEC:
                    extracted_objids.add(objid)
                    extract1(objid, obj)
    return


def dumppdf(
    outfp: TextIO,
    fname: str,
    objids: Iterable[int],
    pagenos: Container[int],
    password: str = '',
    dumpall: bool = False,
    codec: Optional[str] = None,
    extractdir: Optional[str] = None,
    show_fallback_xref: bool = False
) -> None:
    fp = open(fname, 'rb')
    parser = PDFParser(fp)
    doc = PDFDocument(parser, password)
    if objids:
        for objid in objids:
            obj = doc.getobj(objid)
            dumpxml(outfp, obj, codec=codec)
    if pagenos:
        for (pageno, page) in enumerate(PDFPage.create_pages(doc)):
            if pageno in pagenos:
                if codec:
                    for obj in page.contents:
                        obj = stream_value(obj)
                        dumpxml(outfp, obj, codec=codec)
                else:
                    dumpxml(outfp, page.attrs)
    if dumpall:
        dumpallobjs(outfp, doc, codec, show_fallback_xref)
    if (not objids) and (not pagenos) and (not dumpall):
        dumptrailers(outfp, doc, show_fallback_xref)
    fp.close()
    if codec not in ('raw', 'binary'):
        outfp.write('\n')
    return


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument('files', type=str, default=None, nargs='+',
                        help='One or more paths to PDF files.')

    parser.add_argument(
        "--version", "-v", action="version",
        version="pdfminer.six v{}".format(pdfminer.__version__))
    parser.add_argument(
        '--debug', '-d', default=False, action='store_true',
        help='Use debug logging level.')
    procedure_parser = parser.add_mutually_exclusive_group()
    procedure_parser.add_argument(
        '--extract-toc', '-T', default=False, action='store_true',
        help='Extract structure of outline')
    procedure_parser.add_argument(
        '--extract-embedded', '-E', type=str,
        help='Extract embedded files')

    parse_params = parser.add_argument_group(
        'Parser', description='Used during PDF parsing')
    parse_params.add_argument(
        '--page-numbers', type=int, default=None, nargs='+',
        help='A space-seperated list of page numbers to parse.')
    parse_params.add_argument(
        '--pagenos', '-p', type=str,
        help='A comma-separated list of page numbers to parse. Included for '
             'legacy applications, use --page-numbers for more idiomatic '
             'argument entry.')
    parse_params.add_argument(
        '--objects', '-i', type=str,
        help='Comma separated list of object numbers to extract')
    parse_params.add_argument(
        '--all', '-a', default=False, action='store_true',
        help='If the structure of all objects should be extracted')
    parse_params.add_argument(
        '--show-fallback-xref', action='store_true',
        help='Additionally show the fallback xref. Use this if the PDF '
             'has zero or only invalid xref\'s. This setting is ignored if '
             '--extract-toc or --extract-embedded is used.')
    parse_params.add_argument(
        '--password', '-P', type=str, default='',
        help='The password to use for decrypting PDF file.')

    output_params = parser.add_argument_group(
        'Output', description='Used during output generation.')
    output_params.add_argument(
        '--outfile', '-o', type=str, default='-',
        help='Path to file where output is written. Or "-" (default) to '
             'write to stdout.')
    codec_parser = output_params.add_mutually_exclusive_group()
    codec_parser.add_argument(
        '--raw-stream', '-r', default=False, action='store_true',
        help='Write stream objects without encoding')
    codec_parser.add_argument(
        '--binary-stream', '-b', default=False, action='store_true',
        help='Write stream objects with binary encoding')
    codec_parser.add_argument(
        '--text-stream', '-t', default=False, action='store_true',
        help='Write stream objects as plain text')

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = create_parser()
    args = parser.parse_args(args=argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.outfile == '-':
        outfp = sys.stdout
    else:
        outfp = open(args.outfile, 'w')

    if args.objects:
        objids = [int(x) for x in args.objects.split(',')]
    else:
        objids = []

    if args.page_numbers:
        pagenos = {x - 1 for x in args.page_numbers}
    elif args.pagenos:
        pagenos = {int(x) - 1 for x in args.pagenos.split(',')}
    else:
        pagenos = set()

    password = args.password

    if args.raw_stream:
        codec: Optional[str] = 'raw'
    elif args.binary_stream:
        codec = 'binary'
    elif args.text_stream:
        codec = 'text'
    else:
        codec = None

    for fname in args.files:
        if args.extract_toc:
            dumpoutline(
                outfp, fname, objids, pagenos, password=password,
                dumpall=args.all, codec=codec, extractdir=None
            )
        elif args.extract_embedded:
            extractembedded(
                fname, password=password, extractdir=args.extract_embedded
            )
        else:
            dumppdf(
                outfp, fname, objids, pagenos, password=password,
                dumpall=args.all, codec=codec, extractdir=None,
                show_fallback_xref=args.show_fallback_xref
            )

    outfp.close()


if __name__ == '__main__':
    main()
