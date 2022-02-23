import io
import logging
from pdfminer.pdfcolor import PDFColorSpace
from typing import (BinaryIO, Dict, Generic, List, Optional, Sequence, TextIO,
                    Tuple, TypeVar, Union, cast)
import re

from . import utils
from .layout import LAParams, LTComponent, TextGroupElement
from .layout import LTChar
from .layout import LTContainer
from .layout import LTCurve
from .layout import LTFigure
from .layout import LTImage
from .layout import LTItem
from .layout import LTLayoutContainer
from .layout import LTLine
from .layout import LTPage
from .layout import LTRect
from .layout import LTText
from .layout import LTTextBox
from .layout import LTTextBoxVertical
from .layout import LTTextGroup
from .layout import LTTextLine
from .pdfdevice import PDFTextDevice
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfinterp import PDFGraphicState, PDFResourceManager
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import AnyIO, Point, Matrix, Rect, PathSegment
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import enc
from .utils import mult_matrix
from .image import ImageWriter

log = logging.getLogger(__name__)


class PDFLayoutAnalyzer(PDFTextDevice):
    cur_item: LTLayoutContainer
    ctm: Matrix

    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: Optional[LAParams] = None
    ) -> None:
        PDFTextDevice.__init__(self, rsrcmgr)
        self.pageno = pageno
        self.laparams = laparams
        self._stack: List[LTLayoutContainer] = []
        return

    def begin_page(self, page: PDFPage, ctm: Matrix) -> None:
        (x0, y0, x1, y1) = page.mediabox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0-x1), abs(y0-y1))
        self.cur_item = LTPage(self.pageno, mediabox)
        return

    def end_page(self, page: PDFPage) -> None:
        assert not self._stack, str(len(self._stack))
        assert isinstance(self.cur_item, LTPage), str(type(self.cur_item))
        if self.laparams is not None:
            self.cur_item.analyze(self.laparams)
        self.pageno += 1
        self.receive_layout(self.cur_item)
        return

    def begin_figure(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        return

    def end_figure(self, _: str) -> None:
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return

    def render_image(self, name: str, stream: PDFStream) -> None:
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        item = LTImage(name, stream,
                       (self.cur_item.x0, self.cur_item.y0,
                        self.cur_item.x1, self.cur_item.y1))
        self.cur_item.add(item)
        return

    def paint_path(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment]
    ) -> None:
        """Paint paths described in section 4.4 of the PDF reference manual"""
        shape = ''.join(x[0] for x in path)

        if shape.count('m') > 1:
            # recurse if there are multiple m's in this shape
            for m in re.finditer(r'm[^m]+', shape):
                subpath = path[m.start(0):m.end(0)]
                self.paint_path(gstate, stroke, fill, evenodd, subpath)

        else:
            # Although the 'h' command does not not literally provide a
            # point-position, its position is (by definition) equal to the
            # subpath's starting point.
            #
            # And, per Section 4.4's Table 4.9, all other path commands place
            # their point-position in their final two arguments. (Any preceding
            # arguments represent control points on Bézier curves.)
            raw_pts = [cast(Point, p[-2:] if p[0] != 'h' else path[0][-2:])
                       for p in path]
            pts = [apply_matrix_pt(self.ctm, pt) for pt in raw_pts]

            if shape in {'mlh', 'ml'}:
                # single line segment
                #
                # Note: 'ml', in conditional above, is a frequent anomaly
                # that we want to support.
                line = LTLine(gstate.linewidth, pts[0], pts[1], stroke,
                              fill, evenodd, gstate.scolor, gstate.ncolor)
                self.cur_item.add(line)

            elif shape in {'mlllh', 'mllll'}:
                (x0, y0), (x1, y1), (x2, y2), (x3, y3), _ = pts

                is_closed_loop = (pts[0] == pts[4])
                has_square_coordinates = \
                    (x0 == x1 and y1 == y2 and x2 == x3 and y3 == y0) \
                    or (y0 == y1 and x1 == x2 and y2 == y3 and x3 == x0)
                if is_closed_loop and has_square_coordinates:
                    rect = LTRect(gstate.linewidth, (*pts[0], *pts[2]), stroke,
                                  fill, evenodd, gstate.scolor, gstate.ncolor)
                    self.cur_item.add(rect)
                else:
                    curve = LTCurve(gstate.linewidth, pts, stroke, fill,
                                    evenodd, gstate.scolor, gstate.ncolor)
                    self.cur_item.add(curve)

            else:
                curve = LTCurve(gstate.linewidth, pts, stroke, fill, evenodd,
                                gstate.scolor, gstate.ncolor)
                self.cur_item.add(curve)

    def render_char(
        self,
        matrix: Matrix,
        font: PDFFont,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState
    ) -> float:
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(matrix, font, fontsize, scaling, rise, text, textwidth,
                      textdisp, ncs, graphicstate)
        self.cur_item.add(item)
        return item.adv

    def handle_undefined_char(self, font: PDFFont, cid: int) -> str:
        log.info('undefined: %r, %r', font, cid)
        return '(cid:%d)' % cid

    def receive_layout(self, ltpage: LTPage) -> None:
        return


class PDFPageAggregator(PDFLayoutAnalyzer):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: Optional[LAParams] = None
    ) -> None:
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno,
                                   laparams=laparams)
        self.result: Optional[LTPage] = None
        return

    def receive_layout(self, ltpage: LTPage) -> None:
        self.result = ltpage
        return

    def get_result(self) -> LTPage:
        assert self.result is not None
        return self.result


# Some PDFConverter children support only binary I/O
IOType = TypeVar('IOType', TextIO, BinaryIO, AnyIO)


class PDFConverter(PDFLayoutAnalyzer, Generic[IOType]):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: IOType,
        codec: str = 'utf-8',
        pageno: int = 1,
        laparams: Optional[LAParams] = None
    ) -> None:
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno,
                                   laparams=laparams)
        self.outfp: IOType = outfp
        self.codec = codec
        self.outfp_binary = self._is_binary_stream(self.outfp)

    @staticmethod
    def _is_binary_stream(outfp: AnyIO) -> bool:
        """Test if an stream is binary or not"""
        if 'b' in getattr(outfp, 'mode', ''):
            return True
        elif hasattr(outfp, 'mode'):
            # output stream has a mode, but it does not contain 'b'
            return False
        elif isinstance(outfp, io.BytesIO):
            return True
        elif isinstance(outfp, io.StringIO):
            return False
        elif isinstance(outfp, io.TextIOBase):
            return False

        return True


class TextConverter(PDFConverter[AnyIO]):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = 'utf-8',
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
        showpageno: bool = False,
        imagewriter: Optional[ImageWriter] = None
    ) -> None:
        super().__init__(rsrcmgr, outfp, codec=codec, pageno=pageno,
                         laparams=laparams)
        self.showpageno = showpageno
        self.imagewriter = imagewriter
        return

    def write_text(self, text: str) -> None:
        text = utils.compatible_encode_method(text, self.codec, 'ignore')
        if self.outfp_binary:
            cast(BinaryIO, self.outfp).write(text.encode())
        else:
            cast(TextIO, self.outfp).write(text)
        return

    def receive_layout(self, ltpage: LTPage) -> None:
        def render(item: LTItem) -> None:
            if isinstance(item, LTContainer):
                for child in item:
                    render(child)
            elif isinstance(item, LTText):
                self.write_text(item.get_text())
            if isinstance(item, LTTextBox):
                self.write_text('\n')
            elif isinstance(item, LTImage):
                if self.imagewriter is not None:
                    self.imagewriter.export_image(item)
        if self.showpageno:
            self.write_text('Page %s\n' % ltpage.pageid)
        render(ltpage)
        self.write_text('\f')
        return

    # Some dummy functions to save memory/CPU when all that is wanted
    # is text.  This stops all the image and drawing output from being
    # recorded and taking up RAM.
    def render_image(self, name: str, stream: PDFStream) -> None:
        if self.imagewriter is None:
            return
        PDFConverter.render_image(self, name, stream)
        return

    def paint_path(
        self,
        gstate: PDFGraphicState,
        stroke: bool,
        fill: bool,
        evenodd: bool,
        path: Sequence[PathSegment]
    ) -> None:
        return


class HTMLConverter(PDFConverter[AnyIO]):
    RECT_COLORS = {
        'figure': 'yellow',
        'textline': 'magenta',
        'textbox': 'cyan',
        'textgroup': 'red',
        'curve': 'black',
        'page': 'gray',
    }

    TEXT_COLORS = {
        'textbox': 'blue',
        'char': 'black',
    }

    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = 'utf-8',
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
        scale: float = 1,
        fontscale: float = 1.0,
        layoutmode: str = 'normal',
        showpageno: bool = True,
        pagemargin: int = 50,
        imagewriter: Optional[ImageWriter] = None,
        debug: int = 0,
        rect_colors: Optional[Dict[str, str]] = None,
        text_colors: Optional[Dict[str, str]] = None
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, outfp, codec=codec, pageno=pageno,
                              laparams=laparams)

        # write() assumes a codec for binary I/O, or no codec for text I/O.
        if self.outfp_binary == (not self.codec):
            raise ValueError("Codec is required for a binary I/O output")

        if text_colors is None:
            text_colors = {'char': 'black'}
        if rect_colors is None:
            rect_colors = {'curve': 'black', 'page': 'gray'}

        self.scale = scale
        self.fontscale = fontscale
        self.layoutmode = layoutmode
        self.showpageno = showpageno
        self.pagemargin = pagemargin
        self.imagewriter = imagewriter
        self.rect_colors = rect_colors
        self.text_colors = text_colors
        if debug:
            self.rect_colors.update(self.RECT_COLORS)
            self.text_colors.update(self.TEXT_COLORS)
        self._yoffset: float = self.pagemargin
        self._font: Optional[Tuple[str, float]] = None
        self._fontstack: List[Optional[Tuple[str, float]]] = []
        self.write_header()
        return

    def write(self, text: str) -> None:
        if self.codec:
            cast(BinaryIO, self.outfp).write(text.encode(self.codec))
        else:
            cast(TextIO, self.outfp).write(text)
        return

    def write_header(self) -> None:
        self.write('<html><head>\n')
        if self.codec:
            s = '<meta http-equiv="Content-Type" content="text/html; ' \
                'charset=%s">\n' % self.codec
        else:
            s = '<meta http-equiv="Content-Type" content="text/html">\n'
        self.write(s)
        self.write('</head><body>\n')
        return

    def write_footer(self) -> None:
        page_links = ['<a href="#{}">{}</a>'.format(i, i)
                      for i in range(1, self.pageno)]
        s = '<div style="position:absolute; top:0px;">Page: %s</div>\n' % \
            ', '.join(page_links)
        self.write(s)
        self.write('</body></html>\n')
        return

    def write_text(self, text: str) -> None:
        self.write(enc(text))
        return

    def place_rect(
        self,
        color: str,
        borderwidth: int,
        x: float,
        y: float,
        w: float,
        h: float
    ) -> None:
        color2 = self.rect_colors.get(color)
        if color2 is not None:
            s = '<span style="position:absolute; border: %s %dpx solid; ' \
                'left:%dpx; top:%dpx; width:%dpx; height:%dpx;"></span>\n' % \
                (color2, borderwidth, x * self.scale,
                 (self._yoffset - y) * self.scale, w * self.scale,
                 h * self.scale)
            self.write(
                s)
        return

    def place_border(
        self,
        color: str,
        borderwidth: int,
        item: LTComponent
    ) -> None:
        self.place_rect(color, borderwidth, item.x0, item.y1, item.width,
                        item.height)
        return

    def place_image(
        self,
        item: LTImage,
        borderwidth: int,
        x: float,
        y: float,
        w: float,
        h: float
    ) -> None:
        if self.imagewriter is not None:
            name = self.imagewriter.export_image(item)
            s = '<img src="%s" border="%d" style="position:absolute; ' \
                'left:%dpx; top:%dpx;" width="%d" height="%d" />\n' % \
                (enc(name), borderwidth, x * self.scale,
                 (self._yoffset - y) * self.scale, w * self.scale,
                 h * self.scale)
            self.write(s)
        return

    def place_text(
        self,
        color: str,
        text: str,
        x: float,
        y: float,
        size: float
    ) -> None:
        color2 = self.text_colors.get(color)
        if color2 is not None:
            s = '<span style="position:absolute; color:%s; left:%dpx; ' \
                'top:%dpx; font-size:%dpx;">' % \
                (color2, x * self.scale, (self._yoffset - y) * self.scale,
                 size * self.scale * self.fontscale)
            self.write(s)
            self.write_text(text)
            self.write('</span>\n')
        return

    def begin_div(
        self,
        color: str,
        borderwidth: int,
        x: float,
        y: float,
        w: float,
        h: float,
        writing_mode: str = 'False'
    ) -> None:
        self._fontstack.append(self._font)
        self._font = None
        s = '<div style="position:absolute; border: %s %dpx solid; ' \
            'writing-mode:%s; left:%dpx; top:%dpx; width:%dpx; ' \
            'height:%dpx;">' % \
            (color, borderwidth, writing_mode, x * self.scale,
             (self._yoffset - y) * self.scale, w * self.scale, h * self.scale)
        self.write(s)
        return

    def end_div(self, color: str) -> None:
        if self._font is not None:
            self.write('</span>')
        self._font = self._fontstack.pop()
        self.write('</div>')
        return

    def put_text(self, text: str, fontname: str, fontsize: float) -> None:
        font = (fontname, fontsize)
        if font != self._font:
            if self._font is not None:
                self.write('</span>')
            # Remove subset tag from fontname, see PDF Reference 5.5.3
            fontname_without_subset_tag = fontname.split('+')[-1]
            self.write('<span style="font-family: %s; font-size:%dpx">' %
                       (fontname_without_subset_tag,
                        fontsize * self.scale * self.fontscale))
            self._font = font
        self.write_text(text)
        return

    def put_newline(self) -> None:
        self.write('<br>')
        return

    def receive_layout(self, ltpage: LTPage) -> None:
        def show_group(item: Union[LTTextGroup, TextGroupElement]) -> None:
            if isinstance(item, LTTextGroup):
                self.place_border('textgroup', 1, item)
                for child in item:
                    show_group(child)
            return

        def render(item: LTItem) -> None:
            child: LTItem
            if isinstance(item, LTPage):
                self._yoffset += item.y1
                self.place_border('page', 1, item)
                if self.showpageno:
                    self.write('<div style="position:absolute; top:%dpx;">' %
                               ((self._yoffset-item.y1)*self.scale))
                    self.write('<a name="{}">Page {}</a></div>\n'
                               .format(item.pageid, item.pageid))
                for child in item:
                    render(child)
                if item.groups is not None:
                    for group in item.groups:
                        show_group(group)
            elif isinstance(item, LTCurve):
                self.place_border('curve', 1, item)
            elif isinstance(item, LTFigure):
                self.begin_div('figure', 1, item.x0, item.y1, item.width,
                               item.height)
                for child in item:
                    render(child)
                self.end_div('figure')
            elif isinstance(item, LTImage):
                self.place_image(item, 1, item.x0, item.y1, item.width,
                                 item.height)
            else:
                if self.layoutmode == 'exact':
                    if isinstance(item, LTTextLine):
                        self.place_border('textline', 1, item)
                        for child in item:
                            render(child)
                    elif isinstance(item, LTTextBox):
                        self.place_border('textbox', 1, item)
                        self.place_text('textbox', str(item.index+1), item.x0,
                                        item.y1, 20)
                        for child in item:
                            render(child)
                    elif isinstance(item, LTChar):
                        self.place_border('char', 1, item)
                        self.place_text('char', item.get_text(), item.x0,
                                        item.y1, item.size)
                else:
                    if isinstance(item, LTTextLine):
                        for child in item:
                            render(child)
                        if self.layoutmode != 'loose':
                            self.put_newline()
                    elif isinstance(item, LTTextBox):
                        self.begin_div('textbox', 1, item.x0, item.y1,
                                       item.width, item.height,
                                       item.get_writing_mode())
                        for child in item:
                            render(child)
                        self.end_div('textbox')
                    elif isinstance(item, LTChar):
                        self.put_text(item.get_text(), item.fontname,
                                      item.size)
                    elif isinstance(item, LTText):
                        self.write_text(item.get_text())
            return
        render(ltpage)
        self._yoffset += self.pagemargin
        return

    def close(self) -> None:
        self.write_footer()
        return


class XMLConverter(PDFConverter[AnyIO]):

    CONTROL = re.compile('[\x00-\x08\x0b-\x0c\x0e-\x1f]')

    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        outfp: AnyIO,
        codec: str = 'utf-8',
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
        imagewriter: Optional[ImageWriter] = None,
        stripcontrol: bool = False
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, outfp, codec=codec, pageno=pageno,
                              laparams=laparams)

        # write() assumes a codec for binary I/O, or no codec for text I/O.
        if self.outfp_binary == (not self.codec):
            raise ValueError("Codec is required for a binary I/O output")

        self.imagewriter = imagewriter
        self.stripcontrol = stripcontrol
        self.write_header()
        return

    def write(self, text: str) -> None:
        if self.codec:
            cast(BinaryIO, self.outfp).write(text.encode(self.codec))
        else:
            cast(TextIO, self.outfp).write(text)
        return

    def write_header(self) -> None:
        if self.codec:
            self.write('<?xml version="1.0" encoding="%s" ?>\n' % self.codec)
        else:
            self.write('<?xml version="1.0" ?>\n')
        self.write('<pages>\n')
        return

    def write_footer(self) -> None:
        self.write('</pages>\n')
        return

    def write_text(self, text: str) -> None:
        if self.stripcontrol:
            text = self.CONTROL.sub('', text)
        self.write(enc(text))
        return

    def receive_layout(self, ltpage: LTPage) -> None:
        def show_group(item: LTItem) -> None:
            if isinstance(item, LTTextBox):
                self.write('<textbox id="%d" bbox="%s" />\n' %
                           (item.index, bbox2str(item.bbox)))
            elif isinstance(item, LTTextGroup):
                self.write('<textgroup bbox="%s">\n' % bbox2str(item.bbox))
                for child in item:
                    show_group(child)
                self.write('</textgroup>\n')
            return

        def render(item: LTItem) -> None:
            child: LTItem
            if isinstance(item, LTPage):
                s = '<page id="%s" bbox="%s" rotate="%d">\n' % \
                    (item.pageid, bbox2str(item.bbox), item.rotate)
                self.write(s)
                for child in item:
                    render(child)
                if item.groups is not None:
                    self.write('<layout>\n')
                    for group in item.groups:
                        show_group(group)
                    self.write('</layout>\n')
                self.write('</page>\n')
            elif isinstance(item, LTLine):
                s = '<line linewidth="%d" bbox="%s" />\n' % \
                    (item.linewidth, bbox2str(item.bbox))
                self.write(s)
            elif isinstance(item, LTRect):
                s = '<rect linewidth="%d" bbox="%s" />\n' % \
                    (item.linewidth, bbox2str(item.bbox))
                self.write(s)
            elif isinstance(item, LTCurve):
                s = '<curve linewidth="%d" bbox="%s" pts="%s"/>\n' % \
                    (item.linewidth, bbox2str(item.bbox), item.get_pts())
                self.write(s)
            elif isinstance(item, LTFigure):
                s = '<figure name="%s" bbox="%s">\n' % \
                    (item.name, bbox2str(item.bbox))
                self.write(s)
                for child in item:
                    render(child)
                self.write('</figure>\n')
            elif isinstance(item, LTTextLine):
                self.write('<textline bbox="%s">\n' % bbox2str(item.bbox))
                for child in item:
                    render(child)
                self.write('</textline>\n')
            elif isinstance(item, LTTextBox):
                wmode = ''
                if isinstance(item, LTTextBoxVertical):
                    wmode = ' wmode="vertical"'
                s = '<textbox id="%d" bbox="%s"%s>\n' %\
                    (item.index, bbox2str(item.bbox), wmode)
                self.write(s)
                for child in item:
                    render(child)
                self.write('</textbox>\n')
            elif isinstance(item, LTChar):
                s = '<text font="%s" bbox="%s" colourspace="%s" ' \
                    'ncolour="%s" size="%.3f">' % \
                    (enc(item.fontname), bbox2str(item.bbox),
                     item.ncs.name, item.graphicstate.ncolor, item.size)
                self.write(s)
                self.write_text(item.get_text())
                self.write('</text>\n')
            elif isinstance(item, LTText):
                self.write('<text>%s</text>\n' % item.get_text())
            elif isinstance(item, LTImage):
                if self.imagewriter is not None:
                    name = self.imagewriter.export_image(item)
                    self.write('<image src="%s" width="%d" height="%d" />\n' %
                               (enc(name), item.width, item.height))
                else:
                    self.write('<image width="%d" height="%d" />\n' %
                               (item.width, item.height))
            else:
                assert False, str(('Unhandled', item))
            return
        render(ltpage)
        return

    def close(self) -> None:
        self.write_footer()
        return
