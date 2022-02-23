import re
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast
from io import BytesIO
from .cmapdb import CMapDB
from .cmapdb import CMap
from .cmapdb import CMapBase
from .psparser import PSLiteral, PSTypeError
from .psparser import PSStackType
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import literal_name
from .psparser import keyword_name
from .psparser import PSStackParser
from .psparser import LIT
from .psparser import KWD
from . import settings
from .pdfdevice import PDFDevice
from .pdfdevice import PDFTextSeq
from .pdfpage import PDFPage
from .pdftypes import PDFException
from .pdftypes import PDFStream
from .pdftypes import PDFObjRef
from .pdftypes import resolve1
from .pdftypes import list_value
from .pdftypes import dict_value
from .pdftypes import stream_value
from .pdffont import PDFFont
from .pdffont import PDFFontError
from .pdffont import PDFType1Font
from .pdffont import PDFTrueTypeFont
from .pdffont import PDFType3Font
from .pdffont import PDFCIDFont
from .pdfcolor import PDFColorSpace
from .pdfcolor import PREDEFINED_COLORSPACE
from .utils import Matrix, Point, PathSegment, Rect
from .utils import choplist
from .utils import mult_matrix
from .utils import MATRIX_IDENTITY


log = logging.getLogger(__name__)


class PDFResourceError(PDFException):
    pass


class PDFInterpreterError(PDFException):
    pass


LITERAL_PDF = LIT('PDF')
LITERAL_TEXT = LIT('Text')
LITERAL_FONT = LIT('Font')
LITERAL_FORM = LIT('Form')
LITERAL_IMAGE = LIT('Image')


class PDFTextState:
    matrix: Matrix
    linematrix: Point

    def __init__(self) -> None:
        self.font: Optional[PDFFont] = None
        self.fontsize: float = 0
        self.charspace: float = 0
        self.wordspace: float = 0
        self.scaling: float = 100
        self.leading: float = 0
        self.render: int = 0
        self.rise: float = 0
        self.reset()
        # self.matrix is set
        # self.linematrix is set
        return

    def __repr__(self) -> str:
        return '<PDFTextState: font=%r, fontsize=%r, charspace=%r, ' \
               'wordspace=%r, scaling=%r, leading=%r, render=%r, rise=%r, ' \
               'matrix=%r, linematrix=%r>' \
               % (self.font, self.fontsize, self.charspace, self.wordspace,
                  self.scaling, self.leading, self.render, self.rise,
                  self.matrix, self.linematrix)

    def copy(self) -> "PDFTextState":
        obj = PDFTextState()
        obj.font = self.font
        obj.fontsize = self.fontsize
        obj.charspace = self.charspace
        obj.wordspace = self.wordspace
        obj.scaling = self.scaling
        obj.leading = self.leading
        obj.render = self.render
        obj.rise = self.rise
        obj.matrix = self.matrix
        obj.linematrix = self.linematrix
        return obj

    def reset(self) -> None:
        self.matrix = MATRIX_IDENTITY
        self.linematrix = (0, 0)
        return


Color = Union[
    float,                              # Greyscale
    Tuple[float, float, float],         # R, G, B
    Tuple[float, float, float, float]]  # C, M, Y, K


class PDFGraphicState:

    def __init__(self) -> None:
        self.linewidth: float = 0
        self.linecap: Optional[object] = None
        self.linejoin: Optional[object] = None
        self.miterlimit: Optional[object] = None
        self.dash: Optional[Tuple[object, object]] = None
        self.intent: Optional[object] = None
        self.flatness: Optional[object] = None

        # stroking color
        self.scolor: Optional[Color] = None

        # non stroking color
        self.ncolor: Optional[Color] = None
        return

    def copy(self) -> "PDFGraphicState":
        obj = PDFGraphicState()
        obj.linewidth = self.linewidth
        obj.linecap = self.linecap
        obj.linejoin = self.linejoin
        obj.miterlimit = self.miterlimit
        obj.dash = self.dash
        obj.intent = self.intent
        obj.flatness = self.flatness
        obj.scolor = self.scolor
        obj.ncolor = self.ncolor
        return obj

    def __repr__(self) -> str:
        return ('<PDFGraphicState: linewidth=%r, linecap=%r, linejoin=%r, '
                ' miterlimit=%r, dash=%r, intent=%r, flatness=%r, '
                ' stroking color=%r, non stroking color=%r>' %
                (self.linewidth, self.linecap, self.linejoin,
                 self.miterlimit, self.dash, self.intent, self.flatness,
                 self.scolor, self.ncolor))


class PDFResourceManager:
    """Repository of shared resources.

    ResourceManager facilitates reuse of shared resources
    such as fonts and images so that large objects are not
    allocated multiple times.
    """

    def __init__(self, caching: bool = True) -> None:
        self.caching = caching
        self._cached_fonts: Dict[object, PDFFont] = {}
        return

    def get_procset(self, procs: Sequence[object]) -> None:
        for proc in procs:
            if proc is LITERAL_PDF:
                pass
            elif proc is LITERAL_TEXT:
                pass
            else:
                pass
        return

    def get_cmap(self, cmapname: str, strict: bool = False) -> CMapBase:
        try:
            return CMapDB.get_cmap(cmapname)
        except CMapDB.CMapNotFound:
            if strict:
                raise
            return CMap()

    def get_font(self, objid: object, spec: Mapping[str, object]) -> PDFFont:
        if objid and objid in self._cached_fonts:
            font = self._cached_fonts[objid]
        else:
            log.info('get_font: create: objid=%r, spec=%r', objid, spec)
            if settings.STRICT:
                if spec['Type'] is not LITERAL_FONT:
                    raise PDFFontError('Type is not /Font')
            # Create a Font object.
            if 'Subtype' in spec:
                subtype = literal_name(spec['Subtype'])
            else:
                if settings.STRICT:
                    raise PDFFontError('Font Subtype is not specified.')
                subtype = 'Type1'
            if subtype in ('Type1', 'MMType1'):
                # Type1 Font
                font = PDFType1Font(self, spec)
            elif subtype == 'TrueType':
                # TrueType Font
                font = PDFTrueTypeFont(self, spec)
            elif subtype == 'Type3':
                # Type3 Font
                font = PDFType3Font(self, spec)
            elif subtype in ('CIDFontType0', 'CIDFontType2'):
                # CID Font
                font = PDFCIDFont(self, spec)
            elif subtype == 'Type0':
                # Type0 Font
                dfonts = list_value(spec['DescendantFonts'])
                assert dfonts
                subspec = dict_value(dfonts[0]).copy()
                for k in ('Encoding', 'ToUnicode'):
                    if k in spec:
                        subspec[k] = resolve1(spec[k])
                font = self.get_font(None, subspec)
            else:
                if settings.STRICT:
                    raise PDFFontError('Invalid Font spec: %r' % spec)
                font = PDFType1Font(self, spec)  # this is so wrong!
            if objid and self.caching:
                self._cached_fonts[objid] = font
        return font


class PDFContentParser(PSStackParser[Union[PSKeyword, PDFStream]]):

    def __init__(self, streams: Sequence[object]) -> None:
        self.streams = streams
        self.istream = 0
        # PSStackParser.__init__(fp=None) is safe only because we've overloaded
        # all the methods that would attempt to access self.fp without first
        # calling self.fillfp().
        PSStackParser.__init__(self, None)  # type: ignore[arg-type]
        return

    def fillfp(self) -> None:
        if not self.fp:
            if self.istream < len(self.streams):
                strm = stream_value(self.streams[self.istream])
                self.istream += 1
            else:
                raise PSEOF('Unexpected EOF, file truncated?')
            self.fp = BytesIO(strm.get_data())
        return

    def seek(self, pos: int) -> None:
        self.fillfp()
        PSStackParser.seek(self, pos)
        return

    def fillbuf(self) -> None:
        if self.charpos < len(self.buf):
            return
        while 1:
            self.fillfp()
            self.bufpos = self.fp.tell()
            self.buf = self.fp.read(self.BUFSIZ)
            if self.buf:
                break
            self.fp = None  # type: ignore[assignment]
        self.charpos = 0
        return

    def get_inline_data(
        self,
        pos: int,
        target: bytes = b'EI'
    ) -> Tuple[int, bytes]:
        self.seek(pos)
        i = 0
        data = b''
        while i <= len(target):
            self.fillbuf()
            if i:
                ci = self.buf[self.charpos]
                c = bytes((ci,))
                data += c
                self.charpos += 1
                if len(target) <= i and c.isspace():
                    i += 1
                elif i < len(target) and c == (bytes((target[i],))):
                    i += 1
                else:
                    i = 0
            else:
                try:
                    j = self.buf.index(target[0], self.charpos)
                    data += self.buf[self.charpos:j+1]
                    self.charpos = j+1
                    i = 1
                except ValueError:
                    data += self.buf[self.charpos:]
                    self.charpos = len(self.buf)
        data = data[:-(len(target)+1)]  # strip the last part
        data = re.sub(br'(\x0d\x0a|[\x0d\x0a])$', b'', data)
        return (pos, data)

    def flush(self) -> None:
        self.add_results(*self.popall())
        return

    KEYWORD_BI = KWD(b'BI')
    KEYWORD_ID = KWD(b'ID')
    KEYWORD_EI = KWD(b'EI')

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        if token is self.KEYWORD_BI:
            # inline image within a content stream
            self.start_type(pos, 'inline')
        elif token is self.KEYWORD_ID:
            try:
                (_, objs) = self.end_type('inline')
                if len(objs) % 2 != 0:
                    error_msg = 'Invalid dictionary construct: {!r}' \
                        .format(objs)
                    raise PSTypeError(error_msg)
                d = {literal_name(k): v for (k, v) in choplist(2, objs)}
                (pos, data) = self.get_inline_data(pos+len(b'ID '))
                obj = PDFStream(d, data)
                self.push((pos, obj))
                self.push((pos, self.KEYWORD_EI))
            except PSTypeError:
                if settings.STRICT:
                    raise
        else:
            self.push((pos, token))
        return


PDFStackT = PSStackType[PDFStream]
"""Types that may appear on the PDF argument stack."""


class PDFPageInterpreter:
    """Processor for the content of a PDF page

    Reference: PDF Reference, Appendix A, Operator Summary
    """

    def __init__(self, rsrcmgr: PDFResourceManager, device: PDFDevice) -> None:
        self.rsrcmgr = rsrcmgr
        self.device = device
        return

    def dup(self) -> "PDFPageInterpreter":
        return self.__class__(self.rsrcmgr, self.device)

    def init_resources(self, resources: Dict[object, object]) -> None:
        """Prepare the fonts and XObjects listed in the Resource attribute."""
        self.resources = resources
        self.fontmap: Dict[object, PDFFont] = {}
        self.xobjmap = {}
        self.csmap: Dict[str, PDFColorSpace] = PREDEFINED_COLORSPACE.copy()
        if not resources:
            return

        def get_colorspace(spec: object) -> Optional[PDFColorSpace]:
            if isinstance(spec, list):
                name = literal_name(spec[0])
            else:
                name = literal_name(spec)
            if name == 'ICCBased' and isinstance(spec, list) \
                    and 2 <= len(spec):
                return PDFColorSpace(name, stream_value(spec[1])['N'])
            elif name == 'DeviceN' and isinstance(spec, list) \
                    and 2 <= len(spec):
                return PDFColorSpace(name, len(list_value(spec[1])))
            else:
                return PREDEFINED_COLORSPACE.get(name)

        for (k, v) in dict_value(resources).items():
            log.debug('Resource: %r: %r', k, v)
            if k == 'Font':
                for (fontid, spec) in dict_value(v).items():
                    objid = None
                    if isinstance(spec, PDFObjRef):
                        objid = spec.objid
                    spec = dict_value(spec)
                    self.fontmap[fontid] = self.rsrcmgr.get_font(objid, spec)
            elif k == 'ColorSpace':
                for (csid, spec) in dict_value(v).items():
                    colorspace = get_colorspace(resolve1(spec))
                    if colorspace is not None:
                        self.csmap[csid] = colorspace
            elif k == 'ProcSet':
                self.rsrcmgr.get_procset(list_value(v))
            elif k == 'XObject':
                for (xobjid, xobjstrm) in dict_value(v).items():
                    self.xobjmap[xobjid] = xobjstrm
        return

    def init_state(self, ctm: Matrix) -> None:
        """Initialize the text and graphic states for rendering a page."""
        # gstack: stack for graphical states.
        self.gstack: List[Tuple[Matrix, PDFTextState, PDFGraphicState]] = []
        self.ctm = ctm
        self.device.set_ctm(self.ctm)
        self.textstate = PDFTextState()
        self.graphicstate = PDFGraphicState()
        self.curpath: List[PathSegment] = []
        # argstack: stack for command arguments.
        self.argstack: List[PDFStackT] = []
        # set some global states.
        self.scs: Optional[PDFColorSpace] = None
        self.ncs: Optional[PDFColorSpace] = None
        if self.csmap:
            self.scs = self.ncs = next(iter(self.csmap.values()))
        return

    def push(self, obj: PDFStackT) -> None:
        self.argstack.append(obj)
        return

    def pop(self, n: int) -> List[PDFStackT]:
        if n == 0:
            return []
        x = self.argstack[-n:]
        self.argstack = self.argstack[:-n]
        return x

    def get_current_state(
        self
    ) -> Tuple[Matrix, PDFTextState, PDFGraphicState]:
        return (self.ctm, self.textstate.copy(), self.graphicstate.copy())

    def set_current_state(
        self,
        state: Tuple[Matrix, PDFTextState, PDFGraphicState]
    ) -> None:
        (self.ctm, self.textstate, self.graphicstate) = state
        self.device.set_ctm(self.ctm)
        return

    def do_q(self) -> None:
        """Save graphics state"""
        self.gstack.append(self.get_current_state())
        return

    def do_Q(self) -> None:
        """Restore graphics state"""
        if self.gstack:
            self.set_current_state(self.gstack.pop())
        return

    def do_cm(
        self,
        a1: PDFStackT,
        b1: PDFStackT,
        c1: PDFStackT,
        d1: PDFStackT,
        e1: PDFStackT,
        f1: PDFStackT
    ) -> None:
        """Concatenate matrix to current transformation matrix"""
        self.ctm = \
            mult_matrix(cast(Matrix, (a1, b1, c1, d1, e1, f1)), self.ctm)
        self.device.set_ctm(self.ctm)
        return

    def do_w(self, linewidth: PDFStackT) -> None:
        """Set line width"""
        self.graphicstate.linewidth = cast(float, linewidth)
        return

    def do_J(self, linecap: PDFStackT) -> None:
        """Set line cap style"""
        self.graphicstate.linecap = linecap
        return

    def do_j(self, linejoin: PDFStackT) -> None:
        """Set line join style"""
        self.graphicstate.linejoin = linejoin
        return

    def do_M(self, miterlimit: PDFStackT) -> None:
        """Set miter limit"""
        self.graphicstate.miterlimit = miterlimit
        return

    def do_d(self, dash: PDFStackT, phase: PDFStackT) -> None:
        """Set line dash pattern"""
        self.graphicstate.dash = (dash, phase)
        return

    def do_ri(self, intent: PDFStackT) -> None:
        """Set color rendering intent"""
        self.graphicstate.intent = intent
        return

    def do_i(self, flatness: PDFStackT) -> None:
        """Set flatness tolerance"""
        self.graphicstate.flatness = flatness
        return

    def do_gs(self, name: PDFStackT) -> None:
        """Set parameters from graphics state parameter dictionary"""
        # todo
        return

    def do_m(self, x: PDFStackT, y: PDFStackT) -> None:
        """Begin new subpath"""
        self.curpath.append(('m', cast(float, x), cast(float, y)))
        return

    def do_l(self, x: PDFStackT, y: PDFStackT) -> None:
        """Append straight line segment to path"""
        self.curpath.append(('l', cast(float, x), cast(float, y)))
        return

    def do_c(
        self,
        x1: PDFStackT,
        y1: PDFStackT,
        x2: PDFStackT,
        y2: PDFStackT,
        x3: PDFStackT,
        y3: PDFStackT
    ) -> None:
        """Append curved segment to path (three control points)"""
        self.curpath.append(('c', cast(float, x1), cast(float, y1),
                             cast(float, x2), cast(float, y2),
                             cast(float, x3), cast(float, y3)))
        return

    def do_v(
        self,
        x2: PDFStackT,
        y2: PDFStackT,
        x3: PDFStackT,
        y3: PDFStackT
    ) -> None:
        """Append curved segment to path (initial point replicated)"""
        self.curpath.append(('v', cast(float, x2), cast(float, y2),
                             cast(float, x3), cast(float, y3)))
        return

    def do_y(
        self,
        x1: PDFStackT,
        y1: PDFStackT,
        x3: PDFStackT,
        y3: PDFStackT
    ) -> None:
        """Append curved segment to path (final point replicated)"""
        self.curpath.append(('y', cast(float, x1), cast(float, y1),
                             cast(float, x3), cast(float, y3)))
        return

    def do_h(self) -> None:
        """Close subpath"""
        self.curpath.append(('h',))
        return

    def do_re(
        self,
        x: PDFStackT,
        y: PDFStackT,
        w: PDFStackT,
        h: PDFStackT
    ) -> None:
        """Append rectangle to path"""
        x = cast(float, x)
        y = cast(float, y)
        w = cast(float, w)
        h = cast(float, h)
        self.curpath.append(('m', x, y))
        self.curpath.append(('l', x+w, y))
        self.curpath.append(('l', x+w, y+h))
        self.curpath.append(('l', x, y+h))
        self.curpath.append(('h',))
        return

    def do_S(self) -> None:
        """Stroke path"""
        self.device.paint_path(self.graphicstate, True, False, False,
                               self.curpath)
        self.curpath = []
        return

    def do_s(self) -> None:
        """Close and stroke path"""
        self.do_h()
        self.do_S()
        return

    def do_f(self) -> None:
        """Fill path using nonzero winding number rule"""
        self.device.paint_path(self.graphicstate, False, True, False,
                               self.curpath)
        self.curpath = []
        return

    def do_F(self) -> None:
        """Fill path using nonzero winding number rule (obsolete)"""
        return self.do_f()

    def do_f_a(self) -> None:
        """Fill path using even-odd rule"""
        self.device.paint_path(self.graphicstate, False, True, True,
                               self.curpath)
        self.curpath = []
        return

    def do_B(self) -> None:
        """Fill and stroke path using nonzero winding number rule"""
        self.device.paint_path(self.graphicstate, True, True, False,
                               self.curpath)
        self.curpath = []
        return

    def do_B_a(self) -> None:
        """Fill and stroke path using even-odd rule"""
        self.device.paint_path(self.graphicstate, True, True, True,
                               self.curpath)
        self.curpath = []
        return

    def do_b(self) -> None:
        """Close, fill, and stroke path using nonzero winding number rule"""
        self.do_h()
        self.do_B()
        return

    def do_b_a(self) -> None:
        """Close, fill, and stroke path using even-odd rule"""
        self.do_h()
        self.do_B_a()
        return

    def do_n(self) -> None:
        """End path without filling or stroking"""
        self.curpath = []
        return

    def do_W(self) -> None:
        """Set clipping path using nonzero winding number rule"""
        return

    def do_W_a(self) -> None:
        """Set clipping path using even-odd rule"""
        return

    def do_CS(self, name: PDFStackT) -> None:
        """Set color space for stroking operations

        Introduced in PDF 1.1
        """
        try:
            self.scs = self.csmap[literal_name(name)]
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError('Undefined ColorSpace: %r' % name)
        return

    def do_cs(self, name: PDFStackT) -> None:
        """Set color space for nonstroking operations"""
        try:
            self.ncs = self.csmap[literal_name(name)]
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError('Undefined ColorSpace: %r' % name)
        return

    def do_G(self, gray: PDFStackT) -> None:
        """Set gray level for stroking operations"""
        self.graphicstate.scolor = cast(float, gray)
        return

    def do_g(self, gray: PDFStackT) -> None:
        """Set gray level for nonstroking operations"""
        self.graphicstate.ncolor = cast(float, gray)
        return

    def do_RG(self, r: PDFStackT, g: PDFStackT, b: PDFStackT) -> None:
        """Set RGB color for stroking operations"""
        self.graphicstate.scolor = \
            (cast(float, r), cast(float, g), cast(float, b))
        return

    def do_rg(self, r: PDFStackT, g: PDFStackT, b: PDFStackT) -> None:
        """Set RGB color for nonstroking operations"""
        self.graphicstate.ncolor = \
            (cast(float, r), cast(float, g), cast(float, b))
        return

    def do_K(
        self,
        c: PDFStackT,
        m: PDFStackT,
        y: PDFStackT,
        k: PDFStackT
    ) -> None:
        """Set CMYK color for stroking operations"""
        self.graphicstate.scolor = \
            (cast(float, c), cast(float, m), cast(float, y), cast(float, k))
        return

    def do_k(
        self,
        c: PDFStackT,
        m: PDFStackT,
        y: PDFStackT,
        k: PDFStackT
    ) -> None:
        """Set CMYK color for nonstroking operations"""
        self.graphicstate.ncolor = \
            (cast(float, c), cast(float, m), cast(float, y), cast(float, k))
        return

    def do_SCN(self) -> None:
        """Set color for stroking operations."""
        if self.scs:
            n = self.scs.ncomponents
        else:
            if settings.STRICT:
                raise PDFInterpreterError('No colorspace specified!')
            n = 1
        self.graphicstate.scolor = cast(Color, self.pop(n))
        return

    def do_scn(self) -> None:
        """Set color for nonstroking operations"""
        if self.ncs:
            n = self.ncs.ncomponents
        else:
            if settings.STRICT:
                raise PDFInterpreterError('No colorspace specified!')
            n = 1
        self.graphicstate.ncolor = cast(Color, self.pop(n))
        return

    def do_SC(self) -> None:
        """Set color for stroking operations"""
        self.do_SCN()
        return

    def do_sc(self) -> None:
        """Set color for nonstroking operations"""
        self.do_scn()
        return

    def do_sh(self, name: object) -> None:
        """Paint area defined by shading pattern"""
        return

    def do_BT(self) -> None:
        """Begin text object

        Initializing the text matrix, Tm, and the text line matrix, Tlm, to
        the identity matrix. Text objects cannot be nested; a second BT cannot
        appear before an ET.
        """
        self.textstate.reset()
        return

    def do_ET(self) -> None:
        """End a text object"""
        return

    def do_BX(self) -> None:
        """Begin compatibility section"""
        return

    def do_EX(self) -> None:
        """End compatibility section"""
        return

    def do_MP(self, tag: PDFStackT) -> None:
        """Define marked-content point"""
        self.device.do_tag(cast(PSLiteral, tag))
        return

    def do_DP(self, tag: PDFStackT, props: PDFStackT) -> None:
        """Define marked-content point with property list"""
        self.device.do_tag(cast(PSLiteral, tag), props)
        return

    def do_BMC(self, tag: PDFStackT) -> None:
        """Begin marked-content sequence"""
        self.device.begin_tag(cast(PSLiteral, tag))
        return

    def do_BDC(self, tag: PDFStackT, props: PDFStackT) -> None:
        """Begin marked-content sequence with property list"""
        self.device.begin_tag(cast(PSLiteral, tag), props)
        return

    def do_EMC(self) -> None:
        """End marked-content sequence"""
        self.device.end_tag()
        return

    def do_Tc(self, space: PDFStackT) -> None:
        """Set character spacing.

        Character spacing is used by the Tj, TJ, and ' operators.

        :param space: a number expressed in unscaled text space units.
        """
        self.textstate.charspace = cast(float, space)
        return

    def do_Tw(self, space: PDFStackT) -> None:
        """Set the word spacing.

        Word spacing is used by the Tj, TJ, and ' operators.

        :param space: a number expressed in unscaled text space units
        """
        self.textstate.wordspace = cast(float, space)
        return

    def do_Tz(self, scale: PDFStackT) -> None:
        """Set the horizontal scaling.

        :param scale: is a number specifying the percentage of the normal width
        """
        self.textstate.scaling = cast(float, scale)
        return

    def do_TL(self, leading: PDFStackT) -> None:
        """Set the text leading.

        Text leading is used only by the T*, ', and " operators.

        :param leading: a number expressed in unscaled text space units
        """
        self.textstate.leading = -cast(float, leading)
        return

    def do_Tf(self, fontid: PDFStackT, fontsize: PDFStackT) -> None:
        """Set the text font

        :param fontid: the name of a font resource in the Font subdictionary
            of the current resource dictionary
        :param fontsize: size is a number representing a scale factor.
        """
        try:
            self.textstate.font = self.fontmap[literal_name(fontid)]
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError('Undefined Font id: %r' % fontid)
            self.textstate.font = self.rsrcmgr.get_font(None, {})
        self.textstate.fontsize = cast(float, fontsize)
        return

    def do_Tr(self, render: PDFStackT) -> None:
        """Set the text rendering mode"""
        self.textstate.render = cast(int, render)
        return

    def do_Ts(self, rise: PDFStackT) -> None:
        """Set the text rise

        :param rise: a number expressed in unscaled text space units
        """
        self.textstate.rise = cast(float, rise)
        return

    def do_Td(self, tx: PDFStackT, ty: PDFStackT) -> None:
        """Move text position"""
        tx = cast(float, tx)
        ty = cast(float, ty)
        (a, b, c, d, e, f) = self.textstate.matrix
        self.textstate.matrix = (a, b, c, d, tx*a+ty*c+e, tx*b+ty*d+f)
        self.textstate.linematrix = (0, 0)
        return

    def do_TD(self, tx: PDFStackT, ty: PDFStackT) -> None:
        """Move text position and set leading"""
        tx = cast(float, tx)
        ty = cast(float, ty)
        (a, b, c, d, e, f) = self.textstate.matrix
        self.textstate.matrix = (a, b, c, d, tx*a+ty*c+e, tx*b+ty*d+f)
        self.textstate.leading = ty
        self.textstate.linematrix = (0, 0)
        return

    def do_Tm(
        self,
        a: PDFStackT,
        b: PDFStackT,
        c: PDFStackT,
        d: PDFStackT,
        e: PDFStackT,
        f: PDFStackT
    ) -> None:
        """Set text matrix and text line matrix"""
        self.textstate.matrix = cast(Matrix, (a, b, c, d, e, f))
        self.textstate.linematrix = (0, 0)
        return

    def do_T_a(self) -> None:
        """Move to start of next text line"""
        (a, b, c, d, e, f) = self.textstate.matrix
        self.textstate.matrix = (a, b, c, d, self.textstate.leading*c+e,
                                 self.textstate.leading*d+f)
        self.textstate.linematrix = (0, 0)
        return

    def do_TJ(self, seq: PDFStackT) -> None:
        """Show text, allowing individual glyph positioning"""
        if self.textstate.font is None:
            if settings.STRICT:
                raise PDFInterpreterError('No font specified!')
            return
        assert self.ncs is not None
        self.device.render_string(self.textstate, cast(PDFTextSeq, seq),
                                  self.ncs, self.graphicstate.copy())
        return

    def do_Tj(self, s: PDFStackT) -> None:
        """Show text"""
        self.do_TJ([s])
        return

    def do__q(self, s: PDFStackT) -> None:
        """Move to next line and show text

        The ' (single quote) operator.
        """
        self.do_T_a()
        self.do_TJ([s])
        return

    def do__w(self, aw: PDFStackT, ac: PDFStackT, s: PDFStackT) -> None:
        """Set word and character spacing, move to next line, and show text

        The " (double quote) operator.
        """
        self.do_Tw(aw)
        self.do_Tc(ac)
        self.do_TJ([s])
        return

    def do_BI(self) -> None:
        """Begin inline image object"""
        return

    def do_ID(self) -> None:
        """Begin inline image data"""
        return

    def do_EI(self, obj: PDFStackT) -> None:
        """End inline image object"""
        if isinstance(obj, PDFStream) and 'W' in obj and 'H' in obj:
            iobjid = str(id(obj))
            self.device.begin_figure(iobjid, (0, 0, 1, 1), MATRIX_IDENTITY)
            self.device.render_image(iobjid, obj)
            self.device.end_figure(iobjid)
        return

    def do_Do(self, xobjid_arg: PDFStackT) -> None:
        """Invoke named XObject"""
        xobjid = cast(str, literal_name(xobjid_arg))
        try:
            xobj = stream_value(self.xobjmap[xobjid])
        except KeyError:
            if settings.STRICT:
                raise PDFInterpreterError('Undefined xobject id: %r' % xobjid)
            return
        log.info('Processing xobj: %r', xobj)
        subtype = xobj.get('Subtype')
        if subtype is LITERAL_FORM and 'BBox' in xobj:
            interpreter = self.dup()
            bbox = cast(Rect, list_value(xobj['BBox']))
            matrix = cast(Matrix, list_value(
                xobj.get('Matrix', MATRIX_IDENTITY)))
            # According to PDF reference 1.7 section 4.9.1, XObjects in
            # earlier PDFs (prior to v1.2) use the page's Resources entry
            # instead of having their own Resources entry.
            xobjres = xobj.get('Resources')
            if xobjres:
                resources = dict_value(xobjres)
            else:
                resources = self.resources.copy()
            self.device.begin_figure(xobjid, bbox, matrix)
            interpreter.render_contents(resources, [xobj],
                                        ctm=mult_matrix(matrix, self.ctm))
            self.device.end_figure(xobjid)
        elif subtype is LITERAL_IMAGE and 'Width' in xobj and 'Height' in xobj:
            self.device.begin_figure(xobjid, (0, 0, 1, 1), MATRIX_IDENTITY)
            self.device.render_image(xobjid, xobj)
            self.device.end_figure(xobjid)
        else:
            # unsupported xobject type.
            pass
        return

    def process_page(self, page: PDFPage) -> None:
        log.info('Processing page: %r', page)
        (x0, y0, x1, y1) = page.mediabox
        if page.rotate == 90:
            ctm = (0, -1, 1, 0, -y0, x1)
        elif page.rotate == 180:
            ctm = (-1, 0, 0, -1, x1, y1)
        elif page.rotate == 270:
            ctm = (0, 1, -1, 0, y1, -x0)
        else:
            ctm = (1, 0, 0, 1, -x0, -y0)
        self.device.begin_page(page, ctm)
        self.render_contents(page.resources, page.contents, ctm=ctm)
        self.device.end_page(page)
        return

    def render_contents(
        self,
        resources: Dict[object, object],
        streams: Sequence[object],
        ctm: Matrix = MATRIX_IDENTITY
    ) -> None:
        """Render the content streams.

        This method may be called recursively.
        """
        log.info('render_contents: resources=%r, streams=%r, ctm=%r',
                 resources, streams, ctm)
        self.init_resources(resources)
        self.init_state(ctm)
        self.execute(list_value(streams))
        return

    def execute(self, streams: Sequence[object]) -> None:
        try:
            parser = PDFContentParser(streams)
        except PSEOF:
            # empty page
            return
        while 1:
            try:
                (_, obj) = parser.nextobject()
            except PSEOF:
                break
            if isinstance(obj, PSKeyword):
                name = keyword_name(obj)
                method = 'do_%s' % name.replace('*', '_a').replace('"', '_w')\
                    .replace("'", '_q')
                if hasattr(self, method):
                    func = getattr(self, method)
                    nargs = func.__code__.co_argcount-1
                    if nargs:
                        args = self.pop(nargs)
                        log.debug('exec: %s %r', name, args)
                        if len(args) == nargs:
                            func(*args)
                    else:
                        log.debug('exec: %s', name)
                        func()
                else:
                    if settings.STRICT:
                        error_msg = 'Unknown operator: %r' % name
                        raise PDFInterpreterError(error_msg)
            else:
                self.push(obj)
        return
