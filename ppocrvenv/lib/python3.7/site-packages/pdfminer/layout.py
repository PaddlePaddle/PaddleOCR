import heapq
import logging
from typing import (Dict, Generic, Iterable, Iterator, List, Optional,
                    Sequence, Set, Tuple, TypeVar, Union, cast)

from .utils import INF
from .utils import LTComponentT
from .utils import Matrix
from .utils import Plane
from .utils import Point
from .utils import Rect
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import fsplit
from .utils import get_bound
from .utils import matrix2str
from .utils import uniq
from .pdfcolor import PDFColorSpace
from .pdftypes import PDFStream
from .pdfinterp import Color
from .pdfinterp import PDFGraphicState
from .pdffont import PDFFont

logger = logging.getLogger(__name__)


class IndexAssigner:

    def __init__(self, index: int = 0) -> None:
        self.index = index
        return

    def run(self, obj: "LTItem") -> None:
        if isinstance(obj, LTTextBox):
            obj.index = self.index
            self.index += 1
        elif isinstance(obj, LTTextGroup):
            for x in obj:
                self.run(x)
        return


class LAParams:
    """Parameters for layout analysis

    :param line_overlap: If two characters have more overlap than this they
        are considered to be on the same line. The overlap is specified
        relative to the minimum height of both characters.
    :param char_margin: If two characters are closer together than this
        margin they are considered part of the same line. The margin is
        specified relative to the width of the character.
    :param word_margin: If two characters on the same line are further apart
        than this margin then they are considered to be two separate words, and
        an intermediate space will be added for readability. The margin is
        specified relative to the width of the character.
    :param line_margin: If two lines are are close together they are
        considered to be part of the same paragraph. The margin is
        specified relative to the height of a line.
    :param boxes_flow: Specifies how much a horizontal and vertical position
        of a text matters when determining the order of text boxes. The value
        should be within the range of -1.0 (only horizontal position
        matters) to +1.0 (only vertical position matters). You can also pass
        `None` to disable advanced layout analysis, and instead return text
        based on the position of the bottom left corner of the text box.
    :param detect_vertical: If vertical text should be considered during
        layout analysis
    :param all_texts: If layout analysis should be performed on text in
        figures.
    """

    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: Optional[float] = 0.5,
        detect_vertical: bool = False,
        all_texts: bool = False
    ) -> None:
        self.line_overlap = line_overlap
        self.char_margin = char_margin
        self.line_margin = line_margin
        self.word_margin = word_margin
        self.boxes_flow = boxes_flow
        self.detect_vertical = detect_vertical
        self.all_texts = all_texts

        self._validate()
        return

    def _validate(self) -> None:
        if self.boxes_flow is not None:
            boxes_flow_err_msg = ("LAParam boxes_flow should be None, or a "
                                  "number between -1 and +1")
            if not (isinstance(self.boxes_flow, int) or
                    isinstance(self.boxes_flow, float)):
                raise TypeError(boxes_flow_err_msg)
            if not -1 <= self.boxes_flow <= 1:
                raise ValueError(boxes_flow_err_msg)

    def __repr__(self) -> str:
        return '<LAParams: char_margin=%.1f, line_margin=%.1f, ' \
               'word_margin=%.1f all_texts=%r>' % \
               (self.char_margin, self.line_margin, self.word_margin,
                self.all_texts)


class LTItem:
    """Interface for things that can be analyzed"""

    def analyze(self, laparams: LAParams) -> None:
        """Perform the layout analysis."""
        return


class LTText:
    """Interface for things that have text"""

    def __repr__(self) -> str:
        return ('<%s %r>' %
                (self.__class__.__name__, self.get_text()))

    def get_text(self) -> str:
        """Text contained in this object"""
        raise NotImplementedError


class LTComponent(LTItem):
    """Object with a bounding box"""

    def __init__(self, bbox: Rect) -> None:
        LTItem.__init__(self)
        self.set_bbox(bbox)
        return

    def __repr__(self) -> str:
        return ('<%s %s>' %
                (self.__class__.__name__, bbox2str(self.bbox)))

    # Disable comparison.
    def __lt__(self, _: object) -> bool:
        raise ValueError

    def __le__(self, _: object) -> bool:
        raise ValueError

    def __gt__(self, _: object) -> bool:
        raise ValueError

    def __ge__(self, _: object) -> bool:
        raise ValueError

    def set_bbox(self, bbox: Rect) -> None:
        (x0, y0, x1, y1) = bbox
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1-x0
        self.height = y1-y0
        self.bbox = bbox
        return

    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def is_hoverlap(self, obj: "LTComponent") -> bool:
        assert isinstance(obj, LTComponent), str(type(obj))
        return obj.x0 <= self.x1 and self.x0 <= obj.x1

    def hdistance(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_hoverlap(obj):
            return 0
        else:
            return min(abs(self.x0-obj.x1), abs(self.x1-obj.x0))

    def hoverlap(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_hoverlap(obj):
            return min(abs(self.x0-obj.x1), abs(self.x1-obj.x0))
        else:
            return 0

    def is_voverlap(self, obj: "LTComponent") -> bool:
        assert isinstance(obj, LTComponent), str(type(obj))
        return obj.y0 <= self.y1 and self.y0 <= obj.y1

    def vdistance(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_voverlap(obj):
            return 0
        else:
            return min(abs(self.y0-obj.y1), abs(self.y1-obj.y0))

    def voverlap(self, obj: "LTComponent") -> float:
        assert isinstance(obj, LTComponent), str(type(obj))
        if self.is_voverlap(obj):
            return min(abs(self.y0-obj.y1), abs(self.y1-obj.y0))
        else:
            return 0


class LTCurve(LTComponent):
    """A generic Bezier curve"""

    def __init__(
        self,
        linewidth: float,
        pts: List[Point],
        stroke: bool = False,
        fill: bool = False,
        evenodd: bool = False,
        stroking_color: Optional[Color] = None,
        non_stroking_color: Optional[Color] = None
    ) -> None:
        LTComponent.__init__(self, get_bound(pts))
        self.pts = pts
        self.linewidth = linewidth
        self.stroke = stroke
        self.fill = fill
        self.evenodd = evenodd
        self.stroking_color = stroking_color
        self.non_stroking_color = non_stroking_color
        return

    def get_pts(self) -> str:
        return ','.join('%.3f,%.3f' % p for p in self.pts)


class LTLine(LTCurve):
    """A single straight line.

    Could be used for separating text or figures.
    """

    def __init__(
        self,
        linewidth: float,
        p0: Point,
        p1: Point,
        stroke: bool = False,
        fill: bool = False,
        evenodd: bool = False,
        stroking_color: Optional[Color] = None,
        non_stroking_color: Optional[Color] = None
    ) -> None:
        LTCurve.__init__(self, linewidth, [p0, p1], stroke, fill, evenodd,
                         stroking_color, non_stroking_color)
        return


class LTRect(LTCurve):
    """A rectangle.

    Could be used for framing another pictures or figures.
    """

    def __init__(
        self,
        linewidth: float,
        bbox: Rect,
        stroke: bool = False,
        fill: bool = False,
        evenodd: bool = False,
        stroking_color: Optional[Color] = None,
        non_stroking_color: Optional[Color] = None
    ) -> None:
        (x0, y0, x1, y1) = bbox
        LTCurve.__init__(self, linewidth,
                         [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], stroke,
                         fill, evenodd, stroking_color, non_stroking_color)
        return


class LTImage(LTComponent):
    """An image object.

    Embedded images can be in JPEG, Bitmap or JBIG2.
    """

    def __init__(self, name: str, stream: PDFStream, bbox: Rect) -> None:
        LTComponent.__init__(self, bbox)
        self.name = name
        self.stream = stream
        self.srcsize = (stream.get_any(('W', 'Width')),
                        stream.get_any(('H', 'Height')))
        self.imagemask = stream.get_any(('IM', 'ImageMask'))
        self.bits = stream.get_any(('BPC', 'BitsPerComponent'), 1)
        self.colorspace = stream.get_any(('CS', 'ColorSpace'))
        if not isinstance(self.colorspace, list):
            self.colorspace = [self.colorspace]
        return

    def __repr__(self) -> str:
        return ('<%s(%s) %s %r>' %
                (self.__class__.__name__, self.name,
                 bbox2str(self.bbox), self.srcsize))


class LTAnno(LTItem, LTText):
    """Actual letter in the text as a Unicode string.

    Note that, while a LTChar object has actual boundaries, LTAnno objects does
    not, as these are "virtual" characters, inserted by a layout analyzer
    according to the relationship between two characters (e.g. a space).
    """

    def __init__(self, text: str) -> None:
        self._text = text
        return

    def get_text(self) -> str:
        return self._text


class LTChar(LTComponent, LTText):
    """Actual letter in the text as a Unicode string."""

    def __init__(
        self,
        matrix: Matrix,
        font: PDFFont,
        fontsize: float,
        scaling: float,
        rise: float,
        text: str,
        textwidth: float,
        textdisp: Union[float, Tuple[Optional[float], float]],
        ncs: PDFColorSpace,
        graphicstate: PDFGraphicState
    ) -> None:
        LTText.__init__(self)
        self._text = text
        self.matrix = matrix
        self.fontname = font.fontname
        self.ncs = ncs
        self.graphicstate = graphicstate
        self.adv = textwidth * fontsize * scaling
        # compute the boundary rectangle.
        if font.is_vertical():
            # vertical
            assert isinstance(textdisp, tuple)
            (vx, vy) = textdisp
            if vx is None:
                vx = fontsize * 0.5
            else:
                vx = vx * fontsize * .001
            vy = (1000 - vy) * fontsize * .001
            bbox_lower_left = (-vx, vy + rise + self.adv)
            bbox_upper_right = (-vx + fontsize, vy + rise)
        else:
            # horizontal
            descent = font.get_descent() * fontsize
            bbox_lower_left = (0, descent + rise)
            bbox_upper_right = (self.adv, descent + rise + fontsize)
        (a, b, c, d, e, f) = self.matrix
        self.upright = (0 < a*d*scaling and b*c <= 0)
        (x0, y0) = apply_matrix_pt(self.matrix, bbox_lower_left)
        (x1, y1) = apply_matrix_pt(self.matrix, bbox_upper_right)
        if x1 < x0:
            (x0, x1) = (x1, x0)
        if y1 < y0:
            (y0, y1) = (y1, y0)
        LTComponent.__init__(self, (x0, y0, x1, y1))
        if font.is_vertical():
            self.size = self.width
        else:
            self.size = self.height
        return

    def __repr__(self) -> str:
        return ('<%s %s matrix=%s font=%r adv=%s text=%r>' %
                (self.__class__.__name__, bbox2str(self.bbox),
                 matrix2str(self.matrix), self.fontname, self.adv,
                 self.get_text()))

    def get_text(self) -> str:
        return self._text

    def is_compatible(self, obj: object) -> bool:
        """Returns True if two characters can coexist in the same line."""
        return True


LTItemT = TypeVar('LTItemT', bound=LTItem)


class LTContainer(LTComponent, Generic[LTItemT]):
    """Object that can be extended and analyzed"""

    def __init__(self, bbox: Rect) -> None:
        LTComponent.__init__(self, bbox)
        self._objs: List[LTItemT] = []
        return

    def __iter__(self) -> Iterator[LTItemT]:
        return iter(self._objs)

    def __len__(self) -> int:
        return len(self._objs)

    def add(self, obj: LTItemT) -> None:
        self._objs.append(obj)
        return

    def extend(self, objs: Iterable[LTItemT]) -> None:
        for obj in objs:
            self.add(obj)
        return

    def analyze(self, laparams: LAParams) -> None:
        for obj in self._objs:
            obj.analyze(laparams)
        return


class LTExpandableContainer(LTContainer[LTItemT]):
    def __init__(self) -> None:
        LTContainer.__init__(self, (+INF, +INF, -INF, -INF))
        return

    # Incompatible override: we take an LTComponent (with bounding box), but
    # super() LTContainer only considers LTItem (no bounding box).
    def add(self, obj: LTComponent) -> None:  # type: ignore[override]
        LTContainer.add(self, cast(LTItemT, obj))
        self.set_bbox((min(self.x0, obj.x0), min(self.y0, obj.y0),
                       max(self.x1, obj.x1), max(self.y1, obj.y1)))
        return


class LTTextContainer(LTExpandableContainer[LTItemT], LTText):
    def __init__(self) -> None:
        LTText.__init__(self)
        LTExpandableContainer.__init__(self)
        return

    def get_text(self) -> str:
        return ''.join(cast(LTText, obj).get_text() for obj in self
                       if isinstance(obj, LTText))


TextLineElement = Union[LTChar, LTAnno]


class LTTextLine(LTTextContainer[TextLineElement]):
    """Contains a list of LTChar objects that represent a single text line.

    The characters are aligned either horizontally or vertically, depending on
    the text's writing mode.
    """

    def __init__(self, word_margin: float) -> None:
        super().__init__()
        self.word_margin = word_margin
        return

    def __repr__(self) -> str:
        return ('<%s %s %r>' %
                (self.__class__.__name__, bbox2str(self.bbox),
                 self.get_text()))

    def analyze(self, laparams: LAParams) -> None:
        LTTextContainer.analyze(self, laparams)
        LTContainer.add(self, LTAnno('\n'))
        return

    def find_neighbors(self, plane: Plane[LTComponentT], ratio: float
                       ) -> List["LTTextLine"]:
        raise NotImplementedError


class LTTextLineHorizontal(LTTextLine):
    def __init__(self, word_margin: float) -> None:
        LTTextLine.__init__(self, word_margin)
        self._x1: float = +INF
        return

    # Incompatible override: we take an LTComponent (with bounding box), but
    # LTContainer only considers LTItem (no bounding box).
    def add(self, obj: LTComponent) -> None:  # type: ignore[override]
        if isinstance(obj, LTChar) and self.word_margin:
            margin = self.word_margin * max(obj.width, obj.height)
            if self._x1 < obj.x0 - margin:
                LTContainer.add(self, LTAnno(' '))
        self._x1 = obj.x1
        super().add(obj)
        return

    def find_neighbors(
        self,
        plane: Plane[LTComponentT],
        ratio: float
    ) -> List[LTTextLine]:
        """
        Finds neighboring LTTextLineHorizontals in the plane.

        Returns a list of other LTTestLineHorizontals in the plane which are
        close to self. "Close" can be controlled by ratio. The returned objects
        will be the same height as self, and also either left-, right-, or
        centrally-aligned.
        """
        d = ratio * self.height
        objs = plane.find((self.x0, self.y0 - d, self.x1, self.y1 + d))
        return [obj for obj in objs
                if (isinstance(obj, LTTextLineHorizontal) and
                    self._is_same_height_as(obj, tolerance=d) and
                    (self._is_left_aligned_with(obj, tolerance=d) or
                     self._is_right_aligned_with(obj, tolerance=d) or
                     self._is_centrally_aligned_with(obj, tolerance=d)))]

    def _is_left_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0
    ) -> bool:
        """
        Whether the left-hand edge of `other` is within `tolerance`.
        """
        return abs(other.x0 - self.x0) <= tolerance

    def _is_right_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0
    ) -> bool:
        """
        Whether the right-hand edge of `other` is within `tolerance`.
        """
        return abs(other.x1 - self.x1) <= tolerance

    def _is_centrally_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0
    ) -> bool:
        """
        Whether the horizontal center of `other` is within `tolerance`.
        """
        return abs(
            (other.x0 + other.x1) / 2 - (self.x0 + self.x1) / 2) <= tolerance

    def _is_same_height_as(
        self,
        other: LTComponent,
        tolerance: float = 0
    ) -> bool:
        return abs(other.height - self.height) <= tolerance


class LTTextLineVertical(LTTextLine):
    def __init__(self, word_margin: float) -> None:
        LTTextLine.__init__(self, word_margin)
        self._y0: float = -INF
        return

    # Incompatible override: we take an LTComponent (with bounding box), but
    # LTContainer only considers LTItem (no bounding box).
    def add(self, obj: LTComponent) -> None:  # type: ignore[override]
        if isinstance(obj, LTChar) and self.word_margin:
            margin = self.word_margin * max(obj.width, obj.height)
            if obj.y1 + margin < self._y0:
                LTContainer.add(self, LTAnno(' '))
        self._y0 = obj.y0
        super().add(obj)
        return

    def find_neighbors(
        self,
        plane: Plane[LTComponentT],
        ratio: float
    ) -> List[LTTextLine]:
        """
        Finds neighboring LTTextLineVerticals in the plane.

        Returns a list of other LTTextLineVerticals in the plane which are
        close to self. "Close" can be controlled by ratio. The returned objects
        will be the same width as self, and also either upper-, lower-, or
        centrally-aligned.
        """
        d = ratio * self.width
        objs = plane.find((self.x0 - d, self.y0, self.x1 + d, self.y1))
        return [obj for obj in objs
                if (isinstance(obj, LTTextLineVertical) and
                    self._is_same_width_as(obj, tolerance=d) and
                    (self._is_lower_aligned_with(obj, tolerance=d) or
                     self._is_upper_aligned_with(obj, tolerance=d) or
                     self._is_centrally_aligned_with(obj, tolerance=d)))]

    def _is_lower_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0
    ) -> bool:
        """
        Whether the lower edge of `other` is within `tolerance`.
        """
        return abs(other.y0 - self.y0) <= tolerance

    def _is_upper_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0
    ) -> bool:
        """
        Whether the upper edge of `other` is within `tolerance`.
        """
        return abs(other.y1 - self.y1) <= tolerance

    def _is_centrally_aligned_with(
        self,
        other: LTComponent,
        tolerance: float = 0
    ) -> bool:
        """
        Whether the vertical center of `other` is within `tolerance`.
        """
        return abs(
            (other.y0 + other.y1) / 2 - (self.y0 + self.y1) / 2) <= tolerance

    def _is_same_width_as(self, other: LTComponent, tolerance: float) -> bool:
        return abs(other.width - self.width) <= tolerance


class LTTextBox(LTTextContainer[LTTextLine]):
    """Represents a group of text chunks in a rectangular area.

    Note that this box is created by geometric analysis and does not
    necessarily represents a logical boundary of the text. It contains a list
    of LTTextLine objects.
    """

    def __init__(self) -> None:
        LTTextContainer.__init__(self)
        self.index: int = -1
        return

    def __repr__(self) -> str:
        return ('<%s(%s) %s %r>' %
                (self.__class__.__name__,
                 self.index, bbox2str(self.bbox), self.get_text()))

    def get_writing_mode(self) -> str:
        raise NotImplementedError


class LTTextBoxHorizontal(LTTextBox):
    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        self._objs.sort(key=lambda obj: -obj.y1)
        return

    def get_writing_mode(self) -> str:
        return 'lr-tb'


class LTTextBoxVertical(LTTextBox):
    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        self._objs.sort(key=lambda obj: -obj.x1)
        return

    def get_writing_mode(self) -> str:
        return 'tb-rl'


TextGroupElement = Union[LTTextBox, "LTTextGroup"]


class LTTextGroup(LTTextContainer[TextGroupElement]):
    def __init__(self, objs: Iterable[TextGroupElement]) -> None:
        super().__init__()
        self.extend(objs)
        return


class LTTextGroupLRTB(LTTextGroup):
    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        assert laparams.boxes_flow is not None
        boxes_flow = laparams.boxes_flow
        # reorder the objects from top-left to bottom-right.
        self._objs.sort(
            key=lambda obj: (1 - boxes_flow) * obj.x0
            - (1 + boxes_flow) * (obj.y0 + obj.y1))
        return


class LTTextGroupTBRL(LTTextGroup):
    def analyze(self, laparams: LAParams) -> None:
        super().analyze(laparams)
        assert laparams.boxes_flow is not None
        boxes_flow = laparams.boxes_flow
        # reorder the objects from top-right to bottom-left.
        self._objs.sort(
            key=lambda obj: - (1 + boxes_flow) * (obj.x0 + obj.x1)
                            - (1 - boxes_flow) * obj.y1)
        return


class LTLayoutContainer(LTContainer[LTComponent]):
    def __init__(self, bbox: Rect) -> None:
        LTContainer.__init__(self, bbox)
        self.groups: Optional[List[LTTextGroup]] = None
        return

    # group_objects: group text object to textlines.
    def group_objects(
        self,
        laparams: LAParams,
        objs: Iterable[LTComponent]
    ) -> Iterator[LTTextLine]:
        obj0 = None
        line = None
        for obj1 in objs:
            if obj0 is not None:
                # halign: obj0 and obj1 is horizontally aligned.
                #
                #   +------+ - - -
                #   | obj0 | - - +------+   -
                #   |      |     | obj1 |   | (line_overlap)
                #   +------+ - - |      |   -
                #          - - - +------+
                #
                #          |<--->|
                #        (char_margin)
                halign = \
                    obj0.is_compatible(obj1) \
                    and obj0.is_voverlap(obj1) \
                    and min(obj0.height, obj1.height) * laparams.line_overlap \
                    < obj0.voverlap(obj1) \
                    and obj0.hdistance(obj1) \
                    < max(obj0.width, obj1.width) * laparams.char_margin

                # valign: obj0 and obj1 is vertically aligned.
                #
                #   +------+
                #   | obj0 |
                #   |      |
                #   +------+ - - -
                #     |    |     | (char_margin)
                #     +------+ - -
                #     | obj1 |
                #     |      |
                #     +------+
                #
                #     |<-->|
                #   (line_overlap)
                valign = \
                    laparams.detect_vertical \
                    and obj0.is_compatible(obj1) \
                    and obj0.is_hoverlap(obj1) \
                    and min(obj0.width, obj1.width) * laparams.line_overlap \
                    < obj0.hoverlap(obj1) \
                    and obj0.vdistance(obj1) \
                    < max(obj0.height, obj1.height) * laparams.char_margin

                if ((halign and isinstance(line, LTTextLineHorizontal)) or
                        (valign and isinstance(line, LTTextLineVertical))):

                    line.add(obj1)
                elif line is not None:
                    yield line
                    line = None
                else:
                    if valign and not halign:
                        line = LTTextLineVertical(laparams.word_margin)
                        line.add(obj0)
                        line.add(obj1)
                    elif halign and not valign:
                        line = LTTextLineHorizontal(laparams.word_margin)
                        line.add(obj0)
                        line.add(obj1)
                    else:
                        line = LTTextLineHorizontal(laparams.word_margin)
                        line.add(obj0)
                        yield line
                        line = None
            obj0 = obj1
        if line is None:
            line = LTTextLineHorizontal(laparams.word_margin)
            assert obj0 is not None
            line.add(obj0)
        yield line
        return

    def group_textlines(
        self,
        laparams: LAParams,
        lines: Iterable[LTTextLine]
    ) -> Iterator[LTTextBox]:
        """Group neighboring lines to textboxes"""
        plane: Plane[LTTextLine] = Plane(self.bbox)
        plane.extend(lines)
        boxes: Dict[LTTextLine, LTTextBox] = {}
        for line in lines:
            neighbors = line.find_neighbors(plane, laparams.line_margin)
            members = [line]
            for obj1 in neighbors:
                members.append(obj1)
                if obj1 in boxes:
                    members.extend(boxes.pop(obj1))
            if isinstance(line, LTTextLineHorizontal):
                box: LTTextBox = LTTextBoxHorizontal()
            else:
                box = LTTextBoxVertical()
            for obj in uniq(members):
                box.add(obj)
                boxes[obj] = box
        done = set()
        for line in lines:
            if line not in boxes:
                continue
            box = boxes[line]
            if box in done:
                continue
            done.add(box)
            if not box.is_empty():
                yield box
        return

    def group_textboxes(
        self,
        laparams: LAParams,
        boxes: Sequence[LTTextBox]
    ) -> List[LTTextGroup]:
        """Group textboxes hierarchically.

        Get pair-wise distances, via dist func defined below, and then merge
        from the closest textbox pair. Once obj1 and obj2 are merged /
        grouped, the resulting group is considered as a new object, and its
        distances to other objects & groups are added to the process queue.

        For performance reason, pair-wise distances and object pair info are
        maintained in a heap of (idx, dist, id(obj1), id(obj2), obj1, obj2)
        tuples. It ensures quick access to the smallest element. Note that
        since comparison operators, e.g., __lt__, are disabled for
        LTComponent, id(obj) has to appear before obj in element tuples.

        :param laparams: LAParams object.
        :param boxes: All textbox objects to be grouped.
        :return: a list that has only one element, the final top level group.
        """

        ElementT = Union[LTTextBox, LTTextGroup]
        plane: Plane[ElementT] = Plane(self.bbox)

        def dist(obj1: LTComponent, obj2: LTComponent) -> float:
            """A distance function between two TextBoxes.

            Consider the bounding rectangle for obj1 and obj2.
            Return its area less the areas of obj1 and obj2,
            shown as 'www' below. This value may be negative.
                    +------+..........+ (x1, y1)
                    | obj1 |wwwwwwwwww:
                    +------+www+------+
                    :wwwwwwwwww| obj2 |
            (x0, y0) +..........+------+
            """
            x0 = min(obj1.x0, obj2.x0)
            y0 = min(obj1.y0, obj2.y0)
            x1 = max(obj1.x1, obj2.x1)
            y1 = max(obj1.y1, obj2.y1)
            return (x1 - x0) * (y1 - y0) \
                - obj1.width*obj1.height - obj2.width*obj2.height

        def isany(obj1: ElementT, obj2: ElementT) -> Set[ElementT]:
            """Check if there's any other object between obj1 and obj2."""
            x0 = min(obj1.x0, obj2.x0)
            y0 = min(obj1.y0, obj2.y0)
            x1 = max(obj1.x1, obj2.x1)
            y1 = max(obj1.y1, obj2.y1)
            objs = set(plane.find((x0, y0, x1, y1)))
            return objs.difference((obj1, obj2))

        dists: List[Tuple[bool, float, int, int, ElementT, ElementT]] = []
        for i in range(len(boxes)):
            box1 = boxes[i]
            for j in range(i+1, len(boxes)):
                box2 = boxes[j]
                dists.append((False, dist(box1, box2), id(box1), id(box2),
                              box1, box2))
        heapq.heapify(dists)

        plane.extend(boxes)
        done = set()
        while len(dists) > 0:
            (skip_isany, d, id1, id2, obj1, obj2) = heapq.heappop(dists)
            # Skip objects that are already merged
            if (id1 not in done) and (id2 not in done):
                if skip_isany and isany(obj1, obj2):
                    heapq.heappush(dists, (True, d, id1, id2, obj1, obj2))
                    continue
                if isinstance(obj1, (LTTextBoxVertical, LTTextGroupTBRL)) or \
                        isinstance(obj2, (LTTextBoxVertical, LTTextGroupTBRL)):
                    group: LTTextGroup = LTTextGroupTBRL([obj1, obj2])
                else:
                    group = LTTextGroupLRTB([obj1, obj2])
                plane.remove(obj1)
                plane.remove(obj2)
                done.update([id1, id2])

                for other in plane:
                    heapq.heappush(dists, (False, dist(group, other),
                                           id(group), id(other), group, other))
                plane.add(group)
        # By now only groups are in the plane
        return list(cast(LTTextGroup, g) for g in plane)

    def analyze(self, laparams: LAParams) -> None:
        # textobjs is a list of LTChar objects, i.e.
        # it has all the individual characters in the page.
        (textobjs, otherobjs) = fsplit(lambda obj: isinstance(obj, LTChar),
                                       self)
        for obj in otherobjs:
            obj.analyze(laparams)
        if not textobjs:
            return
        textlines = list(self.group_objects(laparams, textobjs))
        (empties, textlines) = fsplit(lambda obj: obj.is_empty(), textlines)
        for obj in empties:
            obj.analyze(laparams)
        textboxes = list(self.group_textlines(laparams, textlines))
        if laparams.boxes_flow is None:
            for textbox in textboxes:
                textbox.analyze(laparams)

            def getkey(box: LTTextBox) -> Tuple[int, float, float]:
                if isinstance(box, LTTextBoxVertical):
                    return (0, -box.x1, -box.y0)
                else:
                    return (1, -box.y0, box.x0)
            textboxes.sort(key=getkey)
        else:
            self.groups = self.group_textboxes(laparams, textboxes)
            assigner = IndexAssigner()
            for group in self.groups:
                group.analyze(laparams)
                assigner.run(group)
            textboxes.sort(key=lambda box: box.index)
        self._objs = (cast(List[LTComponent], textboxes) + otherobjs
                      + cast(List[LTComponent], empties))
        return


class LTFigure(LTLayoutContainer):
    """Represents an area used by PDF Form objects.

    PDF Forms can be used to present figures or pictures by embedding yet
    another PDF document within a page. Note that LTFigure objects can appear
    recursively.
    """

    def __init__(self, name: str, bbox: Rect, matrix: Matrix) -> None:
        self.name = name
        self.matrix = matrix
        (x, y, w, h) = bbox
        bounds = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
        bbox = get_bound(apply_matrix_pt(matrix, (p, q)) for (p, q) in bounds)
        LTLayoutContainer.__init__(self, bbox)
        return

    def __repr__(self) -> str:
        return ('<%s(%s) %s matrix=%s>' %
                (self.__class__.__name__, self.name,
                 bbox2str(self.bbox), matrix2str(self.matrix)))

    def analyze(self, laparams: LAParams) -> None:
        if not laparams.all_texts:
            return
        LTLayoutContainer.analyze(self, laparams)
        return


class LTPage(LTLayoutContainer):
    """Represents an entire page.

    May contain child objects like LTTextBox, LTFigure, LTImage, LTRect,
    LTCurve and LTLine.
    """

    def __init__(self, pageid: int, bbox: Rect, rotate: float = 0) -> None:
        LTLayoutContainer.__init__(self, bbox)
        self.pageid = pageid
        self.rotate = rotate
        return

    def __repr__(self) -> str:
        return ('<%s(%r) %s rotate=%r>' %
                (self.__class__.__name__, self.pageid,
                 bbox2str(self.bbox), self.rotate))
