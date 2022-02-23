import re

from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFPageInterpreter

from . import utils
from .container import Container
from .table import TableFinder
from .utils import resolve_all

lt_pat = re.compile(r"^LT")

ALL_ATTRS = set(
    [
        "adv",
        "height",
        "linewidth",
        "pts",
        "size",
        "srcsize",
        "width",
        "x0",
        "x1",
        "y0",
        "y1",
        "bits",
        "upright",
        "font",
        "fontname",
        "name",
        "text",
        "imagemask",
        "colorspace",
        "evenodd",
        "fill",
        "non_stroking_color",
        "path",
        "stream",
        "stroke",
        "stroking_color",
    ]
)


class Page(Container):
    cached_properties = Container.cached_properties + ["_layout"]
    is_original = True

    def __init__(self, pdf, page_obj, page_number=None, initial_doctop=0):
        self.pdf = pdf
        self.page_obj = page_obj
        self.page_number = page_number
        _rotation = resolve_all(self.page_obj.attrs.get("Rotate", 0))
        self.rotation = _rotation % 360
        self.page_obj.rotate = self.rotation
        self.initial_doctop = initial_doctop

        cropbox = page_obj.attrs.get("CropBox")
        mediabox = page_obj.attrs.get("MediaBox")

        self.cropbox = resolve_all(cropbox) if cropbox is not None else None
        self.mediabox = resolve_all(mediabox) or self.cropbox
        m = self.mediabox

        if self.rotation in [90, 270]:
            self.bbox = (
                min(m[1], m[3]),
                min(m[0], m[2]),
                max(m[1], m[3]),
                max(m[0], m[2]),
            )
        else:
            self.bbox = (
                min(m[0], m[2]),
                min(m[1], m[3]),
                max(m[0], m[2]),
                max(m[1], m[3]),
            )

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def layout(self):
        if hasattr(self, "_layout"):
            return self._layout
        device = PDFPageAggregator(
            self.pdf.rsrcmgr,
            pageno=self.page_number,
            laparams=self.pdf.laparams,
        )
        interpreter = PDFPageInterpreter(self.pdf.rsrcmgr, device)
        interpreter.process_page(self.page_obj)
        self._layout = device.get_result()
        return self._layout

    @property
    def annots(self):
        def parse(annot):
            rect = annot["Rect"]

            a = annot.get("A", {})
            extras = {
                "uri": a.get("URI"),
                "title": annot.get("T"),
                "contents": annot.get("Contents"),
            }
            for k, v in extras.items():
                if v is not None:
                    try:
                        extras[k] = v.decode("utf-8")
                    except UnicodeDecodeError:
                        extras[k] = v.decode("utf-16")

            parsed = {
                "page_number": self.page_number,
                "object_type": "annot",
                "x0": rect[0],
                "y0": rect[1],
                "x1": rect[2],
                "y1": rect[3],
                "doctop": self.initial_doctop + self.height - rect[3],
                "top": self.height - rect[3],
                "bottom": self.height - rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1],
            }
            parsed.update(extras)
            # Replace the indirect reference to the page dictionary
            # with a pointer to our actual page
            if "P" in annot:
                annot["P"] = self
            parsed["data"] = annot
            return parsed

        raw = resolve_all(self.page_obj.annots) or []
        return list(map(parse, raw))

    @property
    def hyperlinks(self):
        return [a for a in self.annots if a["uri"] is not None]

    @property
    def objects(self):
        if hasattr(self, "_objects"):
            return self._objects
        self._objects = self.parse_objects()
        return self._objects

    def process_object(self, obj):
        kind = re.sub(lt_pat, "", obj.__class__.__name__).lower()

        def process_attr(item):
            k, v = item
            if k in ALL_ATTRS:
                res = resolve_all(v)
                return (k, res)
            else:
                return None

        attr = dict(filter(None, map(process_attr, obj.__dict__.items())))

        attr["object_type"] = kind
        attr["page_number"] = self.page_number

        if hasattr(obj, "graphicstate"):
            gs = obj.graphicstate
            attr["stroking_color"] = gs.scolor
            attr["non_stroking_color"] = gs.ncolor

        if hasattr(obj, "get_text"):
            attr["text"] = obj.get_text()

        if kind == "curve":

            def point2coord(pt):
                x, y = pt
                return (x, self.height - y)

            attr["points"] = list(map(point2coord, obj.pts))

        if attr.get("y0") is not None:
            attr["top"] = self.height - attr["y1"]
            attr["bottom"] = self.height - attr["y0"]
            attr["doctop"] = self.initial_doctop + attr["top"]

        return attr

    def iter_layout_objects(self, layout_objects):
        for obj in layout_objects:
            # If object is, like LTFigure, a higher-level object ...
            if hasattr(obj, "_objs"):
                # and LAParams is passed, process the object itself.
                if self.pdf.laparams is not None:
                    yield self.process_object(obj)
                # Regardless, iterate through its children
                yield from self.iter_layout_objects(obj._objs)
            else:
                yield self.process_object(obj)

    def parse_objects(self):
        objects = {}
        for obj in self.iter_layout_objects(self.layout._objs):
            kind = obj["object_type"]
            if kind in ["anno"]:
                continue
            if objects.get(kind) is None:
                objects[kind] = []
            objects[kind].append(obj)
        return objects

    def debug_tablefinder(self, table_settings={}):
        return TableFinder(self, table_settings)

    def find_tables(self, table_settings={}):
        return TableFinder(self, table_settings).tables

    def extract_tables(self, table_settings={}):
        table_settings = TableFinder.resolve_table_settings(table_settings)
        tables = self.find_tables(table_settings)

        extract_kwargs = dict(
            (k, table_settings["text_" + k])
            for k in ["x_tolerance", "y_tolerance"]
            if "text_" + k in table_settings
        )

        return [table.extract(**extract_kwargs) for table in tables]

    def extract_table(self, table_settings={}):
        table_settings = TableFinder.resolve_table_settings(table_settings)
        tables = self.find_tables(table_settings)

        if len(tables) == 0:
            return None

        # Return the largest table, as measured by number of cells.
        def sorter(x):
            return (-len(x.cells), x.bbox[1], x.bbox[0])

        largest = list(sorted(tables, key=sorter))[0]

        extract_kwargs = dict(
            (k, table_settings["text_" + k])
            for k in ["x_tolerance", "y_tolerance"]
            if "text_" + k in table_settings
        )

        return largest.extract(**extract_kwargs)

    def extract_text(self, **kwargs):
        return utils.extract_text(
            self.chars, x_shift=self.bbox[0], y_shift=self.bbox[1], **kwargs
        )

    def extract_words(self, **kwargs):
        return utils.extract_words(self.chars, **kwargs)

    def crop(self, bbox, relative=False):
        return CroppedPage(self, bbox, relative=relative)

    def within_bbox(self, bbox, relative=False):
        """
        Same as .crop, except only includes objects fully within the bbox
        """
        return CroppedPage(self, bbox, relative=relative, crop_fn=utils.within_bbox)

    def filter(self, test_function):
        return FilteredPage(self, test_function)

    def dedupe_chars(self, **kwargs):
        """
        Removes duplicate chars — those sharing the same text, fontname, size,
        and positioning (within `tolerance`) as other characters on the page.
        """
        p = FilteredPage(self, True)
        p._objects = dict((kind, objs) for kind, objs in self.objects.items())
        p._objects["char"] = utils.dedupe_chars(self.chars, **kwargs)
        return p

    def to_image(self, **conversion_kwargs):
        """
        For conversion_kwargs, see:
        http://docs.wand-py.org/en/latest/wand/image.html#wand.image.Image
        """
        from .display import DEFAULT_RESOLUTION, PageImage

        kwargs = dict(conversion_kwargs)
        if "resolution" not in conversion_kwargs:
            kwargs["resolution"] = DEFAULT_RESOLUTION
        return PageImage(self, **kwargs)

    def __repr__(self):
        return f"<Page:{self.page_number}>"


class DerivedPage(Page):
    is_original = False

    def __init__(self, parent_page):
        self.parent_page = parent_page
        self.pdf = parent_page.pdf
        self.page_obj = parent_page.page_obj
        self.page_number = parent_page.page_number
        self.flush_cache(Container.cached_properties)

        if type(parent_page) == Page:
            self.root_page = parent_page
        else:
            self.root_page = parent_page.root_page


def test_proposed_bbox(bbox, parent_bbox):
    bbox_area = utils.calculate_area(bbox)
    if bbox_area == 0:
        raise ValueError(f"Bounding box {bbox} has an area of zero.")

    overlap = utils.get_bbox_overlap(bbox, parent_bbox)
    if overlap is None:
        raise ValueError(
            f"Bounding box {bbox} is entirely outside "
            f"parent page bounding box {parent_bbox}"
        )

    overlap_area = utils.calculate_area(overlap)
    if overlap_area < bbox_area:
        raise ValueError(
            f"Bounding box {bbox} is not fully within "
            f"parent page bounding box {parent_bbox}"
        )


class CroppedPage(DerivedPage):
    def __init__(self, parent_page, bbox, crop_fn=utils.crop_to_bbox, relative=False):
        if relative:
            o_x0, o_top, _, _ = parent_page.bbox
            x0, top, x1, bottom = bbox
            self.bbox = (x0 + o_x0, top + o_top, x1 + o_x0, bottom + o_top)
        else:
            self.bbox = bbox

        test_proposed_bbox(self.bbox, parent_page.bbox)
        self.crop_fn = crop_fn
        super().__init__(parent_page)

    @property
    def objects(self):
        if hasattr(self, "_objects"):
            return self._objects
        self._objects = self.crop_fn(self.parent_page.objects, self.bbox)
        return self._objects


class FilteredPage(DerivedPage):
    def __init__(self, parent_page, filter_fn):
        self.bbox = parent_page.bbox
        self.filter_fn = filter_fn
        super().__init__(parent_page)

    @property
    def objects(self):
        if hasattr(self, "_objects"):
            return self._objects
        self._objects = utils.filter_objects(self.parent_page.objects, self.filter_fn)
        return self._objects
