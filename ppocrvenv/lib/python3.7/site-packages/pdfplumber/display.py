from io import BytesIO

import PIL.Image
import PIL.ImageDraw
import wand.image

from . import utils
from .table import TableFinder


class COLORS(object):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    TRANSPARENT = (0, 0, 0, 0)


DEFAULT_FILL = COLORS.BLUE + (50,)
DEFAULT_STROKE = COLORS.RED + (200,)
DEFAULT_STROKE_WIDTH = 1
DEFAULT_RESOLUTION = 72


def get_page_image(stream, page_no, resolution):
    # If we are working with a file object saved to disk
    if hasattr(stream, "name"):
        spec = dict(filename=f"{stream.name}[{page_no}]")

        def postprocess(img):
            return img

    # If we instead are working with a BytesIO stream
    else:
        stream.seek(0)
        spec = dict(file=stream)

        def postprocess(img):
            return wand.image.Image(image=img.sequence[page_no])

    with wand.image.Image(resolution=resolution, **spec) as img_init:
        img = postprocess(img_init)
        if img.alpha_channel:
            img.background_color = wand.image.Color("white")
            img.alpha_channel = "remove"
        with img.convert("png") as png:
            im = PIL.Image.open(BytesIO(png.make_blob()))
            return im.convert("RGB")


class PageImage(object):
    def __init__(self, page, original=None, resolution=DEFAULT_RESOLUTION):
        self.page = page
        if original is None:
            self.original = get_page_image(
                page.pdf.stream, page.page_number - 1, resolution
            )
        else:
            self.original = original

        if page.is_original:
            self.root = page
            cropped = False
        else:
            self.root = page.root_page
            cropped = page.root_page.bbox != page.bbox
        self.scale = self.original.size[0] / self.root.width
        if cropped:
            cropbox = (
                (page.bbox[0] - page.root_page.bbox[0]) * self.scale,
                (page.bbox[1] - page.root_page.bbox[1]) * self.scale,
                (page.bbox[2] - page.root_page.bbox[0]) * self.scale,
                (page.bbox[3] - page.root_page.bbox[1]) * self.scale,
            )
            self.original = self.original.crop(map(int, cropbox))
        self.reset()

    def _reproject_bbox(self, bbox):
        x0, top, x1, bottom = bbox
        _x0, _top = self._reproject((x0, top))
        _x1, _bottom = self._reproject((x1, bottom))
        return (_x0, _top, _x1, _bottom)

    def _reproject(self, coord):
        """
        Given an (x0, top) tuple from the *root* coordinate system,
        return an (x0, top) tuple in the *image* coordinate system.
        """
        x0, top = coord
        px0, ptop = self.page.bbox[:2]
        rx0, rtop = self.root.bbox[:2]
        _x0 = (x0 + rx0 - px0) * self.scale
        _top = (top + rtop - ptop) * self.scale
        return (_x0, _top)

    def reset(self):
        self.annotated = PIL.Image.new(self.original.mode, self.original.size)
        self.annotated.paste(self.original)
        self.draw = PIL.ImageDraw.Draw(self.annotated, "RGBA")
        return self

    def copy(self):
        return self.__class__(self.page, self.original)

    def draw_line(
        self, points_or_obj, stroke=DEFAULT_STROKE, stroke_width=DEFAULT_STROKE_WIDTH
    ):
        if isinstance(points_or_obj, (tuple, list)):
            points = points_or_obj
        elif type(points_or_obj) == dict and "points" in points_or_obj:
            points = points_or_obj["points"]
        else:
            obj = points_or_obj
            points = ((obj["x0"], obj["top"]), (obj["x1"], obj["bottom"]))
        self.draw.line(
            list(map(self._reproject, points)), fill=stroke, width=stroke_width
        )
        return self

    def draw_lines(self, list_of_lines, **kwargs):
        for x in utils.to_list(list_of_lines):
            self.draw_line(x, **kwargs)
        return self

    def draw_vline(
        self, location, stroke=DEFAULT_STROKE, stroke_width=DEFAULT_STROKE_WIDTH
    ):
        points = (location, self.page.bbox[1], location, self.page.bbox[3])
        self.draw.line(self._reproject_bbox(points), fill=stroke, width=stroke_width)
        return self

    def draw_vlines(self, locations, **kwargs):
        for x in utils.to_list(locations):
            self.draw_vline(x, **kwargs)
        return self

    def draw_hline(
        self, location, stroke=DEFAULT_STROKE, stroke_width=DEFAULT_STROKE_WIDTH
    ):
        points = (self.page.bbox[0], location, self.page.bbox[2], location)
        self.draw.line(self._reproject_bbox(points), fill=stroke, width=stroke_width)
        return self

    def draw_hlines(self, locations, **kwargs):
        for x in utils.to_list(locations):
            self.draw_hline(x, **kwargs)
        return self

    def draw_rect(
        self,
        bbox_or_obj,
        fill=DEFAULT_FILL,
        stroke=DEFAULT_STROKE,
        stroke_width=DEFAULT_STROKE_WIDTH,
    ):
        if isinstance(bbox_or_obj, (tuple, list)):
            bbox = bbox_or_obj
        else:
            obj = bbox_or_obj
            bbox = (obj["x0"], obj["top"], obj["x1"], obj["bottom"])

        x0, top, x1, bottom = bbox
        half = stroke_width / 2
        x0 += half
        top += half
        x1 -= half
        bottom -= half

        self.draw.rectangle(
            self._reproject_bbox((x0, top, x1, bottom)), fill, COLORS.TRANSPARENT
        )

        if stroke_width > 0:
            segments = [
                ((x0, top), (x1, top)),  # top
                ((x0, bottom), (x1, bottom)),  # bottom
                ((x0, top), (x0, bottom)),  # left
                ((x1, top), (x1, bottom)),  # right
            ]
            self.draw_lines(segments, stroke=stroke, stroke_width=stroke_width)
        return self

    def draw_rects(self, list_of_rects, **kwargs):
        for x in utils.to_list(list_of_rects):
            self.draw_rect(x, **kwargs)
        return self

    def draw_circle(
        self, center_or_obj, radius=5, fill=DEFAULT_FILL, stroke=DEFAULT_STROKE
    ):
        if isinstance(center_or_obj, (tuple, list)):
            center = center_or_obj
        else:
            obj = center_or_obj
            center = ((obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2)
        cx, cy = center
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        self.draw.ellipse(self._reproject_bbox(bbox), fill, stroke)
        return self

    def draw_circles(self, list_of_circles, **kwargs):
        for x in utils.to_list(list_of_circles):
            self.draw_circle(x, **kwargs)
        return self

    def save(self, *args, **kwargs):
        return self.annotated.save(*args, **kwargs)

    def debug_table(
        self, table, fill=DEFAULT_FILL, stroke=DEFAULT_STROKE, stroke_width=1
    ):
        """
        Outline all found tables.
        """
        self.draw_rects(
            table.cells, fill=fill, stroke=stroke, stroke_width=stroke_width
        )
        return self

    def debug_tablefinder(self, tf={}):
        if isinstance(tf, TableFinder):
            pass
        elif isinstance(tf, dict):
            tf = self.page.debug_tablefinder(tf)
        else:
            raise ValueError(
                "Argument must be instance of TableFinder"
                "or a TableFinder settings dict."
            )

        for table in tf.tables:
            self.debug_table(table)

        self.draw_lines(tf.edges, stroke_width=1)

        self.draw_circles(
            tf.intersections.keys(),
            fill=COLORS.TRANSPARENT,
            stroke=COLORS.BLUE + (200,),
            radius=3,
        )
        return self

    def outline_words(
        self,
        stroke=DEFAULT_STROKE,
        fill=DEFAULT_FILL,
        stroke_width=DEFAULT_STROKE_WIDTH,
        x_tolerance=utils.DEFAULT_X_TOLERANCE,
        y_tolerance=utils.DEFAULT_Y_TOLERANCE,
    ):

        words = self.page.extract_words(
            x_tolerance=x_tolerance, y_tolerance=y_tolerance
        )
        self.draw_rects(words, stroke=stroke, fill=fill, stroke_width=stroke_width)
        return self

    def outline_chars(
        self,
        stroke=(255, 0, 0, 255),
        fill=(255, 0, 0, int(255 / 4)),
        stroke_width=DEFAULT_STROKE_WIDTH,
    ):

        self.draw_rects(
            self.page.chars, stroke=stroke, fill=fill, stroke_width=stroke_width
        )
        return self

    def _repr_png_(self):
        b = BytesIO()
        self.annotated.save(b, "PNG")
        return b.getvalue()
