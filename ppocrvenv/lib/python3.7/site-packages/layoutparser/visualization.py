from PIL import Image, ImageFont, ImageDraw
from .elements import *
import numpy as np
import functools
import os
import sys
import warnings
import layoutparser
from itertools import cycle

# We need to fix this ugly hack some time in the future
_lib_path = os.path.dirname(sys.modules[layoutparser.__package__].__file__)
_font_path = os.path.join(_lib_path, "misc", "NotoSerifCJKjp-Regular.otf")

DEFAULT_BOX_WIDTH_RATIO = 0.005
DEFAULT_OUTLINE_COLOR = "red"
DEAFULT_COLOR_PALETTE = "#f6bd60-#f7ede2-#f5cac3-#84a59d-#f28482"
# From https://coolors.co/f6bd60-f7ede2-f5cac3-84a59d-f28482

DEFAULT_FONT_PATH = _font_path
DEFAULT_FONT_SIZE = 15
DEFAULT_FONT_OBJECT = ImageFont.truetype(DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE)
DEFAULT_TEXT_COLOR = "black"
DEFAULT_TEXT_BACKGROUND = "white"

__all__ = ["draw_box", "draw_text"]


def _draw_vertical_text(
    text,
    image_font,
    text_color,
    text_background_color,
    character_spacing=2,
    space_width=1,
):
    """Helper function to draw text vertically.
    Ref: https://github.com/Belval/TextRecognitionDataGenerator/blob/7f4c782c33993d2b6f712d01e86a2f342025f2df/trdg/computer_text_generator.py
    """

    space_height = int(image_font.getsize(" ")[1] * space_width)

    char_heights = [
        image_font.getsize(c)[1] if c != " " else space_height for c in text
    ]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGB", (text_width, text_height), color=text_background_color)
    txt_mask = Image.new("RGB", (text_width, text_height), color=text_background_color)

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=text_color,
            font=image_font,
        )

    return txt_img.crop(txt_img.getbbox())


def _calculate_default_box_width(canvas):
    return max(1, int(min(canvas.size) * DEFAULT_BOX_WIDTH_RATIO))


def _create_font_object(font_size=None, font_path=None):

    if font_size is None and font_path is None:
        return DEFAULT_FONT_OBJECT
    else:
        return ImageFont.truetype(
            font_path or DEFAULT_FONT_PATH, font_size or DEFAULT_FONT_SIZE
        )


def _create_new_canvas(canvas, arrangement, text_background_color):

    if arrangement == "lr":
        new_canvas = Image.new(
            "RGB",
            (canvas.width * 2, canvas.height),
            color=text_background_color or DEFAULT_TEXT_BACKGROUND,
        )
        new_canvas.paste(canvas, (canvas.width, 0))

    elif arrangement == "ud":
        new_canvas = Image.new(
            "RGB",
            (canvas.width, canvas.height * 2),
            color=text_background_color or DEFAULT_TEXT_BACKGROUND,
        )
        new_canvas.paste(canvas, (0, canvas.height))

    else:
        raise ValueError(f"Invalid direction {arrangement}")

    return new_canvas


def _create_color_palette(types):
    return {
        type: color
        for type, color in zip(types, cycle(DEAFULT_COLOR_PALETTE.split("-")))
    }


def image_loader(func):
    @functools.wraps(func)
    def wrap(canvas, layout, *args, **kwargs):

        if isinstance(canvas, Image.Image):
            if canvas.mode != "RGB":
                canvas = canvas.convert("RGB")
            canvas = canvas.copy()
        elif isinstance(canvas, np.ndarray):
            canvas = Image.fromarray(canvas)
        out = func(canvas, layout, *args, **kwargs)
        return out

    return wrap


@image_loader
def draw_box(
    canvas,
    layout,
    box_width=None,
    color_map=None,
    show_element_id=False,
    show_element_type=False,
    id_font_size=None,
    id_font_path=None,
    id_text_color=None,
    id_text_background_color=None,
):
    """Draw the layout region on the input canvas(image).

    Args:
        canvas (:obj:`~np.ndarray` or :obj:`~PIL.Image.Image`):
            The canvas to draw the layout boxes.
        layout (:obj:`Layout` or :obj:`list`):
            The layout of the canvas to show.
        box_width (:obj:`int`, optional):
            Set to change the width of the drawn layout box boundary.
            Defaults to None, when the boundary is automatically
            calculated as the the :const:`DEFAULT_BOX_WIDTH_RATIO`
            * the maximum of (height, width) of the canvas.
        color_map (dict, optional):
            A map from `block.type` to the colors, e.g., `{1: 'red'}`.
            You can set it to `{}` to use only the
            :const:`DEFAULT_OUTLINE_COLOR` for the outlines.
            Defaults to None, when a color palette is is automatically
            created based on the input layout.
        show_element_id (bool, optional):
            Whether to display `block.id` on the top-left corner of
            the block.
            Defaults to False.
        show_element_id (bool, optional):
            Whether to display `block.type` on the top-left corner of
            the block.
            Defaults to False.
        id_font_size (int, optional):
            Set to change the font size used for drawing `block.id`.
            Defaults to None, when the size is set to
            :const:`DEFAULT_FONT_SIZE`.
        id_font_path (:obj:`str`, optional):
            Set to change the font used for drawing `block.id`.
            Defaults to None, when the :const:`DEFAULT_FONT_OBJECT` is used.
        id_text_color (:obj:`str`, optional):
            Set to change the text color used for drawing `block.id`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_COLOR`.
        id_text_background_color (:obj:`str`, optional):
            Set to change the text region background used for drawing `block.id`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_BACKGROUND`.
    Returns:
        :obj:`PIL.Image.Image`:
            A Image object containing the `layout` draw upon the input `canvas`.
    """

    draw = ImageDraw.Draw(canvas)

    if box_width is None:
        box_width = _calculate_default_box_width(canvas)

    if show_element_id or show_element_type:
        font_obj = _create_font_object(id_font_size, id_font_path)

    if color_map is None:
        all_types = set([b.type for b in layout if hasattr(b, "type")])
        color_map = _create_color_palette(all_types)

    for idx, ele in enumerate(layout):

        if isinstance(ele, Interval):
            ele = ele.put_on_canvas(canvas)

        outline_color = (
            DEFAULT_OUTLINE_COLOR
            if not isinstance(ele, TextBlock)
            else color_map.get(ele.type, DEFAULT_OUTLINE_COLOR)
        )

        if not isinstance(ele, Quadrilateral):
            draw.rectangle(ele.coordinates, width=box_width, outline=outline_color)

        else:
            p = ele.points.ravel().tolist()
            draw.line(p + p[:2], width=box_width, fill=outline_color)

        if show_element_id or show_element_type:
            text = ""
            if show_element_id:
                ele_id = ele.id or idx
                text += str(ele_id)
            if show_element_type:
                text = str(ele.type) if not text else text + ": " + str(ele.type)

            start_x, start_y = ele.coordinates[:2]
            text_w, text_h = font_obj.getsize(text)

            # Add a small background for the text
            draw.rectangle(
                (start_x, start_y, start_x + text_w, start_y + text_h),
                fill=id_text_background_color or DEFAULT_TEXT_BACKGROUND,
            )

            # Draw the ids
            draw.text(
                (start_x, start_y),
                text,
                fill=id_text_color or DEFAULT_TEXT_COLOR,
                font=font_obj,
            )

    return canvas


@image_loader
def draw_text(
    canvas,
    layout,
    arrangement="lr",
    font_size=None,
    font_path=None,
    text_color=None,
    text_background_color=None,
    vertical_text=False,
    with_box_on_text=False,
    text_box_width=None,
    text_box_color=None,
    with_layout=False,
    **kwargs,
):
    """Draw the (detected) text in the `layout` according to
    their coordinates next to the input `canvas` (image) for better comparison.

    Args:
        canvas (:obj:`~np.ndarray` or :obj:`~PIL.Image.Image`):
            The canvas to draw the layout boxes.
        layout (:obj:`Layout` or :obj:`list`):
            The layout of the canvas to show.
        arrangement (`{'lr', 'ud'}`, optional):
            The arrangement of the drawn text canvas and the original
            image canvas:
            * `lr` - left and right
            * `ud` - up and down

            Defaults to 'lr'.
        font_size (:obj:`str`, optional):
            Set to change the size of the font used for
            drawing `block.text`.
            Defaults to None, when the size is set to
            :const:`DEFAULT_FONT_SIZE`.
        font_path (:obj:`str`, optional):
            Set to change the font used for drawing `block.text`.
            Defaults to None, when the :const:`DEFAULT_FONT_OBJECT` is used.
        text_color ([type], optional):
            Set to change the text color used for drawing `block.text`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_COLOR`.
        text_background_color ([type], optional):
            Set to change the text region background used for drawing
            `block.text`.
            Defaults to None, when the color is set to
            :const:`DEFAULT_TEXT_BACKGROUND`.
        vertical_text (bool, optional):
            Whether the text in a block should be drawn vertically.
            Defaults to False.
        with_box_on_text (bool, optional):
            Whether to draw the layout box boundary of a text region
            on the text canvas.
            Defaults to False.
        text_box_width (:obj:`int`, optional):
            Set to change the width of the drawn layout box boundary.
            Defaults to None, when the boundary is automatically
            calculated as the the :const:`DEFAULT_BOX_WIDTH_RATIO`
            * the maximum of (height, width) of the canvas.
        text_box_color (:obj:`int`, optional):
            Set to change the color of the drawn layout box boundary.
            Defaults to None, when the color is set to
            :const:`DEFAULT_OUTLINE_COLOR`.
        with_layout (bool, optional):
            Whether to draw the layout boxes on the input (image) canvas.
            Defaults to False.
            When set to true, you can pass in the arguments in
            :obj:`draw_box` to change the style of the drawn layout boxes.

    Returns:
        :obj:`PIL.Image.Image`:
            A Image object containing the drawn text from `layout`.
    """
    if with_box_on_text:
        if text_box_width is None:
            text_box_width = _calculate_default_box_width(canvas)

    if with_layout:
        canvas = draw_box(canvas, layout, **kwargs)

    font_obj = _create_font_object(font_size, font_path)
    text_box_color = text_box_color or DEFAULT_OUTLINE_COLOR
    text_color = text_color or DEFAULT_TEXT_COLOR
    text_background_color = text_background_color or DEFAULT_TEXT_BACKGROUND

    canvas = _create_new_canvas(canvas, arrangement, text_background_color)
    draw = ImageDraw.Draw(canvas)

    for idx, ele in enumerate(layout):

        if with_box_on_text:
            p = (
                ele.pad(right=text_box_width, bottom=text_box_width)
                .points.ravel()
                .tolist()
            )

            draw.line(p + p[:2], width=text_box_width, fill=text_box_color)

        if not hasattr(ele, "text") or ele.text == "":
            continue

        (start_x, start_y) = ele.coordinates[:2]
        if not vertical_text:
            draw.text((start_x, start_y), ele.text, font=font_obj, fill=text_color)
        else:
            text_segment = _draw_vertical_text(
                ele.text, font_obj, text_color, text_background_color
            )

            if with_box_on_text:
                # Avoid cover the box regions
                canvas.paste(
                    text_segment, (start_x + text_box_width, start_y + text_box_width)
                )
            else:
                canvas.paste(text_segment, (start_x, start_y))

    return canvas
