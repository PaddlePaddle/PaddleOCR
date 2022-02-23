"""
    pdf2image custom buffer parsers
"""

from io import BytesIO

from PIL import Image


def parse_buffer_to_ppm(data):
    """Parse PPM file bytes to Pillow Image"""

    images = []

    index = 0

    while index < len(data):
        code, size, rgb = tuple(data[index : index + 40].split(b"\n")[0:3])
        size_x, size_y = tuple(size.split(b" "))
        file_size = len(code) + len(size) + len(rgb) + 3 + int(size_x) * int(size_y) * 3
        images.append(Image.open(BytesIO(data[index : index + file_size])))
        index += file_size

    return images


def parse_buffer_to_pgm(data):
    """Parse PGM file bytes to Pillow Image"""

    images = []

    index = 0

    while index < len(data):
        code, size, maxval = tuple(data[index : index + 40].split(b"\n")[0:3])
        size_x, size_y = tuple(size.split(b" "))
        file_size = len(code) + len(size) + len(maxval) + 3 + int(size_x) * int(size_y)
        images.append(Image.open(BytesIO(data[index : index + file_size])))
        index += file_size

    return images


def parse_buffer_to_jpeg(data):
    """Parse JPEG file bytes to Pillow Image"""

    return [
        Image.open(BytesIO(image_data + b"\xff\xd9"))
        for image_data in data.split(b"\xff\xd9")[
            :-1
        ]  # Last element is obviously empty
    ]


def parse_buffer_to_png(data):
    """Parse PNG file bytes to Pillow Image"""

    images = []

    c1 = 0
    c2 = 0
    data_len = len(data)
    while c1 < data_len:
        # IEND can appear in a PNG without being the actual end
        if data[c2 : c2 + 4] == b"IEND" and (
            c2 + 8 == data_len or data[c2 + 9 : c2 + 12] == b"PNG"
        ):
            images.append(Image.open(BytesIO(data[c1 : c2 + 8])))
            c1 = c2 + 8
            c2 = c1
        c2 += 1

    return images
