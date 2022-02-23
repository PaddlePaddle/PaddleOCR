import base64
import csv
import json
from io import StringIO

from .utils import decode_text

COLS_TO_PREPEND = [
    "object_type",
    "page_number",
    "x0",
    "x1",
    "y0",
    "y1",
    "doctop",
    "top",
    "bottom",
    "width",
    "height",
]

ENCODINGS_TO_TRY = [
    "utf-8",
    "latin-1",
    "utf-16",
    "utf-16le",
]


def to_b64(data_bytes):
    return base64.b64encode(data_bytes).decode("ascii")


class Serializer:
    def __init__(self, precision=None):
        self.precision = precision

    def serialize(self, obj):
        if obj is None:
            return None

        t = type(obj)

        # Basic types don't need to be converted
        if t in (int, str):
            return obj

        # Use one of the custom converters, if possible
        fn = getattr(self, f"do_{t.__name__}", None)
        if fn is not None:
            return fn(obj)

        # Otherwise, just use the string-representation
        else:
            return str(obj)

    def do_float(self, x):
        return x if self.precision is None else round(x, self.precision)

    def do_bool(self, x):
        return int(x)

    def do_list(self, obj):
        return list(self.serialize(x) for x in obj)

    def do_tuple(self, obj):
        return tuple(self.serialize(x) for x in obj)

    def do_dict(self, obj):
        return {k: self.serialize(v) for k, v in obj.items()}

    def do_PDFStream(self, obj):
        return {"rawdata": to_b64(obj.rawdata)}

    def do_PSLiteral(self, obj):
        return decode_text(obj.name)

    def do_bytes(self, obj):
        for e in ENCODINGS_TO_TRY:
            try:
                return obj.decode(e)
            except UnicodeDecodeError:  # pragma: no cover
                pass
        # If none of the decodings work, raise whatever error
        # decoding with utf-8 causes
        obj.decode(ENCODINGS_TO_TRY[0])  # pragma: no cover


def to_json(container, stream=None, types=None, precision=None, indent=None):
    if types is None:
        types = list(container.objects.keys()) + ["annot"]

    def page_to_dict(page):
        d = {
            "page_number": page.page_number,
            "initial_doctop": page.initial_doctop,
            "rotation": page.rotation,
            "cropbox": page.cropbox,
            "mediabox": page.mediabox,
            "bbox": page.bbox,
            "width": page.width,
            "height": page.height,
        }
        for t in types:
            d[t + "s"] = getattr(page, t + "s")
        return d

    if hasattr(container, "pages"):
        data = {
            "metadata": container.metadata,
            "pages": list(map(page_to_dict, container.pages)),
        }
    else:
        data = page_to_dict(container)

    serialized = Serializer(precision=precision).serialize(data)

    if stream is None:
        return json.dumps(serialized, indent=indent)
    else:
        return json.dump(serialized, stream, indent=indent)


def to_csv(container, stream=None, types=None, precision=None):
    if stream is None:
        stream = StringIO()
        to_string = True
    else:
        to_string = False

    if types is None:
        types = list(container.objects.keys()) + ["annot"]

    objs = []
    fields = set()

    pages = container.pages if hasattr(container, "pages") else [container]
    for page in pages:
        for t in types:
            new_objs = getattr(page, t + "s")
            if len(new_objs):
                objs += new_objs
                new_keys = [k for k, v in new_objs[0].items() if type(v) is not dict]
                fields = fields.union(set(new_keys))

    serialized = Serializer(precision=precision).serialize(objs)

    cols = COLS_TO_PREPEND + list(sorted(set(fields) - set(COLS_TO_PREPEND)))

    w = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(serialized)

    if to_string:
        stream.seek(0)
        return stream.read()
